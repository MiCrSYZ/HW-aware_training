"""
Model factory and memristor wrapping utilities.

This module provides functions to create models by name and wrap them
with memristor-aware layers.
"""

import torch.nn as nn
from typing import Optional, Dict, Any

try:
    from .resnet20 import ResNet20
    from .vit_tiny import ViTTiny
    from .gru_agnews import GRUAGNews
    from ..memristor.memristor_wrappers import MemristorLinear, MemristorConv2d
    from ..memristor.memristor_gru import MemristorGRU
    from ..memristor.device_model import MemristorDeviceModel
except ImportError:
    from src.models.resnet20 import ResNet20
    from src.models.vit_tiny import ViTTiny
    from src.models.gru_agnews import GRUAGNews
    from src.memristor.memristor_wrappers import MemristorLinear, MemristorConv2d
    from src.memristor.memristor_gru import MemristorGRU
    from src.memristor.device_model import MemristorDeviceModel


def get_model(
    name: str,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        name: Model name ('resnet20', 'vit_tiny', or 'gru_agnews')
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
            - in_channels: Input channels (for ResNet20, default: 3 for RGB, 1 for grayscale)
            - vocab_size: Vocabulary size (for GRU, required)
            - embed_dim: Embedding dimension (for GRU, default: 128)
            - hidden_dim: Hidden dimension (for GRU, default: 256)
            - num_layers: Number of layers (for GRU, default: 2)
        
    Returns:
        Model instance
    """
    if name == 'resnet20':
        in_channels = kwargs.get('in_channels', 3)  # Default to 3 for CIFAR-10
        return ResNet20(num_classes=num_classes, in_channels=in_channels)
    elif name == 'vit_tiny':
        return ViTTiny(num_classes=num_classes, **kwargs)
    elif name == 'gru_agnews':
        vocab_size = kwargs.get('vocab_size', None)
        if vocab_size is None:
            raise ValueError("vocab_size is required for GRU model")
        embed_dim = kwargs.get('embed_dim', 128)
        hidden_dim = kwargs.get('hidden_dim', 256)
        num_layers = kwargs.get('num_layers', 2)
        return GRUAGNews(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model: {name}. Available: 'resnet20', 'vit_tiny', 'gru_agnews'")


def _should_inject_noise_for_layer(
    full_name: str,
    layer_type: str,  # 'linear' or 'conv2d'
    noise_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Determine if noise should be injected for a given layer based on configuration.
    
    Args:
        full_name: Full layer name (e.g., 'patch_embed.proj', 'blocks.0.attn.qkv')
        layer_type: Type of layer ('linear' or 'conv2d')
        noise_config: Noise injection configuration from config file
        
    Returns:
        True if noise should be injected, False otherwise
    """
    # If no noise config provided, default to injecting noise everywhere (backward compatibility)
    if noise_config is None:
        return True
    
    # Helper: get enable from section that can be bool or dict with enable_noise/enable
    def _get_enable_resnet(section: Any) -> bool:
        if section is None:
            return True
        if isinstance(section, bool):
            return section
        if isinstance(section, dict):
            return bool(section.get('enable_noise', section.get('enable', True)))
        return True
    
    # ResNet (CIFAR ResNet20: stem=conv1, layer1/layer2/layer3, head=linear)
    if full_name == 'conv1' and layer_type == 'conv2d':
        return _get_enable_resnet(noise_config.get('stem'))
    if full_name == 'linear' and layer_type == 'linear':
        return _get_enable_resnet(noise_config.get('head'))
    # ResNet shortcut (1x1 conv in layer2.0 / layer3.0): when disabled, keeps residual path clean (ablation)
    if 'shortcut' in full_name and layer_type == 'conv2d':
        sc = noise_config.get('shortcut')
        if sc is False:
            return False
        if isinstance(sc, dict) and not _get_enable_resnet(sc):
            return False
    if 'layer1' in full_name:
        return _get_enable_resnet(noise_config.get('layer1'))
    if 'layer2' in full_name:
        return _get_enable_resnet(noise_config.get('layer2'))
    if 'layer3' in full_name:
        return _get_enable_resnet(noise_config.get('layer3'))
    # 仅当配置为“纯 ResNet”时，未匹配到的层（如 bn/avgpool）才 return False；
    # 若配置里同时有 ViT 键（patch_embed/attn/mlp），说明是 ViT，不要在这里 return False，交给下面 ViT 分支
    if any(k in noise_config for k in ('stem', 'layer1', 'layer2', 'layer3', 'head')):
        if not any(k in noise_config for k in ('patch_embed', 'attn', 'mlp')):
            return False

    # Helper: prefer generic enable_noise/enable (for synth), fallback to enable_ir_drop (memristor)
    def _get_enable(cfg: dict, key_ir_drop: str, key_adc: str = None) -> bool:
        if not cfg:
            return True
        # Synth / generic keys (no IR-drop/ADC naming)
        if 'enable_noise' in cfg:
            return cfg.get('enable_noise', True)
        if 'enable' in cfg:
            return cfg.get('enable', True)
        # Memristor keys
        v = cfg.get(key_ir_drop, True)
        if v is not None:
            return v
        if key_adc is not None:
            return cfg.get(key_adc, True)
        return True

    # Check patch_embed (Conv2d)
    if 'patch_embed' in full_name and layer_type == 'conv2d':
        patch_config = noise_config.get('patch_embed', {})
        if patch_config:
            return _get_enable(patch_config, 'enable_ir_drop', 'enable_adc')
        return True
    
    # Check attention layers (Linear)
    if 'attn' in full_name and layer_type == 'linear':
        attn_config = noise_config.get('attn', {})
        if attn_config:
            if 'qkv' in full_name:
                return attn_config.get('enable_noise_qkv', attn_config.get('enable_adc_qkv', True))
            elif 'proj' in full_name:
                return attn_config.get('enable_noise_w_o', attn_config.get('enable_ir_drop_w_o', True))
            else:
                return False
        return True
    
    # Check MLP layers (Linear)
    if 'mlp' in full_name and layer_type == 'linear':
        mlp_config = noise_config.get('mlp', {})
        if mlp_config:
            return _get_enable(mlp_config, 'enable_ir_drop', 'enable_adc')
        return True
    
    # ViT classification head (same as ResNet head: allow configurable injection)
    if full_name == 'head' and layer_type == 'linear':
        head_cfg = noise_config.get('head')
        if head_cfg is None:
            return True  # default: inject so ViT "all layers" matches ResNet
        return _get_enable_resnet(head_cfg) if isinstance(head_cfg, dict) else bool(head_cfg)
    
    # Other layers (e.g. norm, non-matched names): no noise
    return False


def wrap_model_with_memristor(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    noise_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Wrap a model's layers with memristor-aware versions.
    
    This function recursively replaces nn.Linear and nn.Conv2d layers
    with MemristorLinear and MemristorConv2d equivalents.
    
    Args:
        model: Base model to wrap
        device_model: MemristorDeviceModel instance
        noise_config: Optional noise injection configuration dict from config file.
                     If None, all layers will have noise injected (backward compatibility).
                     Structure: {
                         'patch_embed': {'enable_ir_drop': bool, 'enable_adc': bool},
                         'attn': {'enable_adc_qkv': bool, 'enable_ir_drop_w_o': bool, 'enable_ir_drop_w_v': bool},
                         'mlp': {'enable_ir_drop': bool, 'enable_adc': bool}
                     }
        
    Returns:
        Model with memristor-aware layers
    """
    MemristorLinearCls = MemristorLinear
    MemristorConv2dCls = MemristorConv2d
    
    # Check if model is GRU-based
    is_gru_model = False
    for module in model.modules():
        if isinstance(module, nn.GRU):
            is_gru_model = True
            break
    
    def _replace_in_module(module, parent_name=''):
        """
        Recursively replace Linear, Conv2d, and GRU layers with memristor versions.
        
        This function modifies the module in-place by replacing its Linear/Conv2d/GRU
        children with MemristorLinear/MemristorConv2d/MemristorGRU equivalents.
        
        Args:
            module: Module to process (will be modified in-place)
            parent_name: Full name of parent module (for building full layer names)
        """
        # Get all direct children (not recursive)
        for name, child in list(module.named_children()):
            # Build full name for this layer
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Linear):
                # Determine if noise should be injected for this layer
                enable_noise = _should_inject_noise_for_layer(full_name, 'linear', noise_config)
                # Replace Linear layer
                setattr(module, name, MemristorLinearCls(child, device_model, enable_noise=enable_noise))
            elif isinstance(child, nn.Conv2d):
                # Determine if noise should be injected for this layer
                enable_noise = _should_inject_noise_for_layer(full_name, 'conv2d', noise_config)
                # Replace Conv2d layer
                setattr(module, name, MemristorConv2dCls(child, device_model, enable_noise=enable_noise))
            elif isinstance(child, nn.GRU):
                # For GRU, always enable both weight and output noise
                # The noise_config can control which types are applied via device_model.synthetic_noise_type
                enable_weight_noise = True
                enable_output_noise = True
                # Replace GRU layer
                setattr(module, name, MemristorGRU(child, device_model, enable_weight_noise, enable_output_noise))
            else:
                # Recursively process child modules (Sequential, ModuleList, BasicBlock, etc.)
                _replace_in_module(child, full_name)

    # In-place replace: modify the given model so we only have one copy of parameters.
    # This avoids deepcopy cost and double memory; base_model and memristor_model.base_model
    # become the same object, so no_comp training is fast and sync is a no-op when needed.
    _replace_in_module(model, parent_name='')

    class MemristorModel(nn.Module):
        def __init__(self, base_model, device_model):
            super().__init__()
            self.device_model = device_model
            self.base_model = base_model  # same reference (already in-place replaced above)
        
        def forward(self, x, t=0, seed=None, enable_sanity_check=False, lengths=None):
            # Forward with time parameter and enable_sanity_check
            # Use a counter to only print from the first memristor layer
            self._sanity_check_counter = [0]  # Use list to allow modification in nested calls
            
            # 如果启用累加模式的电导漂移，增加推理次数计数器
            if hasattr(self.device_model, 'drift_time_mode') and self.device_model.drift_time_mode == 'accumulate':
                self.device_model.increment_inference_count()
            
            # DEBUG: Check mapping_net in base_model before forward
            if enable_sanity_check and self._sanity_check_counter[0] == 0:
                print("\n" + "=" * 60)
                print("DEBUG: MemristorModel.forward - checking mapping_net in base_model")
                print("=" * 60)
                print(f"self.base_model id: {id(self.base_model)}")
                mapping_net_count = 0
                for m in self.base_model.modules():
                    if hasattr(m, 'mapping_net'):
                        mapping_net_count += 1
                        mn = getattr(m, 'mapping_net', None)
                        print(f"  {type(m).__name__}: mapping_net = {type(mn).__name__ if mn is not None else 'None'} (id={id(mn) if mn is not None else None})")
                print(f"Total layers with mapping_net: {mapping_net_count}")
                print("=" * 60 + "\n")
            
            return self._forward_with_t(self.base_model, x, t, seed, enable_sanity_check, self._sanity_check_counter, lengths=lengths)
        
        def _forward_with_t(self, module, x, t, seed, enable_sanity_check=False, sanity_check_counter=None, lengths=None):
            """Recursively forward with t parameter and enable_sanity_check where supported."""
            if sanity_check_counter is None:
                sanity_check_counter = [0]
            
            # Check for memristor layers
            is_memristor_linear = isinstance(module, MemristorLinear)
            is_memristor_conv2d = isinstance(module, MemristorConv2d)
            is_memristor_gru = isinstance(module, MemristorGRU)
            if is_memristor_linear or is_memristor_conv2d:
                # Only enable sanity check for the first memristor layer to avoid too much output
                check_this_layer = enable_sanity_check and (sanity_check_counter[0] == 0)
                if check_this_layer:
                    sanity_check_counter[0] += 1
                return module(x, t=t, seed=seed, enable_sanity_check=check_this_layer)
            elif is_memristor_gru:
                # GRU forward with t, seed, and lengths
                return module(x, hx=None, t=t, seed=seed)
            elif isinstance(module, nn.Sequential):
                for m in module:
                    x = self._forward_with_t(m, x, t, seed, enable_sanity_check, sanity_check_counter, lengths=lengths)
                return x
            elif isinstance(module, nn.ModuleList):
                for m in module:
                    x = self._forward_with_t(m, x, t, seed, enable_sanity_check, sanity_check_counter, lengths=lengths)
                return x
            else:
                # For other modules (like GRUAGNews), try to forward with lengths if supported
                if hasattr(module, 'forward'):
                    forward_code = module.forward.__code__
                    forward_varnames = forward_code.co_varnames
                    # Check if forward supports lengths parameter
                    if 'lengths' in forward_varnames:
                        # Check if it also supports t parameter
                        if 't' in forward_varnames:
                            return module(x, lengths=lengths, t=t, seed=seed)
                        else:
                            return module(x, lengths=lengths)
                    elif 't' in forward_varnames:
                        # Check if it also supports enable_sanity_check
                        if 'enable_sanity_check' in forward_varnames:
                            return module(x, t=t, seed=seed, enable_sanity_check=enable_sanity_check)
                        else:
                            return module(x, t=t, seed=seed)
                    else:
                        return module(x)
                else:
                    return module(x)
    
    wrapped_model = MemristorModel(model, device_model)

    # Debug: Print all modules with mapping_net attribute
    # Check both wrapped_model.modules() and wrapped_model.base_model.modules()
    # print("\n" + "=" * 60)
    # print("DEBUG: Checking mapping_net in wrapped model")
    # print("=" * 60)
    # print("Checking wrapped_model.modules():")
    # num_with_mapping_net = 0
    # for m in wrapped_model.modules():
    #     if hasattr(m, 'mapping_net'):
    #         num_with_mapping_net += 1
    #         mapping_net_value = m.mapping_net
    #         mapping_net_str = f"{type(mapping_net_value).__name__}" if mapping_net_value is not None else "None"
    #         print(f"  {type(m).__name__}: mapping_net = {mapping_net_str}")
    #         if mapping_net_value is not None and hasattr(mapping_net_value, 'alpha'):
    #             print(f"    -> alpha = {mapping_net_value.alpha}")
    # print(f"Total in wrapped_model.modules(): {num_with_mapping_net}")
    # 
    # print("\nChecking wrapped_model.base_model.modules():")
    # num_in_base = 0
    # memristor_layers_found = []
    # for m in wrapped_model.base_model.modules():
    #     if hasattr(m, 'mapping_net'):
    #         num_in_base += 1
    #         mapping_net_value = m.mapping_net
    #         mapping_net_str = f"{type(mapping_net_value).__name__}" if mapping_net_value is not None else "None"
    #         print(f"  {type(m).__name__}: mapping_net = {mapping_net_str}")
    #         if mapping_net_value is not None and hasattr(mapping_net_value, 'alpha'):
    #             print(f"    -> alpha = {mapping_net_value.alpha}")
    #     # Also check if it's a Memristor layer
    #     if 'Memristor' in type(m).__name__:
    #         memristor_layers_found.append(type(m).__name__)
    # print(f"Total in base_model.modules() with mapping_net: {num_in_base}")
    # print(f"Total Memristor layers found in base_model: {len(memristor_layers_found)}")
    # if len(memristor_layers_found) > 0:
    #     print(f"  Types: {set(memristor_layers_found)}")
    # 
    # # CRITICAL TEST: Try to set mapping_net directly and verify
    # print("\nTEST: Trying to set mapping_net directly on base_model layers...")
    # test_mapping_net = type('TestMappingNet', (), {'alpha': 999.0})()  # Dummy object for testing
    # test_set_count = 0
    # for m in wrapped_model.base_model.modules():
    #     if hasattr(m, 'set_learned_mapping'):
    #         test_set_count += 1
    #         m.set_learned_mapping(test_mapping_net)
    #         # Verify
    #         if getattr(m, 'mapping_net', None) is test_mapping_net:
    #             print(f"  ✓ Successfully set test mapping_net on {type(m).__name__}")
    #         else:
    #             print(f"  ✗ FAILED to set test mapping_net on {type(m).__name__}!")
    # print(f"Test: Found {test_set_count} layers with set_learned_mapping, tried to set test mapping_net")
    # 
    # # Verify test mapping_net is still there
    # print("\nVerifying test mapping_net after setting...")
    # test_verify_count = 0
    # for m in wrapped_model.base_model.modules():
    #     if hasattr(m, 'mapping_net'):
    #         test_verify_count += 1
    #         mapping_net_value = getattr(m, 'mapping_net', None)
    #         is_test = mapping_net_value is test_mapping_net
    #         status = "✓" if is_test else "✗"
    #         print(f"  {status} {type(m).__name__}: mapping_net = {type(mapping_net_value).__name__ if mapping_net_value is not None else 'None'}")
    # print(f"Verified: {test_verify_count} layers, {sum(1 for m in wrapped_model.base_model.modules() if hasattr(m, 'mapping_net') and getattr(m, 'mapping_net', None) is test_mapping_net)} correctly set")
    # 
    # # Reset test mapping_net back to None
    # for m in wrapped_model.base_model.modules():
    #     if hasattr(m, 'set_learned_mapping'):
    #         m.set_learned_mapping(None)
    # 
    # print("=" * 60 + "\n")
    
    return wrapped_model

