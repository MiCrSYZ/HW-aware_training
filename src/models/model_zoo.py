"""
Model factory and memristor wrapping utilities.

This module provides functions to create models by name and wrap them
with memristor-aware layers.
"""

import torch.nn as nn
from typing import Optional

try:
    from .resnet20 import ResNet20
    from .vit_tiny import ViTTiny
    from .memristor_wrappers import MemristorLinear, MemristorConv2d
    from ..memristor.device_model import MemristorDeviceModel
    from ..memristor.learned_weight_mapping import (
        MemristorLinear as LearnedMappingMemristorLinear,
        MemristorConv2d as LearnedMappingMemristorConv2d
    )
except ImportError:
    from src.models.resnet20 import ResNet20
    from src.models.vit_tiny import ViTTiny
    from src.models.memristor_wrappers import MemristorLinear, MemristorConv2d
    from src.memristor.device_model import MemristorDeviceModel
    from src.memristor.learned_weight_mapping import (
        MemristorLinear as LearnedMappingMemristorLinear,
        MemristorConv2d as LearnedMappingMemristorConv2d
    )


def get_model(
    name: str,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        name: Model name ('resnet20' or 'vit_tiny')
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
            - in_channels: Input channels (for ResNet20, default: 3 for RGB, 1 for grayscale)
        
    Returns:
        Model instance
    """
    if name == 'resnet20':
        in_channels = kwargs.get('in_channels', 3)  # Default to 3 for CIFAR-10
        return ResNet20(num_classes=num_classes, in_channels=in_channels)
    elif name == 'vit_tiny':
        return ViTTiny(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Available: 'resnet20', 'vit_tiny'")


def wrap_model_with_memristor(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    use_learned_mapping: bool = False,
    mapping_max_frac: float = 0.5,
) -> nn.Module:
    """
    Wrap a model's layers with memristor-aware versions.
    
    This function recursively replaces nn.Linear and nn.Conv2d layers
    with MemristorLinear and MemristorConv2d equivalents.
    
    Args:
        model: Base model to wrap
        device_model: MemristorDeviceModel instance
        use_learned_mapping: If True, use learned_weight_mapping.py classes that support mapping_net
        mapping_max_frac: Max fraction for delta clamping in learned mapping (default: 0.5)
        
    Returns:
        Model with memristor-aware layers
    """
    # Select which classes to use based on use_learned_mapping flag
    if use_learned_mapping:
        MemristorLinearCls = LearnedMappingMemristorLinear
        MemristorConv2dCls = LearnedMappingMemristorConv2d
    else:
        MemristorLinearCls = MemristorLinear
        MemristorConv2dCls = MemristorConv2d
    
    def _replace_in_module(module):
        """
        Recursively replace Linear and Conv2d layers with memristor versions.
        
        This function modifies the module in-place by replacing its Linear/Conv2d
        children with MemristorLinear/MemristorConv2d equivalents.
        
        Args:
            module: Module to process (will be modified in-place)
        """
        # Get all direct children (not recursive)
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Replace Linear layer
                if use_learned_mapping:
                    setattr(module, name, MemristorLinearCls(child, device_model, mapping_max_frac=mapping_max_frac))
                else:
                    setattr(module, name, MemristorLinearCls(child, device_model))
            elif isinstance(child, nn.Conv2d):
                # Replace Conv2d layer
                if use_learned_mapping:
                    setattr(module, name, MemristorConv2dCls(child, device_model, mapping_max_frac=mapping_max_frac))
                else:
                    setattr(module, name, MemristorConv2dCls(child, device_model))
            else:
                # Recursively process child modules (Sequential, ModuleList, BasicBlock, etc.)
                _replace_in_module(child)
    
    # Create a wrapper class to maintain model structure
    class MemristorModel(nn.Module):
        def __init__(self, base_model, device_model):
            super().__init__()
            self.device_model = device_model
            # Clone the model structure
            import copy
            self.base_model = copy.deepcopy(base_model)
            # Recursively replace all Linear and Conv2d layers
            _replace_in_module(self.base_model)
        
        def forward(self, x, t=0, seed=None, enable_sanity_check=False):
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
            
            return self._forward_with_t(self.base_model, x, t, seed, enable_sanity_check, self._sanity_check_counter)
        
        def _forward_with_t(self, module, x, t, seed, enable_sanity_check=False, sanity_check_counter=None):
            """Recursively forward with t parameter and enable_sanity_check where supported."""
            if sanity_check_counter is None:
                sanity_check_counter = [0]
            
            # Check for both standard and learned mapping memristor layers
            is_memristor_linear = isinstance(module, (MemristorLinear, LearnedMappingMemristorLinear))
            is_memristor_conv2d = isinstance(module, (MemristorConv2d, LearnedMappingMemristorConv2d))
            if is_memristor_linear or is_memristor_conv2d:
                # Only enable sanity check for the first memristor layer to avoid too much output
                check_this_layer = enable_sanity_check and (sanity_check_counter[0] == 0)
                if check_this_layer:
                    sanity_check_counter[0] += 1
                return module(x, t=t, seed=seed, enable_sanity_check=check_this_layer)
            elif isinstance(module, nn.Sequential):
                for m in module:
                    x = self._forward_with_t(m, x, t, seed, enable_sanity_check, sanity_check_counter)
                return x
            elif isinstance(module, nn.ModuleList):
                for m in module:
                    x = self._forward_with_t(m, x, t, seed, enable_sanity_check, sanity_check_counter)
                return x
            else:
                # For other modules, try to forward with t if supported
                if hasattr(module, 'forward') and 't' in module.forward.__code__.co_varnames:
                    # Check if it also supports enable_sanity_check
                    if 'enable_sanity_check' in module.forward.__code__.co_varnames:
                        return module(x, t=t, seed=seed, enable_sanity_check=enable_sanity_check)
                    else:
                        return module(x, t=t, seed=seed)
                else:
                    return module(x)
    
    wrapped_model = MemristorModel(model, device_model)
    
    # DEBUG CODE COMMENTED OUT
    # # Debug: Print all modules with mapping_net attribute
    # # Check both wrapped_model.modules() and wrapped_model.base_model.modules()
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

