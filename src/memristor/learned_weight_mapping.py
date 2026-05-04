import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging
import random

logger = logging.getLogger(__name__)


class WeightMappingNet(nn.Module):
    """
    Per-weight residual mapping network.

    Behaviour:
      - Accepts noisy weights (same shape as W) and optionally local statistics.
      - Outputs a delta of same shape as W: ΔW.
      - Designed to be lightweight and stable (shared MLP applied per-element).

    Config:
      - hidden_dim: width of per-element MLP
      - per_group: if True, produce a per-(out,in-group) scalar rather than full per-weight
      - group_size: for conv layers, group over k*k as needed
    """

    def __init__(self, hidden_dim: int = 32, per_group: bool = False, alpha: float = 0.5):
        super().__init__()
        self.per_group = per_group
        self.alpha = alpha  # multiplicative scale applied to network output (learnable via hyper)

        # Learnable per-layer scalar for multiplicative calibration
        # W_final = scale * W_fp + delta_W
        self.scale = nn.Parameter(torch.ones(1))
        # Flag to indicate whether to use multiplicative scale (set during training)
        # Default to False (additive_only mode)
        self._use_multiplicative_scale = False

        # shared per-element MLP (input is scalar noisy weight value)
        # we use a tiny MLP applied to each element (via flatten/reshape) for efficiency
        self.elem_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # init last linear to small values to avoid large initial correction
        # nn.init.zeros_(self.elem_mlp[-2].weight)
        nn.init.normal_(self.elem_mlp[-2].weight, mean=0.0, std=0.02)  # small random init
        nn.init.zeros_(self.elem_mlp[-2].bias)

    def forward(self, W_noisy: torch.Tensor, noise_scale: Optional[float] = None,
                layer_type: str = 'linear', conv_shape: Optional[Tuple[int, int, int, int]] = None):
        """
        Args:
            W_noisy: tensor of shape [out, in] for linear, [out, in*k*k] for conv flat
            noise_scale: estimated std/scale of the noise in *weight space* (used to scale output)
            layer_type: 'linear' or 'conv'
            conv_shape: (out_ch, in_ch, k_h, k_w) for conv

        Returns:
            delta_W: same shape as W_noisy
        """
        # Flatten to elements: [N, 1]
        orig_shape = W_noisy.shape
        flat = W_noisy.reshape(-1, 1)

        # Forward shared MLP per element
        delta_flat = self.elem_mlp(flat)  # in (-1,1)

        # Scale delta to desired magnitude. Use noise_scale if provided.
        if noise_scale is None:
            # fallback: estimate from W_noisy std
            noise_scale = float(W_noisy.std().detach().cpu().item() + 1e-12)

        # target correction magnitude should be a fraction of noise_scale (alpha factor)
        # alpha ~ 1.0 means learnable range up to noise_scale
        # delta_flat = delta_flat * (self.alpha * noise_scale)
        scale = self.alpha * W_noisy.std().detach()
        delta_flat = delta_flat * scale
        delta = delta_flat.reshape(orig_shape)
        return delta


# --- Hardware forward that applies weight-space mapping ---
def hardware_linear_forward_with_weight_mapping(
    x: torch.Tensor,
    W: torch.Tensor,
    device_model,
    mapping_net: Optional[nn.Module] = None,
    t: int = 0,
    seed: Optional[int] = None,
    enable_sanity_check: bool = True,
    max_frac: float = 0.5,
    per_tile_quant: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Hardware forward pass with learned weight mapping.
    
    New logic:
    1. delta_W = mapping_net(W_fp)  # mapping_net takes clean weights as input
    2. W_pre = W_fp + delta_W
    3. W_pre = clamp(W_pre, wmin, wmax)
    4. out = hardware_linear_forward_adaptive(x, W_pre)  # Use HAT's forward function
    
    Args:
        x: Input tensor
        W: Clean floating-point weights (W_fp)
        device_model: MemristorDeviceModel instance
        mapping_net: WeightMappingNet instance (optional)
        t: Time/cycle index for drift
        seed: Random seed for reproducibility
        enable_sanity_check: If True, emit extra debug logs (logger.debug) for this forward
        max_frac: Max fraction for delta clamping
        per_tile_quant: Whether to use per-tile quantization
        
    Returns:
        Output tensor and debug info
    """
    debug: Dict[str, Any] = {}
    
    # Import hardware_linear_forward_adaptive from memristor_wrappers
    try:
        from .memristor_wrappers import hardware_linear_forward_adaptive
    except ImportError:
        from src.memristor.memristor_wrappers import hardware_linear_forward_adaptive
    
    # W is W_fp (clean floating-point weights)
    W_fp = W
    
    # If mapping_net is None, directly use HAT's forward (same as HAT)
    if mapping_net is None:
        out, debug_info = hardware_linear_forward_adaptive(
            x, W_fp, device_model, t=t, seed=seed, training=True
        )
        debug['W_fp_mean'] = float(W_fp.mean().detach().cpu().item())
        debug['delta_mean'] = None
        debug['noise_scale'] = None
        return out, {'W_pre': W_fp, 'debug': debug}
    
    # --- STEP 1: Compute delta_W from mapping_net ---
    # mapping_net takes W_fp as input: delta_W = mapping_net(W_fp)
    noise_scale = None
    with torch.no_grad():
        noise_scale_est = float(W_fp.std().detach().cpu().item() + 1e-12)
        min_noise_scale = 1e-8
        max_noise_scale = 0.5 * (device_model.wmax - device_model.wmin)
        noise_scale = float(min(max(noise_scale_est, min_noise_scale), max_noise_scale))
    
    delta_W = mapping_net(W_fp, noise_scale=noise_scale, layer_type='linear', conv_shape=None)
    
    # Apply safety clamping to delta_W
    bound = max_frac * (W_fp.abs() + 1e-9)
    delta_W = torch.max(torch.min(delta_W, bound), -bound)
    abs_clip = (device_model.wmax - device_model.wmin) * 0.5
    delta_W = torch.clamp(delta_W, -abs_clip, abs_clip)
    
    with torch.no_grad():
        debug['delta_mean'] = float(delta_W.mean().detach().cpu().item())
        debug['delta_max_abs'] = float(delta_W.abs().max().detach().cpu().item())
        debug['noise_scale'] = noise_scale
    
    # --- STEP 2: Compute W_pre ---
    # Check if scale should be used
    # In additive_only mode, _use_multiplicative_scale is False
    # In scale_only and scale_plus_delta modes, _use_multiplicative_scale is True
    use_scale = getattr(mapping_net, '_use_multiplicative_scale', False) and hasattr(mapping_net, 'scale')
    
    if use_scale:
        # Use multiplicative + additive calibration: W_pre = scale * W_fp + delta_W
        W_scaled = mapping_net.scale * W_fp
        W_pre = W_scaled + delta_W
    else:
        # Original additive-only mode: W_pre = W_fp + delta_W
        W_pre = W_fp + delta_W
    
    # --- STEP 3: Clamp W_pre to allowed range ---
    if hasattr(device_model, 'wmin') and hasattr(device_model, 'wmax'):
        W_pre = torch.clamp(W_pre, device_model.wmin, device_model.wmax)
    
    with torch.no_grad():
        debug['W_fp_mean'] = float(W_fp.mean().detach().cpu().item())
        debug['W_pre_mean'] = float(W_pre.mean().detach().cpu().item())
        # Only log scale if it's being used (requires_grad=True)
        debug['scale'] = float(mapping_net.scale.item()) if use_scale else None
    
    # Check for NaN/Inf
    if torch.isnan(W_pre).any() or torch.isinf(W_pre).any():
        logger.warning("W_pre contains NaN/Inf, falling back to W_fp")
        W_pre = W_fp
    
    # --- STEP 4: Forward pass using hardware_linear_forward_adaptive (same as HAT) ---
    # Use the same forward function as HAT
    # Note: training flag is passed to hardware_linear_forward_adaptive, which will disable ADC quant during training
    out, debug_info = hardware_linear_forward_adaptive(
        x, W_pre, device_model, t=t, seed=seed, training=True
    )
    
    # Store debug info
    debug.update({
        'Gp_noisy': debug_info[0] if len(debug_info) > 0 else None,
        'Gn_noisy': debug_info[1] if len(debug_info) > 1 else None,
    })

    if enable_sanity_check:
        if debug['delta_mean'] is not None:
            scale_str = f", scale={debug['scale']:.6e}" if debug.get('scale') is not None else ""
            logger.debug(
                "mapping-forward sanity: W_fp_mean=%.6e W_pre_mean=%.6e delta_mean=%.6e "
                "delta_max_abs=%.6e noise_scale=%.6e%s",
                debug['W_fp_mean'],
                debug['W_pre_mean'],
                debug['delta_mean'],
                debug['delta_max_abs'],
                debug['noise_scale'],
                scale_str,
            )
        else:
            logger.debug("mapping-forward sanity: mapping_net inactive (no delta)")

    return out, {'W_pre': W_pre, 'debug': debug}


# --- Wrap into MemristorLinear replacement class ---
class MemristorLinear(nn.Module):
    def __init__(self, linear: nn.Linear, device_model, mapping_max_frac: float = 0.5):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.device_model = device_model
        self.mapping_max_frac = mapping_max_frac  # Max fraction for delta clamping

        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)

        self.mapping_net: Optional[nn.Module] = None

    def set_learned_mapping(self, mapping_net: Optional[nn.Module] = None):
        self.mapping_net = mapping_net

    def forward(self, x: torch.Tensor, t: int = 0, seed: Optional[int] = None, enable_sanity_check: bool = False):
        # For Linear layers: per-tile quant if enable_adc=True
        per_tile_quant = hasattr(self.device_model, 'enable_adc') and self.device_model.enable_adc
        out, info = hardware_linear_forward_with_weight_mapping(
            x, self.weight, self.device_model, mapping_net=self.mapping_net, t=t, seed=seed,
            enable_sanity_check=enable_sanity_check, max_frac=self.mapping_max_frac, per_tile_quant=per_tile_quant
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# --- Wrap into MemristorConv2d replacement class ---
class MemristorConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, device_model, mapping_max_frac: float = 0.5):
        super().__init__()
        self.device_model = device_model
        self.mapping_max_frac = mapping_max_frac  # Max fraction for delta clamping

        # Copy conv parameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,
                                                                                         conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.bias = conv.bias is not None

        # Copy weights and bias
        self.weight = nn.Parameter(conv.weight.data.clone())
        if self.bias:
            self.bias_param = nn.Parameter(conv.bias.data.clone())
        else:
            self.register_parameter('bias_param', None)

        self.mapping_net: Optional[nn.Module] = None

    def set_learned_mapping(self, mapping_net: Optional[nn.Module] = None):
        self.mapping_net = mapping_net

    def forward(self, x: torch.Tensor, t: int = 0, seed: Optional[int] = None, enable_sanity_check: bool = False):
        batch_size = x.size(0)
        in_h, in_w = x.size(2), x.size(3)
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Unfold input into patches
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride
        )

        # Flatten conv weight to matrix form
        W_flat = self.weight.view(self.out_channels, -1)

        # Transpose for linear operation: [batch, num_patches, in_channels*k_h*k_w]
        x_flat = x_unfold.transpose(1, 2)

        # Apply hardware mapping with learned mapping
        # Conv2d: per-patch tiling without quant here; patch-level float32 output
        out_flat, info = hardware_linear_forward_with_weight_mapping(
            x_flat,
            W_flat,
            self.device_model,
            mapping_net=self.mapping_net,
            t=t,
            seed=seed,
            enable_sanity_check=enable_sanity_check,
            max_frac=self.mapping_max_frac,
            per_tile_quant=False  # No per-tile quant for Conv2d patches
        )

        # Transpose back: [batch, out_channels, num_patches]
        out_unfold = out_flat.transpose(1, 2)

        # Calculate output dimensions
        out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1

        # Fold back to feature map
        out = F.fold(
            out_unfold,
            output_size=(out_h, out_w),
            kernel_size=1
        )

        # Add bias
        if self.bias:
            out = out + self.bias_param.view(1, -1, 1, 1)
        
        # Apply ADC quantization on the final feature map (if enabled)
        # Only apply during inference (not during training) to preserve gradients
        # All patches computed → fold back → feature map → one final ADC quant
        if hasattr(self.device_model, 'enable_adc') and self.device_model.enable_adc and not self.training:
            out = self.device_model.adc_quant(out, bits=self.device_model.adc_bits)

        return out


# --- Training helper for mapping net (post-train on small calibration set) ---
def train_weight_mapping(
        mapping_net: nn.Module,
        model: nn.Module,
        calibration_loader,
        device_model,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 10,
        lr: float = 1e-4,
        lambda_reg: float = 1e-4,
        lambda_scale: float = 1e-3,
        t: int = 0,
        use_random_t: bool = False,
        t_max: int = 1000,
        enable_sanity_check: bool = False,
        mode: str = "additive_only",
) -> Dict[str, Any]:
    """
    Post-train mapping_net while freezing the main model weights.
    Mapping net learns ΔW in weight space; model is frozen.
    
    Args:
        use_random_t: If True, use random t for each batch instead of fixed t
        t_max: Maximum t value when use_random_t=True (t will be randomly sampled from [0, t_max])
        mode: Training mode - "additive_only" (original: W_fp + ΔW, no scale, default), 
              "scale_only" (only train scale, freeze delta_W), or "scale_plus_delta" (train both)
        lambda_scale: Regularization weight for scale term: L_scale = lambda_scale * (scale - 1.0)^2
    """
    # Freeze main model
    for p in model.parameters():
        p.requires_grad = False

    # Set up parameter training based on mode
    if mode == "additive_only":
        # Original mode: W_final = W_fp + ΔW (no scale)
        # Freeze scale if it exists, train only delta_W
        for name, p in mapping_net.named_parameters():
            if 'scale' in name:
                p.requires_grad = False
            elif 'elem_mlp' in name:
                p.requires_grad = True
        # Mark that we don't use multiplicative scale
        mapping_net._use_multiplicative_scale = False
        logger.info("train_weight_mapping: mode=additive_only, using original additive mode (W_fp + ΔW), freezing scale")
    elif mode == "scale_only":
        # Freeze all delta_W-related parameters (elem_mlp)
        for name, p in mapping_net.named_parameters():
            if 'elem_mlp' in name:
                p.requires_grad = False
            elif 'scale' in name:
                p.requires_grad = True
        # Mark that we use multiplicative scale
        mapping_net._use_multiplicative_scale = True
        logger.info("train_weight_mapping: mode=scale_only, freezing delta_W parameters, training only scale")
    elif mode == "scale_plus_delta":
        # Train both scale and delta_W
        for p in mapping_net.parameters():
            p.requires_grad = True
        # Mark that we use multiplicative scale
        mapping_net._use_multiplicative_scale = True
        logger.info("train_weight_mapping: mode=scale_plus_delta, training both scale and delta_W")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'additive_only', 'scale_only', or 'scale_plus_delta'")

    # Set mapping net for layers
    # If model is a MemristorModel wrapper, we need to access base_model
    # Check if model has base_model attribute (MemristorModel wrapper)
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
        logger.info(f"train_weight_mapping: Model is MemristorModel wrapper, accessing base_model")
        logger.info(
            f"  model id: {id(model)}, model.base_model id: {id(model.base_model)}, target_model id: {id(target_model)}")
        logger.info(f"  Are model.base_model and target_model the same? {model.base_model is target_model}")
    else:
        logger.info(f"train_weight_mapping: Model is not MemristorModel wrapper, using model directly")
        logger.info(f"  model id: {id(model)}, target_model id: {id(target_model)}")

    # Store target_model reference to verify later
    _train_weight_mapping_target_model = target_model

    num_layers_set = 0
    layers_info = []
    for module in target_model.modules():
        if hasattr(module, 'set_learned_mapping'):
            # Store info before setting
            module_id = id(module)
            module_type = type(module).__name__

            # Set mapping_net
            module.set_learned_mapping(mapping_net)

            # Verify it was set
            new_mapping_net = getattr(module, 'mapping_net', None)
            if new_mapping_net is mapping_net:
                num_layers_set += 1
                layers_info.append((module_type, module_id, True))
                if enable_sanity_check and num_layers_set <= 3:
                    logger.info(f"Set mapping_net for {module_type} (id={module_id})")
            else:
                layers_info.append((module_type, module_id, False))
                logger.error(f"FAILED to set mapping_net for {module_type} (id={module_id})"
                             f"Expected {id(mapping_net)}, got {id(new_mapping_net) if new_mapping_net is not None else None}")

    logger.info(f"train_weight_mapping: Set mapping_net for {num_layers_set} layers")
    if num_layers_set == 0:
        logger.error("ERROR: No layers found with set_learned_mapping! mapping_net will not be used.")
        # Debug: print model structure
        logger.error(f"Model type: {type(model)}")
        if hasattr(model, 'base_model'):
            logger.error(f" Has base_model: {type(model.base_model)}")
            # Try to find any MemristorLinear or MemristorConv2d
            memristor_layers = []
            for name, m in model.named_modules():
                if 'Memristor' in type(m).__name__:
                    memristor_layers.append((name, type(m).__name__))
            logger.error(f"  Found {len(memristor_layers)} Memristor layers: {memristor_layers[:5]}")

    # Verify mapping_net is actually set
    if enable_sanity_check:
        logger.info(f"mapping_net alpha={mapping_net.alpha if hasattr(mapping_net, 'alpha') else 'N/A'}")
        logger.info(f"mapping_net parameters: {sum(p.numel() for p in mapping_net.parameters())}")

    mapping_net.train()

    # Only optimize parameters that require gradients
    trainable_params = [p for p in mapping_net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=0.0)

    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, (data, target) in enumerate(calibration_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()

            # Forward through frozen model (mapping_net applied inside layers)
            # Use random t for each batch if enabled
            if use_random_t:
                batch_t = random.randint(0, t_max)
            else:
                batch_t = t
            
            # Pass enable_sanity_check to model forward
            try:
                out = model(data, t=batch_t, enable_sanity_check=enable_sanity_check and batch_idx == 0)
            except TypeError:
                # Fallback if model doesn't support enable_sanity_check
                out = model(data, t=batch_t)

            loss_task = criterion(out, target)

            # reg on mapping params to keep Δ small (only for elem_mlp parameters)
            reg = torch.tensor(0.0, device=loss_task.device)
            for name, p in mapping_net.named_parameters():
                if 'elem_mlp' in name and p.requires_grad:
                    reg = reg + p.pow(2).sum()
            
            # Scale regularization: L_scale = lambda_scale * (scale - 1.0)^2
            # Only apply if scale exists and is being trained
            if hasattr(mapping_net, 'scale') and mapping_net.scale.requires_grad:
                scale_reg = lambda_scale * (mapping_net.scale - 1.0).pow(2)
            else:
                scale_reg = torch.tensor(0.0, device=loss_task.device)
            
            loss = loss_task + lambda_reg * reg + scale_reg

            # Check if loss requires grad before backward
            if not loss.requires_grad:
                logger.error("Loss does not require grad! Cannot backward.")
                # Check which components don't have grad
                logger.error(f"loss_task.requires_grad: {loss_task.requires_grad}")
                logger.error(f"reg.requires_grad: {reg.requires_grad}")
                logger.error(
                    f"mapping_net parameters require_grad: {[p.requires_grad for p in mapping_net.parameters()]}")
                raise RuntimeError("Loss does not require grad. Check gradient flow from mapping_net to output.")

            loss.backward()
            # Only clip gradients for trainable parameters
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()

            epoch_loss += loss_task.item() * data.size(0)
            epoch_samples += data.size(0)

            # optional sanity logging
            if enable_sanity_check and batch_idx == 0:
                # Print delta scale ratios from first memristor layer
                for m in model.modules():
                    if hasattr(m, 'mapping_net') and m.mapping_net is not None:
                        # compute one forward pass debug
                        _ = m(data[:1], t=t, enable_sanity_check=True)
                        break

        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('inf')

        # validate on calibration set itself
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in calibration_loader:
                data, target = data.to(device), target.to(device)
                out = model(data, t=t)
                pred = out.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        acc = 100.0 * correct / total if total > 0 else 0.0
        model.train()

        if acc > best_acc:
            best_acc = acc
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Log scale value with epoch ID
        scale_value = mapping_net.scale.item() if hasattr(mapping_net, 'scale') else None
        scale_str = f", learned scale: {scale_value:.6f}" if scale_value is not None else ""
        logger.info(f"Post-train mapping epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, acc={acc:.2f}%{scale_str}")

    # Unfreeze main model after training mapping_net
    # This allows the main model to be trained in the outer training loop
    for p in model.parameters():
        p.requires_grad = True
    
    logger.info("train_weight_mapping: Unfroze main model parameters after mapping_net training")

    return {'best_val_acc': best_acc, 'best_loss': best_loss}


# --- Quick debug checklist function ---
def sanity_check_mapping_scale(model, device, sample_loader, t=0):
    """
    Run one batch and report delta / noise magnitudes for the first memristor layer.
    """
    model.eval()
    with torch.no_grad():
        for data, _ in sample_loader:
            data = data.to(device)
            # optional: first memristor layer may log mapping sanity when supported
            try:
                _ = model(data[:32], t=t)
            except Exception:
                try:
                    _ = model(data[:32])
                except Exception:
                    pass
            break
    model.train()
