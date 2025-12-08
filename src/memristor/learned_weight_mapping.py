import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

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
        nn.init.normal_(self.elem_mlp[-2].weight, mean=0.0, std=0.02)  # 小随机值
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
        # inside WeightMappingNet.forward (for debug only)
        # print(f"[MAPPING CALLED] layer forward called, W_noisy device={W_noisy.device}, mapping_net id={id(self)}",flush=True)
        # optionally increment a global counter

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
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    debug: Dict[str, Any] = {}
    eps = 1e-12

    # --- STEP 1: compute a single weight-space noisy estimate W_noisy (ONE non-ideal pass) ---
    if hasattr(device_model, 'apply_noise_to_weights'):
        W_noisy = device_model.apply_noise_to_weights(W, t=t, seed=seed)
    else:
        # map -> apply_nonidealities once -> map back
        Gp, Gn, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W)
        Gp_noisy = device_model.apply_nonidealities(Gp, t=t, seed=seed)
        Gn_noisy = device_model.apply_nonidealities(Gn, t=t, seed=(None if seed is None else seed+1))
        G_range = (device_model.G_max - device_model.G_min)
        scale_back = max_abs / (G_range + eps)
        scale_back = torch.clamp(scale_back, min=1e-9, max=1e9)
        W_noisy = (Gp_noisy - Gn_noisy) * scale_back

    # keep debug
    with torch.no_grad():
        debug['W_noisy_mean'] = float(W_noisy.mean().detach().cpu().item())
        debug['W_mean'] = float(W.mean().detach().cpu().item())

    # --- STEP 2: compute mapping_net delta (if provided)  ---
    delta_W = None
    noise_scale = None
    if mapping_net is not None:
        # compute a robust noise_scale: mean absolute difference, but clamp
        with torch.no_grad():
            noise_est = (W_noisy - W).abs()
            est_mean = float(noise_est.mean().detach().cpu().item() + 1e-12)

            # Bound noise_scale to avoid pathological scale (user-tweakable)
            min_noise_scale = 1e-8
            max_noise_scale = 0.5 * (device_model.wmax - device_model.wmin)  # at most half-range
            noise_scale = float(min(max(est_mean, min_noise_scale), max_noise_scale))

        # mapping_net MUST accept W_noisy (so it can see the corruption)
        # mapping_net returns a delta in weight space
        delta_W = mapping_net(W_noisy, noise_scale=noise_scale, layer_type='linear', conv_shape=None)

        # Safety clamp: don't allow delta to exceed a fraction of current magnitude
        # max_frac can be tuned (start conservative, configurable via parameter)
        # compute per-element bounds: max_frac * (|W_noisy| + small_eps)
        bound = max_frac * (W_noisy.abs() + 1e-9)
        delta_W = torch.max(torch.min(delta_W, bound), -bound)

        # Extra guard: absolute clamp (global) in case W_noisy small
        abs_clip = (device_model.wmax - device_model.wmin) * 0.5
        delta_W = torch.clamp(delta_W, -abs_clip, abs_clip)

        with torch.no_grad():
            debug['delta_mean'] = float(delta_W.mean().detach().cpu().item())
            debug['delta_max_abs'] = float(delta_W.abs().max().detach().cpu().item())
            debug['noise_scale'] = noise_scale
    else:
        debug['delta_mean'] = None
        debug['noise_scale'] = None

    # --- STEP 3: form W_corrected but DO NOT re-apply device non-idealities ---
    if delta_W is not None:
        W_corrected = W_noisy + delta_W
    else:
        W_corrected = W_noisy

    # clamp corrected weights into allowed range (very important)
    if hasattr(device_model, 'wmin') and hasattr(device_model, 'wmax'):
        W_corrected = torch.clamp(W_corrected, device_model.wmin, device_model.wmax)

    with torch.no_grad():
        debug['W_corrected_mean'] = float(W_corrected.mean().detach().cpu().item())

    # If W_corrected contains NaN/Inf, fallback
    if torch.isnan(W_corrected).any() or torch.isinf(W_corrected).any():
        # restore fallback and warn
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("W_corrected contains NaN/Inf, falling back to original W")
        W_corrected = W

    # --- STEP 4: use W_corrected as final effective weight; DO NOT map->apply_nonidealities again ---
    W_eff = W_corrected

    # decide tiling or direct linear
    use_tiling = (hasattr(device_model, 'array_size') and
                  device_model.array_size > 0 and
                  (W_eff.size(0) > device_model.array_size or W_eff.size(1) > device_model.array_size))

    if use_tiling:
        out = device_model.matmul_with_tiling(x, W_eff, adc_bits=device_model.adc_bits)
    else:
        import torch.nn.functional as F
        out = F.linear(x, W_eff)
        if hasattr(device_model, 'enable_adc') and device_model.enable_adc:
            out = device_model.adc_quant(out, bits=device_model.adc_bits)

    # Final debug
    if enable_sanity_check:
        # extra safety checks & prints
        small_warning = False
        if debug['delta_mean'] is not None and abs(debug['delta_mean']) > 0.5 * abs(debug['W_noisy_mean'] + 1e-12):
            small_warning = True

        print("\n=== mapping-forward sanity ===")
        print(f"W mean: {debug['W_mean']:.6e}")
        print(f"W_noisy mean: {debug['W_noisy_mean']:.6e}")
        if debug['delta_mean'] is not None:
            print(f"delta mean: {debug['delta_mean']:.6e}, delta_max_abs: {debug['delta_max_abs']:.6e}, noise_scale: {debug['noise_scale']:.6e}")
            print(f"W_corrected mean: {debug['W_corrected_mean']:.6e}")
            if small_warning:
                print("!!! WARNING: delta magnitude seems large relative to W_noisy mean. Check alpha/max_frac.")
        else:
            print("mapping_net is None (no correction applied).")
        print("=" * 30 + "\n", flush=True)

    return out, {'W_eff': W_eff, 'debug': debug}


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
        out, info = hardware_linear_forward_with_weight_mapping(
            x, self.weight, self.device_model, mapping_net=self.mapping_net, t=t, seed=seed,
            enable_sanity_check=enable_sanity_check, max_frac=self.mapping_max_frac
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
        out_flat, info = hardware_linear_forward_with_weight_mapping(
            x_flat,
            W_flat,
            self.device_model,
            mapping_net=self.mapping_net,
            t=t,
            seed=seed,
            enable_sanity_check=enable_sanity_check,
            max_frac=self.mapping_max_frac
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
        t: int = 0,
        enable_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Post-train mapping_net while freezing the main model weights.
    Mapping net learns ΔW in weight space; model is frozen.
    """
    # Freeze main model
    for p in model.parameters():
        p.requires_grad = False

    # Ensure mapping_net parameters require gradients
    for p in mapping_net.parameters():
        p.requires_grad = True

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

    opt = torch.optim.Adam(mapping_net.parameters(), lr=lr, weight_decay=0.0)

    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, (data, target) in enumerate(calibration_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()

            # Before forward, verify mapping_net is still set
            if batch_idx == 0 and enable_sanity_check:
                print("\n" + "=" * 60)
                print(f"DEBUG: Before forward (epoch {epoch + 1}, batch {batch_idx})")
                print("=" * 60)
                forward_target_model = _train_weight_mapping_target_model
                if hasattr(model, 'base_model'):
                    forward_target_model = model.base_model
                    print(f"Model type: {type(model)}, has base_model: True")
                    print(
                        f"target_model id: {id(_train_weight_mapping_target_model)}, model.base_model id: {id(model.base_model)}")
                    print(f"Are they the same? {_train_weight_mapping_target_model is model.base_model}")
                    if _train_weight_mapping_target_model is not model.base_model:
                        print("WARNING: target_model and model.base_model are NOT the same object")
                        print(" This could cause mapping_net to not be applied during forward")
                else:
                    forward_target_model = model
                    print(f"Model type: {type(model)}, has base_model: False")

                mapping_net_count = 0
                for m in forward_target_model.modules():
                    if hasattr(m, 'mapping_net'):
                        mapping_net_count += 1
                        mn = getattr(m, 'mapping_net', None)
                        is_correct = mn is mapping_net
                        status = "✓" if is_correct else "✗"
                        print(
                            f"  {status} {type(m).__name__}: mapping_net = {type(mn).__name__ if mn is not None else 'None'} "
                            f"(id={id(mn) if mn is not None else None}, expected={id(mapping_net)})")
                print(f"Total layers with mapping_net: {mapping_net_count}")
                print("=" * 60 + "\n")

            # Forward through frozen model (mapping_net applied inside layers)
            # Pass enable_sanity_check to model forward
            try:
                out = model(data, t=t, enable_sanity_check=enable_sanity_check and batch_idx == 0)
            except TypeError:
                # Fallback if model doesn't support enable_sanity_check
                out = model(data, t=t)

            loss_task = criterion(out, target)

            # reg on mapping params to keep Δ small
            reg = sum(p.pow(2).sum() for p in mapping_net.parameters())
            loss = loss_task + lambda_reg * reg

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
            torch.nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=1.0)
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

        logger.info(f"Post-train mapping epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, acc={acc:.2f}%")

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
            # call forward with enable_sanity_check (first memristor layer prints itself)
            try:
                _ = model(data[:32], t=t)
            except Exception:
                try:
                    _ = model(data[:32])
                except Exception:
                    pass
            break
    model.train()
