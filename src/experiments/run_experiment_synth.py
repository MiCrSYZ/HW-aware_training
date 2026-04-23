"""
Synthetic noise experiment runner.

This module provides experiment running functionality for synthetic noise experiments.
Unlike memristor experiments, synthetic noise does NOT use weight-to-conductance mapping.
"""

import copy
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
import logging
import pandas as pd
import math
import time
import numpy as np

try:
    from ..models.model_zoo import get_model, wrap_model_with_synth_noise
    from ..memristor.synth_noise_wrappers import (
        SynthNoiseConfig,
        SynthNoiseLinear,
        SynthNoiseConv2d,
        get_and_reset_backward_diagnostic,
        apply_logits_backward_corruption,
        clear_synth_noise_template_caches,
    )
    from ..utils.metrics import AverageMeter, accuracy
    from ..utils.gradient_quality_metrics import (
        collect_gradient_quality_metrics,
        gradient_reachability_and_consistency,
        gradient_variance_domination,
        gradient_B_mean,
        perturbation_structural_stability,
        sign_coupled_scaling_P,
        gradient_reachability_and_consistency_layerwise,
        gradient_variance_domination_layerwise,
        gradient_B_mean_layerwise,
    )
    from ..utils.checkpoint import save_checkpoint, load_checkpoint
    from ..utils.logger import setup_logger, setup_tensorboard, setup_wandb
    try:
        import wandb
    except ImportError:
        wandb = None
    from ..utils.vit_metrics import (
        collect_gradient_norms_by_tier,
        collect_activation_stats,
        compute_logit_margin,
        register_activation_hooks,
        compute_update_norm_by_tier,
    )
    try:
        from ..utils.gru_metrics import (
            collect_gradient_norms_by_tier as gru_collect_gradient_norms_by_tier,
            collect_activation_stats as gru_collect_activation_stats,
            compute_logit_margin as gru_compute_logit_margin,
            register_activation_hooks as gru_register_activation_hooks,
            compute_update_norm_by_tier as gru_compute_update_norm_by_tier,
        )
    except ImportError:
        gru_collect_gradient_norms_by_tier = None
        gru_collect_activation_stats = None
        gru_compute_logit_margin = None
        gru_register_activation_hooks = None
        gru_compute_update_norm_by_tier = None
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.model_zoo import get_model, wrap_model_with_synth_noise
    from src.memristor.synth_noise_wrappers import (
        SynthNoiseConfig,
        SynthNoiseLinear,
        SynthNoiseConv2d,
        get_and_reset_backward_diagnostic,
        apply_logits_backward_corruption,
        clear_synth_noise_template_caches,
    )
    from src.utils.metrics import AverageMeter, accuracy
    from src.utils.gradient_quality_metrics import (
        collect_gradient_quality_metrics,
        gradient_reachability_and_consistency,
        gradient_variance_domination,
        gradient_B_mean,
        perturbation_structural_stability,
        sign_coupled_scaling_P,
        gradient_reachability_and_consistency_layerwise,
        gradient_variance_domination_layerwise,
        gradient_B_mean_layerwise,
    )
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from src.utils.logger import setup_logger, setup_tensorboard, setup_wandb
    try:
        import wandb
    except ImportError:
        wandb = None
    try:
        from src.utils.vit_metrics import (
            collect_gradient_norms_by_tier,
            collect_activation_stats,
            compute_logit_margin,
            register_activation_hooks,
            compute_update_norm_by_tier,
        )
    except ImportError:
        collect_gradient_norms_by_tier = None
        collect_activation_stats = None
        compute_logit_margin = None
        register_activation_hooks = None
        compute_update_norm_by_tier = None
    try:
        from src.utils.gru_metrics import (
            collect_gradient_norms_by_tier as gru_collect_gradient_norms_by_tier,
            collect_activation_stats as gru_collect_activation_stats,
            compute_logit_margin as gru_compute_logit_margin,
            register_activation_hooks as gru_register_activation_hooks,
            compute_update_norm_by_tier as gru_compute_update_norm_by_tier,
        )
    except ImportError:
        gru_collect_gradient_norms_by_tier = None
        gru_collect_activation_stats = None
        gru_compute_logit_margin = None
        gru_register_activation_hooks = None
        gru_compute_update_norm_by_tier = None

logger = logging.getLogger(__name__)


def _unpack_batch(batch, is_agnews=False):
    """Unpack batch data, handling both image datasets and AG News."""
    if is_agnews:
        labels, texts, lengths = batch
        return texts, labels, lengths
    else:
        data, target = batch
        return data, target, None


def _snapshot_batchnorm_state(model: nn.Module) -> Dict[int, Dict[str, Optional[torch.Tensor]]]:
    """Snapshot BatchNorm running stats so diagnostics don't mutate eval behavior."""
    snap: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            snap[id(m)] = {
                'running_mean': m.running_mean.detach().clone() if m.running_mean is not None else None,
                'running_var': m.running_var.detach().clone() if m.running_var is not None else None,
                'num_batches_tracked': m.num_batches_tracked.detach().clone() if hasattr(m, 'num_batches_tracked') and m.num_batches_tracked is not None else None,
            }
    return snap


def _restore_batchnorm_state(model: nn.Module, snap: Dict[int, Dict[str, Optional[torch.Tensor]]]) -> None:
    """Restore BatchNorm running stats from snapshot."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            st = snap.get(id(m))
            if st is None:
                continue
            if st['running_mean'] is not None and m.running_mean is not None:
                m.running_mean.copy_(st['running_mean'])
            if st['running_var'] is not None and m.running_var is not None:
                m.running_var.copy_(st['running_var'])
            if st['num_batches_tracked'] is not None and hasattr(m, 'num_batches_tracked') and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(st['num_batches_tracked'])


def run_experiment_synth(
    config: Dict[str, Any],
    output_dir: str,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a synthetic noise experiment based on configuration.
    
    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save outputs
        resume: Path to checkpoint to resume from (optional)
        
    Returns:
        Dictionary with experiment results
    """
    # Set up logging
    log_dir = os.path.join(output_dir, 'logs')
    experiment_logger = setup_logger(log_dir, name='experiment')
    
    # Set seed
    try:
        from ..utils.seeds import set_seed
    except ImportError:
        from src.utils.seeds import set_seed
    set_seed(config.get('seed'))
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    experiment_logger.info(f"Using device: {device}")
    
    # Performance optimizations for CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        experiment_logger.info("CUDA optimizations enabled: cudnn.benchmark=True")
    
    # Data loaders
    try:
        from ..data.dataset import get_dataloaders
    except ImportError:
        from src.data.dataset import get_dataloaders
    
    # Determine input channels and num_classes based on dataset
    dataset_name = config['dataset'].lower()
    vocab = None
    if dataset_name == 'mnist':
        in_channels = 1
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar10':
        in_channels = 3
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar100':
        in_channels = 3
        num_classes = config.get('num_classes', 100)
    elif dataset_name == 'agnews':
        num_classes = config.get('num_classes', 4)
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
            seed=config.get('seed'),
        )
        experiment_logger.info(f"AG News vocabulary size: {len(vocab)}")
    else:
        in_channels = config.get('in_channels', 3)
        num_classes = config.get('num_classes', 10)
    
    experiment_logger.info(f"Dataset: {dataset_name}, num_classes: {num_classes}, in_channels: {in_channels if dataset_name != 'agnews' else 'N/A'}")
    
    if dataset_name != 'agnews':
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
            seed=config.get('seed'),
        )
    
    # Validate label ranges
    experiment_logger.info("Validating dataset labels...")
    max_label = -1
    min_label = float('inf')
    is_agnews = (dataset_name == 'agnews')
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:
            break
        data, target, lengths = _unpack_batch(batch, is_agnews=is_agnews)
        batch_max = target.max().item()
        batch_min = target.min().item()
        max_label = max(max_label, batch_max)
        min_label = min(min_label, batch_min)
    
    experiment_logger.info(f"Label range in dataset: [{min_label}, {max_label}], expected: [0, {num_classes-1}]")
    
    if max_label >= num_classes or min_label < 0:
        error_msg = (
            f"Label mismatch detected! Dataset labels range [{min_label}, {max_label}], "
            f"but model expects [0, {num_classes-1}]. "
            f"Please check your config: dataset={dataset_name}, num_classes={num_classes}."
        )
        experiment_logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Model
    model_kwargs = {}
    if config['model_name'] == 'vit_tiny':
        model_kwargs['patch_size'] = config.get('patch_size', 4)
        model_kwargs['embed_dim'] = config.get('embed_dim', 192)
        model_kwargs['depth'] = config.get('depth', 6)
        model_kwargs['num_heads'] = config.get('num_heads', 3)
        model_kwargs['mlp_ratio'] = config.get('mlp_ratio', 4.0)
        model_kwargs['qkv_bias'] = config.get('qkv_bias', False)
    elif config['model_name'] == 'gru_agnews':
        if vocab is None:
            raise ValueError("vocab is required for GRU model but not found")
        model_kwargs['vocab_size'] = len(vocab)
        model_kwargs['embed_dim'] = config.get('embed_dim', 128)
        model_kwargs['hidden_dim'] = config.get('hidden_dim', 256)
        model_kwargs['num_layers'] = config.get('num_layers', 2)
    
    base_model = get_model(
        name=config['model_name'],
        num_classes=num_classes,
        in_channels=in_channels if dataset_name != 'agnews' else None,
        **model_kwargs
    )
    base_model = base_model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    use_compile = config.get('use_torch_compile', False)
    if use_compile and hasattr(torch, 'compile'):
        try:
            base_model = torch.compile(base_model, mode='reduce-overhead')
            experiment_logger.info("Model compiled with torch.compile (PyTorch 2.0+)")
        except Exception as e:
            experiment_logger.warning(f"torch.compile failed: {e}, continuing without compilation")
    
    # Synthetic noise configuration
    synth_noise_config = None
    synth_noise_model = None
    
    if config['experiment']['mode'] != 'baseline':
        synth_config = config.get('synth_noise', {})
        
        # Map old noise type names to new names
        noise_type_map = {
            'full_variability': 'iid_multiplicative',
            'cond1_variance_bounded': 'heavy_tail',
            'cond2_gradient_unbiased': 'input_dependent',
            'cond3_adc_direct': 'gradient_degenerate',
        }
        old_noise_type = synth_config.get('noise_type', 'none')
        noise_type = noise_type_map.get(old_noise_type, old_noise_type)
        
        def _float(key, default):
            v = synth_config.get(key)
            return float(v) if v is not None else default

        def _bool(key, default):
            v = synth_config.get(key)
            return bool(v) if v is not None else default

        var_sigma = _float('variability_sigma', 0.05)
        in_dep_alpha = _float('input_dependent_alpha', _float('cond2_alpha', 0.1))
        synth_noise_config = SynthNoiseConfig(
            noise_type=noise_type,
            variability_sigma=var_sigma,
            heavy_tail_alpha=_float('heavy_tail_alpha', _float('cond1_alpha', 0.1)),
            heavy_tail_nu=_float('heavy_tail_nu', _float('cond1_nu', 2.0)),
            input_dependent_alpha=in_dep_alpha,
            decoupled_consistent_sigma=_float('decoupled_consistent_sigma', var_sigma),
            decoupled_inconsistent_sigma=_float('decoupled_inconsistent_sigma', var_sigma),
            coupled_consistent_alpha=_float('coupled_consistent_alpha', in_dep_alpha),
            coupled_inconsistent_alpha=_float('coupled_inconsistent_alpha', in_dep_alpha),
            adc_bits=_float('adc_bits', 8.0),
            enable_adc=_bool('enable_adc', False),
            adc_backward_mode=synth_config.get('adc_backward_mode'),
            adv_direction_beta=_float('adv_direction_beta', 1.0),
            adv_direction_frozen=_bool('adv_direction_frozen', True),
            adv_direction_random_sign=_bool('adv_direction_random_sign', False),
            sign_corrupt_p=_float('sign_corrupt_p', 0.5),
            sign_corrupt_mode=synth_config.get('sign_corrupt_mode', 'flip'),
            sign_corrupt_noise_sigma=_float('sign_corrupt_noise_sigma', 1.0),
            saturation_gamma=_float('saturation_gamma', 5.0),
            saturation_alpha=_float('saturation_alpha', 1.0),
            drift_beta=_float('drift_beta', 0.3),
            drift_use_norm=_bool('drift_use_norm', False),
            drift_frozen=_bool('drift_frozen', True),
            drift_resample_when_eval=_bool('drift_resample_when_eval', False),
            drift_d_mean=_float('drift_d_mean', 0.0),
            sign_scale_alpha=_float('sign_scale_alpha', 0.5),
            sign_scale_v_resample=_bool('sign_scale_v_resample', False),
            rank_k=int(synth_config.get('rank_k', 4)),
            rank_fill_sigma=_float('rank_fill_sigma', 0.0),
            rank_resample=_bool('rank_resample', False),
            rank_resample_when_eval=_bool('rank_resample_when_eval', False),
            clip_c=_float('clip_c', 1.0),
            clip_c_eval=(float(synth_config['clip_c_eval']) if synth_config.get('clip_c_eval') is not None else None),
            clip_dither=_bool('clip_dither', False),
            input_dependent_v_resample=_bool('input_dependent_v_resample', False),
            seed=config.get('seed'),
            compensation_in_backward=_bool('compensation_in_backward', True),
            backward_corruption_at=synth_config.get('backward_corruption_at'),
        )
        
        experiment_logger.info(
            f"Synthetic noise: type={noise_type}, "
            f"variability_sigma={synth_noise_config.variability_sigma}, "
            f"heavy_tail_alpha={synth_noise_config.heavy_tail_alpha}, "
            f"heavy_tail_nu={synth_noise_config.heavy_tail_nu}, "
            f"input_dependent_alpha={synth_noise_config.input_dependent_alpha}, "
            f"decoupled_consistent_sigma={synth_noise_config.decoupled_consistent_sigma}, "
            f"decoupled_inconsistent_sigma={synth_noise_config.decoupled_inconsistent_sigma}, "
            f"coupled_consistent_alpha={synth_noise_config.coupled_consistent_alpha}, "
            f"coupled_inconsistent_alpha={synth_noise_config.coupled_inconsistent_alpha}, "
            f"adc_bits={synth_noise_config.adc_bits}, enable_adc={synth_noise_config.enable_adc}, adc_backward_mode={synth_noise_config.adc_backward_mode}, "
            f"adv_direction_beta={synth_noise_config.adv_direction_beta}, adv_direction_frozen={synth_noise_config.adv_direction_frozen}, adv_direction_random_sign={synth_noise_config.adv_direction_random_sign}, "
            f"sign_corrupt_p={synth_noise_config.sign_corrupt_p}, sign_corrupt_mode={synth_noise_config.sign_corrupt_mode}, sign_corrupt_noise_sigma={synth_noise_config.sign_corrupt_noise_sigma}, "
            f"saturation_gamma={synth_noise_config.saturation_gamma}, saturation_alpha={synth_noise_config.saturation_alpha}, "
            f"drift_beta={synth_noise_config.drift_beta}, drift_use_norm={synth_noise_config.drift_use_norm}, "
            f"drift_frozen={synth_noise_config.drift_frozen}, drift_d_mean={getattr(synth_noise_config,'drift_d_mean',None)}, "
            f"sign_scale_alpha={synth_noise_config.sign_scale_alpha}, sign_scale_v_resample={getattr(synth_noise_config, 'sign_scale_v_resample', False)}, rank_k={synth_noise_config.rank_k}, rank_fill_sigma={synth_noise_config.rank_fill_sigma}, rank_resample={synth_noise_config.rank_resample}, "
            f"clip_c={synth_noise_config.clip_c}, clip_c_eval={getattr(synth_noise_config, 'clip_c_eval', None)}, clip_dither={synth_noise_config.clip_dither}, input_dependent_v_resample={getattr(synth_noise_config, 'input_dependent_v_resample', False)}"
        )
        
        # Extract noise injection configuration if available
        noise_injection_config = None
        if 'noise_injection' in synth_config:
            noise_injection_config = synth_config['noise_injection']
        
        # wrap_model_with_synth_noise 会就地替换传入模型中的 Linear/Conv2d 为带噪声层。
        # no_comp 要求训练用干净模型、评估用带噪模型，因此 no_comp 时必须包装 base 的深拷贝，
        # 这样 base_model 保持 nn.Linear/nn.Conv2d，训练时无噪声；评估时用包装后的副本加噪声。
        if config['experiment']['mode'] == 'synth_no_comp':
            base_for_noisy = copy.deepcopy(base_model)
            synth_noise_model = wrap_model_with_synth_noise(
                base_for_noisy, synth_noise_config, noise_config=noise_injection_config
            )
        else:
            synth_noise_model = wrap_model_with_synth_noise(
                base_model, synth_noise_config, noise_config=noise_injection_config
            )
        synth_noise_model = synth_noise_model.to(device)
        
        # For synth_with_comp (HAT), use synth noise model for training (with noise)
        # For synth_no_comp, use base model for training (no noise during training),
        # and use synth noise model only for evaluation/testing (noise injected at inference time)
        if config['experiment']['mode'] == 'synth_with_comp':
            model = synth_noise_model
        else:  # synth_no_comp
            model = base_model
            experiment_logger.info("synth_no_comp mode: using base_model for training (no noise), synth_noise_model for eval/test (with noise)")
    else:  # baseline
        model = base_model
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training setup
    use_amp = config.get('mixed_precision', False)
    if isinstance(use_amp, str):
        use_amp = use_amp.lower() in ['fp16', 'bf16', 'true', '1']
    amp_dtype = None
    if use_amp:
        if torch.cuda.is_available():
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                experiment_logger.info("Using mixed precision training with bfloat16")
            else:
                amp_dtype = torch.float16
                experiment_logger.info("Using mixed precision training with float16")
        else:
            experiment_logger.warning("Mixed precision requested but CUDA not available, using FP32")
            use_amp = False
    
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(optimizer_config['lr']),
            momentum=float(optimizer_config.get('momentum', 0.9)),
            weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_config['lr']),
            weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
        )
    elif optimizer_config['type'] == 'adamw':
        betas_config = optimizer_config.get('betas', [0.9, 0.999])
        if isinstance(betas_config, list):
            betas = tuple(betas_config)
        else:
            betas = betas_config
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config['lr']),
            weight_decay=float(optimizer_config.get('weight_decay', 0.01)),
            betas=betas,
            eps=float(optimizer_config.get('eps', 1e-8)),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}. Available: 'sgd', 'adam', 'adamw'")
    
    # Scheduler
    scheduler = None
    epochs = config.get('epochs', 100)
    
    if 'scheduler' in config and config['scheduler']:
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config['type']
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        if scheduler_type == 'cosine':
            if warmup_epochs > 0:
                try:
                    from torch.optim.lr_scheduler import SequentialLR, LinearLR
                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    cosine_T_max = max(1, int(epochs - warmup_epochs))
                    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=cosine_T_max,
                    )
                    scheduler = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_epochs],
                    )
                except ImportError:
                    def lr_lambda(epoch):
                        if epoch < warmup_epochs:
                            return 0.01 + (1.0 - 0.01) * epoch / warmup_epochs
                        else:
                            cosine_epoch = epoch - warmup_epochs
                            cosine_T_max = max(1, epochs - warmup_epochs)
                            return 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_T_max))
                    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(epochs)),
                )
        elif scheduler_type == 'step':
            if warmup_epochs > 0:
                try:
                    from torch.optim.lr_scheduler import SequentialLR, LinearLR
                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    step_scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=int(scheduler_config.get('step_size', 30)),
                        gamma=float(scheduler_config.get('gamma', 0.1)),
                    )
                    scheduler = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, step_scheduler],
                        milestones=[warmup_epochs],
                    )
                except ImportError:
                    step_size = int(scheduler_config.get('step_size', 30))
                    gamma = float(scheduler_config.get('gamma', 0.1))
                    def lr_lambda(epoch):
                        if epoch < warmup_epochs:
                            return 0.01 + (1.0 - 0.01) * epoch / warmup_epochs
                        else:
                            step_epoch = epoch - warmup_epochs
                            return gamma ** (step_epoch // step_size)
                    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(scheduler_config.get('step_size', 30)),
                    gamma=float(scheduler_config.get('gamma', 0.1)),
                )
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    best_epoch = None
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        best_epoch = checkpoint.get('epoch', None)
        experiment_logger.info(f"Resumed from epoch {start_epoch}")
    
    # TensorBoard and W&B
    tb_writer = setup_tensorboard(os.path.join(output_dir, 'tensorboard_logs'))
    wandb_run = setup_wandb(
        project_name=config['logging'].get('project_name', 'synth-noise-nn'),
        run_name=config['logging'].get('run_name'),
        config=config,
        enabled=config['logging'].get('use_wandb', False),
    )
    
    # Training loop
    experiment_mode = config['experiment']['mode']
    metrics_history = []
    epoch_times = []
    training_start_time = time.time()
    
    steps_per_epoch = len(train_loader)
    # Log val/test sizes to detect possible loader swap (e.g. CIFAR100: val=5k, test=10k)
    if val_loader is not None:
        n_val = len(val_loader.dataset)
        experiment_logger.info(f"Val loader: {n_val} samples")
    if test_loader is not None:
        n_test = len(test_loader.dataset)
        experiment_logger.info(f"Test loader: {n_test} samples")
    experiment_logger.info(f"Starting training loop: start_epoch={start_epoch}, epochs={epochs}")
    experiment_logger.info(f"Steps per epoch: {steps_per_epoch}")
    experiment_logger.info(f"Experiment mode: {experiment_mode}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        is_gru = (config['model_name'] == 'gru_agnews')
        
        # Train
        if experiment_mode == 'baseline':
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru
            )
        elif experiment_mode == 'synth_no_comp':
            # no_comp: always train with base_model (no noise during training)
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru
            )
            # Sync weights from base_model to synth_noise_model for evaluation
            if model is base_model:
                _sync_weights_to_synth_model(base_model, synth_noise_model)
        elif experiment_mode == 'synth_with_comp':
            train_metrics = _train_synth_hat(
                model, train_loader, criterion, optimizer, device, epoch, config,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru,
                synth_noise_config=synth_noise_config,
            )
        else:
            raise ValueError(f"Unknown experiment mode: {experiment_mode}")
        
        # Validate
        eval_start_time = time.perf_counter()
        # Optional: override eval-time noise seed (for fixed-but-different template tests)
        eval_seed = None
        if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
            eval_seed = int(config['experiment']['eval_noise_seed'])
        elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
            eval_seed = int(config['synth_noise']['eval_seed'])
        if val_loader is not None:
            if experiment_mode == 'baseline':
                val_metrics = _validate_baseline(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru
                )
            elif experiment_mode == 'synth_no_comp':
                # For no_comp: use synth noise model for validation (apply noise)
                val_metrics = _validate_synth(
                    synth_noise_model, val_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            else:  # synth_with_comp
                # For with_comp: model is already synth noise-wrapped
                val_metrics = _validate_synth(
                    model, val_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
        else:
            val_metrics = {'acc1': 0.0, 'loss': 0.0}
        eval_time = time.perf_counter() - eval_start_time
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc1': train_metrics['acc1'],
            'val_loss': val_metrics['loss'],
            'val_acc1': val_metrics['acc1'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        if 'grad_norm' in train_metrics:
            metrics['grad_norm'] = train_metrics['grad_norm']
            metrics['grad_norm_std'] = train_metrics.get('grad_norm_std', 0.0)
            metrics['grad_var'] = train_metrics.get('grad_var', 0.0)
        if 'update_std' in train_metrics:
            metrics['update_std'] = train_metrics.get('update_std', 0.0)
        if 'template_resampled_ratio' in train_metrics:
            metrics['template_resampled_ratio'] = train_metrics.get('template_resampled_ratio', 0.0)
        
        metrics['data_time_avg'] = train_metrics.get('data_time_avg', 0.0)
        metrics['train_step_time_avg'] = train_metrics.get('train_step_time_avg', 0.0)
        metrics['eval_time'] = eval_time
        
        # Add ViT/GRU-specific metrics if available
        is_vit = config.get('model_name', '') == 'vit_tiny'
        is_gru = config.get('model_name', '') == 'gru_agnews'
        for key in train_metrics:
            if key.startswith('grad_norm_') or key.startswith('update_norm_') or \
               key.startswith('logit_margin_') or key.startswith('act_') or key.startswith('backward_diag_'):
                metrics[key] = train_metrics[key]
        
        # Gradient quality: dead-zone ratio (Jacobian zero proxy) and effective rank / top-k energy
        if experiment_mode in ('synth_no_comp', 'synth_with_comp') and config.get('compute_gradient_quality_metrics', True):
            try:
                grad_model = synth_noise_model if experiment_mode == 'synth_no_comp' else model
                prev_training_mode = grad_model.training
                bn_state_snapshot = _snapshot_batchnorm_state(grad_model)
                batch = next(iter(train_loader))
                gq = collect_gradient_quality_metrics(
                    grad_model,
                    batch,
                    criterion,
                    device,
                    is_gru=is_gru,
                    seed=epoch,
                    synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                )
                metrics['dead_zone_ratio_element_mean'] = gq['dead_zone_ratio_element_mean']
                metrics['dead_zone_ratio_channel_mean'] = gq['dead_zone_ratio_channel_mean']
                metrics['grad_top_k_energy_ratio_mean'] = gq['grad_top_k_energy_ratio_mean']
                metrics['grad_effective_rank_mean'] = gq['grad_effective_rank_mean']

                # New: gradient reachability (A), consistency (C), variance domination (V), perturbation stability (S)
                K_extra = config.get('gradient_metrics_K', 8)
                rc = gradient_reachability_and_consistency(
                    grad_model, batch, criterion, device,
                    synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                    is_gru=is_gru, seed=epoch,
                )
                metrics['gradient_reachability'] = rc['gradient_reachability']
                metrics['gradient_consistency'] = rc['gradient_consistency']
                v = gradient_variance_domination(
                    grad_model, batch, criterion, device,
                    K=K_extra, seed_base=epoch, is_gru=is_gru,
                )
                metrics['gradient_variance_domination'] = v
                b_mean = gradient_B_mean(
                    grad_model, batch, criterion, device,
                    synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                    K=K_extra, seed_base=epoch, is_gru=is_gru,
                )
                metrics['gradient_B_mean'] = b_mean
                ps = perturbation_structural_stability(
                    grad_model, batch, device,
                    synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                    K=K_extra, seed_base=epoch, is_gru=is_gru,
                )
                metrics['perturbation_stability'] = ps['perturbation_stability']
                p_sc = sign_coupled_scaling_P(
                    grad_model, batch, device,
                    synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                    is_gru=is_gru, seed=epoch,
                )
                metrics['sign_coupled_P_positive'] = p_sc['sign_coupled_P_positive']
                metrics['sign_coupled_P_negative'] = p_sc['sign_coupled_P_negative']
                metrics['sign_coupled_P_zero'] = p_sc['sign_coupled_P_zero']

                # Layer-wise A, C, V, B_mean (ResNet: stem/layer1/2/3; ViT: blocks_1/3/5; GRU: embedding/gru_l0/gru_l1/head)
                model_name = config.get('model_name', '')
                if model_name in ('resnet20', 'vit_tiny', 'gru_agnews'):
                    try:
                        rc_lw = gradient_reachability_and_consistency_layerwise(
                            grad_model, batch, criterion, device,
                            synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                            model_name=model_name, is_gru=is_gru, seed=epoch,
                            return_consistency_denom_numer=config.get('log_gradient_consistency_denom', True),
                        )
                        metrics.update(rc_lw)
                        # Log denominator/numerator when layer-wise C is suspiciously large (denom too small)
                        c_large_threshold = config.get('gradient_consistency_large_threshold', 2.0)
                        for k, c_val in rc_lw.items():
                            if not k.startswith('gradient_consistency_') or k.endswith('_numer') or k.endswith('_denom'):
                                continue
                            if not (isinstance(c_val, (int, float)) and not np.isnan(c_val) and abs(c_val) > c_large_threshold):
                                continue
                            t = k.replace('gradient_consistency_', '')
                            numer = rc_lw.get(f'gradient_consistency_{t}_numer')
                            denom = rc_lw.get(f'gradient_consistency_{t}_denom')
                            experiment_logger.info(
                                "Layer-wise C large: %s=%.4g (numer=%.4e, denom=%.4e)",
                                k, c_val, numer if numer is not None else float('nan'), denom if denom is not None else float('nan')
                            )
                        v_lw = gradient_variance_domination_layerwise(
                            grad_model, batch, criterion, device,
                            model_name=model_name, K=K_extra, seed_base=epoch, is_gru=is_gru,
                        )
                        metrics.update(v_lw)
                        b_lw = gradient_B_mean_layerwise(
                            grad_model, batch, criterion, device,
                            synth_layer_types=(SynthNoiseLinear, SynthNoiseConv2d),
                            model_name=model_name, K=K_extra, seed_base=epoch, is_gru=is_gru,
                        )
                        metrics.update(b_lw)
                    except Exception as elw:
                        logger.warning("Layer-wise gradient metrics failed: %s", elw)
                # Prevent diagnostics from polluting subsequent eval via BN running stats.
                _restore_batchnorm_state(grad_model, bn_state_snapshot)
                grad_model.train(prev_training_mode)
                grad_model.zero_grad(set_to_none=True)
            except Exception as e:
                logger.warning("Gradient quality metrics failed: %s", e)
                metrics['dead_zone_ratio_element_mean'] = float('nan')
                metrics['dead_zone_ratio_channel_mean'] = float('nan')
                metrics['grad_top_k_energy_ratio_mean'] = float('nan')
                metrics['grad_effective_rank_mean'] = float('nan')
                metrics['gradient_reachability'] = float('nan')
                metrics['gradient_consistency'] = float('nan')
                metrics['gradient_variance_domination'] = float('nan')
                metrics['gradient_B_mean'] = float('nan')
                metrics['perturbation_stability'] = float('nan')
                metrics['sign_coupled_P_positive'] = float('nan')
                metrics['sign_coupled_P_negative'] = float('nan')
                metrics['sign_coupled_P_zero'] = float('nan')
                try:
                    _restore_batchnorm_state(grad_model, bn_state_snapshot)
                    grad_model.train(prev_training_mode)
                    grad_model.zero_grad(set_to_none=True)
                except Exception:
                    pass
        
        metrics_history.append(metrics)
        
        if tb_writer:
            tb_writer.add_scalar('Train/Loss', metrics['train_loss'], epoch)
            tb_writer.add_scalar('Train/Acc1', metrics['train_acc1'], epoch)
            tb_writer.add_scalar('Val/Loss', metrics['val_loss'], epoch)
            tb_writer.add_scalar('Val/Acc1', metrics['val_acc1'], epoch)
            tb_writer.add_scalar('LR', metrics['lr'], epoch)
            if 'grad_norm' in metrics:
                tb_writer.add_scalar('Train/GradNorm', metrics['grad_norm'], epoch)
                tb_writer.add_scalar('Train/GradNormStd', metrics['grad_norm_std'], epoch)
                tb_writer.add_scalar('Train/GradVar', metrics['grad_var'], epoch)
            if 'update_std' in metrics:
                tb_writer.add_scalar('Train/UpdateStd', metrics['update_std'], epoch)
            if 'template_resampled_ratio' in metrics:
                tb_writer.add_scalar('Train/TemplateResampledRatio', metrics['template_resampled_ratio'], epoch)
            if 'dead_zone_ratio_element_mean' in metrics and not (isinstance(metrics.get('dead_zone_ratio_element_mean'), float) and np.isnan(metrics['dead_zone_ratio_element_mean'])):
                tb_writer.add_scalar('GradQuality/DeadZoneRatioElement', metrics['dead_zone_ratio_element_mean'], epoch)
                tb_writer.add_scalar('GradQuality/DeadZoneRatioChannel', metrics['dead_zone_ratio_channel_mean'], epoch)
                tb_writer.add_scalar('GradQuality/TopKEnergyRatio', metrics['grad_top_k_energy_ratio_mean'], epoch)
                tb_writer.add_scalar('GradQuality/EffectiveRank', metrics['grad_effective_rank_mean'], epoch)
            for key in ('gradient_reachability', 'gradient_consistency', 'gradient_variance_domination', 'gradient_B_mean', 'perturbation_stability', 'sign_coupled_P_positive', 'sign_coupled_P_negative', 'sign_coupled_P_zero'):
                if key in metrics and isinstance(metrics.get(key), (int, float)) and not np.isnan(metrics[key]):
                    tb_writer.add_scalar(f'GradQuality/{key}', metrics[key], epoch)
            for key, value in metrics.items():
                if (key.startswith('gradient_reachability_') or key.startswith('gradient_consistency_')
                        or key.startswith('gradient_variance_domination_') or key.startswith('gradient_B_mean_')):
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        tb_writer.add_scalar(f'GradQuality/{key}', value, epoch)

            prefix = 'ViT' if is_vit else ('GRU' if is_gru else 'Model')
            for key, value in metrics.items():
                if key.startswith('grad_norm_'):
                    tier = key.replace('grad_norm_', '')
                    tb_writer.add_scalar(f'{prefix}/GradNorm_{tier}', value, epoch)
                elif key.startswith('update_norm_'):
                    tier = key.replace('update_norm_', '')
                    tb_writer.add_scalar(f'{prefix}/UpdateNorm_{tier}', value, epoch)
                elif key.startswith('logit_margin_'):
                    tb_writer.add_scalar(f'{prefix}/{key}', value, epoch)
                elif key.startswith('act_'):
                    parts = key.replace('act_', '').split('_', 1)
                    if len(parts) == 2:
                        tier, stat = parts
                        tb_writer.add_scalar(f'{prefix}/Act_{tier}_{stat}', value, epoch)
        
        if wandb_run and wandb is not None:
            wandb.log(metrics, step=epoch)
        
        # Calculate epoch time and ETA
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            eta_str = f"{hours}h{minutes}m"
        
        if epoch_time < 60:
            epoch_time_str = f"{epoch_time:.1f}s"
        elif epoch_time < 3600:
            epoch_time_str = f"{epoch_time / 60:.1f}m"
        else:
            hours = int(epoch_time // 3600)
            minutes = int((epoch_time % 3600) // 60)
            epoch_time_str = f"{hours}h{minutes}m"
        
        log_msg = (
            f"Epoch {epoch}/{epochs-1}: train_loss={metrics['train_loss']:.4f}, "
            f"train_acc={metrics['train_acc1']:.2f}%, "
            f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc1']:.2f}%"
        )
        
        if 'grad_norm' in metrics:
            log_msg += f" | grad_norm={metrics['grad_norm']:.4e}"
            if metrics.get('grad_norm_std', 0.0) > 0:
                log_msg += f"±{metrics['grad_norm_std']:.4e}"
            log_msg += f" | grad_var={metrics.get('grad_var', 0.0):.4e}"
        if 'update_std' in metrics:
            log_msg += f" | update_std={metrics['update_std']:.4e}"
        if 'template_resampled_ratio' in metrics:
            log_msg += f" | template_resampled_ratio={metrics['template_resampled_ratio']:.3f}"
        if metrics.get('backward_diag_sign_corrupt_calls', 0) > 0:
            log_msg += f" | sign_corrupt_calls={metrics['backward_diag_sign_corrupt_calls']:.0f} flip_ratio={metrics.get('backward_diag_sign_corrupt_flip_ratio', 0):.3f}"
        if metrics.get('backward_diag_adv_bias_calls', 0) > 0:
            log_msg += f" | adv_bias_calls={metrics['backward_diag_adv_bias_calls']:.0f}"
        dz_e = metrics.get('dead_zone_ratio_element_mean')
        if dz_e is not None and isinstance(dz_e, (int, float)) and not np.isnan(dz_e):
            log_msg += f" | DZ_ratio={dz_e:.3f} | eff_rank={metrics.get('grad_effective_rank_mean', 0):.2f}"
        if 'gradient_reachability' in metrics and not np.isnan(metrics.get('gradient_reachability', float('nan'))):
            log_msg += f" | A={metrics['gradient_reachability']:.3f} | C={metrics.get('gradient_consistency', 0):.3f}"
        if 'gradient_variance_domination' in metrics and not np.isnan(metrics.get('gradient_variance_domination', float('nan'))):
            log_msg += f" | V={metrics['gradient_variance_domination']:.3f}"
        if 'gradient_B_mean' in metrics and not np.isnan(metrics.get('gradient_B_mean', float('nan'))):
            log_msg += f" | B_mean={metrics['gradient_B_mean']:.3f}"
        if 'perturbation_stability' in metrics and not np.isnan(metrics.get('perturbation_stability', float('nan'))):
            log_msg += f" | S={metrics['perturbation_stability']:.3f}"
        if 'sign_coupled_P_positive' in metrics and not np.isnan(metrics.get('sign_coupled_P_positive', float('nan'))):
            log_msg += f" | P+={metrics['sign_coupled_P_positive']:.2f} P-={metrics.get('sign_coupled_P_negative', 0):.2f}"
        
        log_msg += f" | Time: {epoch_time_str} | ETA: {eta_str}"
        if 'data_time_avg' in metrics and 'train_step_time_avg' in metrics:
            log_msg += (
                f" | data_time={metrics['data_time_avg']*1000:.1f}ms"
                f" | train_step_time={metrics['train_step_time_avg']*1000:.1f}ms"
                f" | eval_time={metrics.get('eval_time', 0):.2f}s"
            )
        
        experiment_logger.info(log_msg)
        
        # Save best model checkpoint
        is_best = val_metrics['acc1'] > best_acc
        if is_best:
            best_acc = val_metrics['acc1']
            best_epoch = epoch
            best_model_path = os.path.join(output_dir, 'model_best.pth')
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                },
                best_model_path,
                is_best=True,
            )
        else:
            best_acc = max(best_acc, val_metrics['acc1'])
    
    # Load best checkpoint so test_acc is for the same model that achieved best_val_acc
    best_model_path = os.path.join(output_dir, 'model_best.pth')
    skip_load_best = (
        best_epoch is not None
        and epochs > 0
        and int(best_epoch) == int(epochs) - 1
    )
    if skip_load_best:
        experiment_logger.info(
            "Skipping model_best.pth load: best_epoch=%s is the last epoch — weights in memory "
            "already match the best checkpoint.",
            best_epoch,
        )
    elif os.path.exists(best_model_path) and test_loader:
        experiment_logger.info(
            f"[DEBUG] best_acc tracked={best_acc:.4f}% at best_epoch={best_epoch} "
            f"before loading model_best.pth"
        )
        # Snapshot current model state so we can rollback if load-best is clearly inconsistent.
        with io.BytesIO() as _buf:
            torch.save(model.state_dict(), _buf)
            _pre_load_state_bytes = _buf.getvalue()

        pre_load_val_acc = None
        eval_seed = None
        if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
            eval_seed = int(config['experiment']['eval_noise_seed'])
        elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
            eval_seed = int(config['synth_noise']['eval_seed'])
        if val_loader is not None:
            if experiment_mode == 'baseline':
                pre_load_val_metrics = _validate_baseline(
                    model, val_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru
                )
            elif experiment_mode == 'synth_no_comp':
                # Must match per-epoch val: noisy forward uses synth_noise_model after weight sync.
                _sync_weights_to_synth_model(base_model, synth_noise_model)
                pre_load_val_metrics = _validate_synth(
                    synth_noise_model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            else:
                pre_load_val_metrics = _validate_synth(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            pre_load_val_acc = float(pre_load_val_metrics['acc1'])

            # Extra diagnostics: evaluate same weights with noise disabled and with fixed eval_seed.
            # This helps distinguish "weights restore problem" vs "noise template/seed application problem".
            if experiment_mode != 'baseline' and synth_noise_config is not None:
                try:
                    _diag_model = synth_noise_model if experiment_mode == 'synth_no_comp' else model
                    noise_modules = [
                        m for m in _diag_model.modules()
                        if isinstance(m, (SynthNoiseLinear, SynthNoiseConv2d))
                    ]
                except Exception:
                    noise_modules = []

                orig_enable = [getattr(m, "enable_noise", True) for m in noise_modules]

                def _set_noise_enable(val: bool) -> None:
                    for m in noise_modules:
                        m.enable_noise = val

                fixed_seed = int(getattr(synth_noise_config, "seed", 0) or 0)
                try:
                    # 1) Noise off
                    _set_noise_enable(False)
                    pre_load_val_noise_off = _validate_synth(
                        _diag_model, val_loader, criterion, device,
                        amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                        synth_noise_config=synth_noise_config,
                    )['acc1']

                    # 2) Fixed eval_seed
                    pre_load_val_seed_fixed = _validate_synth(
                        _diag_model, val_loader, criterion, device,
                        amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=fixed_seed,
                        synth_noise_config=synth_noise_config,
                    )['acc1']
                finally:
                    _set_noise_enable(True)
                    for m, oe in zip(noise_modules, orig_enable):
                        m.enable_noise = oe

                experiment_logger.info(
                    "[DEBUG pre-load extra] val_acc(noise_on)=%.2f%% noise_off=%.2f%% eval_seed_fixed=%d => %.2f%%",
                    pre_load_val_acc, float(pre_load_val_noise_off), fixed_seed, float(pre_load_val_seed_fixed),
                )

        # Weight-level sanity: snapshot one stable parameter norm before load-best.
        # This helps confirm whether `model_best.pth` actually restores the intended tensors.
        weight_key = None
        pre_weight_norm = None
        try:
            sd_model = model.state_dict()
            # Prefer classifier weight if present; otherwise take first state_dict key.
            for k in sd_model.keys():
                if k.endswith("fc.weight") or k.endswith("linear.weight"):
                    weight_key = k
                    break
            if weight_key is None and len(sd_model) > 0:
                weight_key = next(iter(sd_model.keys()))
            if weight_key is not None:
                pre_weight_norm = float(sd_model[weight_key].float().norm().item())
            experiment_logger.info(
                "[DEBUG weight snapshot BEFORE load-best] key=%s norm=%.6f",
                str(weight_key), float(pre_weight_norm) if pre_weight_norm is not None else float("nan"),
            )
        except Exception as e:
            experiment_logger.info("[DEBUG weight snapshot BEFORE load-best] failed: %s", str(e))

        # Template cache sanity: inspect frozen-template keys
        if synth_noise_config is not None:
            drift_keys = []
            try:
                drift_keys = list(getattr(synth_noise_config, "_drift_d_vectors", {}).keys())[:5]
            except Exception:
                drift_keys = []
            experiment_logger.info(
                "[DEBUG drift cache BEFORE load-best] config.seed=%s eval_seed=%s drift_keys_sample=%s drift_cache_len=%s",
                str(getattr(synth_noise_config, "seed", None)),
                str(eval_seed),
                str(drift_keys),
                str(len(getattr(synth_noise_config, "_drift_d_vectors", {})) if hasattr(synth_noise_config, "_drift_d_vectors") else "NA"),
            )

        loaded_ckpt = load_checkpoint(best_model_path, model, optimizer=None, device=device)
        experiment_logger.info(
            "[DEBUG] Loaded best checkpoint meta: epoch=%s best_acc=%s",
            str(loaded_ckpt.get('epoch', 'N/A')) if isinstance(loaded_ckpt, dict) else "N/A",
            str(loaded_ckpt.get('best_acc', 'N/A')) if isinstance(loaded_ckpt, dict) else "N/A",
        )

        # Compare loaded tensor norm with checkpoint stored tensor norm
        try:
            post_weight_norm = None
            if weight_key is not None:
                post_weight_norm = float(model.state_dict()[weight_key].float().norm().item())

            sd_ckpt = loaded_ckpt.get("model_state_dict", loaded_ckpt.get("state_dict", None))
            ckpt_weight_norm = None
            if sd_ckpt is not None and weight_key is not None and weight_key in sd_ckpt:
                ckpt_weight_norm = float(sd_ckpt[weight_key].float().norm().item())

            rel_diff = float("nan")
            if post_weight_norm is not None and ckpt_weight_norm is not None:
                rel_diff = abs(post_weight_norm - ckpt_weight_norm) / (abs(ckpt_weight_norm) + 1e-12)

            experiment_logger.info(
                "[DEBUG weight snapshot AFTER load-best] key=%s norm_post=%.6f norm_ckpt=%.6f rel_diff=%.3e",
                str(weight_key),
                float(post_weight_norm) if post_weight_norm is not None else float("nan"),
                float(ckpt_weight_norm) if ckpt_weight_norm is not None else float("nan"),
                rel_diff,
            )
        except Exception as e:
            experiment_logger.info("[DEBUG weight snapshot AFTER load-best] failed: %s", str(e))

        if synth_noise_config is not None:
            drift_keys = []
            try:
                drift_keys = list(getattr(synth_noise_config, "_drift_d_vectors", {}).keys())[:5]
            except Exception:
                drift_keys = []
            experiment_logger.info(
                "[DEBUG drift cache AFTER load-best meta] config.seed=%s eval_seed=%s drift_keys_sample=%s drift_cache_len=%s",
                str(getattr(synth_noise_config, "seed", None)),
                str(eval_seed),
                str(drift_keys),
                str(len(getattr(synth_noise_config, "_drift_d_vectors", {})) if hasattr(synth_noise_config, "_drift_d_vectors") else "NA"),
            )

        # Template caches live on SynthNoiseConfig (not in .pth). Always clear after loading
        # weights so frozen drift / rank projectors etc. are regenerated deterministically from
        # (dim, seed) — same templates as a fresh run, independent of training-time cache order.
        if synth_noise_config is not None:
            clear_synth_noise_template_caches(synth_noise_config)
            drift_keys = []
            try:
                drift_keys = list(getattr(synth_noise_config, "_drift_d_vectors", {}).keys())[:5]
            except Exception:
                drift_keys = []
            experiment_logger.info(
                "[DEBUG drift cache AFTER clear] config.seed=%s eval_seed=%s drift_keys_sample=%s drift_cache_len=%s",
                str(getattr(synth_noise_config, "seed", None)),
                str(eval_seed),
                str(drift_keys),
                str(len(getattr(synth_noise_config, "_drift_d_vectors", {})) if hasattr(synth_noise_config, "_drift_d_vectors") else "NA"),
            )
        model.eval()
        if experiment_mode == 'synth_no_comp' and model is base_model:
            _sync_weights_to_synth_model(base_model, synth_noise_model)
        # Sanity check: load-best should not catastrophically hurt val accuracy.
        if val_loader is not None:
            if experiment_mode == 'baseline':
                post_load_val_metrics = _validate_baseline(
                    model, val_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru
                )
            elif experiment_mode == 'synth_no_comp':
                post_load_val_metrics = _validate_synth(
                    synth_noise_model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            else:
                post_load_val_metrics = _validate_synth(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            post_load_val_acc = float(post_load_val_metrics['acc1'])
            experiment_logger.info(
                "[DEBUG] load-best sanity: pre_load_val=%.2f%% post_load_val=%.2f%%",
                pre_load_val_acc if pre_load_val_acc is not None else float('nan'),
                post_load_val_acc,
            )

            if experiment_mode != 'baseline' and synth_noise_config is not None:
                try:
                    _diag_model_post = synth_noise_model if experiment_mode == 'synth_no_comp' else model
                    noise_modules = [
                        m for m in _diag_model_post.modules()
                        if isinstance(m, (SynthNoiseLinear, SynthNoiseConv2d))
                    ]
                except Exception:
                    noise_modules = []

                orig_enable = [getattr(m, "enable_noise", True) for m in noise_modules]

                def _set_noise_enable(val: bool) -> None:
                    for m in noise_modules:
                        m.enable_noise = val

                fixed_seed = int(getattr(synth_noise_config, "seed", 0) or 0)
                try:
                    # 1) Noise off
                    _set_noise_enable(False)
                    post_load_val_noise_off = _validate_synth(
                        _diag_model_post, val_loader, criterion, device,
                        amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                        synth_noise_config=synth_noise_config,
                    )['acc1']
                    # 2) Fixed eval_seed
                    post_load_val_seed_fixed = _validate_synth(
                        _diag_model_post, val_loader, criterion, device,
                        amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=fixed_seed,
                        synth_noise_config=synth_noise_config,
                    )['acc1']
                finally:
                    _set_noise_enable(True)
                    for m, oe in zip(noise_modules, orig_enable):
                        m.enable_noise = oe

                experiment_logger.info(
                    "[DEBUG post-load extra] val_acc(noise_on)=%.2f%% noise_off=%.2f%% eval_seed_fixed=%d => %.2f%%",
                    post_load_val_acc, float(post_load_val_noise_off), fixed_seed, float(post_load_val_seed_fixed),
                )
            if pre_load_val_acc is not None and (post_load_val_acc + 20.0) < pre_load_val_acc:
                experiment_logger.warning(
                    "Loaded best checkpoint causes catastrophic val drop "
                    "(pre=%.2f%%, post=%.2f%%). Rolling back to pre-load weights.",
                    pre_load_val_acc,
                    post_load_val_acc,
                )
                model.load_state_dict(torch.load(io.BytesIO(_pre_load_state_bytes), map_location=device))
                if experiment_mode == 'synth_no_comp' and model is base_model:
                    _sync_weights_to_synth_model(base_model, synth_noise_model)
        experiment_logger.info("Loaded best checkpoint for final test evaluation")
    
    # Final evaluation on test set
    if test_loader:
        if experiment_mode == 'baseline':
            test_metrics = _validate_baseline(
                model, test_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru
            )
        elif experiment_mode == 'synth_no_comp':
            # For no_comp: always sync weights from base_model (trained without noise) to synth_noise_model
            _sync_weights_to_synth_model(base_model, synth_noise_model)
            # Use same eval_seed policy as validation
            eval_seed = None
            if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
                eval_seed = int(config['experiment']['eval_noise_seed'])
            elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
                eval_seed = int(config['synth_noise']['eval_seed'])
            test_metrics = _validate_synth(
                synth_noise_model, test_loader, criterion, device,
                amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                synth_noise_config=synth_noise_config,
            )
        else:  # synth_with_comp
            eval_seed = None
            if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
                eval_seed = int(config['experiment']['eval_noise_seed'])
            elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
                eval_seed = int(config['synth_noise']['eval_seed'])
            test_metrics = _validate_synth(
                model, test_loader, criterion, device,
                amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                synth_noise_config=synth_noise_config,
            )
        
        test_log_msg = (
            f"Test accuracy: {test_metrics['acc1']:.2f}%, loss: {test_metrics['loss']:.4f}"
        )
        experiment_logger.info(test_log_msg)
        # Diagnostic: same model on val set to verify val vs test not swapped / overfitting
        if val_loader is not None:
            if experiment_mode == 'baseline':
                val_final_metrics = _validate_baseline(
                    model, val_loader, criterion, device, amp_dtype=amp_dtype, is_gru=is_gru
                )
            elif experiment_mode == 'synth_no_comp':
                eval_seed = None
                if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
                    eval_seed = int(config['experiment']['eval_noise_seed'])
                elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
                    eval_seed = int(config['synth_noise']['eval_seed'])
                val_final_metrics = _validate_synth(
                    synth_noise_model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            else:
                eval_seed = None
                if config.get('experiment', {}).get('eval_noise_seed', None) is not None:
                    eval_seed = int(config['experiment']['eval_noise_seed'])
                elif config.get('synth_noise', {}).get('eval_seed', None) is not None:
                    eval_seed = int(config['synth_noise']['eval_seed'])
                val_final_metrics = _validate_synth(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru, eval_seed=eval_seed,
                    synth_noise_config=synth_noise_config,
                )
            experiment_logger.info(
                f"Final model on val set: {val_final_metrics['acc1']:.2f}%, on test set: {test_metrics['acc1']:.2f}% "
                f"(val_size={len(val_loader.dataset)}, test_size={len(test_loader.dataset)})"
            )
            if best_epoch is not None:
                experiment_logger.info(
                    f"[DEBUG] After load-best: final_val={val_final_metrics['acc1']:.2f}% "
                    f"vs best_acc_tracked={best_acc:.2f}% (best_epoch={best_epoch})"
                )
    else:
        test_metrics = {'acc1': 0.0, 'loss': 0.0}
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'model_final.pth')
    checkpoint_data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'test_acc': test_metrics['acc1'],
        'config': config,
    }
    
    save_checkpoint(checkpoint_data, final_model_path)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Close loggers
    if tb_writer:
        tb_writer.close()
    if wandb_run and wandb is not None:
        wandb.finish()
    
    return {
        'best_val_acc': best_acc,
        'test_acc': test_metrics['acc1'],
        'test_loss': test_metrics['loss'],
        'metrics_history': metrics_history,
    }


def _train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Standard training without noise."""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    grad_norms = []
    grad_vars = []
    update_stds = []
    data_times = []
    step_times = []
    
    use_amp = scaler is not None and amp_dtype is not None
    
    t_prev_end = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        t_iter_start = time.perf_counter()
        data_times.append(t_iter_start - t_prev_end)
        
        data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
        data, target = data.to(device), target.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
        
        weights_before = []
        for param in model.parameters():
            if param.requires_grad:
                weights_before.append(param.data.clone())
        
        optimizer.zero_grad()  # ฅ^•ﻌ•^ฅ
        
        t_step_start = time.perf_counter()
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if is_gru and lengths is not None:
                    output = model(data, lengths=lengths)
                else:
                    output = model(data)
                loss = criterion(output, target)
        else:
            if is_gru and lengths is not None:
                output = model(data, lengths=lengths)
            else:
                output = model(data)
            loss = criterion(output, target)
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()  # ฅ^•ﻌ•^ฅ
        
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten())
        
        if grad_list:
            all_grads = torch.cat(grad_list)
            grad_norm = all_grads.norm().item()
            grad_norms.append(grad_norm)
            grad_var = all_grads.var().item()
            grad_vars.append(grad_var)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()  # ฅ^•ﻌ•^ฅ
        
        t_prev_end = time.perf_counter()
        step_times.append(t_prev_end - t_step_start)

        # 计算更新量
        if weights_before:
            updates = []
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad and param_idx < len(weights_before):
                    update = (param.data - weights_before[param_idx]).flatten()
                    updates.append(update)
                    param_idx += 1
            
            if updates:
                all_updates = torch.cat(updates)
                update_std = all_updates.std().item()
                update_stds.append(update_std)
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    avg_grad_var = np.mean(grad_vars) if grad_vars else 0.0
    std_grad_norm = np.std(grad_norms) if grad_norms else 0.0
    avg_update_std = np.mean(update_stds) if update_stds else 0.0
    
    avg_data_time = np.mean(data_times) if data_times else 0.0
    avg_step_time = np.mean(step_times) if step_times else 0.0
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'grad_norm': avg_grad_norm,
        'grad_norm_std': std_grad_norm,
        'grad_var': avg_grad_var,
        'update_std': avg_update_std,
        'data_time_avg': avg_data_time,
        'train_step_time_avg': avg_step_time,
    }


def _train_synth_hat(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Optional[Dict[str, Any]] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
    synth_noise_config: Optional[Any] = None,
) -> Dict[str, float]:
    """Hardware-aware training with synthetic noise during forward."""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    grad_norms = []
    grad_vars = []
    update_stds = []
    data_times = []
    step_times = []
    
    use_amp = scaler is not None and amp_dtype is not None
    
    # Control noise sampling frequency
    exp_cfg = config.get('experiment', {}) if config else {}
    noise_sampling_interval = exp_cfg.get('noise_sampling_interval', 1)
    resample_mode = exp_cfg.get('noise_resample_mode', None)  # None | 'step' | 'epoch'
    template_mixed_training = bool(exp_cfg.get('template_mixed_training', False))
    template_mix_prob = float(exp_cfg.get('template_mix_prob', 0.5))
    template_mix_prob = max(0.0, min(1.0, template_mix_prob))
    template_total_steps = 0
    template_resampled_steps = 0
    
    t_prev_end = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        t_iter_start = time.perf_counter()
        data_times.append(t_iter_start - t_prev_end)
        
        data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
        data, target = data.to(device), target.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
        
        weights_before = []
        for param in model.parameters():
            if param.requires_grad:
                weights_before.append(param.data.clone())
        
        optimizer.zero_grad()  # ฅ^•ﻌ•^ฅ
        
        t_step_start = time.perf_counter()
        
        # Determine seed for noise sampling
        global_step = epoch * len(train_loader) + batch_idx
        if resample_mode == 'epoch':
            # Same noise template within one epoch, refreshed each epoch.
            seed = 2000000 + epoch
        elif resample_mode == 'step':
            # Force deterministic per-step refresh so it is strictly different from epoch-wise.
            seed = 3000000 + global_step
        elif noise_sampling_interval > 1:
            interval_id = global_step // noise_sampling_interval
            seed = 1000000 + interval_id
        else:
            seed = None

        use_frozen_template = True
        if template_mixed_training:
            # Mixed-template HAT:
            # - with probability p: use frozen template (seed=None, cache/frozen behavior)
            # - with probability 1-p: use deterministic per-step seed + temporary resample flags
            #   to avoid memorizing one fixed template.
            template_total_steps += 1
            use_frozen_template = bool(np.random.rand() < template_mix_prob)
            if use_frozen_template:
                seed = None
            else:
                template_resampled_steps += 1
                seed = 4000000 + global_step
        
        template_attr_restore = {}
        if template_mixed_training and synth_noise_config is not None and not use_frozen_template:
            for attr, value in (
                ('drift_frozen', False),
                ('rank_resample', True),
                ('input_dependent_v_resample', True),
                ('sign_scale_v_resample', True),
                ('adv_direction_frozen', False),
            ):
                if hasattr(synth_noise_config, attr):
                    template_attr_restore[attr] = getattr(synth_noise_config, attr)
                    setattr(synth_noise_config, attr, value)

        logits_wrap = (
            synth_noise_config is not None
            and getattr(synth_noise_config, "backward_corruption_at", None) == "logits"
            and getattr(synth_noise_config, "noise_type", "") in ("sign_gradient_corruption", "adversarial_direction_bias")
        )
        try:
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    try:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths, seed=seed)
                        else:
                            output = model(data, seed=seed)
                    except TypeError:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths)
                        else:
                            output = model(data)
                    if logits_wrap:
                        output = apply_logits_backward_corruption(output, synth_noise_config, seed=seed)
                    loss = criterion(output, target)
            else:
                try:
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths, seed=seed)
                    else:
                        output = model(data, seed=seed)
                except TypeError:
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths)
                    else:
                        output = model(data)
                if logits_wrap:
                    output = apply_logits_backward_corruption(output, synth_noise_config, seed=seed)
                loss = criterion(output, target)
        finally:
            for attr, old_value in template_attr_restore.items():
                setattr(synth_noise_config, attr, old_value)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf detected in loss at epoch {epoch}, batch {batch_idx}. Skipping batch.")
            t_prev_end = time.perf_counter()
            continue
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()  # ฅ^•ﻌ•^ฅ
        
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten())
        
        if grad_list:
            all_grads = torch.cat(grad_list)
            grad_norm = all_grads.norm().item()
            grad_norms.append(grad_norm)
            grad_var = all_grads.var().item()
            grad_vars.append(grad_var)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()  # ฅ^•ﻌ•^ฅ
        
        t_prev_end = time.perf_counter()
        step_times.append(t_prev_end - t_step_start)
        
        if weights_before:
            updates = []
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad and param_idx < len(weights_before):
                    update = (param.data - weights_before[param_idx]).flatten()
                    updates.append(update)
                    param_idx += 1
            
            if updates:
                all_updates = torch.cat(updates)
                update_std = all_updates.std().item()
                update_stds.append(update_std)
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    avg_grad_var = np.mean(grad_vars) if grad_vars else 0.0
    std_grad_norm = np.std(grad_norms) if grad_norms else 0.0
    avg_update_std = np.mean(update_stds) if update_stds else 0.0
    
    avg_data_time = np.mean(data_times) if data_times else 0.0
    avg_step_time = np.mean(step_times) if step_times else 0.0
    result = {
        'loss': losses.avg,
        'acc1': top1.avg,
        'grad_norm': avg_grad_norm,
        'grad_norm_std': std_grad_norm,
        'grad_var': avg_grad_var,
        'update_std': avg_update_std,
        'data_time_avg': avg_data_time,
        'train_step_time_avg': avg_step_time,
    }
    if template_mixed_training and template_total_steps > 0:
        result['template_resampled_ratio'] = float(template_resampled_steps) / float(template_total_steps)
    if os.environ.get("SYNTH_NOISE_DIAGNOSTIC", "").strip() in ("1", "true", "yes") and synth_noise_config is not None:
        diag = get_and_reset_backward_diagnostic()
        if diag.get("sign_corrupt_calls", 0) > 0 or diag.get("adv_bias_calls", 0) > 0:
            result["backward_diag_sign_corrupt_calls"] = diag["sign_corrupt_calls"]
            result["backward_diag_sign_corrupt_flip_ratio"] = diag["sign_corrupt_flip_ratio"]
            result["backward_diag_adv_bias_calls"] = diag["adv_bias_calls"]
    return result


def _validate_baseline(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Standard validation without noise."""
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    use_amp = amp_dtype is not None
    
    with torch.no_grad():
        for batch in val_loader:
            data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
            data, target = data.to(device), target.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths)
                    else:
                        output = model(data)
                    loss = criterion(output, target)
            else:
                if is_gru and lengths is not None:
                    output = model(data, lengths=lengths)
                else:
                    output = model(data)
                loss = criterion(output, target)
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
    
    return {'loss': losses.avg, 'acc1': top1.avg}


def _validate_synth(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
    eval_seed: Optional[int] = None,
    synth_noise_config: Optional[Any] = None,
) -> Dict[str, float]:
    """Validation with synthetic noise applied."""
    old_clip_c = None
    if synth_noise_config is not None and getattr(synth_noise_config, "noise_type", "") == "deterministic_clip":
        ce = getattr(synth_noise_config, "clip_c_eval", None)
        if ce is not None:
            old_clip_c = synth_noise_config.clip_c
            synth_noise_config.clip_c = float(ce)

    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    use_amp = amp_dtype is not None
    
    try:
        with torch.no_grad():
            for batch in val_loader:
                data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
                data, target = data.to(device), target.to(device)
                if lengths is not None:
                    lengths = lengths.to(device)

                # Forward with noise
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        try:
                            if is_gru and lengths is not None:
                                output = model(data, lengths=lengths, seed=eval_seed)
                            else:
                                output = model(data, seed=eval_seed)
                        except TypeError:
                            if is_gru and lengths is not None:
                                output = model(data, lengths=lengths)
                            else:
                                output = model(data)
                        loss = criterion(output, target)
                else:
                    try:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths, seed=eval_seed)
                        else:
                            output = model(data, seed=eval_seed)
                    except TypeError:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths)
                        else:
                            output = model(data)
                    loss = criterion(output, target)

                acc1 = accuracy(output, target, topk=(1,))[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc1, data.size(0))

        return {'loss': losses.avg, 'acc1': top1.avg}
    finally:
        if old_clip_c is not None and synth_noise_config is not None:
            synth_noise_config.clip_c = old_clip_c


def _sync_weights_to_synth_model(base_model: nn.Module, synth_model: nn.Module) -> None:
    """
    Sync weights from base model to synth noise-wrapped model.
    
    Args:
        base_model: Base model (trained without noise)
        synth_model: Synth noise-wrapped model (for evaluation)
    """
    def _raw_model(m: nn.Module) -> nn.Module:
        # torch.compile wraps modules with _orig_mod; sync against raw module keys.
        return getattr(m, "_orig_mod", m)

    raw_base = _raw_model(base_model)

    # Fast path: if base_model and synth_model.base_model are the same object
    if hasattr(synth_model, 'base_model') and raw_base is _raw_model(synth_model.base_model):
        return

    # Preferred path: load directly into wrapped base_model (strict), with compiled/uncompiled fallback.
    if hasattr(synth_model, 'base_model'):
        raw_synth_base = _raw_model(synth_model.base_model)
        base_state = raw_base.state_dict()
        try:
            raw_synth_base.load_state_dict(base_state, strict=True)
            return
        except RuntimeError:
            # Fallback for compiled/uncompiled key prefix mismatch
            normalized = {}
            for k, v in base_state.items():
                if k.startswith("_orig_mod."):
                    normalized[k[len("_orig_mod."):]] = v
                else:
                    normalized[k] = v
            try:
                raw_synth_base.load_state_dict(normalized, strict=True)
                return
            except RuntimeError:
                prefixed = {f"_orig_mod.{k}": v for k, v in normalized.items()}
                raw_synth_base.load_state_dict(prefixed, strict=True)
                return

    # Fallback path for non-standard wrappers
    base_state = raw_base.state_dict()
    synth_state = synth_model.state_dict()
    synced_count = 0
    for base_name, base_param in base_state.items():
        synth_name = f'base_model.{base_name}'
        if synth_name in synth_state and synth_state[synth_name].shape == base_param.shape:
            synth_state[synth_name].data.copy_(base_param.data)
            synced_count += 1
            continue
        if base_name in synth_state and synth_state[base_name].shape == base_param.shape:
            synth_state[base_name].data.copy_(base_param.data)
            synced_count += 1

    missing_keys, unexpected_keys = synth_model.load_state_dict(synth_state, strict=False)
    if missing_keys or unexpected_keys:
        logger.warning(
            "Weight sync fallback mismatch: missing=%d unexpected=%d "
            "(showing up to 10 each): missing=%s unexpected=%s",
            len(missing_keys),
            len(unexpected_keys),
            missing_keys[:10],
            unexpected_keys[:10],
        )
    if synced_count == 0:
        raise RuntimeError(
            "Weight sync FAILED: No parameters synced in fallback path. "
            f"base_tensors={len(base_state)}"
        )


def _remap_bias_param_to_bias(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap keys ending with '.bias_param' to '.bias' so checkpoints saved with
    MemristorConv2d (which uses bias_param) can be loaded into SynthNoiseConv2d (which uses bias).
    """
    out = {}
    suffix = ".bias_param"
    for k, v in state_dict.items():
        if k.endswith(suffix):
            out[k[: -len(suffix)] + ".bias"] = v
        else:
            out[k] = v
    return out


def build_grad_model_and_loader_from_config(
    config: Dict[str, Any],
    device: torch.device,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Build the model and train loader from config (and optionally load state from checkpoint).
    Used to compute gradient metrics (A, C, V, S) from a saved checkpoint without re-running training.

    Returns:
        grad_model: Model to use for gradient metrics (with noise layers for synth modes).
        train_loader: DataLoader for getting batches.
        criterion: Loss criterion.
        is_gru: Whether dataset is AG News (GRU).
    """
    try:
        from ..utils.seeds import set_seed
    except ImportError:
        from src.utils.seeds import set_seed
    set_seed(config.get('seed'))

    try:
        from ..data.dataset import get_dataloaders
    except ImportError:
        from src.data.dataset import get_dataloaders

    dataset_name = config['dataset'].lower()
    vocab = None
    if dataset_name == 'agnews':
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
            seed=config.get('seed'),
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=config['dataset'],
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
            seed=config.get('seed'),
        )

    in_channels = 1 if dataset_name == 'mnist' else (3 if dataset_name in ('cifar10', 'cifar100') else config.get('in_channels', 3))
    num_classes = config.get('num_classes', 10 if dataset_name != 'agnews' else 4)
    if dataset_name == 'agnews':
        num_classes = config.get('num_classes', 4)

    model_kwargs = {}
    if config.get('model_name') == 'vit_tiny':
        model_kwargs['patch_size'] = config.get('patch_size', 4)
        model_kwargs['embed_dim'] = config.get('embed_dim', 192)
        model_kwargs['depth'] = config.get('depth', 6)
        model_kwargs['num_heads'] = config.get('num_heads', 3)
        model_kwargs['mlp_ratio'] = config.get('mlp_ratio', 4.0)
        model_kwargs['qkv_bias'] = config.get('qkv_bias', False)
    elif config.get('model_name') == 'gru_agnews':
        if vocab is None:
            raise ValueError("vocab required for GRU")
        model_kwargs['vocab_size'] = len(vocab)
        model_kwargs['embed_dim'] = config.get('embed_dim', 128)
        model_kwargs['hidden_dim'] = config.get('hidden_dim', 256)
        model_kwargs['num_layers'] = config.get('num_layers', 2)

    base_model = get_model(
        name=config['model_name'],
        num_classes=num_classes,
        in_channels=in_channels if dataset_name != 'agnews' else None,
        **model_kwargs
    )
    base_model = base_model.to(device)
    is_gru = (config.get('model_name') == 'gru_agnews')
    criterion = nn.CrossEntropyLoss()

    experiment_mode = config.get('experiment', {}).get('mode', 'baseline')
    if experiment_mode == 'baseline':
        if state_dict is not None:
            base_model.load_state_dict(state_dict, strict=True)
        return base_model, train_loader, criterion, is_gru

    synth_config = config.get('synth_noise', {})
    noise_type_map = {
        'full_variability': 'iid_multiplicative',
        'cond1_variance_bounded': 'heavy_tail',
        'cond2_gradient_unbiased': 'input_dependent',
        'cond3_adc_direct': 'gradient_degenerate',
    }
    old_nt = synth_config.get('noise_type', 'none')
    noise_type = noise_type_map.get(old_nt, old_nt)

    def _f(key, default):
        v = synth_config.get(key)
        return float(v) if v is not None else default
    def _b(key, default):
        v = synth_config.get(key)
        return bool(v) if v is not None else default

    synth_noise_config = SynthNoiseConfig(
        noise_type=noise_type,
        variability_sigma=_f('variability_sigma', 0.05),
        heavy_tail_alpha=_f('heavy_tail_alpha', _f('cond1_alpha', 0.1)),
        heavy_tail_nu=_f('heavy_tail_nu', _f('cond1_nu', 2.0)),
        input_dependent_alpha=_f('input_dependent_alpha', _f('cond2_alpha', 0.1)),
        decoupled_consistent_sigma=_f('decoupled_consistent_sigma', 0.05),
        decoupled_inconsistent_sigma=_f('decoupled_inconsistent_sigma', 0.05),
        coupled_consistent_alpha=_f('coupled_consistent_alpha', 0.1),
        coupled_inconsistent_alpha=_f('coupled_inconsistent_alpha', 0.1),
        adc_bits=_f('adc_bits', 8.0),
        enable_adc=_b('enable_adc', False),
        adc_backward_mode=synth_config.get('adc_backward_mode'),
        adv_direction_beta=_f('adv_direction_beta', 1.0),
        adv_direction_frozen=_b('adv_direction_frozen', True),
        adv_direction_random_sign=_b('adv_direction_random_sign', False),
        sign_corrupt_p=_f('sign_corrupt_p', 0.5),
        sign_corrupt_mode=synth_config.get('sign_corrupt_mode', 'flip'),
        sign_corrupt_noise_sigma=_f('sign_corrupt_noise_sigma', 1.0),
        saturation_gamma=_f('saturation_gamma', 5.0),
        saturation_alpha=_f('saturation_alpha', 1.0),
        drift_beta=_f('drift_beta', 0.3),
        drift_use_norm=_b('drift_use_norm', False),
        drift_frozen=_b('drift_frozen', True),
        drift_resample_when_eval=_b('drift_resample_when_eval', False),
        drift_d_mean=_f('drift_d_mean', 0.0),
        sign_scale_alpha=_f('sign_scale_alpha', 0.5),
        sign_scale_v_resample=_b('sign_scale_v_resample', False),
        rank_k=int(synth_config.get('rank_k', 4)),
        rank_fill_sigma=_f('rank_fill_sigma', 0.0),
        rank_resample=_b('rank_resample', False),
        rank_resample_when_eval=_b('rank_resample_when_eval', False),
        clip_c=_f('clip_c', 1.0),
        clip_c_eval=(float(synth_config['clip_c_eval']) if synth_config.get('clip_c_eval') is not None else None),
        clip_dither=_b('clip_dither', False),
        input_dependent_v_resample=_b('input_dependent_v_resample', False),
        seed=config.get('seed'),
        compensation_in_backward=_b('compensation_in_backward', True),
        backward_corruption_at=synth_config.get('backward_corruption_at'),
    )
    noise_injection_config = synth_config.get('noise_injection') if 'noise_injection' in synth_config else None

    if experiment_mode == 'synth_no_comp':
        if state_dict is not None:
            base_model.load_state_dict(state_dict, strict=True)
        base_for_noisy = copy.deepcopy(base_model)
        grad_model = wrap_model_with_synth_noise(
            base_for_noisy, synth_noise_config, noise_config=noise_injection_config
        )
    else:
        grad_model = wrap_model_with_synth_noise(
            base_model, synth_noise_config, noise_config=noise_injection_config
        )
        if state_dict is not None:
            # Remap bias_param -> bias so checkpoints from MemristorConv2d can be loaded into SynthNoiseConv2d
            state_dict_to_load = _remap_bias_param_to_bias(state_dict)
            grad_model.load_state_dict(state_dict_to_load, strict=True)
    grad_model = grad_model.to(device)
    grad_model.train()
    return grad_model, train_loader, criterion, is_gru
