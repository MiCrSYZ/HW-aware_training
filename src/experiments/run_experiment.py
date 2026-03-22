import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import os
import logging
import pandas as pd
import math
import time
import numpy as np

try:
    from ..models.model_zoo import get_model, wrap_model_with_memristor
    from ..memristor.device_model import MemristorDeviceModel
    from ..memristor.compensation import (
        hardware_aware_training,
    )
    from ..memristor.energy_estimator import EnergyEstimator
    from ..utils.metrics import AverageMeter, accuracy
    from ..utils.checkpoint import save_checkpoint, load_checkpoint
    from ..utils.logger import setup_logger, setup_tensorboard, setup_wandb
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
    from src.models.model_zoo import get_model, wrap_model_with_memristor
    from src.memristor.device_model import MemristorDeviceModel
    from src.memristor.compensation import (
        hardware_aware_training,
    )
    from src.memristor.energy_estimator import EnergyEstimator
    from src.utils.metrics import AverageMeter, accuracy
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from src.utils.logger import setup_logger, setup_tensorboard, setup_wandb
    try:
        from src.utils.vit_metrics import (
            collect_gradient_norms_by_tier,
            collect_activation_stats,
            compute_logit_margin,
            register_activation_hooks,
            compute_update_norm_by_tier,
        )
    except ImportError:
        # ViT metrics not available, set to None
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
    """
    Unpack batch data, handling both image datasets and AG News.
    
    Args:
        batch: Batch from dataloader
        is_agnews: Whether this is AG News dataset (returns labels, texts, lengths)
        
    Returns:
        Tuple of (data, target, lengths) where lengths is None for non-text datasets
    """
    if is_agnews:
        labels, texts, lengths = batch
        return texts, labels, lengths
    else:
        data, target = batch
        return data, target, None


def compute_boundary_regularization(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    beta: float = 0.8,
) -> torch.Tensor:
    """
    计算远离边界的正则化损失。
    
    L_boundary(W) = (1/N) * sum_i (max(|W_i| - β*w_max, 0))^2
    
    Args:
        model: 模型（可能是 MemristorModel wrapper，需要访问 base_model）
        device_model: MemristorDeviceModel 实例，包含 wmax 信息
        beta: 边界阈值比例，默认 0.8
        
    Returns:
        正则化损失值（标量）
    """
    # 获取实际的模型
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    # 获取 wmax（权重裁剪上界）
    wmax = device_model.wmax
    
    # 计算阈值
    threshold = beta * wmax
    
    total_reg = None
    total_params = 0
    
    # 遍历所有 memristor 层
    for module in target_model.modules():
        # 检查是否是 memristor 层（有权重参数）
        if hasattr(module, 'weight') and module.weight is not None:
            # 检查是否是 memristor 相关的层
            # MemristorLinear, MemristorConv2d, 以及 learned mapping 的层都有 weight
            module_name = type(module).__name__
            if 'Memristor' in module_name or hasattr(module, 'device_model'):
                W = module.weight
                N = W.numel()
                
                # 计算远离边界的正则化
                # max(|W_i| - β*w_max, 0)^2
                abs_W = torch.abs(W)
                excess = torch.clamp(abs_W - threshold, min=0.0)
                layer_reg = (excess ** 2).sum() / N
                
                if total_reg is None:
                    total_reg = layer_reg
                else:
                    total_reg = total_reg + layer_reg
                total_params += 1
    
    if total_params > 0:
        return total_reg
    else:
        # 如果没有找到 memristor 层，返回零张量
        device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        return torch.tensor(0.0, device=device, requires_grad=True)


def run_experiment(
    config: Dict[str, Any],
    output_dir: str,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete experiment based on configuration.
    
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
        # Enable cuDNN benchmark for faster convolutions (finds optimal algorithms)
        torch.backends.cudnn.benchmark = True
        # Disable deterministic mode for better performance (if reproducibility not critical)
        # torch.backends.cudnn.deterministic = False  # Uncomment if you don't need exact reproducibility
        experiment_logger.info("CUDA optimizations enabled: cudnn.benchmark=True")
    
    # Data loaders
    try:
        from ..data.dataset import get_dataloaders
    except ImportError:
        from src.data.dataset import get_dataloaders
    
    # Determine input channels and num_classes based on dataset BEFORE loading data
    dataset_name = config['dataset'].lower()
    vocab = None  # For text datasets
    if dataset_name == 'mnist':
        in_channels = 1  # Grayscale
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar10':
        in_channels = 3  # RGB
        num_classes = config.get('num_classes', 10)
    elif dataset_name == 'cifar100':
        in_channels = 3  # RGB
        num_classes = config.get('num_classes', 100)
    elif dataset_name == 'agnews':
        num_classes = config.get('num_classes', 4)
        # AG News returns vocab as 4th element
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
        in_channels = config.get('in_channels', 3)  # Default to RGB
        num_classes = config.get('num_classes', 10)  # Default to 10
    
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
    
    # Validate label ranges in the dataset
    # This helps catch configuration errors early
    experiment_logger.info("Validating dataset labels...")
    max_label = -1
    min_label = float('inf')
    sample_count = 0
    
    # Check a few batches from train_loader
    is_agnews = (dataset_name == 'agnews')
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Check first 5 batches
            break
        data, target, lengths = _unpack_batch(batch, is_agnews=is_agnews)
        batch_max = target.max().item()
        batch_min = target.min().item()
        max_label = max(max_label, batch_max)
        min_label = min(min_label, batch_min)
        sample_count += len(target)
    
    experiment_logger.info(f"Label range in dataset: [{min_label}, {max_label}], expected: [0, {num_classes-1}]")
    
    if max_label >= num_classes or min_label < 0:
        error_msg = (
            f"Label mismatch detected! Dataset labels range [{min_label}, {max_label}], "
            f"but model expects [0, {num_classes-1}]. "
            f"Please check your config: dataset={dataset_name}, num_classes={num_classes}. "
            f"For CIFAR-100, num_classes should be 100. For CIFAR-10, num_classes should be 10."
        )
        experiment_logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Model
    # Extract model-specific parameters from config
    model_kwargs = {}
    if config['model_name'] == 'vit_tiny':
        # ViT-specific parameters
        model_kwargs['patch_size'] = config.get('patch_size', 4)
        model_kwargs['embed_dim'] = config.get('embed_dim', 192)
        model_kwargs['depth'] = config.get('depth', 6)
        model_kwargs['num_heads'] = config.get('num_heads', 3)
        model_kwargs['mlp_ratio'] = config.get('mlp_ratio', 4.0)
        model_kwargs['qkv_bias'] = config.get('qkv_bias', False)
    elif config['model_name'] == 'gru_agnews':
        # GRU-specific parameters
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
    
    # Device model (for memristor experiments)
    device_model = None
    memristor_model = None  # Wrapped model for memristor evaluation
    
    if config['experiment']['mode'] != 'baseline':
        memristor_config = config['memristor']
        
        # 读取参数
        array_size = memristor_config.get('array_size', 128)
        adc_bits = memristor_config.get('adc_bits', 8)
        enable_update_model = memristor_config.get('enable_update_model', False)
        enable_adc = memristor_config.get('enable_adc', False)
        adc_add_noise = memristor_config.get('adc_add_noise', False)
        enable_energy = memristor_config.get('enable_energy', False)
        
        # 读取更新模型参数
        update_params = memristor_config.get('update_params', None)
        if update_params is None:
            update_params = {
                'A_plus': memristor_config.get('A_plus', 1e-5),
                'A_minus': memristor_config.get('A_minus', 1e-5),
                'p_plus': memristor_config.get('p_plus', 1.0),
                'p_minus': memristor_config.get('p_minus', 1.0),
                'gamma': memristor_config.get('gamma', 1.0),
                'write_noise_ratio': memristor_config.get('write_noise_ratio', 0.05),
            }
        else:
            # 确保所有参数都存在
            default_update = {
                'A_plus': 1e-5, 'A_minus': 1e-5, 'p_plus': 1.0, 'p_minus': 1.0,
                'gamma': 1.0, 'write_noise_ratio': 0.05
            }
            for key, val in default_update.items():
                if key not in update_params:
                    update_params[key] = val
        
        # 读取能耗系数
        energy_coefs = memristor_config.get('energy_coefs', None)
        if energy_coefs is None:
            energy_coefs = {
                'alpha': memristor_config.get('energy_alpha', 1.0),
                'beta': memristor_config.get('energy_beta', 1.0),
            }
        else:
            default_energy = {'alpha': 1.0, 'beta': 1.0}
            for key, val in default_energy.items():
                if key not in energy_coefs:
                    energy_coefs[key] = val
        
        # 读取电导漂移时间设置方式
        drift_time_mode = memristor_config.get('drift_time_mode', 'accumulate')
        drift_time_fixed = memristor_config.get('drift_time_fixed', 0)
        
        # 读取IR-drop模型参数
        ir_drop_mode = memristor_config.get('ir_drop_mode', 'none')
        ir_drop_gamma = memristor_config.get('ir_drop_gamma', 0.35)
        ir_drop_scaling = memristor_config.get('ir_drop_scaling', 1.0)
        # crossbar模式参数
        ir_drop_eta = memristor_config.get('ir_drop_eta', 1.0)
        ir_drop_cap = memristor_config.get('ir_drop_cap', 0.10)
        ir_drop_norm = memristor_config.get('ir_drop_norm', 'mean')
        ir_drop_train_enabled = memristor_config.get('ir_drop_train_enabled', True)  # 默认True保持向后兼容
        
        # 读取写入更新参数
        write_config = memristor_config.get('write', {})
        write_t_min = float(write_config.get('t_min', 5e-9))  # 最小脉冲宽度
        write_t_scale = float(write_config.get('t_scale', 50e-9))  # 脉冲时间的放大因子
        write_V_write = float(write_config.get('V_write', 1.2))  # 写入电压
        write_max_pulses = int(write_config.get('max_pulses', 200))  # 最大脉冲数
        write_tolerance = float(write_config.get('tolerance', 0.02))  # 容差（相对于电导范围的比例）
        # write_interval 默认等于 epochs（训练完成后一次性写入）
        # 会在后面读取 epochs 后设置默认值
        
        device_model = MemristorDeviceModel(
            G_min=float(memristor_config['G_min']),
            G_max=float(memristor_config['G_max']),
            weight_clip=tuple(float(x) for x in memristor_config['weight_clip']),
            variability_sigma=float(memristor_config['variability_sigma']),
            read_noise_sigma=float(memristor_config['read_noise_sigma']),
            drift_alpha=float(memristor_config['drift_alpha']),
            stuck_ratio=float(memristor_config['stuck_ratio']),
            stuck_low_prob=float(memristor_config['stuck_low_prob']),
            ir_drop_beta=float(memristor_config['ir_drop_beta']),
            mapping=str(memristor_config['mapping']),
            seed=config.get('seed'),

            array_size=int(array_size),
            adc_bits=int(adc_bits),
            enable_update_model=bool(enable_update_model),
            enable_adc=bool(enable_adc),
            adc_add_noise=bool(adc_add_noise),
            enable_energy=bool(enable_energy),
            update_params=update_params,
            energy_coefs=energy_coefs,
            drift_time_mode=str(drift_time_mode),
            drift_time_fixed=int(drift_time_fixed),

            ir_drop_mode=str(ir_drop_mode),
            ir_drop_gamma=float(ir_drop_gamma),
            ir_drop_scaling=float(ir_drop_scaling),
            ir_drop_eta=float(ir_drop_eta),
            ir_drop_cap=float(ir_drop_cap),
            ir_drop_norm=str(ir_drop_norm),
            ir_drop_train_enabled=bool(ir_drop_train_enabled),
            enable_adc_during_training=bool(memristor_config.get('enable_adc_during_training', False)),
            adc_training_mode=str(memristor_config.get('adc_training_mode', 'ste')),  # Default to 'ste' for backward compatibility
            enable_ir_drop_paper_during_training=bool(memristor_config.get('enable_ir_drop_paper_during_training', False)),
            # 合成噪声参数（从配置传入，否则合成噪声永远不会生效）
            synthetic_noise_type=str(memristor_config.get('synthetic_noise_type', 'none')),
            cond1_alpha=float(memristor_config.get('cond1_alpha', 0.1)),
            cond1_nu=float(memristor_config.get('cond1_nu', 2.0)),
            cond2_alpha=float(memristor_config.get('cond2_alpha', 0.1)),
        )
        
        # Log ADC training mode and synthetic noise for debugging
        adc_mode_from_config = memristor_config.get('adc_training_mode', None)
        print(f"[DEBUG CONFIG] adc_training_mode from config: {adc_mode_from_config}")
        print(f"[DEBUG CONFIG] adc_training_mode in device_model: {device_model.adc_training_mode}")
        synthetic_type = getattr(device_model, 'synthetic_noise_type', 'none')
        if synthetic_type != 'none':
            experiment_logger.info(
                f"Synthetic noise: type={synthetic_type}, cond1_alpha={device_model.cond1_alpha}, "
                f"cond1_nu={device_model.cond1_nu}, cond2_alpha={device_model.cond2_alpha}"
            )
            print(f"[DEBUG CONFIG] synthetic_noise_type={synthetic_type} (cond1_alpha={device_model.cond1_alpha}, cond2_alpha={device_model.cond2_alpha})")
        if device_model.enable_adc_during_training:
            logger.info(f"ADC training enabled with mode: {device_model.adc_training_mode}")
            print(f"[DEBUG] Device model ADC settings: enable_adc={device_model.enable_adc}, "
                  f"enable_adc_during_training={device_model.enable_adc_during_training}, "
                  f"adc_training_mode={device_model.adc_training_mode}")
        
        # Create memristor-wrapped model for evaluation
        # Extract noise injection configuration if available
        noise_config = None
        if 'memristor' in config and 'noise_injection' in config['memristor']:
            noise_config = config['memristor']['noise_injection']
        
        # wrap_model_with_memristor 会就地替换传入模型中的 Linear/Conv2d 为 Memristor 层。
        # no_comp 要求训练用干净模型、评估用带噪模型，因此 no_comp 时必须包装 base 的深拷贝，
        # 这样 base_model 保持 nn.Linear/nn.Conv2d，训练时无噪声；评估时用包装后的副本加噪声。
        if config['experiment']['mode'] == 'memristor_no_comp':
            base_for_noisy = copy.deepcopy(base_model)
            memristor_model = wrap_model_with_memristor(
                base_for_noisy, device_model, noise_config=noise_config
            )
        else:
            memristor_model = wrap_model_with_memristor(
                base_model, device_model, noise_config=noise_config
            )
        memristor_model = memristor_model.to(device)
        
        # For memristor_with_comp (HAT), use memristor model for training (with noise + compensation)
        # For memristor_no_comp, ALWAYS use base model for training (no noise during training),
        # and use memristor model only for evaluation/testing (noise injected at inference time)
        if config['experiment']['mode'] == 'memristor_with_comp':
            model = memristor_model
        else:  # memristor_no_comp
            # no_comp: train clean (base_model), eval with noise (memristor_model)
            model = base_model
            experiment_logger.info("memristor_no_comp mode: using base_model for training (no noise), memristor_model for eval/test (with noise)")
    else:  # baseline
        model = base_model
        # baseline模式不需要写入参数，设置默认值
        write_t_min = 5e-9
        write_t_scale = 50e-9
        write_V_write = 1.2
        write_max_pulses = 200
        write_tolerance = 0.02
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training setup
    use_amp = config.get('mixed_precision', False)  # Default to False for backward compatibility
    if isinstance(use_amp, str):
        # Support 'fp16', 'bf16', 'true', 'false'
        use_amp = use_amp.lower() in ['fp16', 'bf16', 'true', '1']
    amp_dtype = None
    if use_amp:
        if torch.cuda.is_available():
            # Check if bfloat16 is supported (Ampere+ GPUs)
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                # Prefer bfloat16 for better numerical stability
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
        # AdamW uses decoupled weight decay (better for transformers)
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
    epochs = config.get('epochs', 100)  # Default to 100 epochs if not specified
    
    # 读取写入间隔（如果未设置，默认为epochs，即训练完成后一次性写入）
    if config['experiment']['mode'] != 'baseline':
        write_config = config['memristor'].get('write', {})
        write_interval = write_config.get('write_interval', epochs)
        write_interval = int(write_interval)
    else:
        write_interval = epochs  # baseline模式不需要写入
    
    if 'scheduler' in config and config['scheduler']:
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config['type']
        
        # Check if warmup is enabled
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        if scheduler_type == 'cosine':
            if warmup_epochs > 0:
                # Use warmup + cosine annealing
                # Try to use SequentialLR if available (PyTorch 1.13+), otherwise use LambdaLR
                try:
                    from torch.optim.lr_scheduler import SequentialLR, LinearLR
                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=0.01,  # Start from 1% of base LR
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
                    # Fallback for PyTorch < 1.13: use LambdaLR for warmup
                    def lr_lambda(epoch):
                        if epoch < warmup_epochs:
                            # Linear warmup from 0.01 to 1.0
                            return 0.01 + (1.0 - 0.01) * epoch / warmup_epochs
                        else:
                            # Cosine annealing after warmup
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
                # Use warmup + step decay
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
                    # Fallback for PyTorch < 1.13: use LambdaLR for warmup
                    step_size = int(scheduler_config.get('step_size', 30))
                    gamma = float(scheduler_config.get('gamma', 0.1))
                    def lr_lambda(epoch):
                        if epoch < warmup_epochs:
                            # Linear warmup from 0.01 to 1.0
                            return 0.01 + (1.0 - 0.01) * epoch / warmup_epochs
                        else:
                            # Step decay after warmup
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
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        experiment_logger.info(f"Resumed from epoch {start_epoch}")
    
    # TensorBoard and W&B
    tb_writer = setup_tensorboard(os.path.join(output_dir, 'tensorboard_logs'))
    wandb_run = setup_wandb(
        project_name=config['logging'].get('project_name', 'memristor-nn'),
        run_name=config['logging'].get('run_name'),
        config=config,
        enabled=config['logging'].get('use_wandb', False),
    )

    # Training loop
    experiment_mode = config['experiment']['mode']
    metrics_history = []
    
    
    # Time tracking for ETA calculation
    epoch_times = []
    training_start_time = time.time()
    
    # Log training loop start
    steps_per_epoch = len(train_loader)
    experiment_logger.info(f"Starting training loop: start_epoch={start_epoch}, epochs={epochs}")
    experiment_logger.info(f"Steps per epoch: {steps_per_epoch} (same for baseline/comp/no_comp)")
    experiment_logger.info(f"Experiment mode: {experiment_mode}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        #experiment_logger.info(f"Epoch {epoch}/{epochs-1} starting...")
        
        # 重置能耗统计（如果启用，每个epoch开始时重置）
        if device_model and hasattr(device_model, 'enable_energy') and device_model.enable_energy:
            device_model.reset_energy_stats()
        
        # Determine if GRU model
        is_gru = (config['model_name'] == 'gru_agnews')
        
        # Train
        if experiment_mode == 'baseline':
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru
            )
        elif experiment_mode == 'memristor_no_comp':
            # no_comp: always train with base_model (no noise during training)
            # Noise is only injected during evaluation/testing via memristor_model
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru
            )
            # Sync weights from base_model to memristor_model for evaluation
            # This ensures memristor_model has the latest trained weights before eval/test
            if model is base_model:  # Should always be true for no_comp
                _sync_weights_to_memristor_model(base_model, memristor_model)
        elif experiment_mode == 'memristor_with_comp':
            train_metrics = _train_hat(
                model, train_loader, criterion, optimizer, device, epoch, device_model, config,
                scaler=scaler, amp_dtype=amp_dtype, is_gru=is_gru
            )
        else:
            raise ValueError(f"Unknown experiment mode: {experiment_mode}")
        
        # 应用写入更新（根据 write_interval）
        # memristor_no_comp 不执行 writeback：no_comp 仅“训练+推理时加噪声”，不模拟训练后写入硬件，
        # 且 writeback 会覆盖权重导致准确率崩溃，且耗时极长（多脉冲仿真）。
        # 如果 write_interval = epochs，则在训练循环结束后执行（避免重复）
        # 否则，在训练循环中按间隔执行
        if (experiment_mode != 'baseline' and 
            experiment_mode != 'memristor_no_comp' and
            device_model and 
            device_model.enable_update_model and 
            write_interval < epochs and  # 只在非一次性写入时在训练循环中执行
            (epoch + 1) % write_interval == 0):
            experiment_logger.info(f"Applying writeback at epoch {epoch}")
            _apply_writeback(
                model, device_model, 
                write_t_min, write_t_scale, write_V_write,
                max_pulses=write_max_pulses,
                tolerance=write_tolerance
            )
            # memristor_no_comp 不在此分支内，无需同步
        
        # Validate (with timing)
        eval_start_time = time.perf_counter()
        if val_loader is not None:
            if experiment_mode == 'baseline':
                val_metrics = _validate_baseline(
                    model, val_loader, criterion, device,
                    amp_dtype=amp_dtype, is_gru=is_gru
                )
            elif experiment_mode == 'memristor_no_comp':
                # For no_comp: use memristor model for validation (apply non-idealities)
                val_metrics = _validate_memristor(
                    memristor_model, val_loader, criterion, device, device_model, amp_dtype=amp_dtype, is_gru=is_gru
                )
            else:  # memristor_with_comp
                # For with_comp: model is already memristor-wrapped
                val_metrics = _validate_memristor(
                    model, val_loader, criterion, device, device_model, amp_dtype=amp_dtype, is_gru=is_gru
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
        
        # Add gradient statistics if available (for training dynamics analysis)
        if 'grad_norm' in train_metrics:
            metrics['grad_norm'] = train_metrics['grad_norm']
            metrics['grad_norm_std'] = train_metrics.get('grad_norm_std', 0.0)
            metrics['grad_var'] = train_metrics.get('grad_var', 0.0)
        if 'update_std' in train_metrics:
            metrics['update_std'] = train_metrics.get('update_std', 0.0)
        # Timing
        metrics['data_time_avg'] = train_metrics.get('data_time_avg', 0.0)
        metrics['train_step_time_avg'] = train_metrics.get('train_step_time_avg', 0.0)
        metrics['eval_time'] = eval_time
        
        # Add ViT/GRU-specific metrics if available
        is_vit = config.get('model_name', '') == 'vit_tiny'
        is_gru = config.get('model_name', '') == 'gru_agnews'
        for key in train_metrics:
            if key.startswith('grad_norm_') or key.startswith('update_norm_') or \
               key.startswith('logit_margin_') or key.startswith('act_'):
                metrics[key] = train_metrics[key]
        
        # Use appropriate metrics collection based on model type
        if is_vit and collect_gradient_norms_by_tier is not None:
            # ViT metrics already collected in _train_hat
            pass
        elif is_gru and gru_collect_gradient_norms_by_tier is not None:
            # GRU metrics already collected in _train_hat
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
            if 'synthetic_noise_avg_g' in metrics:
                tb_writer.add_scalar('SyntheticNoise/AvgG', metrics['synthetic_noise_avg_g'], epoch)
                tb_writer.add_scalar('SyntheticNoise/AvgR', metrics['synthetic_noise_avg_r'], epoch)
                tb_writer.add_scalar('SyntheticNoise/EMANorm', metrics['synthetic_noise_ema_norm'], epoch)
            
            # Log ViT/GRU-specific metrics
            is_vit = config.get('model_name', '') == 'vit_tiny'
            is_gru = config.get('model_name', '') == 'gru_agnews'
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
                    # Format: act_{tier}_{stat_name}
                    parts = key.replace('act_', '').split('_', 1)
                    if len(parts) == 2:
                        tier, stat = parts
                        tb_writer.add_scalar(f'{prefix}/Act_{tier}_{stat}', value, epoch)
        
        if wandb_run:
            wandb.log(metrics, step=epoch)
        
        # Calculate epoch time and ETA
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate average epoch time and ETA
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            eta_str = f"{hours}h{minutes}m"
        
        # Format epoch time
        if epoch_time < 60:
            epoch_time_str = f"{epoch_time:.1f}s"
        elif epoch_time < 3600:
            epoch_time_str = f"{epoch_time / 60:.1f}m"
        else:
            hours = int(epoch_time // 3600)
            minutes = int((epoch_time % 3600) // 60)
            epoch_time_str = f"{hours}h{minutes}m"
        
        # 构建日志信息
        log_msg = (
            f"Epoch {epoch}/{epochs-1}: train_loss={metrics['train_loss']:.4f}, "
            f"train_acc={metrics['train_acc1']:.2f}%, "
            f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc1']:.2f}%"
        )
        
        # 添加梯度统计信息和更新量统计（如果可用）
        if 'grad_norm' in metrics:
            log_msg += f" | grad_norm={metrics['grad_norm']:.4e}"
            if metrics.get('grad_norm_std', 0.0) > 0:
                log_msg += f"±{metrics['grad_norm_std']:.4e}"
            log_msg += f" | grad_var={metrics.get('grad_var', 0.0):.4e}"
        if 'update_std' in metrics:
            log_msg += f" | update_std={metrics['update_std']:.4e}"
        
        log_msg += f" | Time: {epoch_time_str} | ETA: {eta_str}"
        # Timing breakdown
        if 'data_time_avg' in metrics and 'train_step_time_avg' in metrics:
            log_msg += (
                f" | data_time={metrics['data_time_avg']*1000:.1f}ms"
                f" | train_step_time={metrics['train_step_time_avg']*1000:.1f}ms"
                f" | eval_time={metrics.get('eval_time', 0):.2f}s"
            )
        
        # 如果启用了能耗估计，添加能耗信息
        if device_model and hasattr(device_model, 'enable_energy') and device_model.enable_energy:
            energy_stats = device_model.get_energy_stats()
            if energy_stats:
                log_msg += (
                    f" | Energy: write={energy_stats['write']:.6e}, "
                    f"read={energy_stats['read']:.6e}, "
                    f"total={energy_stats['write'] + energy_stats['read']:.6e}"
                )
        
        experiment_logger.info(log_msg)
        
        # Save best model checkpoint
        is_best = val_metrics['acc1'] > best_acc
        if is_best:
            best_acc = val_metrics['acc1']
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
    
    # 在训练循环结束后、最终测试前，如果 write_interval = epochs，执行最后一次 writeback
    # memristor_no_comp 不执行：会覆盖权重导致准确率崩溃，且多脉冲仿真耗时极长
    if (experiment_mode != 'baseline' and 
        experiment_mode != 'memristor_no_comp' and
        device_model and 
        device_model.enable_update_model and 
        write_interval == epochs):
        experiment_logger.info("Applying final writeback before test evaluation")
        _apply_writeback(
            model, device_model, 
            write_t_min, write_t_scale, write_V_write,
            max_pulses=write_max_pulses,
            tolerance=write_tolerance
        )
        # memristor_no_comp 不在此分支内，无需同步
    
    # Final evaluation on test set
    if test_loader:
        if experiment_mode == 'baseline':
            test_metrics = _validate_baseline(
                model, test_loader, criterion, device, is_gru=is_gru
            )
        elif experiment_mode == 'memristor_no_comp':
            # For no_comp: always sync weights from base_model (trained without noise) to memristor_model
            # Then use memristor_model for test evaluation (with noise injected)
            # Note: model should always be base_model for no_comp (we train clean, eval with noise)
            _sync_weights_to_memristor_model(base_model, memristor_model)
            # Use memristor model for test (apply non-idealities)
            test_metrics = _validate_memristor(memristor_model, test_loader, criterion, device, device_model, is_gru=is_gru)
        else:  # memristor_with_comp
            # For with_comp: model is already memristor-wrapped
            test_metrics = _validate_memristor(model, test_loader, criterion, device, device_model, is_gru=is_gru)
        # 构建测试日志信息
        test_log_msg = (
            f"Test accuracy: {test_metrics['acc1']:.2f}%, loss: {test_metrics['loss']:.4f}"
        )
        
        # 如果启用了能耗估计，添加能耗信息
        if 'energy_stats' in test_metrics and test_metrics['energy_stats']:
            energy_stats = test_metrics['energy_stats']
            test_log_msg += (
                f" | Energy: write={energy_stats['write']:.6e}, "
                f"read={energy_stats['read']:.6e}, "
                f"total={energy_stats['write'] + energy_stats['read']:.6e}"
            )
        
        experiment_logger.info(test_log_msg)
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
    
    # Energy estimation (if enabled)
    energy_metrics = None
    if config['experiment'].get('energy_estimation', False) and device_model:
        energy_config = config['experiment'].get('energy_params', {})
        estimator = EnergyEstimator(
            subarray_size=energy_config.get('subarray_size', 128),
            num_subarrays=energy_config.get('num_subarrays', 1),
            technology_node_nm=energy_config.get('technology_node_nm', 45),
        )
        energy_metrics = estimator.estimate(model, device_model, test_loader, num_samples=100)
        experiment_logger.info(f"Energy: {energy_metrics['energy_joules']*1e9:.2f} nJ")
    
    # Close loggers
    if tb_writer:
        tb_writer.close()
    if wandb_run:
        wandb.finish()
    
    return {
        'best_val_acc': best_acc,
        'test_acc': test_metrics['acc1'],
        'test_loss': test_metrics['loss'],
        'energy_metrics': energy_metrics,
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
    """Standard training without memristor non-idealities."""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Gradient statistics for training dynamics analysis
    grad_norms = []  # List of gradient norms per batch
    grad_vars = []  # List of gradient variances per batch
    update_stds = []  # List of update standard deviations per batch
    
    # Timing: data_time (per batch), train_step_time (forward+backward+opt per batch)
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
        
        # Validate labels before forward pass (only on first batch of first epoch to avoid overhead)
        if epoch == 0 and batch_idx == 0:
            max_label = target.max().item()
            min_label = target.min().item()
            # Get model output size
            with torch.no_grad():
                if is_gru and lengths is not None:
                    sample_lengths = lengths[:1] if lengths is not None else None
                    sample_output = model(data[:1], lengths=sample_lengths)
                else:
                    sample_output = model(data[:1])
                model_output_size = sample_output.shape[1]
            
            if max_label >= model_output_size or min_label < 0:
                error_msg = (
                    f"Label mismatch in training batch! Labels range [{min_label}, {max_label}], "
                    f"but model output size is {model_output_size}. "
                    f"This will cause CUDA device-side assert error. "
                    f"Please check your config: num_classes should match the dataset. "
                    f"For CIFAR-100, num_classes must be 100. For CIFAR-10, num_classes must be 10."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Save weights before update (for computing update statistics)
        weights_before = []
        for param in model.parameters():
            if param.requires_grad:
                weights_before.append(param.data.clone())
        
        optimizer.zero_grad()
        
        t_step_start = time.perf_counter()
        # Mixed precision forward pass (GRU needs lengths for pack_padded_sequence)
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
        
        # Backward pass with gradient scaling if using AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute gradient statistics before optimizer.step()
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten())
        
        if grad_list:
            # Concatenate all gradients
            all_grads = torch.cat(grad_list)
            
            # Compute gradient norm
            grad_norm = all_grads.norm().item()
            grad_norms.append(grad_norm)
            
            # Compute gradient variance (across all parameters)
            grad_var = all_grads.var().item()
            grad_vars.append(grad_var)
            
            # Print gradient statistics for each layer (only on first batch of first few epochs)
            if batch_idx == 0 and epoch < 3:
                print("=" * 80)
                print(f"Gradient Statistics (Epoch {epoch}, Batch {batch_idx}) - Baseline:")
                print("-" * 80)
                
                # Get base model if wrapped
                base_model = model
                if hasattr(model, 'base_model'):
                    base_model = model.base_model
                
                layer_grad_stats = {}
                for name, param in base_model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_max = param.grad.abs().max().item()
                        grad_mean = param.grad.mean().item()
                        param_norm = param.data.norm().item()
                        
                        # Determine layer type
                        layer_type = "unknown"
                        if 'embedding' in name:
                            layer_type = "embedding"
                        elif 'gru' in name:
                            if 'weight_ih' in name:
                                layer_type = "gru_weight_ih"
                            elif 'weight_hh' in name:
                                layer_type = "gru_weight_hh"
                            elif 'bias' in name:
                                layer_type = "gru_bias"
                            else:
                                layer_type = "gru_other"
                        elif 'head' in name:
                            layer_type = "head"
                        
                        # Group by layer type
                        if layer_type not in layer_grad_stats:
                            layer_grad_stats[layer_type] = {
                                'names': [],
                                'grad_norms': [],
                                'grad_maxs': [],
                                'grad_means': [],
                                'param_norms': [],
                            }
                        
                        layer_grad_stats[layer_type]['names'].append(name)
                        layer_grad_stats[layer_type]['grad_norms'].append(grad_norm)
                        layer_grad_stats[layer_type]['grad_maxs'].append(grad_max)
                        layer_grad_stats[layer_type]['grad_means'].append(grad_mean)
                        layer_grad_stats[layer_type]['param_norms'].append(param_norm)
                    else:
                        # No gradient
                        layer_type = "unknown"
                        if 'embedding' in name:
                            layer_type = "embedding"
                        elif 'gru' in name:
                            layer_type = "gru"
                        elif 'head' in name:
                            layer_type = "head"
                        
                        if layer_type not in layer_grad_stats:
                            layer_grad_stats[layer_type] = {
                                'names': [],
                                'grad_norms': [],
                                'grad_maxs': [],
                                'grad_means': [],
                                'param_norms': [],
                            }
                        layer_grad_stats[layer_type]['names'].append(name)
                        layer_grad_stats[layer_type]['grad_norms'].append(0.0)
                        layer_grad_stats[layer_type]['grad_maxs'].append(0.0)
                        layer_grad_stats[layer_type]['grad_means'].append(0.0)
                        layer_grad_stats[layer_type]['param_norms'].append(param.data.norm().item())
                
                # Print statistics by layer type
                for layer_type in sorted(layer_grad_stats.keys()):
                    stats = layer_grad_stats[layer_type]
                    if not stats['names']:
                        continue
                    
                    avg_grad_norm = np.mean(stats['grad_norms'])
                    max_grad_norm = np.max(stats['grad_norms'])
                    avg_grad_max = np.mean(stats['grad_maxs'])
                    avg_grad_mean = np.mean(stats['grad_means'])
                    zero_grad_count = sum(1 for n in stats['grad_norms'] if n < 1e-8)
                    
                    status = "✓" if avg_grad_norm > 1e-6 else "✗"
                    print(f"{status} {layer_type:20s}: "
                          f"avg_grad_norm={avg_grad_norm:.2e}, "
                          f"max_grad_norm={max_grad_norm:.2e}, "
                          f"avg_grad_max={avg_grad_max:.2e}, "
                          f"zero_grad_layers={zero_grad_count}/{len(stats['names'])}")
                    
                    # Print individual layer details for GRU (to see which gates are blocked)
                    if 'gru' in layer_type and epoch == 0:
                        for name, grad_norm in zip(stats['names'], stats['grad_norms']):
                            gate_info = ""
                            if 'weight_ih' in name or 'weight_hh' in name:
                                gate_info = " (gates: reset, update, candidate)"
                            
                            grad_status = "✓" if grad_norm > 1e-6 else "✗ BLOCKED"
                            print(f"    {grad_status} {name}: grad_norm={grad_norm:.2e}{gate_info}")
                
                print("=" * 80)
        
        # Optimizer step with gradient scaling
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        t_prev_end = time.perf_counter()
        step_times.append(t_prev_end - t_step_start)
        
        # Compute update statistics (weight change after optimizer.step())
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
    
    # Compute average gradient statistics
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


def _sync_weights_to_memristor_model(base_model: nn.Module, memristor_model: nn.Module) -> None:
    """
    Sync weights from base model to memristor-wrapped model.
    
    This is needed for memristor_no_comp mode where we train the base model
    but evaluate with the memristor-wrapped model.
    
    After removing deepcopy, base_model and memristor_model.base_model are the same object,
    so sync is a no-op (weights are already shared). This function includes a fast path
    to skip unnecessary work when they're the same object.
    
    Args:
        base_model: Base model (trained without non-idealities)
        memristor_model: Memristor-wrapped model (for evaluation)
    """
    # Fast path: if base_model and memristor_model.base_model are the same object,
    # weights are already shared, no sync needed
    if hasattr(memristor_model, 'base_model') and base_model is memristor_model.base_model:
        # Same object - weights are already shared, sync is a no-op
        return
    
    # Get the wrapped base_model from memristor_model
    if not hasattr(memristor_model, 'base_model'):
        # If memristor_model doesn't have base_model attribute, try direct sync
        memristor_model.load_state_dict(base_model.state_dict(), strict=False)
        return
    
    # Build a mapping from base_model parameter names to memristor_model parameter names
    # The memristor_model wraps base_model, so structure is: base_model.{base_name}
    # MemristorLinear/MemristorConv2d have the same parameter names (weight, bias) as original layers
    base_state = base_model.state_dict()
    memristor_state = memristor_model.state_dict()
    
    # Create mapping: for each parameter in base_model, find corresponding in memristor_model
    synced_count = 0
    for base_name, base_param in base_state.items():
        # The memristor model wraps base_model, so path should be: base_model.{base_name}
        memristor_name = f'base_model.{base_name}'
        if memristor_name in memristor_state:
            # Check if shapes match
            if memristor_state[memristor_name].shape == base_param.shape:
                memristor_state[memristor_name].data.copy_(base_param.data)
                synced_count += 1
            else:
                logger.warning(
                    f"Weight sync: Shape mismatch for {base_name}: "
                    f"base {base_param.shape} vs memristor {memristor_state[memristor_name].shape}"
                )
        else:
            # If not found with base_model prefix, try direct name
            if base_name in memristor_state:
                if memristor_state[base_name].shape == base_param.shape:
                    memristor_state[base_name].data.copy_(base_param.data)
                    synced_count += 1
    
    # Load the updated state dict
    missing_keys, unexpected_keys = memristor_model.load_state_dict(memristor_state, strict=False)
    
    # Log sync status (only if there are issues)
    total_base_params = len(base_state)
    if synced_count == 0:
        logger.error(
            f"Weight sync FAILED: No parameters synced! "
            f"Base model has {total_base_params} parameters, "
            f"synced {synced_count}. "
            f"Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
        )
        if len(missing_keys) > 0:
            logger.error(f"First few missing keys: {list(missing_keys)[:5]}")
    elif synced_count < total_base_params * 0.5:
        logger.warning(
            f"Weight sync: Only {synced_count}/{total_base_params} parameters synced. "
            f"This may cause incorrect behavior."
        )


def _program_conductance(
    G_current: torch.Tensor,
    G_target: torch.Tensor,
    device_model: MemristorDeviceModel,
    write_V: float,
    write_t_min: float,
    write_t_scale: float,
    max_iters: int = 200,
    tol: float = 0.02,
) -> Tuple[torch.Tensor, int]:
    """
    多脉冲写入 → 验证循环。
    
    逐步将每个电导值推向目标值，使用迭代脉冲序列。
    
    Args:
        G_current: 当前电导值张量
        G_target: 目标电导值张量
        device_model: 忆阻器器件模型
        write_V: 写入电压
        write_t_min: 最小脉冲宽度
        write_t_scale: 脉冲时间缩放因子
        max_iters: 最大迭代次数
        tol: 容差（相对于电导范围的比例）
        
    Returns:
        G_final: 最终电导值张量
        num_pulses: 实际使用的脉冲数（平均）
    """
    G_range = device_model.G_max - device_model.G_min
    tolerance = tol * G_range  # 绝对容差
    
    G_work = G_current.clone()
    
    # 计算初始误差
    error = G_target - G_work
    abs_error = torch.abs(error)
    
    # 使用掩码来跟踪哪些元素已经收敛
    converged_mask = abs_error < tolerance
    
    # 记录迭代次数（用于统计平均脉冲数）
    num_iterations = 0
    
    for iter_step in range(max_iters):
        # 如果所有元素都收敛，提前退出
        if converged_mask.all():
            break
        
        # 计算方向：sign(error)
        direction = torch.sign(error)
        
        # 计算归一化误差（用于动态脉冲宽度）
        # 归一化到 [0.1, 1.0] 范围，确保小误差时也有足够的脉冲宽度
        error_norm = abs_error / (G_range + 1e-12)
        error_norm = torch.clamp(error_norm, min=0.1, max=1.0)
        
        # 计算动态脉冲宽度：write_t_min + write_t_scale * error_norm
        # 误差大时使用更大的脉冲宽度，误差小时使用较小的脉冲宽度
        pulse_t = write_t_min + write_t_scale * error_norm
        
        # 脉冲电压：方向 * write_V
        pulse_V = direction * write_V
        
        # 应用写入更新（向量化操作，对所有元素同时进行）
        G_work = device_model.write_update(
            G_work, pulse_V, pulse_t, direction, seed=None
        )
        
        # 更新迭代计数
        num_iterations += 1
        
        # 重新计算误差和收敛掩码
        error = G_target - G_work
        abs_error = torch.abs(error)
        converged_mask = abs_error < tolerance
    
    return G_work, num_iterations


def _apply_writeback(
    model: nn.Module,
    device_model: MemristorDeviceModel,
    write_t_min: float,
    write_t_scale: float,
    write_V_write: float,
    max_pulses: int = 200,
    tolerance: float = 0.02,
) -> None:
    """
    应用写入更新模型，将目标权重写入到忆阻器器件中。
    
    使用多脉冲写入-验证循环来模拟真实的忆阻器编程过程。
    
    对于每个有权重的模块：
    1. 获取目标权重 W_target（optimizer更新后的权重）
    2. 将当前权重映射到电导值 Gp_current, Gn_current（从硬件状态）
    3. 将目标权重映射到目标电导值 Gp_target, Gn_target
    4. 使用多脉冲循环将 (Gp_current, Gn_current) → (Gp_target, Gn_target)
    5. 将编程后的电导值转换回权重并更新模块权重
    
    Args:
        model: 模型（可能是 MemristorModel wrapper，需要访问 base_model）
        device_model: 忆阻器器件模型
        write_t_min: 最小脉冲宽度
        write_t_scale: 脉冲时间的放大因子
        write_V_write: 写入电压
        max_pulses: 最大脉冲数（默认200）
        tolerance: 容差（相对于电导范围的比例，默认0.02即2%）
    """
    if not device_model.enable_update_model:
        return
    
    # 获取实际的模型
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    with torch.no_grad():
        total_modules = 0
        total_pulses_p = 0.0
        total_pulses_n = 0.0
        
        for module in target_model.modules():
            # 只处理有权重的模块（Linear, Conv2d等）
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            
            total_modules += 1
            
            # 获取目标权重（optimizer更新后的权重，这是我们要写入的目标）
            W_target = module.weight.data.clone()
            
            # 1. 将当前权重映射到当前电导值（从硬件状态）
            # 这代表忆阻器上的实际电导值（上一轮写入后的状态）
            Gp_current, Gn_current, max_abs_current = device_model.map_weights_to_conductance_diff_adaptive(
                module.weight.data
            )
            
            # 2. 将目标权重映射到目标电导值
            Gp_target, Gn_target, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W_target)
            
            # 计算 scale（用于后续转换回权重）
            G_range = (device_model.G_max - device_model.G_min)
            scale = max_abs / (G_range + 1e-12)
            scale = torch.clamp(scale, min=1e-3, max=1e6)
            
            # 3. 使用多脉冲循环编程电导值
            Gp_prog, pulses_p = _program_conductance(
                G_current=Gp_current,
                G_target=Gp_target,
                device_model=device_model,
                write_V=write_V_write,
                write_t_min=write_t_min,
                write_t_scale=write_t_scale,
                max_iters=max_pulses,
                tol=tolerance,
            )
            
            Gn_prog, pulses_n = _program_conductance(
                G_current=Gn_current,
                G_target=Gn_target,
                device_model=device_model,
                write_V=write_V_write,
                write_t_min=write_t_min,
                write_t_scale=write_t_scale,
                max_iters=max_pulses,
                tol=tolerance,
            )
            
            total_pulses_p += pulses_p
            total_pulses_n += pulses_n
            
            # 4. 将编程后的电导值转换回权重
            W_new = (Gp_prog - Gn_prog) * scale
            
            # 检查是否有 NaN 或 Inf
            if torch.isnan(W_new).any() or torch.isinf(W_new).any():
                logger.warning(f"W_new contains NaN/Inf in {type(module).__name__}, using target weights")
                W_new = W_target.clone()
            
            # 更新模块权重
            module.weight.data.copy_(W_new)
        
        # 记录统计信息
        if total_modules > 0:
            avg_pulses_p = total_pulses_p / total_modules
            avg_pulses_n = total_pulses_n / total_modules
            avg_pulses = (avg_pulses_p + avg_pulses_n) / 2.0
            
            logger.info(
                f"Writeback applied to {total_modules} modules: "
                f"avg pulses per layer: {avg_pulses:.1f} "
                f"(Gp: {avg_pulses_p:.1f}, Gn: {avg_pulses_n:.1f})"
            )


def _train_hat(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    device_model: MemristorDeviceModel,
    config: Optional[Dict[str, Any]] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Hardware-aware training with non-idealities during forward.
    
    This function ensures HAT training uses memristor_wrappers.py layers
    """
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Gradient statistics for training dynamics analysis
    grad_norms = []  # List of gradient norms per batch
    grad_vars = []  # List of gradient variances per batch
    update_stds = []  # List of update standard deviations per batch
    
    # ViT/GRU-specific metrics
    is_vit = config is not None and config.get('model_name', '') == 'vit_tiny'
    is_gru = is_gru or (config is not None and config.get('model_name', '') == 'gru_agnews')
    
    if is_vit:
        tier_grad_norms = {'patch_embed': [], 'attn_proj': [], 'mlp': [], 'head': []}
        tier_update_norms = {'patch_embed': [], 'attn_proj': [], 'mlp': [], 'head': []}
    elif is_gru:
        tier_grad_norms = {'embedding': [], 'gru_weight': [], 'head': []}
        tier_update_norms = {'embedding': [], 'gru_weight': [], 'head': []}
    else:
        tier_grad_norms = {}
        tier_update_norms = {}
    
    logit_margins = []
    activation_hooks = None
    hook_handles = []
    
    # Timing: data_time (per batch), train_step_time (forward+backward+opt per batch)
    data_times = []
    step_times = []
    
    use_amp = scaler is not None and amp_dtype is not None
    
    # Register activation hooks for ViT/GRU if available
    if is_vit and register_activation_hooks is not None:
        activation_hooks, hook_handles = register_activation_hooks(model)
    elif is_gru and gru_register_activation_hooks is not None:
        activation_hooks, hook_handles = gru_register_activation_hooks(model)
    
    t_prev_end = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        t_iter_start = time.perf_counter()
        data_times.append(t_iter_start - t_prev_end)
        
        data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
        data, target = data.to(device), target.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
        
        # Save weights before update (for computing update statistics)
        weights_before = []
        for param in model.parameters():
            if param.requires_grad:
                weights_before.append(param.data.clone())
        
        optimizer.zero_grad()
        
        t_step_start = time.perf_counter()
        # Forward with non-idealities (t increases with each batch)
        t = epoch * len(train_loader) + batch_idx
        
        # Control noise sampling frequency
        # If noise_sampling_interval > 1, only sample noise every k steps
        # Within the same interval, reuse the same noise pattern
        noise_sampling_interval = config.get('experiment', {}).get('noise_sampling_interval', 1) if config else 1
        global_step = epoch * len(train_loader) + batch_idx
        
        if noise_sampling_interval > 1:
            # Calculate which interval we're in
            interval_id = global_step // noise_sampling_interval
            # Use interval_id as seed to ensure same noise pattern within the interval
            # Add a large offset to avoid conflicts with other uses of seed
            seed = 1000000 + interval_id  # Fixed seed for this interval
        else:
            # Default: sample noise every step (seed=None means random each time)
            seed = None  # Let randomness vary naturally for HAT
        
        # Mixed precision forward pass
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                try:
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths, t=t, seed=seed)
                    else:
                        output = model(data, t=t, seed=seed)
                except TypeError:
                    # Fallback if model doesn't accept t parameter
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths)
                    else:
                        output = model(data)
        else:
            try:
                if is_gru and lengths is not None:
                    output = model(data, lengths=lengths, t=t, seed=seed)
                else:
                    output = model(data, t=t, seed=seed)
            except TypeError:
                # Fallback if model doesn't accept t parameter
                if is_gru and lengths is not None:
                    output = model(data, lengths=lengths)
                else:
                    output = model(data)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            logger.warning(f"NaN detected in output at epoch {epoch}, batch {batch_idx}. Skipping batch.")
            t_prev_end = time.perf_counter()
            continue
        
        # Mixed precision loss computation
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss_task = criterion(output, target)
        else:
            loss_task = criterion(output, target)
        
        # Check for NaN in loss
        if torch.isnan(loss_task) or torch.isinf(loss_task):
            logger.warning(f"NaN/Inf detected in loss at epoch {epoch}, batch {batch_idx}. Skipping batch.")
            t_prev_end = time.perf_counter()
            continue
        
        # Boundary regularization (if enabled)
        loss = loss_task
        if config is not None:
            boundary_reg_config = config.get('experiment', {}).get('boundary_regularization', {})
            if boundary_reg_config.get('enabled', False):
                lambda_boundary = float(boundary_reg_config.get('lambda', 1e-4))
                beta = float(boundary_reg_config.get('beta', 0.8))
                boundary_reg = compute_boundary_regularization(model, device_model, beta=beta)
                loss = loss_task + lambda_boundary * boundary_reg
        
        # Backward pass with gradient scaling if using AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute gradient statistics before optimizer.step()
        # Get all parameter gradients
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten())
        
        if grad_list:
            # Concatenate all gradients
            all_grads = torch.cat(grad_list)
            
            # Compute gradient norm
            grad_norm = all_grads.norm().item()
            grad_norms.append(grad_norm)
            
            # Compute gradient variance (across all parameters)
            grad_var = all_grads.var().item()
            grad_vars.append(grad_var)
            
            # ViT/GRU: Collect tier-based gradient norms
            if is_vit and collect_gradient_norms_by_tier is not None:
                tier_grads = collect_gradient_norms_by_tier(model)
                for tier, norm in tier_grads.items():
                    if tier in tier_grad_norms:
                        tier_grad_norms[tier].append(norm)
            elif is_gru and gru_collect_gradient_norms_by_tier is not None:
                tier_grads = gru_collect_gradient_norms_by_tier(model)
                for tier, norm in tier_grads.items():
                    if tier in tier_grad_norms:
                        tier_grad_norms[tier].append(norm)
            
            # Debug: log very small gradients (potential gradient vanishing)
            if grad_norm < 1e-6 and batch_idx == 0:
                logger.warning(f"Very small gradient norm detected: {grad_norm:.2e} at epoch {epoch}, batch {batch_idx}. "
                             f"This may indicate gradient vanishing (e.g., direct ADC mode).")
            
            # Print gradient statistics for each layer (only on first batch of first few epochs)
            if batch_idx == 0 and epoch < 3:
                print("=" * 80)
                print(f"Gradient Statistics (Epoch {epoch}, Batch {batch_idx}):")
                print("-" * 80)
                
                # Get base model if wrapped
                base_model = model
                if hasattr(model, 'base_model'):
                    base_model = model.base_model
                
                layer_grad_stats = {}
                for name, param in base_model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_max = param.grad.abs().max().item()
                        grad_mean = param.grad.mean().item()
                        param_norm = param.data.norm().item()
                        
                        # Determine layer type
                        layer_type = "unknown"
                        if 'embedding' in name:
                            layer_type = "embedding"
                        elif 'gru' in name:
                            if 'weight_ih' in name:
                                layer_type = "gru_weight_ih"
                            elif 'weight_hh' in name:
                                layer_type = "gru_weight_hh"
                            elif 'bias' in name:
                                layer_type = "gru_bias"
                            else:
                                layer_type = "gru_other"
                        elif 'head' in name:
                            layer_type = "head"
                        
                        # Group by layer type
                        if layer_type not in layer_grad_stats:
                            layer_grad_stats[layer_type] = {
                                'names': [],
                                'grad_norms': [],
                                'grad_maxs': [],
                                'grad_means': [],
                                'param_norms': [],
                            }
                        
                        layer_grad_stats[layer_type]['names'].append(name)
                        layer_grad_stats[layer_type]['grad_norms'].append(grad_norm)
                        layer_grad_stats[layer_type]['grad_maxs'].append(grad_max)
                        layer_grad_stats[layer_type]['grad_means'].append(grad_mean)
                        layer_grad_stats[layer_type]['param_norms'].append(param_norm)
                    else:
                        # No gradient
                        layer_type = "unknown"
                        if 'embedding' in name:
                            layer_type = "embedding"
                        elif 'gru' in name:
                            layer_type = "gru"
                        elif 'head' in name:
                            layer_type = "head"
                        
                        if layer_type not in layer_grad_stats:
                            layer_grad_stats[layer_type] = {
                                'names': [],
                                'grad_norms': [],
                                'grad_maxs': [],
                                'grad_means': [],
                                'param_norms': [],
                            }
                        layer_grad_stats[layer_type]['names'].append(name)
                        layer_grad_stats[layer_type]['grad_norms'].append(0.0)
                        layer_grad_stats[layer_type]['grad_maxs'].append(0.0)
                        layer_grad_stats[layer_type]['grad_means'].append(0.0)
                        layer_grad_stats[layer_type]['param_norms'].append(param.data.norm().item())
                
                # Print statistics by layer type
                for layer_type in sorted(layer_grad_stats.keys()):
                    stats = layer_grad_stats[layer_type]
                    if not stats['names']:
                        continue
                    
                    avg_grad_norm = np.mean(stats['grad_norms'])
                    max_grad_norm = np.max(stats['grad_norms'])
                    avg_grad_max = np.mean(stats['grad_maxs'])
                    avg_grad_mean = np.mean(stats['grad_means'])
                    zero_grad_count = sum(1 for n in stats['grad_norms'] if n < 1e-8)
                    
                    status = "✓" if avg_grad_norm > 1e-6 else "✗"
                    print(f"{status} {layer_type:20s}: "
                          f"avg_grad_norm={avg_grad_norm:.2e}, "
                          f"max_grad_norm={max_grad_norm:.2e}, "
                          f"avg_grad_max={avg_grad_max:.2e}, "
                          f"zero_grad_layers={zero_grad_count}/{len(stats['names'])}")
                    
                    # Print individual layer details for GRU (to see which gates are blocked)
                    if 'gru' in layer_type and epoch == 0:
                        for name, grad_norm in zip(stats['names'], stats['grad_norms']):
                            gate_info = ""
                            if 'weight_ih' in name or 'weight_hh' in name:
                                # GRU weights are organized as [reset, update, candidate] gates
                                # Each gate is hidden_size rows
                                gate_info = " (gates: reset, update, candidate)"
                            
                            grad_status = "✓" if grad_norm > 1e-6 else "✗ BLOCKED"
                            print(f"    {grad_status} {name}: grad_norm={grad_norm:.2e}{gate_info}")
                
                print("=" * 80)
            
            # Check for gradient explosion
            if grad_norm > 1e6:
                logger.warning(f"Large gradient norm detected: {grad_norm:.2e} at epoch {epoch}, batch {batch_idx}")
                # Clip gradients to prevent explosion
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with gradient scaling
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        t_prev_end = time.perf_counter()
        step_times.append(t_prev_end - t_step_start)
        
        # Compute update statistics (weight change after optimizer.step())
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
            
            # ViT/GRU: Collect tier-based update norms
            if is_vit and compute_update_norm_by_tier is not None:
                tier_updates = compute_update_norm_by_tier(model, weights_before)
                for tier, norm in tier_updates.items():
                    if tier in tier_update_norms:
                        tier_update_norms[tier].append(norm)
            elif is_gru and gru_compute_update_norm_by_tier is not None:
                tier_updates = gru_compute_update_norm_by_tier(model, weights_before)
                for tier, norm in tier_updates.items():
                    if tier in tier_update_norms:
                        tier_update_norms[tier].append(norm)
        
        # ViT/GRU: Compute logit margin
        if is_vit and compute_logit_margin is not None:
            margin_stats = compute_logit_margin(output.detach())
            logit_margins.append(margin_stats.get('mean', 0.0))
        elif is_gru and gru_compute_logit_margin is not None:
            margin_stats = gru_compute_logit_margin(output.detach())
            logit_margins.append(margin_stats.get('mean', 0.0))
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    # Compute average gradient statistics
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    avg_grad_var = np.mean(grad_vars) if grad_vars else 0.0
    std_grad_norm = np.std(grad_norms) if grad_norms else 0.0
    avg_update_std = np.mean(update_stds) if update_stds else 0.0
    
    # Remove activation hooks
    if hook_handles:
        for handle in hook_handles:
            handle.remove()
    
    avg_data_time = np.mean(data_times) if data_times else 0.0
    avg_step_time = np.mean(step_times) if step_times else 0.0
    # Build return dictionary
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
    
    # Add ViT-specific metrics
    if is_vit:
        # Tier-based gradient norms
        for tier, norms in tier_grad_norms.items():
            if norms:
                result[f'grad_norm_{tier}'] = np.mean(norms)
        
        # Tier-based update norms
        for tier, norms in tier_update_norms.items():
            if norms:
                result[f'update_norm_{tier}'] = np.mean(norms)
        
        # Logit margin
        if logit_margins:
            result['logit_margin_mean'] = np.mean(logit_margins)
            result['logit_margin_std'] = np.std(logit_margins)
        
        # Activation statistics (collected from hooks)
        if activation_hooks:
            if is_vit and collect_activation_stats is not None:
                act_stats = collect_activation_stats(model, activation_hooks)
                for tier, stats in act_stats.items():
                    for stat_name, stat_value in stats.items():
                        result[f'act_{tier}_{stat_name}'] = stat_value
            elif is_gru and gru_collect_activation_stats is not None:
                act_stats = gru_collect_activation_stats(model, activation_hooks)
                for tier, stats in act_stats.items():
                    for stat_name, stat_value in stats.items():
                        result[f'act_{tier}_{stat_name}'] = stat_value
    
    return result


def _validate_baseline(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
) -> Dict[str, float]:
    """Standard validation without memristor non-idealities."""
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


def _validate_memristor(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    device_model: MemristorDeviceModel,
    t: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
    is_gru: bool = False,
) -> Dict[str, float]:
    """
    Validation with memristor non-idealities applied.
    
    Args:
        t: Drift time for evaluation. If None, uses t_fixed from post_train config
           (stored in run_experiment._post_train_t_fixed), or 0 if not available.
        is_gru: Whether model is GRU (AG News), for unpacking batch and passing lengths.
    
    Note: Each forward pass will apply non-idealities with different random noise
    (unless seed is fixed). This simulates realistic memristor behavior where
    each read operation has different noise.
    """
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Determine t value for evaluation
    # Priority: explicit t arg > post_train t_fixed > default 0
    if t is None:
        if hasattr(run_experiment, '_post_train_t_fixed') and run_experiment._post_train_t_fixed is not None:
            eval_t = run_experiment._post_train_t_fixed
        else:
            eval_t = 0
    else:
        eval_t = t
    
    # 重置能耗统计（如果启用）
    if hasattr(device_model, 'enable_energy') and device_model.enable_energy:
        device_model.reset_energy_stats()
    
    # 重置推理次数计数器（如果使用累加模式）
    if hasattr(device_model, 'drift_time_mode') and device_model.drift_time_mode == 'accumulate':
        device_model.reset_inference_count()
    
    use_amp = amp_dtype is not None
    
    with torch.no_grad():
        for batch in val_loader:
            data, target, lengths = _unpack_batch(batch, is_agnews=is_gru)
            data, target = data.to(device), target.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            
            # Forward with non-idealities
            # Don't use seed here to allow natural randomness in non-idealities
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    try:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths, t=eval_t, seed=None)
                        else:
                            output = model(data, t=eval_t, seed=None)
                    except TypeError:
                        if is_gru and lengths is not None:
                            output = model(data, lengths=lengths)
                        else:
                            output = model(data)
                    loss = criterion(output, target)
            else:
                try:
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths, t=eval_t, seed=None)
                    else:
                        output = model(data, t=eval_t, seed=None)
                except TypeError:
                    if is_gru and lengths is not None:
                        output = model(data, lengths=lengths)
                    else:
                        output = model(data)
                loss = criterion(output, target)
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
    
    # 获取能耗统计（如果启用）
    energy_stats = None
    if hasattr(device_model, 'enable_energy') and device_model.enable_energy:
        energy_stats = device_model.get_energy_stats()
    
    result = {'loss': losses.avg, 'acc1': top1.avg}
    if energy_stats is not None:
        result['energy_stats'] = energy_stats
    
    return result

