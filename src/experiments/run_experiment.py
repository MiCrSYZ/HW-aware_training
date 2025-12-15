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

try:
    from ..models.model_zoo import get_model, wrap_model_with_memristor
    from ..memristor.device_model import MemristorDeviceModel
    from ..memristor.compensation import (
        hardware_aware_training,
    )
    from ..memristor.learned_weight_mapping import (
        WeightMappingNet,
        train_weight_mapping,
    )
    from ..memristor.energy_estimator import EnergyEstimator
    from ..utils.metrics import AverageMeter, accuracy
    from ..utils.checkpoint import save_checkpoint, load_checkpoint
    from ..utils.logger import setup_logger, setup_tensorboard, setup_wandb
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.model_zoo import get_model, wrap_model_with_memristor
    from src.memristor.device_model import MemristorDeviceModel
    from src.memristor.compensation import (
        hardware_aware_training,
    )
    from src.memristor.learned_weight_mapping import (
        WeightMappingNet,
        train_weight_mapping,
    )
    from src.memristor.energy_estimator import EnergyEstimator
    from src.utils.metrics import AverageMeter, accuracy
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from src.utils.logger import setup_logger, setup_tensorboard, setup_wandb

logger = logging.getLogger(__name__)


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
    
    # Data loaders
    try:
        from ..data.dataset import get_dataloaders
    except ImportError:
        from src.data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=config['dataset'],
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        val_split=config.get('val_split', 0.1),
        seed=config.get('seed'),
    )
    
    # Model
    # Determine input channels based on dataset
    dataset_name = config['dataset'].lower()
    if dataset_name == 'mnist':
        in_channels = 1  # Grayscale
    elif dataset_name == 'cifar10':
        in_channels = 3  # RGB
    else:
        in_channels = config.get('in_channels', 3)  # Default to RGB
    
    base_model = get_model(
        name=config['model_name'],
        num_classes=config.get('num_classes', 10),
        in_channels=in_channels,
    )
    base_model = base_model.to(device)
    
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
        )
        
        # Check if learned mapping is used
        compensation_method = config['experiment'].get('compensation_method', 'hat')
        use_learned_mapping = (compensation_method == 'learned_mapping')
        
        # Get mapping parameters from config
        mapping_max_frac = float(config['experiment'].get('mapping_max_frac', 0.5))
        
        # Create memristor-wrapped model for evaluation
        # Use learned mapping classes if compensation_method is learned_mapping or hybrid
        memristor_model = wrap_model_with_memristor(
            base_model, device_model, 
            use_learned_mapping=use_learned_mapping,
            mapping_max_frac=mapping_max_frac
        )
        memristor_model = memristor_model.to(device)
        
        # For memristor_with_comp (HAT), use memristor model for training
        # For memristor_no_comp, use base model for training, memristor model for eval
        if config['experiment']['mode'] == 'memristor_with_comp':
            model = memristor_model
        else:  # memristor_no_comp
            model = base_model
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
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
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
        if scheduler_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(epochs),
            )
        elif scheduler_config['type'] == 'step':
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
    
    # Check if post_train is enabled for learned_mapping
    post_train_config = config['experiment'].get('post_train', {})
    use_post_train = (experiment_mode == 'memristor_with_comp' and 
                     config['experiment'].get('compensation_method') == 'learned_mapping' and
                     post_train_config.get('enabled', False))
    
    # Time tracking for ETA calculation
    epoch_times = []
    training_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        # 重置能耗统计（如果启用，每个epoch开始时重置）
        if device_model and hasattr(device_model, 'enable_energy') and device_model.enable_energy:
            device_model.reset_energy_stats()
        
        # Train
        if experiment_mode == 'baseline':
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch
            )
        elif experiment_mode == 'memristor_no_comp':
            # Train normally, apply non-idealities only at eval
            train_metrics = _train_baseline(
                model, train_loader, criterion, optimizer, device, epoch
            )
            # Sync weights from base_model to memristor_model for evaluation
            # This ensures memristor_model has the latest trained weights
            _sync_weights_to_memristor_model(base_model, memristor_model)
        elif experiment_mode == 'memristor_with_comp':
            compensation_method = config['experiment'].get('compensation_method', 'hat')
            if compensation_method == 'hat':
                train_metrics = _train_hat(
                    model, train_loader, criterion, optimizer, device, epoch, device_model, config
                )
            elif compensation_method == 'learned_mapping':
                train_metrics = _train_learned_mapping(
                    model, train_loader, val_loader, device_model, criterion, 
                    optimizer, device, epoch, config
                )
            else:
                raise ValueError(f"Unknown compensation method: {compensation_method}. "
                               f"Use 'hat' or 'learned_mapping'.")
        else:
            raise ValueError(f"Unknown experiment mode: {experiment_mode}")
        
        # 应用写入更新（根据 write_interval）
        # 如果 write_interval = epochs，则在训练循环结束后执行（避免重复）
        # 否则，在训练循环中按间隔执行
        # 注意：对于 learned_mapping (post_train模式)，跳过训练循环中的 writeback，
        #       因为最终会在 post-train 阶段写入 W_final = W_fp + ΔW
        compensation_method = config['experiment'].get('compensation_method', 'hat')
        post_train_config = config['experiment'].get('post_train', {})
        use_post_train = post_train_config.get('enabled', False)
        skip_writeback_in_training = (compensation_method == 'learned_mapping' and use_post_train)
        
        if (experiment_mode != 'baseline' and 
            device_model and 
            device_model.enable_update_model and 
            not skip_writeback_in_training and  # 跳过 learned_mapping post_train 模式的训练循环写入
            write_interval < epochs and  # 只在非一次性写入时在训练循环中执行
            (epoch + 1) % write_interval == 0):
            experiment_logger.info(f"Applying writeback at epoch {epoch}")
            _apply_writeback(
                model, device_model, 
                write_t_min, write_t_scale, write_V_write,
                max_pulses=write_max_pulses,
                tolerance=write_tolerance
            )
            # 如果是 memristor_no_comp 模式，还需要同步到 memristor_model
            if experiment_mode == 'memristor_no_comp':
                _sync_weights_to_memristor_model(base_model, memristor_model)
        
        # Validate
        if val_loader is not None:
            if experiment_mode == 'baseline':
                val_metrics = _validate_baseline(model, val_loader, criterion, device)
            elif experiment_mode == 'memristor_no_comp':
                # For no_comp: use memristor model for validation (apply non-idealities)
                val_metrics = _validate_memristor(
                    memristor_model, val_loader, criterion, device, device_model
                )
            else:  # memristor_with_comp
                # For with_comp: model is already memristor-wrapped
                val_metrics = _validate_memristor(
                    model, val_loader, criterion, device, device_model
                )
        else:
            val_metrics = {'acc1': 0.0, 'loss': 0.0}
        
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
        metrics_history.append(metrics)
        
        if tb_writer:
            tb_writer.add_scalar('Train/Loss', metrics['train_loss'], epoch)
            tb_writer.add_scalar('Train/Acc1', metrics['train_acc1'], epoch)
            tb_writer.add_scalar('Val/Loss', metrics['val_loss'], epoch)
            tb_writer.add_scalar('Val/Acc1', metrics['val_acc1'], epoch)
            tb_writer.add_scalar('LR', metrics['lr'], epoch)
        
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
            f" | Time: {epoch_time_str} | ETA: {eta_str}"
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
    
    # Post-train learned mapping if enabled
    # Flow: Main network training (same as HAT) → W_fp → Mapping network training (no write_update) 
    # → W_final = W_fp + ΔW → write_update → inference
    if use_post_train:
        experiment_logger.info("=" * 60)
        experiment_logger.info("Starting post-train learned mapping phase")
        experiment_logger.info("Flow: Main training (HAT) → W_fp → Mapping training → W_final → write_update → inference")
        experiment_logger.info("=" * 60)
        
        # Initialize mapping network
        mapping_alpha = float(config['experiment'].get('mapping_alpha', 0.5))
        mapping_net = WeightMappingNet(
            hidden_dim=config['experiment'].get('mapping_hidden_dim', 32),
            alpha=mapping_alpha
        ).to(device)
        
        # Set mapping network for all memristor layers
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
        
        num_set = 0
        for module in target_model.modules():
            if hasattr(module, 'set_learned_mapping'):
                module.set_learned_mapping(mapping_net)
                num_set += 1
        experiment_logger.info(f"Post-train: Set mapping_net for {num_set} layers")
        
        # Get post_train parameters from config
        post_train_num_epochs = int(post_train_config.get('num_epochs', 20))
        post_train_lr = float(post_train_config.get('lr', 3e-4))
        post_train_lambda_reg = float(post_train_config.get('lambda_reg', 1e-4))
        enable_sanity_check = config['experiment'].get('enable_sanity_check', False)
        
        experiment_logger.info(f"Post-train parameters: epochs={post_train_num_epochs}, "
                              f"lr={post_train_lr}, lambda_reg={post_train_lambda_reg}")
        experiment_logger.info("Starting post-train mapping_net training:")
        
        # Train mapping_net while freezing main model
        # train_weight_mapping will log each epoch internally
        mapping_results = train_weight_mapping(
            mapping_net=mapping_net,
            model=model,
            calibration_loader=train_loader,
            device_model=device_model,
            criterion=criterion,
            device=device,
            num_epochs=post_train_num_epochs,
            lr=post_train_lr,
            lambda_reg=post_train_lambda_reg,
            t=0,  # Use t=0 for post-train
            enable_sanity_check=enable_sanity_check,
        )
        
        experiment_logger.info(f"Post-train mapping_net training completed. "
                              f"Best: val_acc={mapping_results.get('best_val_acc', 0.0):.2f}%, "
                              f"loss={mapping_results.get('best_loss', 0.0):.4f}")
        
        # Compute W_final = W_fp + ΔW for learned mapping
        # Flow: Main network training (same as HAT) → W_fp → Mapping network training (no write_update) 
        # → W_final = W_fp + ΔW → write_update → inference
        # W_fp is the trained main model weights (already in model, adapted to non-idealities via HAT)
        # ΔW is computed by mapping_net from W_fp (mapping_net takes W_fp as input, not W_noisy)
        experiment_logger.info("Computing W_final = W_fp + ΔW for all layers...")
        mapping_net.eval()
        with torch.no_grad():
            for module in target_model.modules():
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                
                # W_fp: current weight (trained main model, same as HAT result)
                W_fp = module.weight.data.clone()
                
                # Compute W_noisy: W_fp with non-idealities applied (for noise_scale estimation)
                # This simulates what would be read from hardware
                Gp, Gn, max_abs = device_model.map_weights_to_conductance_diff_adaptive(W_fp)
                Gp_noisy = device_model.apply_nonidealities(Gp, t=0, seed=None)
                Gn_noisy = device_model.apply_nonidealities(Gn, t=0, seed=None)
                G_range = device_model.G_max - device_model.G_min
                scale_back = max_abs / (G_range + 1e-12)
                scale_back = torch.clamp(scale_back, min=1e-9, max=1e9)
                W_noisy = (Gp_noisy - Gn_noisy) * scale_back
                
                # Estimate noise_scale for mapping_net
                noise_est = (W_noisy - W_fp).abs()
                est_mean = float(noise_est.mean().detach().cpu().item() + 1e-12)
                min_noise_scale = 1e-8
                max_noise_scale = 0.5 * (device_model.wmax - device_model.wmin)
                noise_scale = float(min(max(est_mean, min_noise_scale), max_noise_scale))
                
                # Get delta from mapping_net
                # Note: mapping_net takes W_fp as input (not W_noisy), as per learned_weight_mapping.py design
                # Determine layer type and conv_shape
                if isinstance(module, nn.Conv2d):
                    layer_type = 'conv'
                    conv_shape = (module.out_channels, module.in_channels, 
                                 module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                                 module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size)
                else:
                    layer_type = 'linear'
                    conv_shape = None
                
                # mapping_net takes W_fp as input (not W_noisy)
                delta_W = mapping_net(W_fp, noise_scale=noise_scale, layer_type=layer_type, conv_shape=conv_shape)
                
                # Apply safety clamping (same as in hardware_linear_forward_with_weight_mapping)
                mapping_max_frac = float(config['experiment'].get('mapping_max_frac', 0.5))
                bound = mapping_max_frac * (W_noisy.abs() + 1e-9)
                delta_W = torch.max(torch.min(delta_W, bound), -bound)
                abs_clip = (device_model.wmax - device_model.wmin) * 0.5
                delta_W = torch.clamp(delta_W, -abs_clip, abs_clip)
                
                # Compute W_final = W_fp + ΔW
                W_final = W_fp + delta_W
                
                # Clamp to allowed weight range
                W_final = torch.clamp(W_final, device_model.wmin, device_model.wmax)
                
                # Update module weight to W_final
                module.weight.data.copy_(W_final)
        
        experiment_logger.info("W_final = W_fp + ΔW computed and stored in model weights")
        
        # Write W_final to hardware using write_update (if enabled)
        # This simulates the hardware programming process after computing W_final
        if (device_model and device_model.enable_update_model):
            experiment_logger.info("Writing W_final to hardware using write_update...")
            _apply_writeback(
                model, device_model,
                write_t_min, write_t_scale, write_V_write,
                max_pulses=write_max_pulses,
                tolerance=write_tolerance
            )
            experiment_logger.info("W_final written to hardware")
        else:
            experiment_logger.info("Writeback disabled (enable_update_model=False), skipping hardware write")
        
        # Disable mapping_net for inference (compensation is now baked into weights)
        experiment_logger.info("Disabling mapping_net for inference (compensation baked into weights)")
        for module in target_model.modules():
            if hasattr(module, 'set_learned_mapping'):
                module.set_learned_mapping(None)
        
        # Re-validate with W_final (mapping_net disabled, using hardware weights)
        if val_loader is not None:
            val_metrics = _validate_memristor(
                model, val_loader, criterion, device, device_model
            )
            experiment_logger.info(f"Post-writeback validation: acc={val_metrics['acc1']:.2f}%, "
                                  f"loss={val_metrics['loss']:.4f}")
            
            # Update best_acc if improved
            if val_metrics['acc1'] > best_acc:
                best_acc = val_metrics['acc1']
                best_model_path = os.path.join(output_dir, 'model_best.pth')
                checkpoint_data = {
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                }
                # Save mapping_net state (for reference, though it's disabled)
                checkpoint_data['mapping_net_state_dict'] = mapping_net.state_dict()
                save_checkpoint(checkpoint_data, best_model_path, is_best=True)
                experiment_logger.info(f"Saved best model after post-train with acc={best_acc:.2f}%")
        
        experiment_logger.info("=" * 60)
        experiment_logger.info("Post-train learned mapping phase completed")
        experiment_logger.info("=" * 60)
        
        # Store mapping_net for checkpoint saving
        if use_post_train:
            # Store mapping_net state for checkpoint
            if not hasattr(run_experiment, '_post_train_mapping_net_state'):
                run_experiment._post_train_mapping_net_state = mapping_net.state_dict()
    
    # 在训练循环结束后、最终测试前，如果 write_interval = epochs，执行最后一次 writeback
    # 这确保最终测试使用的是经过写入更新后的权重
    # 注意：对于 learned_mapping (post_train模式)，跳过这里的 writeback，
    #       因为已经在 post-train 阶段完成了：W_final = W_fp + ΔW → write_update
    compensation_method = config['experiment'].get('compensation_method', 'hat')
    post_train_config = config['experiment'].get('post_train', {})
    use_post_train = post_train_config.get('enabled', False)
    skip_writeback_after_training = (compensation_method == 'learned_mapping' and use_post_train)
    
    if (experiment_mode != 'baseline' and 
        device_model and 
        device_model.enable_update_model and 
        not skip_writeback_after_training and  # 跳过 learned_mapping post_train 模式的训练后写入
        write_interval == epochs):
        experiment_logger.info("Applying final writeback before test evaluation")
        _apply_writeback(
            model, device_model, 
            write_t_min, write_t_scale, write_V_write,
            max_pulses=write_max_pulses,
            tolerance=write_tolerance
        )
        # 如果是 memristor_no_comp 模式，还需要同步到 memristor_model
        if experiment_mode == 'memristor_no_comp':
            _sync_weights_to_memristor_model(base_model, memristor_model)
    
    # Final evaluation on test set
    if test_loader:
        if experiment_mode == 'baseline':
            test_metrics = _validate_baseline(model, test_loader, criterion, device)
        elif experiment_mode == 'memristor_no_comp':
            # For no_comp: sync weights one more time before final test evaluation
            _sync_weights_to_memristor_model(base_model, memristor_model)
            # Use memristor model for test (apply non-idealities)
            test_metrics = _validate_memristor(memristor_model, test_loader, criterion, device, device_model)
        else:  # memristor_with_comp
            # For with_comp: model is already memristor-wrapped
            test_metrics = _validate_memristor(model, test_loader, criterion, device, device_model)
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
    
    # Save mapping_net state if post_train was used
    if use_post_train and hasattr(run_experiment, '_post_train_mapping_net_state'):
        checkpoint_data['mapping_net_state_dict'] = run_experiment._post_train_mapping_net_state
        experiment_logger.info("Saved mapping_net state in checkpoint")
    
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
) -> Dict[str, float]:
    """Standard training without memristor non-idealities."""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    return {'loss': losses.avg, 'acc1': top1.avg}


def _train_learned_mapping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_model: MemristorDeviceModel,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Train with learned mapping compensation strategy.
    
    Supports two modes:
    1. Post-train mode (post_train.enabled=True): 
       - Train main model same as HAT (non-idealities applied during forward, 
         weights updated via normal gradient descent) → get W_fp
       - After main training, freeze main model and train mapping_net (no write_update)
       - Compute W_final = W_fp + ΔW
       - Call write_update to write W_final to hardware
       - Inference with hardware weights
    2. Joint training mode (post_train.enabled=False):
       - Alternate between training mapping_net and main model each epoch
    """
    # Check if post_train is enabled
    post_train_config = config['experiment'].get('post_train', {})
    use_post_train = post_train_config.get('enabled', False)
    
    if use_post_train:
        # Post-train mode: Train main model same as HAT (non-idealities applied during forward, 
        # but weights updated via normal gradient descent to get W_fp)
        # mapping_net will be trained after main model training completes
        logger.info(f"Post-train mode: Training main model (same as HAT, epoch {epoch})")
        
        # Ensure no mapping_net is set during main model training
        # This ensures we use non-idealities during forward but update weights normally
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
        
        # Explicitly disable mapping_net in all layers
        for module in target_model.modules():
            if hasattr(module, 'mapping_net'):
                module.mapping_net = None
            if hasattr(module, 'set_learned_mapping'):
                module.set_learned_mapping(None)
        
        # Train main model same as HAT: non-idealities applied during forward pass,
        # but weights are updated via normal gradient descent (not W_noisy)
        # This produces W_fp (float-point weights adapted to non-idealities)
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward with non-idealities (same as HAT)
            # Weights are updated via normal gradient descent, producing W_fp
            t = epoch * len(train_loader) + batch_idx
            seed = None  # Let randomness vary naturally
            try:
                output = model(data, t=t, seed=seed)
            except TypeError:
                output = model(data)
            
            loss_task = criterion(output, target)
            
            # Boundary regularization (if enabled)
            loss = loss_task
            boundary_reg_config = config.get('experiment', {}).get('boundary_regularization', {})
            if boundary_reg_config.get('enabled', False):
                lambda_boundary = float(boundary_reg_config.get('lambda', 1e-4))
                beta = float(boundary_reg_config.get('beta', 0.8))
                boundary_reg = compute_boundary_regularization(model, device_model, beta=beta)
                loss = loss_task + lambda_boundary * boundary_reg
            
            loss.backward()
            optimizer.step()
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'mapping_loss': 0.0,
        }
    else:
        # Joint training mode: Alternate between mapping_net and main model
        # Initialize mapping network (create once, reuse across epochs)
        if not hasattr(_train_learned_mapping, 'mapping_net'):
            mapping_alpha = float(config['experiment'].get('mapping_alpha', 0.5))
            _train_learned_mapping.mapping_net = WeightMappingNet(
                hidden_dim=config['experiment'].get('mapping_hidden_dim', 32),
                alpha=mapping_alpha
            ).to(device)
        
        mapping_net = _train_learned_mapping.mapping_net
        
        # Set mapping network for all memristor layers
        # IMPORTANT: If model is a MemristorModel wrapper, we need to access base_model
        target_model = model
        if hasattr(model, 'base_model'):
            target_model = model.base_model
            logger.info("_train_learned_mapping: Model is MemristorModel wrapper, accessing base_model")
        
        num_set = 0
        for module in target_model.modules():
            if hasattr(module, 'set_learned_mapping'):
                module.set_learned_mapping(mapping_net)
                num_set += 1
                # Verify immediately
                if getattr(module, 'mapping_net', None) is not mapping_net:
                    logger.error(f"ERROR: Failed to set mapping_net for {type(module).__name__}!")
        logger.info(f"_train_learned_mapping: Set mapping_net for {num_set} layers (before train_weight_mapping)")
        
        # Use train_weight_mapping from learned_weight_mapping.py
        # This trains the mapping network while freezing the main model
        mapping_epochs = int(config['experiment'].get('mapping_epochs_per_main_epoch', 1))
        t_step = epoch * len(train_loader)
        
        # Use calibration loader (can use train_loader or a subset)
        calibration_loader = train_loader
        
        # Get enable_sanity_check from config, default to False
        enable_sanity_check = config['experiment'].get('enable_sanity_check', False)
        logger.info(f"Learned mapping training: enable_sanity_check={enable_sanity_check}, "
                    f"mapping_epochs={mapping_epochs}, lr={config['experiment'].get('mapping_lr', 1e-4)}")
        
        mapping_results = train_weight_mapping(
            mapping_net=mapping_net,
            model=model,
            calibration_loader=calibration_loader,
            device_model=device_model,
            criterion=criterion,
            device=device,
            num_epochs=mapping_epochs,
            lr=float(config['experiment'].get('mapping_lr', 1e-4)),
            lambda_reg=float(config['experiment'].get('mapping_lambda_reg', 1e-4)),
            t=t_step,
            enable_sanity_check=enable_sanity_check,
        )
        
        # Get training metrics with learned mapping applied
        # Ensure mapping_net is still set after train_weight_mapping
        if hasattr(model, 'base_model'):
            verify_target = model.base_model
        else:
            verify_target = model
        
        # Re-verify mapping_net is set (train_weight_mapping should have set it)
        verify_count = 0
        for module in verify_target.modules():
            if hasattr(module, 'mapping_net'):
                verify_count += 1
                mn = getattr(module, 'mapping_net', None)
                if mn is not mapping_net:
                    logger.warning(f"WARNING: mapping_net mismatch for {type(module).__name__} after train_weight_mapping! "
                                 f"Expected {id(mapping_net)}, got {id(mn) if mn is not None else None}")
                    # Re-set it to be safe
                    if hasattr(module, 'set_learned_mapping'):
                        module.set_learned_mapping(mapping_net)
        
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        # Train the main model with learned mapping applied
        # The mapping_net is already trained and set in layers
        # Now we need to train the main model weights while keeping mapping_net fixed
        # Ensure mapping_net parameters are frozen but gradients can flow through
        for p in mapping_net.parameters():
            p.requires_grad = False
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with learned mapping (mapping_net is already set in layers)
            # Gradients will flow through mapping_net to main model weights
            try:
                output = model(data, t=t_step + batch_idx)
            except TypeError:
                output = model(data)
            
            loss_task = criterion(output, target)
            
            # Boundary regularization (if enabled)
            loss = loss_task
            boundary_reg_config = config.get('experiment', {}).get('boundary_regularization', {})
            if boundary_reg_config.get('enabled', False):
                lambda_boundary = float(boundary_reg_config.get('lambda', 1e-4))
                beta = float(boundary_reg_config.get('beta', 0.8))
                boundary_reg = compute_boundary_regularization(model, device_model, beta=beta)
                loss = loss_task + lambda_boundary * boundary_reg
            
            # Backward pass to update main model weights
            # mapping_net parameters are frozen, so only main model weights will be updated
            loss.backward()
            optimizer.step()
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'mapping_loss': mapping_results.get('best_loss', 0.0),
        }


def _sync_weights_to_memristor_model(base_model: nn.Module, memristor_model: nn.Module) -> None:
    """
    Sync weights from base model to memristor-wrapped model.
    
    This is needed for memristor_no_comp mode where we train the base model
    but evaluate with the memristor-wrapped model.
    
    The memristor_model has MemristorLinear/MemristorConv2d layers that wrap
    the original layers. We need to copy weights from base_model's layers to
    the corresponding memristor layers.
    
    Args:
        base_model: Base model (trained without non-idealities)
        memristor_model: Memristor-wrapped model (for evaluation)
    """
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
) -> Dict[str, float]:
    """Hardware-aware training with non-idealities during forward.
    
    This function ensures HAT training uses memristor_wrappers.py layers
    (which don't have mapping_net) and never uses learned_weight_mapping.py layers.
    """
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Ensure no mapping_net is set for HAT training
    # If model has base_model (MemristorModel wrapper), check both
    target_model = model
    if hasattr(model, 'base_model'):
        target_model = model.base_model
    
    # Explicitly disable mapping_net in all layers
    for module in target_model.modules():
        if hasattr(module, 'mapping_net'):
            module.mapping_net = None
        # Also try set_learned_mapping if it exists (for learned_weight_mapping layers)
        if hasattr(module, 'set_learned_mapping'):
            module.set_learned_mapping(None)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward with non-idealities (t increases with each batch)
        t = epoch * len(train_loader) + batch_idx
        # For memristor-wrapped models, always pass t parameter
        # Use different seed for each batch to ensure non-idealities vary
        # This is important for HAT training to see diverse non-ideality effects
        seed = None  # Let randomness vary naturally for HAT
        try:
            output = model(data, t=t, seed=seed)
        except TypeError:
            # Fallback if model doesn't accept t parameter
            output = model(data)
        
        loss_task = criterion(output, target)
        
        # Boundary regularization (if enabled)
        loss = loss_task
        if config is not None:
            boundary_reg_config = config.get('experiment', {}).get('boundary_regularization', {})
            if boundary_reg_config.get('enabled', False):
                lambda_boundary = float(boundary_reg_config.get('lambda', 1e-4))
                beta = float(boundary_reg_config.get('beta', 0.8))
                boundary_reg = compute_boundary_regularization(model, device_model, beta=beta)
                loss = loss_task + lambda_boundary * boundary_reg
        
        loss.backward()
        optimizer.step()
        
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
    
    return {'loss': losses.avg, 'acc1': top1.avg}


def _validate_baseline(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Standard validation without memristor non-idealities."""
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
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
) -> Dict[str, float]:
    """
    Validation with memristor non-idealities applied.
    
    Note: Each forward pass will apply non-idealities with different random noise
    (unless seed is fixed). This simulates realistic memristor behavior where
    each read operation has different noise.
    """
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # 重置能耗统计（如果启用）
    if hasattr(device_model, 'enable_energy') and device_model.enable_energy:
        device_model.reset_energy_stats()
    
    # 重置推理次数计数器（如果使用累加模式）
    if hasattr(device_model, 'drift_time_mode') and device_model.drift_time_mode == 'accumulate':
        device_model.reset_inference_count()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward with non-idealities (use t=0 for evaluation)
            # For memristor-wrapped models, always pass t parameter
            # Don't use seed here to allow natural randomness in non-idealities
            try:
                output = model(data, t=0, seed=None)
            except TypeError:
                # Fallback if model doesn't accept t parameter
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

