"""
Evaluation script for trained models.

This script loads a checkpoint and evaluates it on test/validation sets.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .models.model_zoo import get_model, wrap_model_with_memristor
    from .memristor.device_model import MemristorDeviceModel
    from .data.dataset import get_dataloaders
    from .utils.checkpoint import load_checkpoint
    from .utils.metrics import AverageMeter, accuracy
    from .utils.seeds import set_seed
except ImportError:
    from models.model_zoo import get_model, wrap_model_with_memristor
    from memristor.device_model import MemristorDeviceModel
    from data.dataset import get_dataloaders
    from utils.checkpoint import load_checkpoint
    from utils.metrics import AverageMeter, accuracy
    from utils.seeds import set_seed
import logging

logger = logging.getLogger(__name__)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Evaluate trained model')
        parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
        parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
        parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                           help='Dataset split to evaluate on')
        args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed'))
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Data loader
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=config['dataset'],
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        val_split=0.0,  # Don't split for evaluation
        seed=config.get('seed'),
    )
    
    if args.split == 'train':
        eval_loader = train_loader
    elif args.split == 'val':
        eval_loader = val_loader if val_loader else test_loader
    else:
        eval_loader = test_loader
    
    # Model
    model = get_model(
        name=config['model_name'],
        num_classes=config.get('num_classes', 10),
    )
    
    # Device model (if memristor experiment)
    device_model = None
    if config['experiment']['mode'] != 'baseline':
        memristor_config = config['memristor']
        
        # 读取新参数（向后兼容）
        array_size = memristor_config.get('array_size', 128)
        adc_bits = memristor_config.get('adc_bits', 6)
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
        
        # 读取新的IR-drop模型参数
        ir_drop_mode = memristor_config.get('ir_drop_mode', 'none')
        ir_drop_gamma = memristor_config.get('ir_drop_gamma', 0.35)
        ir_drop_scaling = memristor_config.get('ir_drop_scaling', 1.0)
        # crossbar模式参数
        ir_drop_eta = memristor_config.get('ir_drop_eta', 1.0)
        ir_drop_cap = memristor_config.get('ir_drop_cap', 0.10)
        ir_drop_norm = memristor_config.get('ir_drop_norm', 'mean')
        ir_drop_train_enabled = memristor_config.get('ir_drop_train_enabled', True)  # 默认True保持向后兼容
        
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
            # 新增参数
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
            # 新的IR-drop参数
            ir_drop_mode=str(ir_drop_mode),
            ir_drop_gamma=float(ir_drop_gamma),
            ir_drop_scaling=float(ir_drop_scaling),
            ir_drop_eta=float(ir_drop_eta),
            ir_drop_cap=float(ir_drop_cap),
            ir_drop_norm=str(ir_drop_norm),
            ir_drop_train_enabled=bool(ir_drop_train_enabled),
            enable_adc_during_training=bool(memristor_config.get('enable_adc_during_training', False)),
            adc_training_mode=str(memristor_config.get('adc_training_mode', 'ste')),
            enable_ir_drop_paper_during_training=bool(memristor_config.get('enable_ir_drop_paper_during_training', False)),
            synthetic_noise_type=str(memristor_config.get('synthetic_noise_type', 'none')),
            cond1_alpha=float(memristor_config.get('cond1_alpha', 0.1)),
            cond1_nu=float(memristor_config.get('cond1_nu', 2.0)),
            cond2_alpha=float(memristor_config.get('cond2_alpha', 0.1)),
        )
        model = wrap_model_with_memristor(model, device_model)
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        logger.info(f"Checkpoint best acc: {checkpoint['best_acc']:.2f}%")
    
    # Evaluate
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # 重置能耗统计（如果启用）
    if device_model and hasattr(device_model, 'enable_energy') and device_model.enable_energy:
        device_model.reset_energy_stats()
    
    # 重置推理次数计数器（如果使用累加模式）
    if device_model and hasattr(device_model, 'drift_time_mode') and device_model.drift_time_mode == 'accumulate':
        device_model.reset_inference_count()
    
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 't' in model.forward.__code__.co_varnames:
                output = model(data, t=0)
            else:
                output = model(data)
            
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))
    
    logger.info(f"Evaluation on {args.split} set:")
    logger.info(f"  Loss: {losses.avg:.4f}")
    logger.info(f"  Top-1 Accuracy: {top1.avg:.2f}%")
    logger.info(f"  Top-5 Accuracy: {top5.avg:.2f}%")
    
    # 打印能耗统计（如果启用）
    if device_model and hasattr(device_model, 'enable_energy') and device_model.enable_energy:
        energy_stats = device_model.get_energy_stats()
        if energy_stats:
            logger.info(f"  Energy Statistics:")
            logger.info(f"    Write Energy: {energy_stats['write']:.6e}")
            logger.info(f"    Read Energy: {energy_stats['read']:.6e}")
            logger.info(f"    Total Energy: {energy_stats['write'] + energy_stats['read']:.6e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

