"""
TinyImageNet dataset loading hook.

This module provides dataloaders for TinyImageNet (64x64 images, 200 classes).
This is an optional dataset that can be added to the framework.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple, Optional
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TinyImageNetDataset(Dataset):
    """TinyImageNet dataset loader."""
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load class names
        with open(os.path.join(root, 'wnids.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        if split == 'train':
            for cls in self.classes:
                cls_dir = os.path.join(root, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.samples.append((
                            os.path.join(cls_dir, img_name),
                            self.class_to_idx[cls]
                        ))
        else:  # val or test
            val_dir = os.path.join(root, 'val', 'images')
            annotations_file = os.path.join(root, 'val', 'val_annotations.txt')
            with open(annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    cls = parts[1]
                    if cls in self.class_to_idx:
                        self.samples.append((
                            os.path.join(val_dir, img_name),
                            self.class_to_idx[cls]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_tinyimagenet_dataloaders(
    data_root: str = './datasets/tiny-imagenet-200',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get TinyImageNet dataloaders.
    
    Args:
        data_root: Root directory for TinyImageNet data
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split (currently unused, uses provided val set)
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Load datasets
    train_dataset = TinyImageNetDataset(
        root=data_root,
        split='train',
        transform=train_transform,
    )
    
    val_dataset = TinyImageNetDataset(
        root=data_root,
        split='val',
        transform=val_test_transform,
    )
    
    test_dataset = TinyImageNetDataset(
        root=data_root,
        split='val',  # TinyImageNet uses val set as test
        transform=val_test_transform,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(
        f"TinyImageNet dataloaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader


