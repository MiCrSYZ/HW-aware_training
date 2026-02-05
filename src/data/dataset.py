"""
Data loading and preprocessing for CIFAR-10, CIFAR-100, MNIST, and TinyImageNet.

This module provides dataloaders with appropriate data augmentation
for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _create_optimized_dataloader(dataset, batch_size, shuffle, num_workers):
    """Create an optimized DataLoader with performance settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches for better GPU utilization
    )


def get_cifar10_dataloaders(
    data_root: str = './datasets/cifar-10',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 dataloaders for training, validation, and testing.
    
    Args:
        data_root: Root directory for CIFAR-10 data
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        val_split: Fraction of training data to use for validation
        seed: Random seed for train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full training set without transform first (for splitting)
    full_train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=None,  # Apply transform after splitting
    )
    
    # Split into train and validation
    if val_split > 0:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Create train dataset with train transform
        train_dataset = _TransformDataset(train_subset, train_transform)
        
        # Create validation dataset with val transform
        val_dataset = _TransformDataset(val_subset, val_test_transform)
    else:
        train_dataset = _TransformDataset(full_train_dataset, train_transform)
        val_dataset = None
    
    # Test dataset
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=val_test_transform,
    )
    
    # Create dataloaders with performance optimizations
    train_loader = _create_optimized_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    
    if val_dataset is not None:
        val_loader = _create_optimized_dataloader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = None
    
    test_loader = _create_optimized_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(
        f"CIFAR-10 dataloaders: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders(
    data_root: str = './datasets/cifar-100',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 dataloaders for training, validation, and testing.
    
    Args:
        data_root: Root directory for CIFAR-100 data
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        val_split: Fraction of training data to use for validation
        seed: Random seed for train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation for training
    # CIFAR-100 uses the same normalization as CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full training set without transform first (for splitting)
    full_train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=None,  # Apply transform after splitting
    )
    
    # Split into train and validation
    if val_split > 0:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Create train dataset with train transform
        train_dataset = _TransformDataset(train_subset, train_transform)
        
        # Create validation dataset with val transform
        val_dataset = _TransformDataset(val_subset, val_test_transform)
    else:
        train_dataset = _TransformDataset(full_train_dataset, train_transform)
        val_dataset = None
    
    # Test dataset
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=val_test_transform,
    )
    
    # Create dataloaders with performance optimizations
    train_loader = _create_optimized_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    
    if val_dataset is not None:
        val_loader = _create_optimized_dataloader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = None
    
    test_loader = _create_optimized_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(
        f"CIFAR-100 dataloaders: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader


def get_mnist_dataloaders(
    data_root: str = './datasets/mnist',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get MNIST dataloaders for training, validation, and testing.
    
    Args:
        data_root: Root directory for MNIST data
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        val_split: Fraction of training data to use for validation
        seed: Random seed for train/val split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Small rotation for augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Load full training set without transform first (for splitting)
    full_train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=None,  # Apply transform after splitting
    )
    
    # Split into train and validation
    if val_split > 0:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Create train dataset with train transform
        train_dataset = _TransformDataset(train_subset, train_transform)
        
        # Create validation dataset with val transform
        val_dataset = _TransformDataset(val_subset, val_test_transform)
    else:
        train_dataset = _TransformDataset(full_train_dataset, train_transform)
        val_dataset = None
    
    # Test dataset
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=val_test_transform,
    )
    
    # Create dataloaders with performance optimizations
    train_loader = _create_optimized_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    
    if val_dataset is not None:
        val_loader = _create_optimized_dataloader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = None
    
    test_loader = _create_optimized_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(
        f"MNIST dataloaders: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader


class _TransformDataset(torch.utils.data.Dataset):
    """Wrapper to apply different transform to a subset of a dataset."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)


def get_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get dataloaders for a dataset by name.
    
    Args:
        dataset_name: Dataset name ('cifar10', 'cifar100', 'mnist', or 'tinyimagenet')
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split fraction
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_name.lower() == 'mnist':
        return get_mnist_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_name.lower() == 'tinyimagenet':
        # Import here to avoid circular dependency
        from .tinyimagenet_hook import get_tinyimagenet_dataloaders
        return get_tinyimagenet_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
        )
    elif dataset_name.lower() == 'agnews':
        # Import here to avoid circular dependency
        from .agnews import get_agnews_dataloaders
        train_loader, val_loader, test_loader, vocab = get_agnews_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
        )
        # Return 4 values so caller can unpack (train_loader, val_loader, test_loader, vocab)
        import sys
        sys.modules[__name__]._agnews_vocab = vocab
        return train_loader, val_loader, test_loader, vocab
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: 'cifar10', 'cifar100', 'mnist', 'tinyimagenet', 'agnews'")


