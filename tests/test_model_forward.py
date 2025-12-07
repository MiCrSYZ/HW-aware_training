"""
Unit tests for model forward passes.
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.resnet20 import ResNet20
from models.vit_tiny import ViTTiny
from models.model_zoo import get_model, wrap_model_with_memristor
from memristor.device_model import MemristorDeviceModel


def test_resnet20_baseline_forward():
    """Test ResNet-20 forward pass in baseline mode."""
    model = ResNet20(num_classes=10)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_vit_tiny_baseline_forward():
    """Test ViT-Tiny forward pass in baseline mode."""
    model = ViTTiny(num_classes=10)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_resnet20_memristor_forward():
    """Test ResNet-20 forward pass with memristor wrappers."""
    # Create baseline model
    base_model = ResNet20(num_classes=10)
    
    # Create device model
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.05,
        seed=42,
    )
    
    # Wrap with memristor
    model = wrap_model_with_memristor(base_model, device_model)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(x, t=0)
    
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_vit_tiny_memristor_forward():
    """Test ViT-Tiny forward pass with memristor wrappers."""
    # Create baseline model
    base_model = ViTTiny(num_classes=10)
    
    # Create device model
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.05,
        seed=42,
    )
    
    # Wrap with memristor
    model = wrap_model_with_memristor(base_model, device_model)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(x, t=0)
    
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_zoo_get_model():
    """Test model factory function."""
    model1 = get_model('resnet20', num_classes=10)
    assert isinstance(model1, ResNet20)
    
    model2 = get_model('vit_tiny', num_classes=10)
    assert isinstance(model2, ViTTiny)
    
    with pytest.raises(ValueError):
        get_model('unknown_model')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


