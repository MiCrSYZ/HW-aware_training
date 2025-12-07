"""
Unit tests for MemristorDeviceModel.
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memristor.device_model import MemristorDeviceModel


def test_map_weights_to_conductance_linear():
    """Test linear weight-to-conductance mapping."""
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        weight_clip=(-1.0, 1.0),
        mapping='linear',
    )
    
    # Test with weight tensor
    W = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]])
    G = device_model.map_weights_to_conductance(W)
    
    # Check shape
    assert G.shape == W.shape
    
    # Check bounds
    assert torch.all(G >= device_model.G_min)
    assert torch.all(G <= device_model.G_max)
    
    # Check mapping: -1.0 -> G_min, 1.0 -> G_max
    assert torch.allclose(G[0, 0], torch.tensor(device_model.G_min), atol=1e-8)
    assert torch.allclose(G[0, 2], torch.tensor(device_model.G_max), atol=1e-8)


def test_map_weights_to_conductance_log():
    """Test logarithmic weight-to-conductance mapping."""
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        weight_clip=(-1.0, 1.0),
        mapping='log',
    )
    
    W = torch.tensor([[-1.0, 0.0, 1.0]])
    G = device_model.map_weights_to_conductance(W)
    
    assert G.shape == W.shape
    assert torch.all(G >= device_model.G_min)
    assert torch.all(G <= device_model.G_max)


def test_apply_nonidealities_variability():
    """Test variability application."""
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.1,
        read_noise_sigma=0.0,
        drift_alpha=0.0,
        stuck_ratio=0.0,
        ir_drop_beta=0.0,
    )
    
    G = torch.ones(10, 10) * 5e-5  # Middle of range
    G_noisy = device_model.apply_nonidealities(G, t=0, seed=42)
    
    assert G_noisy.shape == G.shape
    assert torch.all(G_noisy >= device_model.G_min)
    assert torch.all(G_noisy <= device_model.G_max)
    # Should have some variation
    assert not torch.allclose(G_noisy, G, atol=1e-8)


def test_apply_nonidealities_drift():
    """Test drift application over time."""
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.0,
        read_noise_sigma=0.0,
        drift_alpha=1e-3,
        stuck_ratio=0.0,
        ir_drop_beta=0.0,
    )
    
    G = torch.ones(5, 5) * 5e-5
    G_t0 = device_model.apply_nonidealities(G, t=0, seed=42)
    G_t100 = device_model.apply_nonidealities(G, t=100, seed=42)
    
    # With drift, G should decrease over time
    assert torch.all(G_t100 <= G_t0)


def test_apply_nonidealities_stuck_at():
    """Test stuck-at fault application."""
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.0,
        read_noise_sigma=0.0,
        drift_alpha=0.0,
        stuck_ratio=0.5,  # 50% stuck
        stuck_low_prob=0.5,
        ir_drop_beta=0.0,
    )
    
    G = torch.ones(100, 100) * 5e-5
    G_noisy = device_model.apply_nonidealities(G, t=0, seed=42)
    
    # Some values should be stuck at G_min or G_max
    stuck_low = (G_noisy == device_model.G_min).sum()
    stuck_high = (G_noisy == device_model.G_max).sum()
    assert stuck_low > 0 or stuck_high > 0


def test_save_load_state():
    """Test saving and loading device model state."""
    device_model1 = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.05,
        seed=42,
    )
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        temp_path = f.name
    
    try:
        device_model1.save_state(temp_path)
        
        device_model2 = MemristorDeviceModel()
        device_model2.load_state(temp_path)
        
        assert device_model2.G_min == device_model1.G_min
        assert device_model2.G_max == device_model1.G_max
        assert device_model2.variability_sigma == device_model1.variability_sigma
    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


