"""
Test script to verify that non-idealities are actually being applied.

This script helps debug why comp and no_comp modes produce identical results.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memristor.device_model import MemristorDeviceModel
from models.resnet20 import ResNet20
from models.model_zoo import get_model, wrap_model_with_memristor


def test_nonidealities_effect():
    """Test that non-idealities actually change the output."""
    print("Testing non-idealities effect...")
    
    # Create device model with noticeable non-idealities
    device_model = MemristorDeviceModel(
        G_min=1e-6,
        G_max=1e-4,
        variability_sigma=0.1,  # 10% variability (more noticeable)
        read_noise_sigma=1e-6,  # Larger read noise
        drift_alpha=0.0,  # No drift for this test
        stuck_ratio=0.0,
        ir_drop_beta=0.0,
        mapping='linear',
        seed=None,  # Random seed
    )
    
    # Create a simple model
    model = ResNet20(num_classes=10)
    model.eval()
    
    # Wrap with memristor
    memristor_model = wrap_model_with_memristor(model, device_model)
    memristor_model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 32, 32)
    
    # Test 1: Same input, different forward passes should give different outputs
    # (due to randomness in non-idealities)
    with torch.no_grad():
        output1 = memristor_model(x, t=0, seed=None)
        output2 = memristor_model(x, t=0, seed=None)
        
        diff = torch.abs(output1 - output2).mean().item()
        print(f"Output difference between two forward passes: {diff:.6f}")
        
        if diff < 1e-6:
            print("WARNING: Outputs are identical! Non-idealities may not be working.")
        else:
            print("✓ Non-idealities are producing different outputs (good!)")
    
    # Test 2: Compare baseline vs memristor output
    with torch.no_grad():
        baseline_output = model(x)
        memristor_output = memristor_model(x, t=0, seed=None)
        
        diff = torch.abs(baseline_output - memristor_output).mean().item()
        print(f"Output difference between baseline and memristor: {diff:.6f}")
        
        if diff < 1e-6:
            print("WARNING: Baseline and memristor outputs are identical!")
        else:
            print("✓ Memristor model produces different output than baseline (good!)")
    
    # Test 3: Check weight mapping
    print("\nTesting weight-to-conductance mapping...")
    test_weight = torch.tensor([[0.5, -0.3], [0.1, -0.8]])
    G = device_model.map_weights_to_conductance(test_weight)
    print(f"Input weight range: [{test_weight.min():.2f}, {test_weight.max():.2f}]")
    print(f"Output conductance range: [{G.min():.6e}, {G.max():.6e}]")
    print(f"Expected range: [{device_model.G_min:.6e}, {device_model.G_max:.6e}]")
    
    # Test 4: Check non-idealities application
    print("\nTesting non-idealities application...")
    G_clean = torch.ones(10, 10) * 5e-5  # Middle of range
    G_noisy1 = device_model.apply_nonidealities(G_clean, t=0, seed=None)
    G_noisy2 = device_model.apply_nonidealities(G_clean, t=0, seed=None)
    
    diff = torch.abs(G_noisy1 - G_noisy2).mean().item()
    print(f"Conductance difference between two applications: {diff:.6e}")
    
    if diff < 1e-10:
        print("WARNING: Non-idealities are not introducing variation!")
    else:
        print("✓ Non-idealities are introducing variation (good!)")


if __name__ == '__main__':
    test_nonidealities_effect()

