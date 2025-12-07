"""
Unit tests for mapping utilities.
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memristor.mapping import (
    map_weights_linear,
    map_weights_log,
    differential_pair_mapping,
    reshape_conv_to_matrix,
    reshape_matrix_to_conv,
)


def test_map_weights_linear():
    """Test linear mapping function."""
    W = torch.tensor([[-1.0, 0.0, 1.0]])
    G = map_weights_linear(W, G_min=1e-6, G_max=1e-4, wmin=-1.0, wmax=1.0)
    
    assert G.shape == W.shape
    assert torch.allclose(G[0, 0], torch.tensor(1e-6), atol=1e-8)
    assert torch.allclose(G[0, 2], torch.tensor(1e-4), atol=1e-8)


def test_map_weights_log():
    """Test logarithmic mapping function."""
    W = torch.tensor([[-1.0, 0.0, 1.0]])
    G = map_weights_log(W, G_min=1e-6, G_max=1e-4, wmin=-1.0, wmax=1.0)
    
    assert G.shape == W.shape
    assert torch.all(G >= 1e-6)
    assert torch.all(G <= 1e-4)


def test_differential_pair_mapping():
    """Test differential pair mapping."""
    W = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    W_pos, W_neg = differential_pair_mapping(W)
    
    assert W_pos.shape == W.shape
    assert W_neg.shape == W.shape
    assert torch.all(W_pos >= 0)
    assert torch.all(W_neg >= 0)
    # Check that W_pos - W_neg reconstructs original (approximately)
    W_recon = W_pos - W_neg
    assert torch.allclose(W_recon, W, atol=1e-6)


def test_reshape_conv_to_matrix():
    """Test convolutional weight reshaping."""
    # Create a conv weight: [out_ch, in_ch, k, k]
    weight_conv = torch.randn(32, 16, 3, 3)
    W_flat = reshape_conv_to_matrix(weight_conv)
    
    assert W_flat.shape == (32, 16 * 3 * 3)
    assert W_flat.numel() == weight_conv.numel()


def test_reshape_matrix_to_conv():
    """Test matrix to convolutional weight reshaping."""
    out_ch, in_ch, k_h, k_w = 32, 16, 3, 3
    W_flat = torch.randn(out_ch, in_ch * k_h * k_w)
    weight_conv = reshape_matrix_to_conv(W_flat, out_ch, in_ch, k_h, k_w)
    
    assert weight_conv.shape == (out_ch, in_ch, k_h, k_w)
    assert weight_conv.numel() == W_flat.numel()


def test_reshape_roundtrip():
    """Test that reshape_conv_to_matrix and reshape_matrix_to_conv are inverses."""
    weight_conv_orig = torch.randn(32, 16, 3, 3)
    W_flat = reshape_conv_to_matrix(weight_conv_orig)
    weight_conv_recon = reshape_matrix_to_conv(W_flat, 32, 16, 3, 3)
    
    assert torch.allclose(weight_conv_orig, weight_conv_recon)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


