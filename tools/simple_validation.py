#!/usr/bin/env python3
"""
Simple validation comparing our estimates with actual parameter counts
from basic PyTorch modules to verify our calculation methodology.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from estimate_model_complexity import mFormerV1ComplexityEstimator


def test_convnext_block_estimation():
    """Test ConvNeXt block parameter estimation."""
    from linnaeus.models.blocks.convnext import ConvNeXtBlock
    
    estimator = mFormerV1ComplexityEstimator()
    
    # Test different dimensions
    dims = [96, 192, 384, 768]
    
    print("ConvNeXt Block Parameter Estimation Validation:")
    print("-" * 50)
    
    for dim in dims:
        # Create actual block
        block = ConvNeXtBlock(dim=dim, drop_path=0.0, layer_scale_init_value=1e-6)
        actual_params = sum(p.numel() for p in block.parameters())
        
        # Get estimate
        estimated_params = estimator.estimate_convnext_block_params(dim)
        
        error_pct = abs(estimated_params - actual_params) / actual_params * 100
        status = "✓" if error_pct < 5 else "⚠" if error_pct < 10 else "✗"
        
        print(f"  Dim {dim:3d}: Est={estimated_params:7d} | Act={actual_params:7d} | Err={error_pct:5.1f}% {status}")


def test_rope_block_estimation():
    """Test RoPE block parameter estimation."""
    from linnaeus.models.blocks.rope_2d_mhsa import RoPE2DMHSABlock
    
    estimator = mFormerV1ComplexityEstimator()
    
    print("\nRoPE Block Parameter Estimation Validation:")
    print("-" * 50)
    
    test_cases = [
        (384, 6, 4.0),   # SM stage 3
        (768, 12, 4.0),  # SM stage 4
        (1024, 16, 4.0), # XL stage 3
        (2048, 32, 4.0), # XL stage 4
    ]
    
    for dim, num_heads, mlp_ratio in test_cases:
        # Create actual block
        block = RoPE2DMHSABlock(
            dim=dim,
            img_grid_size=(12, 12),  # Arbitrary grid size
            extra_token_num=4,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            rope_theta=10000.0,
            rope_mixed=True,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_flash_attn=False  # Disable for testing
        )
        actual_params = sum(p.numel() for p in block.parameters())
        
        # Get estimate
        estimated_params = estimator.estimate_rope_mhsa_block_params(dim, num_heads, mlp_ratio)
        
        error_pct = abs(estimated_params - actual_params) / actual_params * 100
        status = "✓" if error_pct < 5 else "⚠" if error_pct < 10 else "✗"
        
        print(f"  Dim {dim:4d}, H={num_heads:2d}: Est={estimated_params:8d} | Act={actual_params:8d} | Err={error_pct:5.1f}% {status}")


def test_linear_head_estimation():
    """Test classification head parameter estimation."""
    estimator = mFormerV1ComplexityEstimator()
    
    print("\nClassification Head Parameter Estimation Validation:")
    print("-" * 50)
    
    # Test reptilia task configuration
    num_classes_dict = {
        'taxa_L10': 684,
        'taxa_L20': 179,
        'taxa_L30': 40,
        'taxa_L40': 4
    }
    
    in_features_list = [768, 1536, 2048]  # Different model sizes
    
    for in_features in in_features_list:
        # Create actual linear heads
        actual_params = 0
        for task, num_classes in num_classes_dict.items():
            head = nn.Linear(in_features, num_classes, bias=True)
            actual_params += sum(p.numel() for p in head.parameters())
        
        # Get estimate
        estimated_params = estimator.estimate_conditional_classifier_params(in_features, num_classes_dict)
        
        error_pct = abs(estimated_params - actual_params) / actual_params * 100
        status = "✓" if error_pct < 1 else "⚠" if error_pct < 5 else "✗"
        
        print(f"  In_feat {in_features:4d}: Est={estimated_params:7d} | Act={actual_params:7d} | Err={error_pct:5.1f}% {status}")


def test_downsampler_estimation():
    """Test ConvNeXt downsampler parameter estimation."""
    from linnaeus.models.blocks.convnext import ConvNeXtDownsampleLayer
    
    estimator = mFormerV1ComplexityEstimator()
    
    print("\nDownsampler Parameter Estimation Validation:")
    print("-" * 50)
    
    test_cases = [
        (96, 192),    # First downsample
        (192, 384),   # Second downsample
        (384, 768),   # Third downsample (SM)
        (768, 1536),  # Third downsample (LG)
        (1024, 2048), # Third downsample (XL)
    ]
    
    for in_dim, out_dim in test_cases:
        # Create actual downsampler
        downsampler = ConvNeXtDownsampleLayer(in_dim, out_dim)
        actual_params = sum(p.numel() for p in downsampler.parameters())
        
        # Get estimate
        estimated_params = estimator.estimate_convnext_downsample_params(in_dim, out_dim)
        
        error_pct = abs(estimated_params - actual_params) / actual_params * 100
        status = "✓" if error_pct < 5 else "⚠" if error_pct < 10 else "✗"
        
        print(f"  {in_dim:4d}→{out_dim:4d}: Est={estimated_params:6d} | Act={actual_params:6d} | Err={error_pct:5.1f}% {status}")


def main():
    print("Parameter Estimation Validation")
    print("=" * 60)
    
    test_convnext_block_estimation()
    test_rope_block_estimation()
    test_linear_head_estimation()
    test_downsampler_estimation()
    
    print("\nValidation Summary:")
    print("✓ = Error < 5% (Excellent)")
    print("⚠ = Error 5-10% (Good)")
    print("✗ = Error > 10% (Needs improvement)")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    main()