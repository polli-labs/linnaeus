#!/usr/bin/env python3
"""
Validation script to compare estimated vs actual model complexity.

This script builds actual models and compares parameter counts with estimates
to validate the accuracy of the estimation tool.
"""

import argparse
import os
import sys
import yaml
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    from yacs.config import CfgNode as CN
    from linnaeus.models.build import build_model
    from tools.estimate_model_complexity import mFormerV1ComplexityEstimator, format_number
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running in the correct environment with all dependencies installed.")
    sys.exit(1)


def create_minimal_config(model_config_path: str) -> CN:
    """Create a minimal config for model building."""
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)['MODEL']
    
    # Create minimal config with required fields
    config = CN()
    
    # Model config - need to convert nested dicts to CN recursively
    config.MODEL = CN()
    for key, value in model_config.items():
        if isinstance(value, dict):
            config.MODEL[key] = CN(value)
        else:
            config.MODEL[key] = value
    config.MODEL.IMG_SIZE = 384
    
    # Set up classification heads
    if 'CLASSIFICATION' not in config.MODEL or not config.MODEL.CLASSIFICATION.HEADS:
        config.MODEL.CLASSIFICATION = CN()
        config.MODEL.CLASSIFICATION.HEADS = CN()
        for task in ['taxa_L10', 'taxa_L20', 'taxa_L30', 'taxa_L40']:
            config.MODEL.CLASSIFICATION.HEADS[task] = CN({
                'TYPE': 'ConditionalClassifier',
                'ROUTING_STRATEGY': 'soft',
                'TEMPERATURE': 1.0,
                'USE_BIAS': True
            })
    
    # Required DATA config
    config.DATA = CN()
    config.DATA.TASK_KEYS_H5 = ['taxa_L10', 'taxa_L20', 'taxa_L30', 'taxa_L40']
    config.DATA.META = CN()
    config.DATA.META.ACTIVE = True
    config.DATA.META.COMPONENTS = CN()
    config.DATA.META.COMPONENTS.TEMPORAL = CN({
        'ENABLED': True, 'DIM': 2, 'IDX': 0
    })
    config.DATA.META.COMPONENTS.SPATIAL = CN({
        'ENABLED': True, 'DIM': 3, 'IDX': 1  
    })
    config.DATA.META.COMPONENTS.ELEVATION = CN({
        'ENABLED': True, 'DIM': 10, 'IDX': 2
    })
    
    # Required TRAIN config
    config.TRAIN = CN()
    config.TRAIN.GRADIENT_CHECKPOINTING = CN()
    config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS = False
    
    # Add DEBUG config to suppress warnings
    config.DEBUG = CN()
    config.DEBUG.MODEL_BUILD = False
    
    return config


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count parameters in different parts of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count head parameters
    head_params = 0
    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters())
    
    # Count meta head parameters
    meta_head_params = 0
    for name, param in model.named_parameters():
        if 'meta_' in name and '_head_' in name:
            meta_head_params += param.numel()
    
    # Backbone is everything else
    backbone_params = total_params - head_params - meta_head_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'backbone_params': backbone_params,
        'head_params': head_params,
        'meta_head_params': meta_head_params
    }


def validate_variant(variant: str) -> Dict[str, Any]:
    """Validate estimates for a specific model variant."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(project_root, f"configs/model/archs/mFormerV1/mFormerV1_{variant}.yaml")
    
    if not os.path.exists(config_path):
        return {'error': f"Config not found: {config_path}"}
    
    try:
        # Get estimates
        estimator = mFormerV1ComplexityEstimator()
        estimated = estimator.estimate_from_config(config_path)
        
        # Build actual model
        config = create_minimal_config(config_path)
        num_classes = {
            'taxa_L10': 684,
            'taxa_L20': 179,
            'taxa_L30': 40,
            'taxa_L40': 4
        }
        
        model = build_model(config, num_classes=num_classes, taxonomy_tree=None)
        actual = count_model_parameters(model)
        
        # Calculate errors
        param_error = abs(estimated.total_params - actual['total_params']) / actual['total_params'] * 100
        backbone_error = abs(estimated.backbone_params - actual['backbone_params']) / actual['backbone_params'] * 100
        head_error = abs(estimated.head_params - actual['head_params']) / actual['head_params'] * 100
        meta_head_error = abs(estimated.meta_head_params - actual['meta_head_params']) / actual['meta_head_params'] * 100
        
        return {
            'variant': variant,
            'estimated': {
                'total_params': estimated.total_params,
                'backbone_params': estimated.backbone_params,
                'head_params': estimated.head_params,
                'meta_head_params': estimated.meta_head_params
            },
            'actual': actual,
            'errors': {
                'total_params_error_pct': param_error,
                'backbone_params_error_pct': backbone_error,
                'head_params_error_pct': head_error,
                'meta_head_params_error_pct': meta_head_error
            }
        }
    
    except Exception as e:
        return {'error': f"Failed to validate {variant}: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Validate model complexity estimates")
    parser.add_argument("--variant", choices=['sm', 'md', 'lg', 'xl'], help="Validate specific variant")
    parser.add_argument("--all", action="store_true", help="Validate all variants")
    
    args = parser.parse_args()
    
    if args.all:
        variants = ['sm', 'md', 'lg', 'xl']
    elif args.variant:
        variants = [args.variant]
    else:
        parser.print_help()
        return 1
    
    print("Validating Model Complexity Estimates")
    print("=" * 60)
    
    for variant in variants:
        result = validate_variant(variant)
        
        if 'error' in result:
            print(f"\n{variant.upper()}: {result['error']}")
            continue
        
        print(f"\nmFormerV1_{variant.upper()} Validation Results:")
        print("-" * 40)
        
        est = result['estimated']
        act = result['actual']
        err = result['errors']
        
        print(f"Parameter Count Comparison:")
        print(f"  Total:      Est: {format_number(est['total_params']):>12} | Act: {format_number(act['total_params']):>12} | Error: {err['total_params_error_pct']:>6.2f}%")
        print(f"  Backbone:   Est: {format_number(est['backbone_params']):>12} | Act: {format_number(act['backbone_params']):>12} | Error: {err['backbone_params_error_pct']:>6.2f}%")
        print(f"  Heads:      Est: {format_number(est['head_params']):>12} | Act: {format_number(act['head_params']):>12} | Error: {err['head_params_error_pct']:>6.2f}%")
        print(f"  Meta Heads: Est: {format_number(est['meta_head_params']):>12} | Act: {format_number(act['meta_head_params']):>12} | Error: {err['meta_head_params_error_pct']:>6.2f}%")
        
        # Overall assessment
        if err['total_params_error_pct'] < 5:
            assessment = "✓ Excellent"
        elif err['total_params_error_pct'] < 10:
            assessment = "✓ Good"
        elif err['total_params_error_pct'] < 20:
            assessment = "⚠ Fair"
        else:
            assessment = "✗ Poor"
        
        print(f"  Assessment: {assessment}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())