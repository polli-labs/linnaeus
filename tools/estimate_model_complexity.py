#!/usr/bin/env python3
"""
Model Parameter/FLOP Estimation Tool for mFormerV1 architectures.

This script estimates parameters and FLOPs for mFormerV1 model variants
without requiring actual model instantiation, making it useful for quick
architecture comparison and planning.

Usage:
    python tools/estimate_model_complexity.py --config path/to/config.yaml
    python tools/estimate_model_complexity.py --all-variants
    python tools/estimate_model_complexity.py --variant xl --task-config path/to/task_config.yaml
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any
import yaml
from dataclasses import dataclass

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class ModelComplexity:
    """Container for model complexity metrics."""
    total_params: int
    trainable_params: int
    backbone_params: int
    head_params: int
    meta_head_params: int
    flops: int
    memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'backbone_params': self.backbone_params,
            'head_params': self.head_params,
            'meta_head_params': self.meta_head_params,
            'flops': self.flops,
            'memory_mb': self.memory_mb,
        }

class mFormerV1ComplexityEstimator:
    """Estimates parameters and FLOPs for mFormerV1 architecture."""
    
    def __init__(self, img_size: int = 384, in_chans: int = 3):
        self.img_size = img_size
        self.in_chans = in_chans
        
    def estimate_convnext_block_params(self, dim: int) -> int:
        """Estimate parameters for a ConvNeXt block."""
        # ConvNeXt block: 
        # - Depthwise Conv 7x7: dim * (7*7 + 1)
        # - LayerNorm: 2 * dim
        # - Linear 1 (expand): dim * (4*dim + 1)  # mlp_ratio = 4
        # - Linear 2 (contract): 4*dim * (dim + 1)
        # - Layer scale: dim (if used)
        
        dw_conv = dim * (7 * 7 + 1)  # 7x7 depthwise + bias
        norm = 2 * dim  # LayerNorm weight + bias
        mlp_expand = dim * (4 * dim + 1)
        mlp_contract = 4 * dim * (dim + 1)
        layer_scale = dim  # layer scale parameter
        
        return dw_conv + norm + mlp_expand + mlp_contract + layer_scale
    
    def estimate_rope_mhsa_block_params(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> int:
        """Estimate parameters for a RoPE2D MHSA block."""
        # Multi-head attention:
        # - qkv projection: dim * (3 * dim + 3)  # 3 for q,k,v with bias
        # - output projection: dim * (dim + 1)
        # - LayerNorm (pre-attn): 2 * dim
        # - LayerNorm (pre-mlp): 2 * dim
        # MLP:
        # - Linear 1: dim * (mlp_ratio * dim + 1)
        # - Linear 2: mlp_ratio * dim * (dim + 1)
        
        qkv_proj = dim * (3 * dim + 3)
        attn_out_proj = dim * (dim + 1)
        norm1 = 2 * dim
        norm2 = 2 * dim
        
        mlp_hidden = int(dim * mlp_ratio)
        mlp_in = dim * (mlp_hidden + 1)
        mlp_out = mlp_hidden * (dim + 1)
        
        return qkv_proj + attn_out_proj + norm1 + norm2 + mlp_in + mlp_out
    
    def estimate_convnext_downsample_params(self, in_dim: int, out_dim: int) -> int:
        """Estimate parameters for ConvNeXt downsampling layer."""
        # LayerNorm + 2x2 Conv stride 2
        norm = 2 * in_dim
        conv = out_dim * (in_dim * 4 + 1)  # 2x2 conv + bias
        return norm + conv
    
    def estimate_conditional_classifier_params(self, in_features: int, num_classes_dict: Dict[str, int]) -> int:
        """Estimate parameters for ConditionalClassifier head."""
        total_params = 0
        
        # Each level gets a linear classifier
        for task_key, num_classes in num_classes_dict.items():
            # Linear layer: in_features * num_classes + num_classes (bias)
            total_params += in_features * (num_classes + 1)
        
        return total_params
    
    def estimate_meta_head_params(self, meta_dims: List[int], rope_dims: List[int]) -> int:
        """Estimate parameters for metadata heads."""
        total_params = 0
        
        for meta_dim in meta_dims:
            if meta_dim > 0:
                # For each RoPE stage (2 stages)
                for rope_dim in rope_dims:
                    # Linear + ReLU + LayerNorm + ResNormLayer
                    linear = meta_dim * (rope_dim + 1)
                    norm = 2 * rope_dim  # LayerNorm
                    res_norm = rope_dim  # ResNormLayer (approximately)
                    total_params += linear + norm + res_norm
        
        return total_params
    
    def estimate_backbone_flops(self, convnext_config: Dict, rope_config: Dict) -> int:
        """Estimate FLOPs for the backbone (ConvNeXt + RoPE stages)."""
        H, W = self.img_size, self.img_size
        total_flops = 0
        
        # Stem: 4x4 conv stride 4
        stem_flops = (H // 4) * (W // 4) * convnext_config['DIMS'][0] * self.in_chans * 16
        total_flops += stem_flops
        H, W = H // 4, W // 4
        
        # ConvNeXt stages (first 2 only)
        current_dim = convnext_config['DIMS'][0]
        
        # Stage 1
        for _ in range(convnext_config['DEPTHS'][0]):
            # Depthwise conv 7x7
            stage1_dw_flops = H * W * current_dim * 49
            # MLP (4x expansion)
            stage1_mlp_flops = H * W * (current_dim * 4 * current_dim + 4 * current_dim * current_dim)
            total_flops += stage1_dw_flops + stage1_mlp_flops
        
        # Downsample 1
        total_flops += (H // 2) * (W // 2) * convnext_config['DIMS'][1] * current_dim * 4
        H, W = H // 2, W // 2
        current_dim = convnext_config['DIMS'][1]
        
        # Stage 2
        for _ in range(convnext_config['DEPTHS'][1]):
            stage2_dw_flops = H * W * current_dim * 49
            stage2_mlp_flops = H * W * (current_dim * 4 * current_dim + 4 * current_dim * current_dim)
            total_flops += stage2_dw_flops + stage2_mlp_flops
        
        # Downsample 2
        total_flops += (H // 2) * (W // 2) * rope_config['DIMS'][0] * current_dim * 4
        H, W = H // 2, W // 2
        
        # RoPE stages
        seq_len_stage3 = H * W + 4  # patch tokens + 4 extra tokens
        for stage_idx in range(2):
            dim = rope_config['DIMS'][stage_idx]
            depths = rope_config['DEPTHS'][stage_idx]
            
            seq_len = seq_len_stage3 if stage_idx == 0 else (H // 2) * (W // 2) + 4
            
            for _ in range(depths):
                # Attention FLOPs: qkv projection + attention computation + output projection
                qkv_flops = seq_len * dim * 3 * dim
                attn_flops = seq_len * seq_len * dim  # Simplified attention
                out_proj_flops = seq_len * dim * dim
                
                # MLP FLOPs
                mlp_hidden = int(dim * rope_config['MLP_RATIO'][stage_idx])
                mlp_flops = seq_len * (dim * mlp_hidden + mlp_hidden * dim)
                
                total_flops += qkv_flops + attn_flops + out_proj_flops + mlp_flops
            
            if stage_idx == 0:  # After stage 3, downsample again
                H, W = H // 2, W // 2
        
        return total_flops
    
    def estimate_head_flops(self, in_features: int, num_classes_dict: Dict[str, int], batch_size: int = 1) -> int:
        """Estimate FLOPs for classification heads."""
        total_flops = 0
        
        for num_classes in num_classes_dict.values():
            # Linear layer FLOPs
            total_flops += batch_size * in_features * num_classes
        
        return total_flops
    
    def estimate_from_config(self, config_path: str, num_classes_dict: Dict[str, int] = None) -> ModelComplexity:
        """Estimate complexity from a model config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['MODEL']
        
        # Default reptilia classes if not provided
        if num_classes_dict is None:
            num_classes_dict = {
                'taxa_L10': 684,
                'taxa_L20': 179, 
                'taxa_L30': 40,
                'taxa_L40': 4
            }
        
        return self.estimate_from_dict(model_config, num_classes_dict)
    
    def estimate_from_dict(self, model_config: Dict, num_classes_dict: Dict[str, int]) -> ModelComplexity:
        """Estimate complexity from a model config dictionary."""
        convnext_config = model_config['CONVNEXT_STAGES']
        rope_config = model_config['ROPE_STAGES']
        
        # Extract metadata configuration
        meta_dims = []
        if 'EXTRA_TOKEN_NUM' in model_config:
            # Assume EXTRA_TOKEN_NUM = 1 (CLS) + len(meta_dims)
            extra_tokens = model_config['EXTRA_TOKEN_NUM']
            if extra_tokens > 1:
                # Default reptilia meta dims: [2, 3, 10]
                meta_dims = [2, 3, 10]  # TEMPORAL, SPATIAL, ELEVATION
        
        backbone_params = self._estimate_backbone_params(convnext_config, rope_config)
        head_params = self.estimate_conditional_classifier_params(rope_config['DIMS'][-1], num_classes_dict)
        meta_head_params = self.estimate_meta_head_params(meta_dims, rope_config['DIMS'])
        
        # Additional components
        cls_tokens = 2 * sum(rope_config['DIMS'])  # 2 CLS tokens
        aggregation_params = 0
        if not model_config.get('ONLY_LAST_CLS', False):
            # cl_1_fc MLP + Conv1d aggregation + final norm
            rope_dim_final = rope_config['DIMS'][-1]
            aggregation_params = (rope_config['DIMS'][0] * rope_dim_final + rope_dim_final * rope_dim_final +  # MLP
                                2 * rope_dim_final +  # LayerNorm  
                                2 * 1 * rope_dim_final +  # Conv1d aggregation
                                2 * rope_dim_final)  # final norm
        else:
            aggregation_params = 2 * rope_config['DIMS'][-1]  # Just final norm
        
        total_params = backbone_params + head_params + meta_head_params + cls_tokens + aggregation_params
        
        # Estimate FLOPs
        backbone_flops = self.estimate_backbone_flops(convnext_config, rope_config)
        head_flops = self.estimate_head_flops(rope_config['DIMS'][-1], num_classes_dict)
        total_flops = backbone_flops + head_flops
        
        # Estimate memory (rough approximation)
        memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
        
        return ModelComplexity(
            total_params=total_params,
            trainable_params=total_params,  # Assume all trainable
            backbone_params=backbone_params,
            head_params=head_params,
            meta_head_params=meta_head_params,
            flops=total_flops,
            memory_mb=memory_mb
        )
    
    def _estimate_backbone_params(self, convnext_config: Dict, rope_config: Dict) -> int:
        """Estimate backbone parameters."""
        total_params = 0
        
        # Stem
        stem_params = convnext_config['DIMS'][0] * (self.in_chans * 16 + 1) + 2 * convnext_config['DIMS'][0]
        total_params += stem_params
        
        # ConvNeXt stages (first 2)
        for stage_idx in range(2):
            depth = convnext_config['DEPTHS'][stage_idx]
            dim = convnext_config['DIMS'][stage_idx]
            for _ in range(depth):
                total_params += self.estimate_convnext_block_params(dim)
        
        # Downsamplers
        for i in range(3):
            if i < 2:
                in_dim, out_dim = convnext_config['DIMS'][i], convnext_config['DIMS'][i+1]
            else:
                in_dim, out_dim = convnext_config['DIMS'][i], rope_config['DIMS'][i-2]
            total_params += self.estimate_convnext_downsample_params(in_dim, out_dim)
        
        # RoPE stages
        for stage_idx in range(2):
            depth = rope_config['DEPTHS'][stage_idx]
            dim = rope_config['DIMS'][stage_idx]
            num_heads = rope_config['NUM_HEADS'][stage_idx]
            mlp_ratio = rope_config['MLP_RATIO'][stage_idx]
            
            for _ in range(depth):
                total_params += self.estimate_rope_mhsa_block_params(dim, num_heads, mlp_ratio)
        
        # Norms between stages
        total_params += 2 * rope_config['DIMS'][0]  # norm_1
        total_params += 2 * rope_config['DIMS'][1]  # norm_2
        
        return total_params


def load_config_from_experiment(experiment_config_path: str) -> Tuple[Dict, Dict[str, int]]:
    """Load model config and extract num_classes from an experiment config."""
    with open(experiment_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Load base model config
    base_config_path = exp_config['MODEL']['BASE'][0]
    if not os.path.isabs(base_config_path):
        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_config_path = os.path.join(project_root, base_config_path)
    
    with open(base_config_path, 'r') as f:
        model_config = yaml.safe_load(f)['MODEL']
    
    # Override with experiment-specific settings
    if 'MODEL' in exp_config:
        exp_model_config = exp_config['MODEL']
        for key, value in exp_model_config.items():
            if key != 'BASE':
                model_config[key] = value
    
    # Extract task dimensions (use default reptilia if not found)
    num_classes_dict = {
        'taxa_L10': 684,
        'taxa_L20': 179,
        'taxa_L30': 40, 
        'taxa_L40': 4
    }
    
    return model_config, num_classes_dict


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def main():
    parser = argparse.ArgumentParser(description="Estimate mFormerV1 model complexity")
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--experiment-config", help="Path to experiment config file")
    parser.add_argument("--variant", choices=['sm', 'md', 'lg', 'xl'], help="Model variant to analyze")
    parser.add_argument("--all-variants", action="store_true", help="Analyze all model variants")
    parser.add_argument("--img-size", type=int, default=384, help="Input image size")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    estimator = mFormerV1ComplexityEstimator(img_size=args.img_size)
    results = {}
    
    if args.all_variants:
        # Analyze all variants
        project_root = os.path.dirname(os.path.dirname(__file__))
        variants = ['sm', 'md', 'lg', 'xl']
        
        for variant in variants:
            config_path = os.path.join(project_root, f"configs/model/archs/mFormerV1/mFormerV1_{variant}.yaml")
            if os.path.exists(config_path):
                complexity = estimator.estimate_from_config(config_path)
                results[f"mFormerV1_{variant}"] = complexity.to_dict()
                
                print(f"\n{'='*60}")
                print(f"mFormerV1_{variant.upper()} Complexity Estimate")
                print(f"{'='*60}")
                print(f"Total Parameters:     {format_number(complexity.total_params):>12}")
                print(f"Backbone Parameters:  {format_number(complexity.backbone_params):>12}")
                print(f"Head Parameters:      {format_number(complexity.head_params):>12}")
                print(f"Meta Head Parameters: {format_number(complexity.meta_head_params):>12}")
                print(f"FLOPs:               {format_number(complexity.flops):>12}")
                print(f"Memory (approx):     {complexity.memory_mb:>8.1f} MB")
            else:
                print(f"Config not found: {config_path}")
    
    elif args.variant:
        # Analyze specific variant
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, f"configs/model/archs/mFormerV1/mFormerV1_{args.variant}.yaml")
        
        if os.path.exists(config_path):
            complexity = estimator.estimate_from_config(config_path)
            results[f"mFormerV1_{args.variant}"] = complexity.to_dict()
            
            print(f"\nmFormerV1_{args.variant.upper()} Complexity Estimate")
            print(f"{'='*40}")
            print(f"Total Parameters:     {format_number(complexity.total_params)}")
            print(f"Backbone Parameters:  {format_number(complexity.backbone_params)}")
            print(f"Head Parameters:      {format_number(complexity.head_params)}")
            print(f"Meta Head Parameters: {format_number(complexity.meta_head_params)}")
            print(f"FLOPs:               {format_number(complexity.flops)}")
            print(f"Memory (approx):     {complexity.memory_mb:.1f} MB")
        else:
            print(f"Config not found: {config_path}")
            return 1
    
    elif args.config:
        # Analyze from config file
        complexity = estimator.estimate_from_config(args.config)
        model_name = os.path.basename(args.config).replace('.yaml', '')
        results[model_name] = complexity.to_dict()
        
        print(f"\n{model_name} Complexity Estimate")
        print(f"{'='*40}")
        print(f"Total Parameters:     {format_number(complexity.total_params)}")
        print(f"Backbone Parameters:  {format_number(complexity.backbone_params)}")
        print(f"Head Parameters:      {format_number(complexity.head_params)}")
        print(f"Meta Head Parameters: {format_number(complexity.meta_head_params)}")
        print(f"FLOPs:               {format_number(complexity.flops)}")
        print(f"Memory (approx):     {complexity.memory_mb:.1f} MB")
    
    elif args.experiment_config:
        # Analyze from experiment config
        model_config, num_classes_dict = load_config_from_experiment(args.experiment_config)
        complexity = estimator.estimate_from_dict(model_config, num_classes_dict)
        
        exp_name = os.path.basename(args.experiment_config).replace('.yaml', '')
        results[exp_name] = complexity.to_dict()
        
        print(f"\n{exp_name} Complexity Estimate")
        print(f"{'='*50}")
        print(f"Total Parameters:     {format_number(complexity.total_params)}")
        print(f"Backbone Parameters:  {format_number(complexity.backbone_params)}")
        print(f"Head Parameters:      {format_number(complexity.head_params)}")
        print(f"Meta Head Parameters: {format_number(complexity.meta_head_params)}")
        print(f"FLOPs:               {format_number(complexity.flops)}")
        print(f"Memory (approx):     {complexity.memory_mb:.1f} MB")
        
        if args.verbose:
            print(f"\nTask Configuration:")
            for task, num_classes in num_classes_dict.items():
                print(f"  {task}: {num_classes} classes")
    else:
        parser.print_help()
        return 1
    
    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())