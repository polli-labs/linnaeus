# linnaeus/loss/loss_optimizer.py
"""
Loss path optimization module that provides configuration and integration
for optimized loss operations including masking and weighting.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class OptimizedLossProcessor:
    """
    Manages optimized loss processing with configurable optimizations.
    Integrates vectorized masking, fused operations, and torch.compile support.
    """
    
    def __init__(self, config: CN, num_classes_per_task: dict[str, int], device: torch.device):
        """
        Initialize the optimized loss processor.
        
        Args:
            config: Experiment configuration
            num_classes_per_task: Number of classes for each task
            device: Device for tensor operations
        """
        self.config = config
        self.device = device
        self.num_classes_per_task = num_classes_per_task
        
        # Check optimization flags
        self.use_optimized = getattr(config.LOSS, "USE_OPTIMIZED_PATH", False)
        self.use_torch_compile = getattr(config.LOSS, "USE_TORCH_COMPILE", False)
        self.compile_mode = getattr(config.LOSS, "COMPILE_MODE", "default")
        
        # Pre-compute class weight tensors if using optimized path
        self.class_weight_tensors = None
        if self.use_optimized:
            self._prepare_class_weights()
        
        # Compile loss functions if requested
        if self.use_torch_compile:
            self._compile_loss_functions()
        
        logger.info(f"OptimizedLossProcessor initialized: optimized={self.use_optimized}, compile={self.use_torch_compile}")
    
    def _prepare_class_weights(self):
        """Pre-compute class weight tensors for optimized lookup."""
        from linnaeus.loss.masking_optimized import prepare_class_weights_tensors
        
        # Get class weights from config if available
        class_weights_dict = self.config.LOSS.GRAD_WEIGHTING.CLASS.get("WEIGHTS", None)
        if class_weights_dict:
            self.class_weight_tensors = prepare_class_weights_tensors(
                class_weights_dict,
                self.num_classes_per_task,
                self.device,
                dtype=torch.float32
            )
            logger.debug(f"Pre-computed class weight tensors for {len(self.class_weight_tensors)} tasks")
    
    def _compile_loss_functions(self):
        """Apply torch.compile to loss processing functions."""
        if not self.use_optimized:
            logger.warning("torch.compile requested but optimized path not enabled. Skipping compilation.")
            return
        
        try:
            from linnaeus.loss import masking_optimized
            
            # Compile the optimized functions
            compile_config = {
                'backend': 'inductor',
                'mode': self.compile_mode,
                'fullgraph': False,
            }
            
            masking_optimized.apply_null_masking_optimized = torch.compile(
                masking_optimized.apply_null_masking_optimized,
                **compile_config
            )
            
            masking_optimized.apply_class_weighting_optimized = torch.compile(
                masking_optimized.apply_class_weighting_optimized,
                **compile_config
            )
            
            logger.info(f"Successfully compiled loss functions with mode='{self.compile_mode}'")
        except Exception as e:
            logger.warning(f"Failed to compile loss functions: {e}. Falling back to eager mode.")
    
    def process_losses(
        self,
        per_task_losses: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        ops_schedule: Any,
        current_step: int,
        is_validation: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Process losses with optimized or standard path based on configuration.
        
        Args:
            per_task_losses: Per-sample losses for each task
            targets: Target labels for each task
            ops_schedule: Schedule for operations
            current_step: Current training step
            is_validation: Whether this is validation
        
        Returns:
            Tuple of processed losses and statistics
        """
        if self.use_optimized:
            from linnaeus.loss.masking_optimized import apply_loss_masking_optimized
            
            return apply_loss_masking_optimized(
                per_task_losses,
                targets,
                ops_schedule,
                current_step,
                class_weights=self.class_weight_tensors,
                is_validation=is_validation,
                logger=logger,
                config=self.config,
            )
        else:
            from linnaeus.loss.masking import apply_loss_masking
            
            # Convert tensor weights back to dict format for legacy path
            class_weights_dict = None
            if self.class_weight_tensors:
                class_weights_dict = {}
                for task_key, weight_tensor in self.class_weight_tensors.items():
                    class_weights_dict[task_key] = {
                        i: weight_tensor[i].item()
                        for i in range(len(weight_tensor))
                    }
            
            return apply_loss_masking(
                per_task_losses,
                targets,
                ops_schedule,
                current_step,
                class_weights=class_weights_dict,
                is_validation=is_validation,
                logger=logger,
                config=self.config,
            )