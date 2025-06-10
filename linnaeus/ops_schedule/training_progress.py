"""
Training progress tracking for robust checkpoint resumption.

This module implements a TrainingProgress class that tracks the current training stage
(training vs different validation types) and ensures correct resumption after interruptions.
"""

from enum import Enum, auto
from typing import Any


class TrainingStage(Enum):
    """Enum representing the different stages in the training and validation process."""

    TRAINING = auto()
    VALIDATION_NORMAL = auto()
    VALIDATION_MASK_META = auto()
    VALIDATION_PARTIAL_MASK_META = auto()

    @classmethod
    def validation_stages(cls) -> set["TrainingStage"]:
        """Return all validation stages."""
        return {
            cls.VALIDATION_NORMAL,
            cls.VALIDATION_MASK_META,
            cls.VALIDATION_PARTIAL_MASK_META,
        }

    def is_validation(self) -> bool:
        """Check if this stage is any kind of validation."""
        return self in TrainingStage.validation_stages()


class TrainingProgress:
    """
    Tracks the progress of training and validation for robust checkpoint resumption.

    This class maintains the current training stage and scheduled validations to ensure
    that when training is resumed from a checkpoint, it correctly continues from the
    appropriate stage (training or validation).
    """

    def __init__(self):
        """Initialize a new TrainingProgress instance."""
        self.current_stage: TrainingStage = TrainingStage.TRAINING
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.expected_total_steps: int | None = None
        self.pending_validations: list[TrainingStage] = []
        self.completed_validations: list[TrainingStage] = []
        self.partial_validation_indices: list[int] = []

    def start_training_epoch(self, epoch: int):
        """Mark the start of a training epoch."""
        self.current_stage = TrainingStage.TRAINING
        self.current_epoch = epoch
        self.pending_validations = []
        self.completed_validations = []
        self.partial_validation_indices = []

    def schedule_validation(
        self, validation_type: TrainingStage, partial_index: int | None = None
    ):
        """
        Schedule a validation to be performed after the current epoch.

        Args:
            validation_type: The type of validation to schedule
            partial_index: For partial mask-meta validation, the index of the component to validate
        """
        if validation_type != TrainingStage.TRAINING:
            if validation_type not in self.pending_validations:
                self.pending_validations.append(validation_type)

            if (
                validation_type == TrainingStage.VALIDATION_PARTIAL_MASK_META
                and partial_index is not None
            ):
                if partial_index not in self.partial_validation_indices:
                    self.partial_validation_indices.append(partial_index)

    def start_validation(self, validation_type: TrainingStage):
        """Mark the start of a validation phase."""
        self.current_stage = validation_type

    def complete_validation(
        self, validation_type: TrainingStage, partial_index: int | None = None
    ):
        """
        Mark the completion of a validation phase.

        Args:
            validation_type: The type of validation completed
            partial_index: For partial mask-meta validation, the index of the component validated
        """
        if validation_type not in self.completed_validations:
            self.completed_validations.append(validation_type)

        # Remove from pending if all related validations are complete
        if validation_type == TrainingStage.VALIDATION_PARTIAL_MASK_META:
            if (
                partial_index is not None
                and partial_index in self.partial_validation_indices
            ):
                self.partial_validation_indices.remove(partial_index)

            # Only remove from pending if all partial validations are complete
            if not self.partial_validation_indices:
                if validation_type in self.pending_validations:
                    self.pending_validations.remove(validation_type)
        else:
            # For non-partial validations, simply remove from pending
            if validation_type in self.pending_validations:
                self.pending_validations.remove(validation_type)

        # Return to training as default state after validation if no more pending
        if not self.has_pending_validations():
            self.current_stage = TrainingStage.TRAINING

    def has_pending_validations(self) -> bool:
        """Check if there are pending validations to be performed."""
        return len(self.pending_validations) > 0

    def get_pending_validations(self) -> list[TrainingStage]:
        """Get a list of pending validation stages."""
        return self.pending_validations.copy()

    def get_partial_validation_indices(self) -> list[int]:
        """Get indices for pending partial mask-meta validations."""
        return self.partial_validation_indices.copy()

    def state_dict(self) -> dict[str, Any]:
        """Convert the training progress to a state dictionary for checkpoint saving."""
        return {
            "current_stage": self.current_stage.name,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "expected_total_steps": self.expected_total_steps,
            "pending_validations": [val.name for val in self.pending_validations],
            "completed_validations": [val.name for val in self.completed_validations],
            "partial_validation_indices": self.partial_validation_indices,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the training progress from a state dictionary.

        Args:
            state_dict: Dictionary containing training progress state
        """
        self.current_stage = TrainingStage[state_dict["current_stage"]]
        self.current_epoch = state_dict["current_epoch"]
        self.global_step = state_dict["global_step"]
        self.expected_total_steps = state_dict.get("expected_total_steps", None)
        self.pending_validations = [
            TrainingStage[name] for name in state_dict["pending_validations"]
        ]
        self.completed_validations = [
            TrainingStage[name] for name in state_dict["completed_validations"]
        ]
        self.partial_validation_indices = state_dict["partial_validation_indices"]

    def __str__(self) -> str:
        """Return a string representation of the training progress state."""
        return (
            f"TrainingProgress(stage={self.current_stage.name}, "
            f"epoch={self.current_epoch}, step={self.global_step}, "
            f"expected_total_steps={self.expected_total_steps}, "
            f"pending={[v.name for v in self.pending_validations]}, "
            f"completed={[v.name for v in self.completed_validations]}, "
            f"partial_indices={self.partial_validation_indices})"
        )
