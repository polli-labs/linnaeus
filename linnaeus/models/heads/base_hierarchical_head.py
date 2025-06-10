import torch.nn as nn


class BaseHierarchicalHead(nn.Module):
    """Base class for hierarchical heads providing GradNorm mode switching."""

    def __init__(self):
        super().__init__()
        self._gradnorm_mode = False

    def set_gradnorm_mode(self, mode: bool) -> None:
        """Enable or disable GradNorm mode."""
        self._gradnorm_mode = bool(mode)

    def is_gradnorm_mode(self) -> bool:
        """Return whether GradNorm mode is active."""
        return self._gradnorm_mode

