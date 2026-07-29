"""Object-aware supervision and pruning for mask-guided 2DGS."""

from .losses import (
    alpha_polarization_loss,
    masked_l1_loss,
    object_mark_polarization_loss,
)
from .pruning import prune_gaussians_by_object_mark
from .schedules import (
    default_object_mark_pruning_iterations,
    object_mark_active_at_iteration,
)

__all__ = [
    "alpha_polarization_loss",
    "default_object_mark_pruning_iterations",
    "masked_l1_loss",
    "object_mark_active_at_iteration",
    "object_mark_polarization_loss",
    "prune_gaussians_by_object_mark",
]
