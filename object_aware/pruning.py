"""ObjectMark-based Gaussian pruning helpers."""

from typing import List, Protocol

import torch


OBJECT_MARK_SCORE_BIN_LABELS = (
    "0~0.1",
    "0.1~0.2",
    "0.2~0.3",
    "0.3~0.4",
    "0.4~0.5",
    "0.5~0.6",
    "0.6~0.7",
    "0.7~0.8",
    "0.8~0.9",
    "0.9~1",
)


class ObjectMarkGaussianModel(Protocol):
    """Minimal model interface needed by the pruning helpers."""

    @property
    def get_objectmark_score_prob(self) -> torch.Tensor:
        ...

    def prune_background_by_objectmark_score(self, threshold: float) -> int:
        ...


@torch.no_grad()
def prune_gaussians_by_object_mark(
    gaussians: ObjectMarkGaussianModel,
    threshold: float,
) -> int:
    """Prune Gaussians whose ObjectMark probability is at most ``threshold``.

    The model method updates every optimizable Gaussian tensor, Adam state,
    densification accumulator, and screen-space radius consistently. Training
    invokes this function at each configured pruning iteration; with the paper
    defaults this is repeated every 500 iterations from 2,500 through 29,500.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"ObjectMark pruning threshold must be in [0, 1], got {threshold}")
    return gaussians.prune_background_by_objectmark_score(threshold)


@torch.no_grad()
def object_mark_score_bin_counts(
    gaussians: ObjectMarkGaussianModel,
) -> List[int]:
    """Count ObjectMark probabilities in ten diagnostic bins."""
    scores = gaussians.get_objectmark_score_prob.detach().flatten().clamp(0.0, 1.0)
    if scores.numel() == 0:
        return [0] * len(OBJECT_MARK_SCORE_BIN_LABELS)

    bin_edges = scores.new_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bin_indices = torch.bucketize(scores, bin_edges, right=True)
    bin_counts = torch.bincount(bin_indices, minlength=len(OBJECT_MARK_SCORE_BIN_LABELS))
    return bin_counts[: len(OBJECT_MARK_SCORE_BIN_LABELS)].cpu().tolist()
