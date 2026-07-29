"""Losses introduced by the object-aware reconstruction method.

All functions operate on tensors that are already on the same device. They do
not move data implicitly, which keeps device placement visible in the training
loop.
"""

from typing import Optional

import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    mask_denominator: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the mean absolute RGB error over foreground pixels.

    Args:
        prediction: Rendered image with shape ``[C, H, W]`` and values normally
            in ``[0, 1]``.
        target: Ground-truth image with the same shape as ``prediction``.
        foreground_mask: Single-channel mask broadcastable to ``prediction``;
            foreground is 1 and background is 0.
        mask_denominator: Optional cached value equal to
            ``foreground_mask.sum() * C + eps``.
        eps: Denominator guard used when a mask contains no foreground pixels.

    The reduction and epsilon match the original method implementation.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.device != target.device or prediction.device != foreground_mask.device:
        raise ValueError("prediction, target, and foreground_mask must be on the same device")

    foreground_mask = foreground_mask.to(dtype=prediction.dtype)
    absolute_error = torch.sub(prediction, target)
    absolute_error.abs_().mul_(foreground_mask)
    if mask_denominator is None:
        mask_denominator = foreground_mask.sum() * prediction.shape[0] + eps
    return absolute_error.sum() / mask_denominator


def _polarization_loss(
    rendered_value: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    if rendered_value.device != foreground_mask.device:
        raise ValueError("rendered_value and foreground_mask must be on the same device")
    if rendered_value.shape != foreground_mask.shape:
        raise ValueError(
            "rendered_value and foreground_mask must have identical shapes; "
            f"got {tuple(rendered_value.shape)} and {tuple(foreground_mask.shape)}"
        )

    rendered_value = rendered_value.clamp(0.0, 1.0)
    foreground_mask = foreground_mask.to(dtype=rendered_value.dtype)
    # Equivalent to v * (1 - m) + (1 - v) * m, using the existing reduction.
    loss_map = torch.addcmul(rendered_value, rendered_value, foreground_mask, value=-2.0)
    loss_map.add_(foreground_mask)
    return loss_map.mean()


def alpha_polarization_loss(
    rendered_alpha: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """Polarize rendered alpha toward a binary foreground mask.

    Both tensors have shape ``[1, H, W]`` and expected values in ``[0, 1]``.
    """
    return _polarization_loss(rendered_alpha, foreground_mask)


def object_mark_polarization_loss(
    rendered_object_mark: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """Polarize the rendered ObjectMark response toward the input mask.

    Both tensors have shape ``[1, H, W]`` and expected values in ``[0, 1]``.
    """
    return _polarization_loss(rendered_object_mark, foreground_mask)
