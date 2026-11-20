"""Direction-bin decoding helper for the opt-in native CoAlign path.

The generic VoxelPostprocessor intentionally remains unchanged so accepted
legacy checkpoints keep their exact decoding behavior.  V2V4Real's adapter
can call :func:`apply_direction_bins` after decoding and before score/NMS.
"""

from __future__ import annotations

import math

import torch


def apply_direction_bins(
    boxes3d: torch.Tensor,
    direction_logits: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    dir_offset: float,
    num_bins: int,
) -> torch.Tensor:
    """Correct decoded yaw using the official CoAlign direction convention.

    Parameters
    ----------
    boxes3d:
        Decoded candidates with shape ``[B, H*W*A, 7]``.
    direction_logits:
        Direction head output ``[B, A*num_bins, H, W]``.
    selected_mask:
        Boolean candidate mask with shape ``[B, H*W*A]``; it is used only to
        select direction logits corresponding to boxes that survived score
        thresholding.
    dir_offset, num_bins:
        The postprocessor direction settings.  The public CoAlign recipe uses
        offset ``0.7853`` and two bins.
    """

    if boxes3d.ndim != 3 or boxes3d.shape[-1] != 7:
        raise ValueError("boxes3d must have shape [B, candidates, 7]")
    if direction_logits.ndim != 4 or selected_mask.shape != boxes3d.shape[:2]:
        raise ValueError("direction logits/mask do not align with boxes3d")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    batch, _, height, width = direction_logits.shape
    channels = direction_logits.shape[1]
    if channels % num_bins != 0:
        raise ValueError("direction channels must be divisible by num_bins")
    anchors = channels // num_bins
    expected_candidates = height * width * anchors
    if boxes3d.shape[1] != expected_candidates:
        raise ValueError(
            "direction logits imply a different candidate count: "
            f"{expected_candidates} versus {boxes3d.shape[1]}"
        )
    direction = direction_logits.permute(0, 2, 3, 1).reshape(
        batch, expected_candidates, num_bins
    )
    selected_direction = direction[selected_mask]
    selected_boxes = boxes3d[selected_mask].clone()
    if selected_boxes.numel() == 0:
        return selected_boxes
    labels = selected_direction.argmax(dim=-1).to(selected_boxes.dtype)
    period = 2.0 * math.pi / float(num_bins)
    yaw = selected_boxes[:, 6]
    yaw = torch.remainder(yaw - float(dir_offset), period)
    yaw = yaw + float(dir_offset) + period * labels
    # Match the public OpenCOOD helper's [-pi, pi) normalization convention.
    yaw = torch.remainder(yaw + math.pi, 2.0 * math.pi) - math.pi
    selected_boxes[:, 6] = yaw
    return selected_boxes

