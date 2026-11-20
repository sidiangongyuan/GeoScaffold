"""Dense, class-agnostic BEV footprint targets for a training-only control.

The target is rasterized directly from the already filtered receiver-frame GT
boxes carried by the intermediate-fusion dataset.  It deliberately has no
anchor, proposal, or detector-output dependency; the only discretization is
the native fused BEV grid on which the auxiliary head is evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def rasterize_global_bev_footprint(
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    height: int,
    width: int,
    lidar_range: Sequence[float],
) -> torch.Tensor:
    """Rasterize the union of valid receiver-frame GT footprints.

    Parameters
    ----------
    gt_boxes:
        Receiver-frame boxes with shape ``[B, N, 8]`` or ``[B, N, 7]``.
        The first seven values use the repository's ``hwl`` convention:
        ``[x, y, z, h, w, l, yaw]``.  An optional eighth class value is
        ignored because this control is class agnostic.
    gt_mask:
        Native validity mask with shape ``[B, N]``.  The intermediate-fusion
        dataset supplies this mask after the same range and augmentation
        filtering used by the detector labels.
    height, width:
        Spatial size of the fused BEV feature map.
    lidar_range:
        ``[x_min, y_min, z_min, x_max, y_max, z_max]`` for the receiver frame.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[B, height, width]``.  A cell is positive
        when its center lies inside at least one valid oriented footprint.
        Strict inequalities match the existing anchor-local rasterizer's
        boundary convention.
    """

    if not isinstance(gt_boxes, torch.Tensor) or not isinstance(gt_mask, torch.Tensor):
        raise TypeError("gt_boxes and gt_mask must be torch.Tensor instances.")
    if gt_boxes.ndim != 3 or gt_boxes.shape[-1] not in (7, 8):
        raise ValueError("gt_boxes must have shape [B, N, 7] or [B, N, 8].")
    if gt_mask.ndim != 2 or gt_mask.shape[:2] != gt_boxes.shape[:2]:
        raise ValueError("gt_mask must align with gt_boxes as [B, N].")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    if len(lidar_range) != 6:
        raise ValueError("lidar_range must contain six values.")

    x_min, y_min, _, x_max, y_max, _ = (float(value) for value in lidar_range)
    if not x_max > x_min or not y_max > y_min:
        raise ValueError("BEV lidar range must have positive x and y extents.")

    boxes = gt_boxes[..., :7]
    valid = gt_mask.to(device=boxes.device, dtype=torch.bool)
    if valid.any():
        valid_boxes = boxes[valid]
        if not torch.isfinite(valid_boxes).all():
            raise ValueError("Valid GT boxes must be finite.")
        if bool((valid_boxes[:, 3:6] <= 0.0).any()):
            raise ValueError("Valid GT boxes must have positive h, w, and l.")

    # The first feature-map column/row represents the center of the first BEV
    # cell, matching the convention used by anchor_aligned_grid.
    dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
    device = boxes.device
    x_axis = (
        torch.arange(width, device=device, dtype=dtype) + 0.5
    ) * ((x_max - x_min) / float(width)) + x_min
    y_axis = (
        torch.arange(height, device=device, dtype=dtype) + 0.5
    ) * ((y_max - y_min) / float(height)) + y_min
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")

    if boxes.shape[1] == 0 or not bool(valid.any()):
        return torch.zeros(
            (boxes.shape[0], height, width),
            device=device,
            dtype=torch.bool,
        )

    center_x = boxes[..., 0].unsqueeze(-1).unsqueeze(-1)
    center_y = boxes[..., 1].unsqueeze(-1).unsqueeze(-1)
    delta_x = grid_x.view(1, 1, height, width) - center_x
    delta_y = grid_y.view(1, 1, height, width) - center_y
    yaw = boxes[..., 6].unsqueeze(-1).unsqueeze(-1)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    local_u = cosine * delta_x + sine * delta_y
    local_v = -sine * delta_x + cosine * delta_y
    inside = (
        (local_u.abs() < 0.5 * boxes[..., 5].abs().unsqueeze(-1).unsqueeze(-1))
        & (local_v.abs() < 0.5 * boxes[..., 4].abs().unsqueeze(-1).unsqueeze(-1))
    )
    return (inside & valid.unsqueeze(-1).unsqueeze(-1)).any(dim=1)

