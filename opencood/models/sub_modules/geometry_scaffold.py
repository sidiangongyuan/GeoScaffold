from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from opencood.data_utils.global_bev_footprint import (
    rasterize_global_bev_footprint,
)
from opencood.models.sub_modules.global_bev_footprint import (
    GlobalBEVFootprintHead,
    resolve_global_bev_footprint_config,
)


DEFAULT_ROI_SIZE = 7
DEFAULT_MASK_SIZE = 28
VALID_GEOMETRY_SCAFFOLD_MODES = frozenset({"disabled", "ego", "fused"})
VALID_GEOMETRY_SCAFFOLD_AUXILIARIES = frozenset({"field", "box_code"})
LEGACY_GEOMETRY_SCAFFOLD_MODE_MAP = {
    "ego_geoscaffold": "ego",
    "fused_geoscaffold": "fused",
}


def normalize_geometry_scaffold_mode(mode_value: object) -> str:
    """Normalize canonical and legacy GeoScaffold mode names."""

    mode = str(mode_value).lower()
    mode = LEGACY_GEOMETRY_SCAFFOLD_MODE_MAP.get(mode, mode)
    if mode not in VALID_GEOMETRY_SCAFFOLD_MODES:
        raise ValueError(
            "geometry_scaffold.mode must be one of disabled, ego, or fused; "
            f"got {mode_value!r}."
        )
    return mode


def normalize_geometry_scaffold_auxiliary(auxiliary_value: object) -> str:
    """Normalize the training target used by the shared GeoScaffold trunk."""

    auxiliary = str(auxiliary_value).lower().replace("-", "_")
    aliases = {
        "mask": "field",
        "footprint": "field",
        "native_box_code": "box_code",
        "box_code_7d": "box_code",
        "7d_box_code": "box_code",
    }
    auxiliary = aliases.get(auxiliary, auxiliary)
    if auxiliary not in VALID_GEOMETRY_SCAFFOLD_AUXILIARIES:
        raise ValueError(
            "geometry_scaffold.auxiliary must be one of field or box_code; "
            f"got {auxiliary_value!r}."
        )
    return auxiliary


def resolve_geometry_scaffold_config(
    args: Mapping[str, object],
) -> dict[str, int | str]:
    """Return one canonical GeoScaffold config with Phase-57 fallback."""

    geometry_scaffold = args.get("geometry_scaffold", {})
    if geometry_scaffold is None:
        geometry_scaffold = {}
    if not isinstance(geometry_scaffold, Mapping):
        raise TypeError("geometry_scaffold must be a mapping when provided.")

    counterfactual = args.get("counterfactual_training", {})
    if counterfactual is None:
        counterfactual = {}
    if not isinstance(counterfactual, Mapping):
        raise TypeError("counterfactual_training must be a mapping when provided.")

    mode_value = geometry_scaffold.get("mode")
    if mode_value is None and "enabled" in geometry_scaffold:
        mode_value = "fused" if bool(geometry_scaffold["enabled"]) else "disabled"
    if mode_value is None:
        legacy_mode = counterfactual.get("mode", "disabled")
        # Other counterfactual plugins share this argument namespace.  They
        # must not accidentally be interpreted as GeoScaffold modes.
        mode_value = (
            legacy_mode
            if str(legacy_mode).lower() in {
                "disabled",
                "ego",
                "fused",
                "ego_geoscaffold",
                "fused_geoscaffold",
            }
            else "disabled"
        )

    config = {
        "mode": normalize_geometry_scaffold_mode(mode_value),
        "roi_size": int(geometry_scaffold.get("roi_size", DEFAULT_ROI_SIZE)),
        "mask_size": int(geometry_scaffold.get("mask_size", DEFAULT_MASK_SIZE)),
        "hidden_channels": int(geometry_scaffold.get("hidden_channels", 128)),
        "checkpoint_chunk_size": int(
            geometry_scaffold.get("checkpoint_chunk_size", 0)
        ),
        "auxiliary": normalize_geometry_scaffold_auxiliary(
            geometry_scaffold.get(
                "auxiliary",
                geometry_scaffold.get("target", "field"),
            )
        ),
    }
    if config["roi_size"] <= 0 or config["mask_size"] <= 0:
        raise ValueError("GeoScaffold roi_size and mask_size must be positive.")
    if config["hidden_channels"] <= 0:
        raise ValueError("GeoScaffold hidden_channels must be positive.")
    return config


def extract_ego_bev_features(
    packed_bev: torch.Tensor,
    record_len: torch.Tensor,
) -> torch.Tensor:
    """Select the ego feature for every coalition from packed BEV maps."""

    if packed_bev.ndim != 4:
        raise ValueError("packed_bev must have shape [sum(record_len), C, H, W].")
    if record_len.ndim != 1:
        raise ValueError(
            f"record_len must be one-dimensional, got {tuple(record_len.shape)}."
        )
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Every coalition must contain at least one feature map.")
    if sum(lengths) != packed_bev.shape[0]:
        raise ValueError(
            f"sum(record_len)={sum(lengths)} does not match packed BEV count "
            f"{packed_bev.shape[0]}."
        )
    starts = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length
    indices = torch.tensor(starts, device=packed_bev.device, dtype=torch.long)
    return packed_bev.index_select(0, indices)


def collect_positive_scaffold_targets(
    label_dict: dict[str, torch.Tensor],
    all_anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect official positive anchors and decode their assigned GT boxes."""

    labels = torch.as_tensor(label_dict["pos_equal_one"])
    targets = torch.as_tensor(label_dict["targets"])
    if labels.ndim != 4 or targets.ndim != 5:
        raise ValueError("Expected [B, H, W, A] labels and [B, H, W, A, 7] targets.")
    if labels.shape[:4] != targets.shape[:4]:
        raise ValueError("pos_equal_one and targets must be anchor-aligned.")
    flat_labels = labels.to(dtype=torch.int64).reshape(labels.shape[0], -1)
    flat_targets = targets.to(dtype=torch.float32).reshape(targets.shape[0], -1, 7)
    positive_locations = (flat_labels > 0).nonzero(as_tuple=False)
    if positive_locations.numel() == 0:
        empty_long = torch.empty((0,), device=flat_labels.device, dtype=torch.long)
        empty_float = torch.empty((0, 7), device=flat_targets.device, dtype=flat_targets.dtype)
        return empty_long, empty_long, empty_float, empty_float
    sample_indices = positive_locations[:, 0].to(dtype=torch.long)
    anchor_indices = positive_locations[:, 1].to(dtype=torch.long)
    class_ids = flat_labels[sample_indices, anchor_indices].to(dtype=torch.long)
    flat_anchors = merge_anchor_sets(all_anchors).to(
        device=flat_targets.device,
        dtype=flat_targets.dtype,
    ).reshape(-1, 7)
    anchor_boxes = flat_anchors.index_select(0, anchor_indices)
    gt_boxes = decode_box_targets(
        flat_targets[sample_indices, anchor_indices],
        anchor_boxes,
    )
    return sample_indices, class_ids, anchor_boxes, gt_boxes


def merge_anchor_sets(all_anchors: torch.Tensor) -> torch.Tensor:
    """Merge per-class anchor sets into the flattened label order."""

    if all_anchors.ndim == 4 and all_anchors.shape[-1] == 7:
        return all_anchors.contiguous()
    if all_anchors.ndim != 5 or all_anchors.shape[-1] != 7:
        raise ValueError(
            "all_anchors must have shape [classes, H, W, anchors, 7] or "
            "[H, W, anchors, 7]."
        )
    return all_anchors.permute(1, 2, 0, 3, 4).reshape(
        all_anchors.shape[1],
        all_anchors.shape[2],
        -1,
        7,
    ).contiguous()


def decode_box_targets(
    target_codes: torch.Tensor,
    anchor_boxes: torch.Tensor,
) -> torch.Tensor:
    """Invert the native raw-anchor box coding at positive anchors."""

    if target_codes.shape != anchor_boxes.shape or target_codes.shape[-1] != 7:
        raise ValueError("target_codes and anchor_boxes must share shape [N, 7].")
    anchors_d = torch.sqrt(anchor_boxes[:, 4].square() + anchor_boxes[:, 5].square())
    decoded = torch.empty_like(target_codes)
    decoded[:, 0] = target_codes[:, 0] * anchors_d + anchor_boxes[:, 0]
    decoded[:, 1] = target_codes[:, 1] * anchors_d + anchor_boxes[:, 1]
    decoded[:, 2] = target_codes[:, 2] * anchor_boxes[:, 3] + anchor_boxes[:, 2]
    decoded[:, 3] = torch.exp(target_codes[:, 3]) * anchor_boxes[:, 3]
    decoded[:, 4] = torch.exp(target_codes[:, 4]) * anchor_boxes[:, 4]
    decoded[:, 5] = torch.exp(target_codes[:, 5]) * anchor_boxes[:, 5]
    decoded[:, 6] = target_codes[:, 6] + anchor_boxes[:, 6]
    return decoded


def rasterize_anchor_gt_intersections(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    mask_size: int = DEFAULT_MASK_SIZE,
) -> torch.Tensor:
    """Rasterize raw-anchor cell centers that lie inside the assigned GT box."""

    if anchor_boxes.shape != gt_boxes.shape or anchor_boxes.shape[-1] != 7:
        raise ValueError("anchor_boxes and gt_boxes must share shape [N, 7].")
    if mask_size <= 0:
        raise ValueError("mask_size must be positive.")
    if anchor_boxes.numel() == 0:
        return torch.empty(
            (0, mask_size, mask_size),
            device=anchor_boxes.device,
            dtype=torch.bool,
        )

    dtype = anchor_boxes.dtype
    device = anchor_boxes.device
    axis = (
        torch.arange(mask_size, device=device, dtype=dtype) + 0.5
    ) / float(mask_size) - 0.5
    local_v, local_u = torch.meshgrid(axis, axis, indexing="ij")
    local_u = local_u.unsqueeze(0) * anchor_boxes[:, 5].view(-1, 1, 1)
    local_v = local_v.unsqueeze(0) * anchor_boxes[:, 4].view(-1, 1, 1)

    delta_x = gt_boxes[:, 0] - anchor_boxes[:, 0]
    delta_y = gt_boxes[:, 1] - anchor_boxes[:, 1]
    anchor_cos = torch.cos(anchor_boxes[:, 6])
    anchor_sin = torch.sin(anchor_boxes[:, 6])
    gt_center_u = anchor_cos.view(-1, 1, 1) * delta_x.view(-1, 1, 1) + \
        anchor_sin.view(-1, 1, 1) * delta_y.view(-1, 1, 1)
    gt_center_v = -anchor_sin.view(-1, 1, 1) * delta_x.view(-1, 1, 1) + \
        anchor_cos.view(-1, 1, 1) * delta_y.view(-1, 1, 1)

    relative_u = local_u - gt_center_u
    relative_v = local_v - gt_center_v
    relative_yaw = gt_boxes[:, 6] - anchor_boxes[:, 6]
    rel_cos = torch.cos(relative_yaw).view(-1, 1, 1)
    rel_sin = torch.sin(relative_yaw).view(-1, 1, 1)
    gt_u = rel_cos * relative_u + rel_sin * relative_v
    gt_v = -rel_sin * relative_u + rel_cos * relative_v
    half_length = 0.5 * gt_boxes[:, 5].abs().clamp_min(1e-6).view(-1, 1, 1)
    half_width = 0.5 * gt_boxes[:, 4].abs().clamp_min(1e-6).view(-1, 1, 1)
    return (gt_u.abs() < half_length) & (gt_v.abs() < half_width)


def anchor_aligned_grid(
    anchor_boxes: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    x_min: float,
    y_min: float,
    cell_x: float,
    cell_y: float,
    roi_size: int = DEFAULT_ROI_SIZE,
) -> torch.Tensor:
    """Build one anchor-aligned sampling grid for native ``grid_sample``."""

    if anchor_boxes.ndim != 2 or anchor_boxes.shape[1] != 7:
        raise ValueError("anchor_boxes must have shape [N, 7].")
    if min(feature_height, feature_width, roi_size) <= 1:
        raise ValueError("Feature and ROI dimensions must exceed one.")
    if cell_x <= 0.0 or cell_y <= 0.0:
        raise ValueError("Feature cell sizes must be positive.")
    if anchor_boxes.numel() == 0:
        return anchor_boxes.new_empty((0, roi_size, roi_size, 2))

    axis = (
        torch.arange(roi_size, device=anchor_boxes.device, dtype=anchor_boxes.dtype) + 0.5
    ) / float(roi_size) - 0.5
    local_v, local_u = torch.meshgrid(axis, axis, indexing="ij")
    local_u = local_u.unsqueeze(0) * anchor_boxes[:, 5].view(-1, 1, 1)
    local_v = local_v.unsqueeze(0) * anchor_boxes[:, 4].view(-1, 1, 1)
    cosine = torch.cos(anchor_boxes[:, 6]).view(-1, 1, 1)
    sine = torch.sin(anchor_boxes[:, 6]).view(-1, 1, 1)
    global_x = anchor_boxes[:, 0].view(-1, 1, 1) + cosine * local_u - sine * local_v
    global_y = anchor_boxes[:, 1].view(-1, 1, 1) + sine * local_u + cosine * local_v
    # Feature index zero represents the first BEV cell center, not the range
    # boundary.  Account for that half-cell offset before normalization.
    index_x = (global_x - float(x_min)) / float(cell_x) - 0.5
    index_y = (global_y - float(y_min)) / float(cell_y) - 0.5
    normalized_x = 2.0 * index_x / float(feature_width - 1) - 1.0
    normalized_y = 2.0 * index_y / float(feature_height - 1) - 1.0
    return torch.stack((normalized_x, normalized_y), dim=-1)


def sample_anchor_rois(
    source_bev: torch.Tensor,
    anchor_boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    x_min: float,
    y_min: float,
    cell_x: float,
    cell_y: float,
    roi_size: int = DEFAULT_ROI_SIZE,
) -> torch.Tensor:
    """Sample one canonical anchor-aligned ROI from a batch of BEV maps."""

    if source_bev.ndim != 4:
        raise ValueError("source_bev must have shape [batch, channels, H, W].")
    if batch_indices.ndim != 1 or batch_indices.dtype != torch.long:
        raise ValueError("batch_indices must be a one-dimensional torch.long tensor.")
    if anchor_boxes.shape != (batch_indices.numel(), 7):
        raise ValueError("Every anchor must have one matching batch index.")
    if anchor_boxes.numel() == 0:
        return source_bev.new_empty((0, source_bev.shape[1], roi_size, roi_size))
    if int(batch_indices.min()) < 0 or int(batch_indices.max()) >= source_bev.shape[0]:
        raise ValueError("A batch index lies outside the source BEV tensor.")

    grid = anchor_aligned_grid(
        anchor_boxes,
        feature_height=int(source_bev.shape[2]),
        feature_width=int(source_bev.shape[3]),
        x_min=x_min,
        y_min=y_min,
        cell_x=cell_x,
        cell_y=cell_y,
        roi_size=roi_size,
    )
    roi_chunks = []
    position_chunks = []
    for batch_index in torch.unique(batch_indices, sorted=True).tolist():
        positions = torch.nonzero(
            batch_indices == int(batch_index),
            as_tuple=False,
        ).flatten()
        sample_grid = grid.index_select(0, positions)
        # Stack all output ROIs for one sample into grid_sample's output height.
        # The source BEV is therefore read once rather than copied per anchor.
        stacked_grid = sample_grid.reshape(
            1,
            positions.numel() * roi_size,
            roi_size,
            2,
        )
        sampled = F.grid_sample(
            source_bev[int(batch_index):int(batch_index) + 1],
            stacked_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(
            source_bev.shape[1],
            positions.numel(),
            roi_size,
            roi_size,
        ).permute(1, 0, 2, 3)
        roi_chunks.append(sampled)
        position_chunks.append(positions)
    concatenated = torch.cat(roi_chunks, dim=0)
    concatenated_positions = torch.cat(position_chunks, dim=0)
    return concatenated.index_select(
        0,
        torch.argsort(concatenated_positions),
    )


class GeometryScaffoldHead(nn.Module):
    """Predict an anchor-local field or a native seven-value box code.

    Both variants use the same ROI stem, class embedding, and convolutional
    trunk.  The box-code variant changes only the prediction target and keeps
    the feature representation matched to the field auxiliary.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        roi_size: int = DEFAULT_ROI_SIZE,
        mask_size: int = DEFAULT_MASK_SIZE,
        checkpoint_chunk_size: int = 0,
        auxiliary: str = "field",
    ) -> None:
        super().__init__()
        if min(in_channels, num_classes, hidden_channels, roi_size, mask_size) <= 0:
            raise ValueError("GeoScaffold dimensions must be positive.")
        if roi_size != DEFAULT_ROI_SIZE:
            raise ValueError("GeoScaffold requires a 7x7 canonical ROI.")
        self.auxiliary = normalize_geometry_scaffold_auxiliary(auxiliary)
        self.roi_size = int(roi_size)
        self.mask_size = int(mask_size)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.checkpoint_chunk_size = int(checkpoint_chunk_size)
        if self.checkpoint_chunk_size < 0:
            raise ValueError("checkpoint_chunk_size must be non-negative.")
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)
        self.body = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.upsample = None
        self.mask_logits = None
        self.box_code_head = None
        if self.auxiliary == "field":
            # Keep the decoder capacity and checkpoint schema fixed across the
            # resolution ablations. The logits are resized after this native
            # 28x28 decoder in ``_forward_impl`` when a different mask is used.
            self.upsample = self._build_upsample(hidden_channels)
            self.mask_logits = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
            nn.init.normal_(self.mask_logits.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.mask_logits.bias)
        else:
            # This decoder keeps the box-code control close in capacity to the
            # field decoder while sharing its ROI/trunk/class representation.
            self.box_code_head = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels // 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels // 2, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 7),
            )

    def _build_upsample(self, hidden_channels: int) -> nn.Sequential:
        """Build the fixed-capacity decoder calibrated for the default mask."""

        if DEFAULT_MASK_SIZE != self.roi_size * 4:
            raise RuntimeError(
                "GeoScaffold default decoder expects a 4x ROI upsampling ratio."
            )
        layers: list[nn.Module] = []
        layers.extend(
            (
                nn.GELU(),
                nn.ConvTranspose2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=2,
                    stride=2,
                ),
                nn.GELU(),
                nn.ConvTranspose2d(
                    hidden_channels,
                    hidden_channels // 2,
                    kernel_size=2,
                    stride=2,
                ),
            )
        )
        layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def _forward_features(
        self,
        rois: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.stem(rois)
        hidden = hidden + self.class_embedding(class_ids - 1).view(
            class_ids.numel(),
            -1,
            1,
            1,
        )
        return self.body(hidden)

    def _forward_impl(
        self,
        rois: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.auxiliary != "field" or self.upsample is None or self.mask_logits is None:
            raise RuntimeError("Field logits requested from a box-code GeoScaffold head.")
        logits = self.mask_logits(self.upsample(self._forward_features(rois, class_ids)))
        if logits.shape[-2:] != (self.mask_size, self.mask_size):
            logits = F.interpolate(
                logits,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def _validate_inputs(self, rois: torch.Tensor, class_ids: torch.Tensor) -> None:
        if rois.ndim != 4:
            raise ValueError("rois must have shape [N, channels, roi, roi].")
        if class_ids.ndim != 1 or class_ids.dtype != torch.long:
            raise ValueError("class_ids must be a one-dimensional torch.long tensor.")
        if rois.shape[0] != class_ids.numel():
            raise ValueError("Every ROI must have one class id.")
        if rois.shape[2:] != (self.roi_size, self.roi_size):
            raise ValueError("ROI size does not match the configured GeoScaffold head.")
        if rois.shape[0] and (
            int(class_ids.min()) < 1 or int(class_ids.max()) > self.num_classes
        ):
            raise ValueError("Class ids must follow the repository's one-indexed convention.")

    def forward(self, rois: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(rois, class_ids)
        if self.auxiliary != "field":
            raise RuntimeError("Use forward_box_code for a box-code GeoScaffold head.")
        if rois.shape[0] == 0:
            return rois.new_empty((0, 2, self.mask_size, self.mask_size))

        chunk_size = self.checkpoint_chunk_size
        use_checkpoint = (
            self.training
            and torch.is_grad_enabled()
            and rois.requires_grad
            and chunk_size > 0
            and rois.shape[0] > chunk_size
        )
        if use_checkpoint:
            chunks = []
            for start in range(0, int(rois.shape[0]), chunk_size):
                stop = min(start + chunk_size, int(rois.shape[0]))
                chunks.append(
                    checkpoint(
                        self._forward_impl,
                        rois[start:stop],
                        class_ids[start:stop],
                    )
                )
            logits = torch.cat(chunks, dim=0)
        else:
            logits = self._forward_impl(rois, class_ids)
        if logits.shape != (rois.shape[0], 2, self.mask_size, self.mask_size):
            raise RuntimeError("GeoScaffold head violated its configured field shape.")
        return logits

    def forward_box_code(
        self,
        rois: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict native anchor-to-GT box codes for positive-anchor ROIs."""

        self._validate_inputs(rois, class_ids)
        if self.auxiliary != "box_code" or self.box_code_head is None:
            raise RuntimeError("This GeoScaffold head is configured for fields.")
        if rois.shape[0] == 0:
            return rois.new_empty((0, 7))
        return self.box_code_head(self._forward_features(rois, class_ids))


def field_ce_dice_loss(
    field_logits: torch.Tensor,
    field_targets: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Positive-anchor normalized cellwise CE and/or Dice."""

    if not math.isfinite(float(ce_weight)) or not math.isfinite(float(dice_weight)):
        raise ValueError("GeoScaffold CE/Dice weights must be finite.")
    if ce_weight < 0.0 or dice_weight < 0.0:
        raise ValueError("GeoScaffold CE/Dice weights must be non-negative.")

    if field_logits.ndim != 4 or field_logits.shape[1] != 2:
        raise ValueError("field_logits must have shape [N, 2, H, W].")
    if field_targets.shape != (
        field_logits.shape[0],
        field_logits.shape[2],
        field_logits.shape[3],
    ):
        raise ValueError("field_targets must align with field_logits.")
    if field_logits.shape[0] == 0:
        return field_logits.sum()

    targets = field_targets.to(device=field_logits.device, dtype=torch.bool)
    target_class = targets.to(dtype=torch.long)
    loss = field_logits.new_zeros((field_logits.shape[0],))
    if ce_weight != 0.0:
        ce = F.cross_entropy(
            field_logits,
            target_class,
            reduction="none",
        ).flatten(1).mean(dim=1)
        loss = loss + float(ce_weight) * ce
    if dice_weight != 0.0:
        occupied = torch.softmax(field_logits, dim=1)[:, 1]
        target_float = targets.to(dtype=field_logits.dtype)
        intersection = (occupied * target_float).flatten(1).sum(dim=1)
        denominator = occupied.flatten(1).sum(dim=1) + target_float.flatten(1).sum(dim=1)
        epsilon = torch.finfo(field_logits.dtype).eps
        dice = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
        nonempty = targets.flatten(1).any(dim=1)
        loss = loss + float(dice_weight) * torch.where(
            nonempty,
            dice,
            torch.zeros_like(dice),
        )
    return loss.mean()


def native_box_code_smooth_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 1.0 / 9.0,
) -> torch.Tensor:
    """Native box-code SmoothL1 with the repository's sine yaw convention."""

    if (
        predictions.shape != targets.shape
        or predictions.ndim != 2
        or predictions.shape[1] != 7
    ):
        raise ValueError("Box-code predictions and targets must have shape [N, 7].")
    if beta <= 0.0 or not math.isfinite(float(beta)):
        raise ValueError("SmoothL1 beta must be positive and finite.")
    if predictions.shape[0] == 0:
        return predictions.sum()
    pred_yaw = torch.sin(predictions[:, 6:7]) * torch.cos(targets[:, 6:7])
    target_yaw = torch.cos(predictions[:, 6:7]) * torch.sin(targets[:, 6:7])
    pred = torch.cat((predictions[:, :6], pred_yaw), dim=1)
    target = torch.cat((targets[:, :6], target_yaw), dim=1)
    return F.smooth_l1_loss(pred, target, beta=float(beta), reduction="mean")


class GeometryScaffoldMixin:
    """Training-only geometry auxiliaries shared by BEV detectors.

    GeoScaffold remains the anchor-canonical sparse auxiliary.  The optional
    global footprint control is initialized and emitted independently, so it
    cannot change the native detector's decoding path or the GeoScaffold
    target representation.
    """

    geometry_scaffold_mode: str
    geometry_scaffold_auxiliary: str
    geometry_scaffold_roi_size: int
    geometry_scaffold_mask_size: int
    geometry_scaffold_checkpoint_chunk_size: int
    geometry_scaffold_lidar_range: tuple[float, float, float, float, float, float]
    geometry_scaffold_head: GeometryScaffoldHead | None
    global_bev_footprint_head: GlobalBEVFootprintHead | None
    global_bev_footprint_lidar_range: tuple[float, float, float, float, float, float]

    def init_geometry_scaffold(
        self,
        args: Mapping[str, object],
        *,
        in_channels: int,
        num_classes: int,
    ) -> None:
        config = resolve_geometry_scaffold_config(args)
        self.geometry_scaffold_mode = str(config["mode"])
        self.geometry_scaffold_auxiliary = str(config["auxiliary"])
        self.geometry_scaffold_roi_size = int(config["roi_size"])
        self.geometry_scaffold_mask_size = int(config["mask_size"])
        self.geometry_scaffold_checkpoint_chunk_size = int(
            config["checkpoint_chunk_size"]
        )
        self.geometry_scaffold_lidar_range = tuple(
            float(value) for value in args["lidar_range"]
        )
        self.geometry_scaffold_head = None
        if self.geometry_scaffold_mode != "disabled":
            self.geometry_scaffold_head = GeometryScaffoldHead(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_channels=int(config["hidden_channels"]),
                roi_size=self.geometry_scaffold_roi_size,
                mask_size=self.geometry_scaffold_mask_size,
                checkpoint_chunk_size=self.geometry_scaffold_checkpoint_chunk_size,
                auxiliary=self.geometry_scaffold_auxiliary,
            )

        global_config = resolve_global_bev_footprint_config(args)
        self.global_bev_footprint_lidar_range = self.geometry_scaffold_lidar_range
        self.global_bev_footprint_head = None
        if bool(global_config["enabled"]):
            self.global_bev_footprint_head = GlobalBEVFootprintHead(
                in_channels=in_channels,
                hidden_channels=int(global_config["hidden_channels"]),
            )

    def build_geometry_scaffold_output(
        self,
        packed_bev: torch.Tensor,
        fused_feature: torch.Tensor,
        record_len: torch.Tensor,
        data_dict: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        if not self.training:
            return {}

        outputs: dict[str, torch.Tensor] = {}
        if self.geometry_scaffold_head is not None:
            outputs.update(
                self._build_geometry_scaffold_output(
                    packed_bev,
                    fused_feature,
                    record_len,
                    data_dict,
                )
            )
        if self.global_bev_footprint_head is not None:
            outputs.update(
                self.build_global_bev_footprint_output(
                    fused_feature,
                    data_dict,
                )
            )
        return outputs

    def _build_geometry_scaffold_output(
        self,
        packed_bev: torch.Tensor,
        fused_feature: torch.Tensor,
        record_len: torch.Tensor,
        data_dict: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        if self.geometry_scaffold_head is None:
            return {}
        if "geoscaffold" not in data_dict:
            raise KeyError(
                "GeoScaffold training requires sparse positive-anchor targets."
            )

        if self.geometry_scaffold_mode == "fused":
            source_bev = fused_feature
        elif self.geometry_scaffold_mode == "ego":
            source_bev = extract_ego_bev_features(packed_bev, record_len)
        else:
            raise RuntimeError(
                "build_geometry_scaffold_output called while GeoScaffold is disabled."
            )

        sparse_targets = data_dict["geoscaffold"]
        if not isinstance(sparse_targets, Mapping):
            raise TypeError("geoscaffold targets must be a mapping.")
        sample_indices = sparse_targets["batch_indices"].to(
            device=source_bev.device,
            dtype=torch.long,
        )
        class_ids = sparse_targets["class_ids"].to(
            device=source_bev.device,
            dtype=torch.long,
        )
        anchor_boxes = sparse_targets["anchor_boxes_hwl"].to(
            device=source_bev.device,
            dtype=source_bev.dtype,
        )
        targets = sparse_targets["field_targets"].to(
            device=source_bev.device,
            dtype=torch.bool,
        )
        box_code_targets = sparse_targets.get("box_code_targets")
        if box_code_targets is not None:
            box_code_targets = box_code_targets.to(
                device=source_bev.device,
                dtype=source_bev.dtype,
            )
        if sample_indices.numel() == 0:
            if self.geometry_scaffold_auxiliary == "box_code":
                return {
                    "geoscaffold_box_code_preds": source_bev.new_empty((0, 7)),
                    "geoscaffold_box_code_targets": source_bev.new_empty((0, 7)),
                    "geoscaffold_positive_count": torch.zeros(
                        (),
                        device=source_bev.device,
                        dtype=torch.int64,
                    ),
                }
            return {
                "geoscaffold_logits": source_bev.new_empty(
                    (0, 2, self.geometry_scaffold_mask_size, self.geometry_scaffold_mask_size)
                ),
                "geoscaffold_targets": torch.empty(
                    (0, self.geometry_scaffold_mask_size, self.geometry_scaffold_mask_size),
                    device=source_bev.device,
                    dtype=torch.bool,
                ),
                "geoscaffold_positive_count": torch.zeros(
                    (),
                    device=source_bev.device,
                    dtype=torch.int64,
                ),
            }

        lidar_range = self.geometry_scaffold_lidar_range
        rois = sample_anchor_rois(
            source_bev,
            anchor_boxes,
            sample_indices,
            x_min=lidar_range[0],
            y_min=lidar_range[1],
            cell_x=(lidar_range[3] - lidar_range[0]) / float(source_bev.shape[3]),
            cell_y=(lidar_range[4] - lidar_range[1]) / float(source_bev.shape[2]),
            roi_size=self.geometry_scaffold_roi_size,
        )
        if self.geometry_scaffold_auxiliary == "box_code":
            if box_code_targets is None:
                raise KeyError(
                    "GeoScaffold box-code auxiliary requires native box-code targets."
                )
            predictions = self.geometry_scaffold_head.forward_box_code(
                rois,
                class_ids,
            )
            return {
                "geoscaffold_box_code_preds": predictions,
                "geoscaffold_box_code_targets": box_code_targets,
                "geoscaffold_positive_count": torch.tensor(
                    sample_indices.numel(),
                    device=source_bev.device,
                    dtype=torch.int64,
                ),
            }
        return {
            "geoscaffold_logits": self.geometry_scaffold_head(rois, class_ids),
            "geoscaffold_targets": targets,
            "geoscaffold_positive_count": torch.tensor(
                sample_indices.numel(),
                device=source_bev.device,
                dtype=torch.int64,
            ),
        }

    def build_global_bev_footprint_output(
        self,
        fused_feature: torch.Tensor,
        data_dict: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Build the independent dense global-footprint auxiliary output.

        ``object_bbx_center`` and ``object_bbx_mask`` are the receiver-frame
        boxes already filtered by the intermediate-fusion dataset.  The model
        only rasterizes those existing labels at training time; evaluation
        returns before this method is called.
        """

        if self.global_bev_footprint_head is None:
            return {}
        target_data = data_dict.get("global_bev_footprint_target")
        if target_data is not None:
            if not isinstance(target_data, torch.Tensor):
                raise TypeError("Global BEV footprint targets must be torch tensors.")
            expected_shape = (
                fused_feature.shape[0],
                fused_feature.shape[-2],
                fused_feature.shape[-1],
            )
            if tuple(target_data.shape) != expected_shape:
                raise ValueError(
                    "Global BEV footprint targets must align with fused BEV: "
                    f"{tuple(target_data.shape)} versus {expected_shape}."
                )
            targets = target_data.to(
                device=fused_feature.device,
                dtype=torch.bool,
            )
        else:
            # Keep direct model-level smoke tests useful when a caller supplies
            # the existing filtered boxes without running the dataset collator.
            if "object_bbx_center" not in data_dict or "object_bbx_mask" not in data_dict:
                raise KeyError(
                    "Global BEV footprint training requires a collated target or "
                    "filtered object boxes and their validity mask."
                )
            boxes = data_dict["object_bbx_center"]
            box_mask = data_dict["object_bbx_mask"]
            if not isinstance(boxes, torch.Tensor) or not isinstance(box_mask, torch.Tensor):
                raise TypeError("Global BEV footprint boxes and mask must be torch tensors.")
            if boxes.shape[0] != fused_feature.shape[0]:
                raise ValueError(
                    "Global BEV footprint boxes must align with fused feature batch: "
                    f"{boxes.shape[0]} versus {fused_feature.shape[0]}."
                )
            targets = rasterize_global_bev_footprint(
                boxes.to(device=fused_feature.device),
                box_mask.to(device=fused_feature.device),
                height=int(fused_feature.shape[-2]),
                width=int(fused_feature.shape[-1]),
                lidar_range=self.global_bev_footprint_lidar_range,
            )
        logits = self.global_bev_footprint_head(fused_feature)
        return {
            "global_bev_footprint_logits": logits,
            "global_bev_footprint_targets": targets,
            "global_bev_footprint_positive_count": targets.sum(
                dtype=torch.int64,
            ),
        }
