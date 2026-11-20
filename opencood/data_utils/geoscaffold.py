"""Sparse training targets for GeoScaffold.

The native label generator already records the exact GT-relative box code for
every official positive anchor.  This module decodes only those positives and
rasterizes anchor-local GT support without carrying the static dense anchor
grid through the training loader.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


MASK_SIZE = 28


def _flatten_anchors(
    all_anchors: Sequence[np.ndarray] | np.ndarray,
    label_shape: tuple[int, int, int],
) -> np.ndarray:
    anchors = np.asarray(all_anchors)
    if anchors.ndim != 5 or anchors.shape[-1] != 7:
        raise ValueError(
            "all_anchors must have shape [classes, H, W, anchors, 7]."
        )
    # Label generation concatenates class-specific anchor rotations along its
    # final axis.  Mirror that order exactly: H, W, class, rotation.
    flattened = anchors.transpose(1, 2, 0, 3, 4).reshape(*label_shape, 7)
    if flattened.shape[:3] != label_shape:
        raise RuntimeError("The native anchor grid does not align with labels.")
    return np.ascontiguousarray(flattened, dtype=np.float32)


def _decode_positive_targets(
    anchors_hwl: np.ndarray,
    encoded_targets: np.ndarray,
) -> np.ndarray:
    if anchors_hwl.shape != encoded_targets.shape or anchors_hwl.shape[-1] != 7:
        raise ValueError("Positive anchors and regression targets must be [N, 7].")
    anchors = anchors_hwl.astype(np.float64, copy=False)
    targets = encoded_targets.astype(np.float64, copy=False)
    diagonal = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)
    decoded = np.empty_like(anchors)
    decoded[:, 0] = targets[:, 0] * diagonal + anchors[:, 0]
    decoded[:, 1] = targets[:, 1] * diagonal + anchors[:, 1]
    decoded[:, 2] = targets[:, 2] * anchors[:, 3] + anchors[:, 2]
    decoded[:, 3:6] = np.exp(targets[:, 3:6]) * anchors[:, 3:6]
    decoded[:, 6] = targets[:, 6] + anchors[:, 6]
    if not np.isfinite(decoded).all() or np.any(decoded[:, 3:6] <= 0.0):
        raise RuntimeError("Decoded official positive targets are invalid.")
    return decoded


def _binary_intersection_fields(
    anchors_hwl: np.ndarray,
    gt_boxes_hwl: np.ndarray,
    *,
    mask_size: int,
) -> np.ndarray:
    if mask_size <= 0:
        raise ValueError("mask_size must be positive.")
    count = int(anchors_hwl.shape[0])
    if count == 0:
        return np.zeros((0, mask_size, mask_size), dtype=np.bool_)

    axis = (
        (np.arange(mask_size, dtype=np.float64) + 0.5) / float(mask_size)
        - 0.5
    )
    local_v_unit, local_u_unit = np.meshgrid(axis, axis, indexing="ij")
    local_u = local_u_unit[None] * anchors_hwl[:, None, None, 5]
    local_v = local_v_unit[None] * anchors_hwl[:, None, None, 4]

    anchor_cos = np.cos(anchors_hwl[:, 6])[:, None, None]
    anchor_sin = np.sin(anchors_hwl[:, 6])[:, None, None]
    global_x = (
        anchors_hwl[:, None, None, 0]
        + anchor_cos * local_u
        - anchor_sin * local_v
    )
    global_y = (
        anchors_hwl[:, None, None, 1]
        + anchor_sin * local_u
        + anchor_cos * local_v
    )

    delta_x = global_x - gt_boxes_hwl[:, None, None, 0]
    delta_y = global_y - gt_boxes_hwl[:, None, None, 1]
    gt_cos = np.cos(gt_boxes_hwl[:, 6])[:, None, None]
    gt_sin = np.sin(gt_boxes_hwl[:, 6])[:, None, None]
    gt_u = gt_cos * delta_x + gt_sin * delta_y
    gt_v = -gt_sin * delta_x + gt_cos * delta_y
    # Strict inequalities match shapely.contains_xy used by the accepted
    # Phase-54 cell-center target, including its boundary convention.
    return np.ascontiguousarray(
        (np.abs(gt_u) < 0.5 * gt_boxes_hwl[:, None, None, 5])
        & (np.abs(gt_v) < 0.5 * gt_boxes_hwl[:, None, None, 4]),
        dtype=np.bool_,
    )


def build_sparse_geoscaffold_targets(
    label_dict: Mapping[str, np.ndarray],
    all_anchors: Sequence[np.ndarray] | np.ndarray,
    *,
    mask_size: int = MASK_SIZE,
) -> dict[str, np.ndarray]:
    """Build positive-only fields in native flattened-anchor order."""

    if "pos_equal_one" not in label_dict or "targets" not in label_dict:
        raise KeyError("GeoScaffold requires native labels and regression targets.")
    labels = np.asarray(label_dict["pos_equal_one"])
    targets = np.asarray(label_dict["targets"])
    if labels.ndim != 3 or targets.shape != (*labels.shape, 7):
        raise ValueError("Native labels must be [H, W, A] with [H, W, A, 7] targets.")
    if not np.isfinite(labels).all() or not np.isfinite(targets).all():
        raise ValueError("Native label tensors must be finite.")

    anchors = _flatten_anchors(all_anchors, tuple(int(v) for v in labels.shape))
    flat_labels = labels.reshape(-1)
    positive_indices = np.flatnonzero(flat_labels > 0).astype(np.int64)
    class_values = flat_labels[positive_indices]
    if not np.equal(class_values, np.round(class_values)).all():
        raise RuntimeError("Official positive class IDs must be integral.")
    class_ids = class_values.astype(np.int64)
    positive_anchors = anchors.reshape(-1, 7)[positive_indices]
    positive_targets = targets.reshape(-1, 7)[positive_indices]
    decoded_gt = _decode_positive_targets(positive_anchors, positive_targets)
    fields = _binary_intersection_fields(
        positive_anchors.astype(np.float64, copy=False),
        decoded_gt,
        mask_size=mask_size,
    )
    return {
        "anchor_indices": positive_indices,
        "anchor_boxes_hwl": positive_anchors.astype(np.float32, copy=False),
        "class_ids": class_ids,
        # Preserve the native target code for the parameter-matched box-code
        # control.  The field target and the code target share the exact same
        # official positive-anchor population.
        "box_code_targets": positive_targets.astype(np.float32, copy=False),
        "field_targets": fields,
    }


def collate_sparse_geoscaffold_targets(
    samples: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Concatenate variable positive-anchor sets and attach batch indices."""

    if not samples:
        raise ValueError("GeoScaffold collation requires at least one sample.")
    batch_indices: list[np.ndarray] = []
    anchor_indices: list[np.ndarray] = []
    anchor_boxes: list[np.ndarray] = []
    class_ids: list[np.ndarray] = []
    box_code_targets: list[np.ndarray] = []
    field_targets: list[np.ndarray] = []
    has_box_code_targets: list[bool] = []
    for batch_index, sample in enumerate(samples):
        required = {
            "anchor_indices",
            "anchor_boxes_hwl",
            "class_ids",
            "field_targets",
        }
        missing = required.difference(sample)
        if missing:
            raise KeyError(f"GeoScaffold sample is missing {sorted(missing)}.")
        indices = np.asarray(sample["anchor_indices"], dtype=np.int64).reshape(-1)
        boxes = np.asarray(sample["anchor_boxes_hwl"], dtype=np.float32)
        labels = np.asarray(sample["class_ids"], dtype=np.int64).reshape(-1)
        fields = np.asarray(sample["field_targets"], dtype=np.bool_)
        count = int(indices.shape[0])
        if boxes.shape != (count, 7) or labels.shape != (count,):
            raise ValueError("GeoScaffold sparse metadata is not anchor-aligned.")
        if fields.ndim != 3 or fields.shape[0] != count or fields.shape[1] != fields.shape[2]:
            raise ValueError("GeoScaffold fields must be square [N, mask_size, mask_size].")
        has_box_code = "box_code_targets" in sample
        has_box_code_targets.append(has_box_code)
        if has_box_code:
            codes = np.asarray(sample["box_code_targets"], dtype=np.float32)
            if codes.shape != (count, 7):
                raise ValueError("GeoScaffold box-code targets must be [N, 7].")
            if not np.isfinite(codes).all():
                raise ValueError("GeoScaffold box-code targets must be finite.")
            box_code_targets.append(codes)
        else:
            box_code_targets.append(np.empty((0, 7), dtype=np.float32))
        batch_indices.append(np.full(count, batch_index, dtype=np.int64))
        anchor_indices.append(indices)
        anchor_boxes.append(boxes)
        class_ids.append(labels)
        field_targets.append(fields)

    mask_sizes = {int(fields.shape[1]) for fields in field_targets}
    if len(mask_sizes) != 1:
        raise ValueError("GeoScaffold fields must use one mask size per batch.")
    batch_mask_size = next(iter(mask_sizes))

    def concatenate(parts: Sequence[np.ndarray], shape: tuple[int, ...], dtype: np.dtype):
        if any(len(part) for part in parts):
            return np.concatenate(parts, axis=0)
        return np.empty(shape, dtype=dtype)

    total_fields = concatenate(
        field_targets,
        (0, batch_mask_size, batch_mask_size),
        np.dtype(np.bool_),
    )
    if any(has_box_code_targets) and not all(has_box_code_targets):
        raise ValueError("GeoScaffold box-code targets must be present for every sample or none.")
    return {
        "batch_indices": torch.from_numpy(
            concatenate(batch_indices, (0,), np.dtype(np.int64))
        ),
        "anchor_indices": torch.from_numpy(
            concatenate(anchor_indices, (0,), np.dtype(np.int64))
        ),
        "anchor_boxes_hwl": torch.from_numpy(
            concatenate(anchor_boxes, (0, 7), np.dtype(np.float32))
        ),
        "class_ids": torch.from_numpy(
            concatenate(class_ids, (0,), np.dtype(np.int64))
        ),
        "field_targets": torch.from_numpy(total_fields),
        **(
            {
                "box_code_targets": torch.from_numpy(
                    concatenate(box_code_targets, (0, 7), np.dtype(np.float32))
                )
            }
            if all(has_box_code_targets)
            else {}
        ),
    }
