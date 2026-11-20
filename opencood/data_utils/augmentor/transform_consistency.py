"""Keep local point augmentation and feature-space pose transforms consistent."""

from __future__ import annotations

import numpy as np


def sampled_augmentation_matrix(augmentor, flip, rotation, scale) -> np.ndarray:
    """Recover the already-sampled affine map using the actual augmentation operator.

    This consumes no new randomness: the intermediate dataset has already sampled
    every active flip, rotation, and scale. Seven-column empty boxes avoid treating
    the class column of this repository's box API as a velocity component.
    """
    if augmentor is None or not augmentor.train:
        return np.eye(4, dtype=np.float64)
    for config in augmentor.augment_config:
        name = config["NAME"]
        if name == "random_world_flip":
            if flip is None or any(flip[i] is None for i, _ in enumerate(config["ALONG_AXIS_LIST"])):
                raise ValueError("Feature transforms require already-sampled flips")
        elif name == "random_world_rotation":
            if rotation is None:
                raise ValueError("Feature transforms require an already-sampled rotation")
        elif name == "random_world_scaling":
            if scale is None:
                raise ValueError("Feature transforms require an already-sampled scale")
        else:
            raise ValueError(f"Unsupported feature-transform augmentation: {name}")
    points = np.zeros((4, 4), dtype=np.float32)
    points[1:, :3] = np.eye(3, dtype=np.float32)
    transformed = augmentor.forward({
        "lidar_np": points,
        "object_bbx_center": np.empty((0, 7), dtype=np.float32),
        "object_bbx_mask": np.empty((0,), dtype=np.float32),
        "flip": flip, "noise_rotation": rotation, "noise_scale": scale,
    })["lidar_np"][:, :3].astype(np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (transformed[1:] - transformed[0]).T
    matrix[:3, 3] = transformed[0]
    return matrix


def conjugate_pairwise_transforms(transforms: np.ndarray, augmentation: np.ndarray) -> np.ndarray:
    """For augmented local points A p, use A T A^{-1}, not the old T."""
    return augmentation @ np.asarray(transforms) @ np.linalg.inv(augmentation)
