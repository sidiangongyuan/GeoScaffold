"""Opt-in coordinate-consistent local BEV inputs for UrbanIng fusion models."""

import copy
import math

from opencood.data_utils.augmentor.transform_consistency import (
    conjugate_pairwise_transforms,
    sampled_augmentation_matrix,
)
from opencood.data_utils.datasets.lidar.intermediate_fusion_dataset import (
    LidarIntermediateFusionDataset,
)


class UrbanIngLocalFeatureDataset(LidarIntermediateFusionDataset):
    """Use the same sampled augmentation for points and inter-agent transforms.

    The legacy dataset and accepted checkpoints retain their original path.
    This class is selected only for an explicitly corrected local-feature recipe.
    """

    def __init__(self, params, visualize=False, train=True, **kwargs):
        # One training config also constructs a validation loader. A validation
        # split label must never be used to join the training alignment cache.
        params = copy.deepcopy(params)
        if train:
            params["dataset_split"] = "train"
            if "box_align" in params:
                params["box_align"]["split"] = "train"
        elif (params.get("dataset_split") == "train"
              and not params.get("phase59_stage1_precalc", False)):
            params["dataset_split"] = "val"
            if "box_align" in params:
                params["box_align"]["split"] = "val"
        super().__init__(params, visualize, train, **kwargs)
        if (params.get("phase59_stage1_precalc", False)
                and params.get("phase59_stage1_precalc_all_agents", False)):
            # Stage-1 caches must cover every potential training ego, not only
            # collaborators within the deterministic precalculation ego's range.
            # Normal fusion retains the dataset's original communication range.
            self.communication_range = float("inf")

    def generate_augment(self):
        sampled = super().generate_augment()
        self._local_point_augmentation = None
        if self.train and not self.proj_first:
            self._local_point_augmentation = sampled_augmentation_matrix(
                self.data_augmentor, *sampled,
            )
        return sampled

    def retrieve_base_data(self, *args, **kwargs):
        base = super().retrieve_base_data(*args, **kwargs)
        if not self.params['fusion']['args'].get('pack_active_agents', False):
            return base
        # Feature tensors are packed after range filtering. Construct pose matrices
        # from that same ordered source set, not the first N unfiltered agents.
        # Keep the old path opt-in for reproducibility of superseded checkpoints.
        from opencood.data_utils.datasets import COM_RANGE

        receiver = next(value for value in base.values() if value['ego'])
        ego_pose = receiver['params']['lidar_pose']
        radius = getattr(self, 'communication_range', COM_RANGE)
        return type(base)((agent, value) for agent, value in base.items()
                          if math.hypot(value['params']['lidar_pose'][0] - ego_pose[0],
                                        value['params']['lidar_pose'][1] - ego_pose[1]) <= radius)

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        transforms = super().get_pairwise_transformation(base_data_dict, max_cav)
        augmentation = getattr(self, "_local_point_augmentation", None)
        if augmentation is not None:
            transforms = conjugate_pairwise_transforms(transforms, augmentation)
        return transforms
