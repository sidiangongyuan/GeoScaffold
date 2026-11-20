# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.lidar.late_fusion_dataset import LidarLateFusionDataset
from opencood.data_utils.datasets.lidar.early_fusion_dataset import LidarEarlyFusionDataset
from opencood.data_utils.datasets.lidar.intermediate_fusion_dataset import LidarIntermediateFusionDataset
from opencood.data_utils.datasets.v2v4real import V2V4REAL_DATASETS
from opencood.data_utils.datasets.lidar.local_feature_dataset import UrbanIngLocalFeatureDataset

__all__ = {
    'LidarLateFusionDataset': LidarLateFusionDataset,
    'LidarEarlyFusionDataset': LidarEarlyFusionDataset,
    'LidarIntermediateFusionDataset': LidarIntermediateFusionDataset
}

# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True, **kwargs):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    dataset_classes = (
        V2V4REAL_DATASETS
        if dataset_cfg.get('dataset_format') == 'v2v4real'
        else __all__
    )
    assert dataset_name in dataset_classes.keys(), error_message

    dataset_class = dataset_classes[dataset_name]
    fusion_args = dataset_cfg['fusion'].get('args', {})
    if (dataset_cfg.get('dataset_format') != 'v2v4real'
            and dataset_name == 'LidarIntermediateFusionDataset'
            and isinstance(fusion_args, dict)
            and fusion_args.get('correct_augmented_transforms', False)):
        dataset_class = UrbanIngLocalFeatureDataset
    dataset = dataset_class(
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        **kwargs
    )

    return dataset
