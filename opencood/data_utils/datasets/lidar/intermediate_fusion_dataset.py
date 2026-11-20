# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Edited: Dominik Roessle <dominik.roessle@thi.de>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion
"""
from __future__ import annotations

import random
import math
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.geoscaffold import (
    MASK_SIZE,
    build_sparse_geoscaffold_targets,
    collate_sparse_geoscaffold_targets,
)
from opencood.data_utils.global_bev_footprint import (
    rasterize_global_bev_footprint,
)
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.models.sub_modules.coalign_pose import (
    load_stage1_artifact,
    refine_base_data_poses,
    select_stage1_frame,
)
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum


def _object_key(object_id: Any) -> str:
    """Return one stable key for integer-like IDs from YAML and NumPy."""
    return str(object_id)


def _read_point_cloud_hits(object_content: Any) -> tuple[float, bool]:
    """Read non-negative point hits, returning zero and a miss flag if absent."""
    if not isinstance(object_content, Mapping):
        return 0.0, True
    try:
        hits = float(object_content["point_cloud_hits"])
    except (KeyError, TypeError, ValueError):
        return 0.0, True
    if not math.isfinite(hits) or hits < 0.0:
        return 0.0, True
    return hits, False


def compute_receiver_relative_evidence(
    object_ids: Sequence[Any],
    active_agent_objects: Sequence[tuple[bool, Mapping[Any, Any]]],
) -> tuple[np.ndarray, int]:
    """Compute receiver-relative evidence using only forward-active agents.

    The agent marked ``ego`` is the receiver selected for this sample. Missing
    object records and missing/invalid hit fields contribute zero and are
    counted so the training run can audit metadata coverage.
    """
    ego_count = sum(int(is_ego) for is_ego, _ in active_agent_objects)
    if ego_count != 1:
        raise ValueError(
            "Receiver-relative evidence requires exactly one active ego; "
            f"found {ego_count}."
        )

    normalized_agents = []
    for is_ego, objects in active_agent_objects:
        normalized_agents.append(
            (is_ego, {_object_key(object_id): value for object_id, value in objects.items()})
        )

    evidence = np.zeros(len(object_ids), dtype=np.float32)
    missing_hit_count = 0
    for object_index, object_id in enumerate(object_ids):
        key = _object_key(object_id)
        ego_hits = 0.0
        remote_hits = 0.0
        for is_ego, objects in normalized_agents:
            hits, missing = _read_point_cloud_hits(objects.get(key))
            missing_hit_count += int(missing)
            if is_ego:
                ego_hits = hits
            else:
                remote_hits += hits
        evidence[object_index] = math.log1p(remote_hits) - math.log1p(ego_hits)

    return evidence, missing_hit_count


def build_teacher_processed_lidar(
    projected_lidar_stack: Sequence[np.ndarray],
    pre_processor: Any,
    cav_lidar_range: Sequence[float],
) -> Mapping[str, Any]:
    """Voxelize the all-CAV ego-aligned cloud used by DiscoNet's teacher."""
    if not projected_lidar_stack:
        raise ValueError('DiscoNet teacher requires at least one active CAV.')
    stacked = np.vstack(projected_lidar_stack)
    stacked = mask_points_by_range(stacked, cav_lidar_range)
    return pre_processor.preprocess(stacked)


def teacher_preprocess_config(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve DiscoNet's frozen teacher voxelization contract."""
    config = params.get('teacher_preprocess', params['preprocess'])
    if not isinstance(config, Mapping):
        raise TypeError('teacher_preprocess must be a mapping when provided.')
    return config


def phase59_requires_anchor_grid(params: Mapping[str, Any]) -> bool:
    core_method = str(params.get('model', {}).get('core_method', '')).lower()
    return core_method.split('.')[-1] == 'point_pillar_mmcooper'


def phase59_requires_ego_targets(params: Mapping[str, Any]) -> bool:
    """Return whether the native method supervises an ego-only detector head."""
    core_method = str(params.get('model', {}).get('core_method', '')).lower()
    return core_method.split('.')[-1] == 'point_pillar_ermvp'


def phase64_requires_single_agent_targets(params: Mapping[str, Any]) -> bool:
    """Return whether the native fusion loss needs labels in each CAV frame.

    HEAL's foreground-aware pyramid predicts one occupancy field per packed
    agent before feature warping.  Its supervision therefore cannot reuse the
    receiver-frame collaborative labels.
    """
    model = params.get('model', {})
    if not isinstance(model, Mapping):
        return False
    model_args = model.get('args', {})
    if not isinstance(model_args, Mapping):
        return False
    core_method = str(model.get('core_method', '')).lower().split('.')[-1]
    return (
        core_method == 'point_pillar_heal'
        and bool(model_args.get('supervise_single', False))
    )


def _geometry_scaffold_enabled(params: Mapping[str, Any]) -> bool:
    """Return whether this loader should emit GeoScaffold sparse targets."""

    model_args = params.get('model', {}).get('args', {})
    if not isinstance(model_args, Mapping):
        model_args = {}
    geometry_scaffold = model_args.get('geometry_scaffold', {})
    if not isinstance(geometry_scaffold, Mapping):
        geometry_scaffold = {}
    mode = geometry_scaffold.get('mode')
    if mode is None and 'enabled' in geometry_scaffold:
        mode = 'fused' if bool(geometry_scaffold['enabled']) else 'disabled'
    if mode is None:
        counterfactual = model_args.get('counterfactual_training', {})
        if not isinstance(counterfactual, Mapping):
            counterfactual = {}
        mode = counterfactual.get('mode', 'disabled')
    normalized_mode = str(mode).lower()
    if normalized_mode in {'ego_geoscaffold', 'fused_geoscaffold'}:
        return True
    if normalized_mode in {'ego', 'fused'}:
        return True

    loss_args = params.get('loss', {}).get('args', {})
    if not isinstance(loss_args, Mapping):
        return False
    geometry_scaffold_loss = loss_args.get('geometry_scaffold', {})
    if isinstance(geometry_scaffold_loss, Mapping) and 'weight' in geometry_scaffold_loss:
        return float(geometry_scaffold_loss.get('weight', 0.0)) != 0.0
    counterfactual_loss = loss_args.get('counterfactual_training', {})
    if not isinstance(counterfactual_loss, Mapping):
        return False
    return float(counterfactual_loss.get('geoscaffold_weight', 0.0)) != 0.0


def _geometry_scaffold_mask_size(params: Mapping[str, Any]) -> int:
    """Resolve the canonical field resolution used by model and sparse labels."""

    model = params.get('model', {})
    model_args = model.get('args', {}) if isinstance(model, Mapping) else {}
    geometry_scaffold = (
        model_args.get('geometry_scaffold', {})
        if isinstance(model_args, Mapping)
        else {}
    )
    if not isinstance(geometry_scaffold, Mapping):
        geometry_scaffold = {}
    mask_size = int(geometry_scaffold.get('mask_size', MASK_SIZE))
    if mask_size <= 0:
        raise ValueError('geometry_scaffold.mask_size must be positive')
    return mask_size


def _global_bev_footprint_enabled(params: Mapping[str, Any]) -> bool:
    """Return whether training batches should carry dense footprint targets."""

    model = params.get("model", {})
    model_args = model.get("args", {}) if isinstance(model, Mapping) else {}
    config = (
        model_args.get("global_bev_footprint", {})
        if isinstance(model_args, Mapping)
        else {}
    )
    if not isinstance(config, Mapping):
        config = {}
    return bool(config.get("enabled", False))


class LidarIntermediateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize=False, train=True, **kwargs):
        super(LidarIntermediateFusionDataset, self). \
            __init__(params, visualize, train, **kwargs)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            self.object_types,
            train)
        self.geometry_scaffold_enabled = (
            bool(train) and _geometry_scaffold_enabled(params)
        )
        self.geometry_scaffold_mask_size = _geometry_scaffold_mask_size(params)
        self.global_bev_footprint_enabled = (
            bool(train) and _global_bev_footprint_enabled(params)
        )
        self.coalign_stage1 = None
        self.coalign_args = {}
        self.coalign_split = str(params.get('dataset_split', 'train' if train else 'val'))
        if 'box_align' in params:
            box_align = params['box_align']
            result_key = 'train_result' if train else 'val_result'
            if result_key not in box_align:
                raise KeyError(f'box_align.{result_key} is required')
            self.coalign_stage1 = load_stage1_artifact(box_align[result_key])
            self.coalign_args = dict(box_align.get('args', {}))
            self.coalign_split = str(box_align.get('split', self.coalign_split))
        self.phase59_stage1_precalc = bool(
            params.get('phase59_stage1_precalc', False)
        )
        self.kd_flag = bool(params.get('kd_flag', False))
        if self.kd_flag and not self.proj_first:
            raise ValueError(
                'DiscoNet teacher_processed_lidar requires proj_first=true so '
                'all CAV points share the receiver frame.'
            )
        self.teacher_pre_processor = None
        self.teacher_lidar_range = None
        if self.kd_flag:
            teacher_preprocess = teacher_preprocess_config(params)
            self.teacher_pre_processor = build_preprocessor(
                teacher_preprocess,
                train,
            )
            self.teacher_lidar_range = teacher_preprocess['cav_lidar_range']
        self.phase59_anchor_grid = phase59_requires_anchor_grid(params)
        self.phase59_ego_targets = phase59_requires_ego_targets(params)
        self.supervise_single = phase64_requires_single_agent_targets(params)
    
    def generate_augment(self):
        flip = [None, None, None]
        noise_rotation = None
        noise_scale = None

        if self.data_augmentor is None:
            return flip, noise_rotation, noise_scale

        for aug_ele in self.data_augmentor.augment_config:
            # for intermediate fusion only
            if 'random_world_rotation' in aug_ele['NAME']:
                rot_range = \
                    aug_ele['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0],
                                                        rot_range[1])

            if 'random_world_flip' in aug_ele['NAME']:
                for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                    enable = np.random.choice([False, True], replace=False,
                                              p=[0.5, 0.5])
                    flip[i] = enable

            if 'random_world_scaling' in aug_ele['NAME']:
                scale_range = \
                    aug_ele['WORLD_SCALE_RANGE']
                noise_scale = \
                    np.random.uniform(scale_range[0], scale_range[1])

        return flip, noise_rotation, noise_scale

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx,
                                                 cur_ego_pose_flag=self.cur_ego_pose_flag,
                                                 load_lidar_data=True)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        # augmentation related
        flip, noise_rotation, noise_scale = self.generate_augment()

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        coalign_report = None
        if self.coalign_stage1 is not None:
            active_for_alignment = OrderedDict()
            for cav_id, cav_content in base_data_dict.items():
                distance = math.sqrt(
                    (cav_content['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2
                    + (cav_content['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )
                if distance <= getattr(
                    self,
                    "communication_range",
                    opencood.data_utils.datasets.COM_RANGE,
                ):
                    active_for_alignment[cav_id] = cav_content
            scenario_ids = {
                str(cav_content.get('scenario_id'))
                for cav_content in active_for_alignment.values()
            }
            if len(scenario_ids) != 1:
                raise ValueError(
                    'CoAlign sample must have exactly one scenario identity; '
                    f'got {sorted(scenario_ids)}.'
                )
            scenario_id = next(iter(scenario_ids))
            current_timestamp = str(
                base_data_dict[ego_id].get('current_timestamp')
            )
            expected_timestamps = {
                str(cav_id): str(
                    cav_content.get('current_timestamp')
                    if bool(cav_content['ego'])
                    else cav_content.get('history_timestamp')
                )
                for cav_id, cav_content in active_for_alignment.items()
            }
            dataset_roots = {
                str(cav_content.get('dataset_root'))
                for cav_content in active_for_alignment.values()
            }
            if len(dataset_roots) != 1:
                raise ValueError(
                    'CoAlign sample must have exactly one dataset_root identity; '
                    f'got {sorted(dataset_roots)}.'
                )
            dataset_root = next(iter(dataset_roots))
            frame = select_stage1_frame(
                self.coalign_stage1,
                sample_idx=str(idx),
                dataset_root=dataset_root,
                split=self.coalign_split,
                scenario_id=scenario_id,
                current_timestamp=current_timestamp,
                expected_agent_timestamps=expected_timestamps,
                expected_agent_ids=list(active_for_alignment),
            )
            coalign_report = refine_base_data_poses(
                active_for_alignment,
                frame,
                **self.coalign_args,
            )
            ego_lidar_pose = base_data_dict[ego_id]['params']['lidar_pose']

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []
        active_agent_objects = []
        ego_object_boxes = None
        single_agent_object_boxes = []
        single_agent_object_masks = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []
        active_cav_ids = []
        active_lidar_poses = []
        active_current_timestamps = []
        active_history_timestamps = []
        active_scenario_ids = []
        active_dataset_roots = []
        active_dataset_splits = []

        if self.visualize or self.kd_flag:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > getattr(
                self,
                "communication_range",
                opencood.data_utils.datasets.COM_RANGE,
            ):
                continue

            if self.train:
                # Evidence must use exactly the agents entering this forward pass.
                active_agent_objects.append(
                    (
                        bool(selected_cav_base['ego']),
                        selected_cav_base['params'].get('vehicles', {}),
                    )
                )

            # augment related
            selected_cav_base['flip'] = flip
            selected_cav_base['noise_rotation'] = noise_rotation
            selected_cav_base['noise_scale'] = noise_scale

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            if self.phase59_ego_targets and bool(selected_cav_base['ego']):
                ego_object_boxes = np.asarray(
                    selected_cav_processed['object_bbx_center']
                )

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])
            if self.supervise_single:
                single_agent_object_boxes.append(
                    selected_cav_processed['single_object_bbx_center']
                )
                single_agent_object_masks.append(
                    selected_cav_processed['single_object_bbx_mask']
                )

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            # this is only useful when proj_first = True, and communication
            # delay is considered. Right now only V2X-ViT utilizes the
            # spatial_correction. There is a time delay when the cavs project
            # their lidar to ego and when the ego receives the feature, and
            # this variable is used to correct such pose difference (ego_t-1 to
            # ego_t)
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)
            active_cav_ids.append(str(cav_id))
            active_lidar_poses.append(selected_cav_base['params']['lidar_pose'])
            active_current_timestamps.append(
                str(selected_cav_base.get('current_timestamp'))
            )
            active_history_timestamps.append(
                str(selected_cav_base.get('history_timestamp'))
            )
            active_scenario_ids.append(
                str(selected_cav_base.get('scenario_id'))
            )
            active_dataset_roots.append(
                str(selected_cav_base.get('dataset_root'))
            )
            active_dataset_splits.append(
                str(selected_cav_base.get('dataset_split'))
            )

            if self.visualize or self.kd_flag:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        selected_object_ids = [object_id_stack[i] for i in unique_indices]
        if self.train:
            receiver_evidence_q, missing_hit_count = \
                compute_receiver_relative_evidence(
                    selected_object_ids,
                    active_agent_objects,
                )
        else:
            # Evaluation remains the original, unweighted objective and avoids
            # receiver-evidence preprocessing at inference time.
            receiver_evidence_q = np.zeros(
                len(selected_object_ids),
                dtype=np.float32,
            )
            missing_hit_count = 0

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 8))
        mask = np.zeros(self.params['postprocess']['max_num'])
        padded_receiver_evidence_q = np.zeros(
            self.params['postprocess']['max_num'],
            dtype=np.float32,
        )
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1
        padded_receiver_evidence_q[:object_stack.shape[0]] = receiver_evidence_q

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)
        teacher_processed_lidar = None
        if self.kd_flag:
            if self.teacher_pre_processor is None or self.teacher_lidar_range is None:
                raise RuntimeError('DiscoNet teacher preprocessor was not initialized.')
            teacher_processed_lidar = build_teacher_processed_lidar(
                projected_lidar_stack,
                self.teacher_pre_processor,
                self.teacher_lidar_range,
            )

        # generate the anchor boxes
        all_anchors, num_anchors_per_location = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=all_anchors,
                num_anchors_per_location=num_anchors_per_location,
                mask=mask,
                receiver_evidence_q=padded_receiver_evidence_q,
                receiver_evidence_missing_hit_count=missing_hit_count)
        single_label_dicts = None
        if self.supervise_single:
            if len(single_agent_object_boxes) != cav_num:
                raise RuntimeError(
                    'HEAL single-agent targets must match the packed CAV count.'
                )
            single_label_dicts = [
                self.post_processor.generate_label(
                    gt_box_center=single_boxes,
                    anchors=all_anchors,
                    num_anchors_per_location=num_anchors_per_location,
                    mask=single_mask,
                    include_all_negative_channels=True,
                )
                for single_boxes, single_mask in zip(
                    single_agent_object_boxes,
                    single_agent_object_masks,
                )
            ]
        ego_label_dict = None
        if self.phase59_ego_targets:
            if ego_object_boxes is None:
                raise RuntimeError('ERMVP requires the active ego object targets.')
            ego_object_bbx_center = np.zeros(
                (self.params['postprocess']['max_num'], 8)
            )
            ego_mask = np.zeros(self.params['postprocess']['max_num'])
            ego_count = min(
                ego_object_boxes.shape[0],
                self.params['postprocess']['max_num'],
            )
            ego_object_bbx_center[:ego_count] = ego_object_boxes[:ego_count]
            ego_mask[:ego_count] = 1
            ego_label_dict = self.post_processor.generate_label(
                gt_box_center=ego_object_bbx_center,
                anchors=all_anchors,
                num_anchors_per_location=num_anchors_per_location,
                mask=ego_mask,
            )
        geoscaffold_targets = None
        if self.geometry_scaffold_enabled:
            # Keep only positive-anchor geometry after the native assigner has
            # consumed the static grid.  This avoids shipping the dense anchor
            # field through every training batch.
            geoscaffold_targets = build_sparse_geoscaffold_targets(
                label_dict,
                all_anchors,
                mask_size=self.geometry_scaffold_mask_size,
            )
        global_bev_footprint_target = None
        if self.global_bev_footprint_enabled:
            target_height, target_width = label_dict["pos_equal_one"].shape[:2]
            global_bev_footprint_target = rasterize_global_bev_footprint(
                torch.from_numpy(object_bbx_center[None]).float(),
                torch.from_numpy(mask[None]).bool(),
                height=int(target_height),
                width=int(target_width),
                lidar_range=self.params["postprocess"]["anchor_args"][
                    "cav_lidar_range"
                ],
            )[0].numpy()

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                               spatial_correction_matrix),1,1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                   padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': selected_object_ids,
             'all_anchors': all_anchors,
             'num_anchors_per_location': num_anchors_per_location,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'pairwise_t_matrix': pairwise_t_matrix})
        if geoscaffold_targets is not None:
            processed_data_dict['ego']['geoscaffold'] = geoscaffold_targets
        if global_bev_footprint_target is not None:
            processed_data_dict['ego']['global_bev_footprint_target'] = \
                global_bev_footprint_target
        if ego_label_dict is not None:
            processed_data_dict['ego']['ego_label_dict'] = ego_label_dict
        if single_label_dicts is not None:
            processed_data_dict['ego']['single_label_dicts'] = single_label_dicts
        if coalign_report is not None:
            processed_data_dict['ego']['coalign_report'] = coalign_report
        if self.phase59_stage1_precalc:
            metadata_identity = {
                'scenario': set(active_scenario_ids),
                'dataset_root': set(active_dataset_roots),
                'dataset_split': set(active_dataset_splits),
                'current_timestamp': set(active_current_timestamps),
            }
            for field, values in metadata_identity.items():
                if len(values) != 1:
                    raise ValueError(
                        'Stage-1 metadata requires one sample-level '
                        f'{field} identity, got {sorted(values)}.'
                    )
            processed_data_dict['ego']['phase59_stage1_metadata'] = {
                'sample_idx': str(idx),
                'cav_ids': active_cav_ids,
                'lidar_poses': np.asarray(active_lidar_poses, dtype=np.float64),
                'dataset_root': active_dataset_roots[0],
                'dataset_split': active_dataset_splits[0],
                'scenario': active_scenario_ids[0],
                'timestamp': active_current_timestamps[0],
                'current_timestamps': active_current_timestamps,
                'history_timestamps': active_history_timestamps,
                'source_hist': active_history_timestamps,
            }
        if teacher_processed_lidar is not None:
            processed_data_dict['ego']['teacher_processed_lidar'] = \
                teacher_processed_lidar

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        single_object_bbx_center = None
        single_object_bbx_mask = None
        if self.supervise_single:
            single_object_bbx_center, single_object_bbx_mask, _ = \
                self.post_processor.generate_object_center(
                    [selected_cav_base],
                    selected_cav_base['params']['lidar_pose'],
                    self.gt_range,
                )

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center(
                [selected_cav_base], ego_pose, self.gt_range)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
        # data augmentation
        if self.data_augmentor:
            lidar_np, object_bbx_center, object_bbx_mask = \
                self.augment(lidar_np, object_bbx_center, object_bbx_mask,
                            selected_cav_base['flip'],
                            selected_cav_base['noise_rotation'],
                            selected_cav_base['noise_scale'])
            if single_object_bbx_center is not None:
                empty_points = np.empty((0, lidar_np.shape[1]), dtype=lidar_np.dtype)
                _, single_object_bbx_center, single_object_bbx_mask = self.augment(
                    empty_points,
                    single_object_bbx_center,
                    single_object_bbx_mask,
                    selected_cav_base['flip'],
                    selected_cav_base['noise_rotation'],
                    selected_cav_base['noise_scale'],
                )

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        
        object_bbx_center_valid = object_bbx_center[object_bbx_mask == 1]

        if self.data_augmentor:
            # filter out the augmented bbx that is out of range
            object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True)
            object_ids = [int(x) for x in list(np.array(object_ids)[range_mask])]

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center_valid,
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})
        if single_object_bbx_center is not None:
            selected_cav_processed.update(
                {
                    'single_object_bbx_center': single_object_bbx_center,
                    'single_object_bbx_mask': single_object_bbx_mask,
                }
            )

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        ego_label_dict_list = []
        geoscaffold_target_list = []
        global_bev_footprint_target_list = []
        coalign_report_list = []
        stage1_metadata_list = []
        teacher_processed_lidar_list = []
        single_label_dicts = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            if 'ego_label_dict' in ego_dict:
                ego_label_dict_list.append(ego_dict['ego_label_dict'])
            if 'geoscaffold' in ego_dict:
                geoscaffold_target_list.append(ego_dict['geoscaffold'])
            if 'global_bev_footprint_target' in ego_dict:
                global_bev_footprint_target_list.append(
                    ego_dict['global_bev_footprint_target']
                )
            if 'coalign_report' in ego_dict:
                coalign_report_list.append(ego_dict['coalign_report'])
            if 'phase59_stage1_metadata' in ego_dict:
                stage1_metadata_list.append(ego_dict['phase59_stage1_metadata'])
            if 'teacher_processed_lidar' in ego_dict:
                teacher_processed_lidar_list.append(
                    ego_dict['teacher_processed_lidar']
                )
            if 'single_label_dicts' in ego_dict:
                single_label_dicts.extend(ego_dict['single_label_dicts'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        ego_label_torch_dict = None
        if ego_label_dict_list:
            if len(ego_label_dict_list) != len(batch):
                raise ValueError(
                    'ERMVP ego targets must be present for every sample or none.'
                )
            ego_label_torch_dict = self.post_processor.collate_batch(
                ego_label_dict_list
            )
        geoscaffold_torch_dict = None
        if geoscaffold_target_list:
            if len(geoscaffold_target_list) != len(batch):
                raise ValueError(
                    'GeoScaffold targets must be present for every sample or none.'
                )
            geoscaffold_torch_dict = collate_sparse_geoscaffold_targets(
                geoscaffold_target_list
            )

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'pairwise_t_matrix': pairwise_t_matrix})
        if geoscaffold_torch_dict is not None:
            output_dict['ego']['geoscaffold'] = geoscaffold_torch_dict
        if global_bev_footprint_target_list:
            if len(global_bev_footprint_target_list) != len(batch):
                raise ValueError(
                    'Global BEV footprint targets must be present for every sample or none.'
                )
            output_dict['ego']['global_bev_footprint_target'] = torch.from_numpy(
                np.asarray(global_bev_footprint_target_list, dtype=np.bool_)
            )
        if ego_label_torch_dict is not None:
            output_dict['ego']['ego_label_dict'] = ego_label_torch_dict
        if single_label_dicts:
            expected_agents = int(record_len.sum().item())
            if len(single_label_dicts) != expected_agents:
                raise ValueError(
                    'HEAL single-agent labels must preserve packed-agent order: '
                    f'expected {expected_agents}, found {len(single_label_dicts)}.'
                )
            output_dict['ego']['label_dict_single'] = \
                self.post_processor.collate_batch(single_label_dicts)
        if coalign_report_list:
            if len(coalign_report_list) != len(batch):
                raise ValueError('CoAlign reports must be present for every sample or none.')
            output_dict['ego']['coalign_report'] = coalign_report_list
        if stage1_metadata_list:
            if len(stage1_metadata_list) != len(batch):
                raise ValueError(
                    'Stage-1 metadata must be present for every sample or none.'
                )
            collated_stage1_metadata = []
            for item in stage1_metadata_list:
                metadata = {
                    'sample_idx': item['sample_idx'],
                    'cav_ids': item['cav_ids'],
                    'lidar_poses': torch.from_numpy(item['lidar_poses']),
                }
                for key in (
                    'dataset_root',
                    'dataset_split',
                    'scenario',
                    'timestamp',
                    'current_timestamps',
                    'history_timestamps',
                    'source_hist',
                ):
                    if key in item:
                        metadata[key] = item[key]
                collated_stage1_metadata.append(metadata)
            output_dict['ego']['phase59_stage1_metadata'] = collated_stage1_metadata
        if teacher_processed_lidar_list:
            if len(teacher_processed_lidar_list) != len(batch):
                raise ValueError(
                    'Teacher LiDAR must be present for every sample or none.'
                )
            merged_teacher = self.merge_features_to_dict(
                teacher_processed_lidar_list
            )
            if self.teacher_pre_processor is None:
                raise RuntimeError('DiscoNet teacher preprocessor was not initialized.')
            output_dict['ego']['teacher_processed_lidar'] = \
                self.teacher_pre_processor.collate_batch(merged_teacher)
        if self.phase59_anchor_grid:
            output_dict['ego']['all_anchors'] = torch.from_numpy(
                np.asarray(batch[0]['ego']['all_anchors'])
            )

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['all_anchors'] is not None:
            # (num_class, H, W, num_anchor, 7)
            all_anchors = torch.from_numpy(
                np.array(batch[0]['ego']['all_anchors'] ))
            # [(H, W, num_anchor, 7)] * num_class
            # all_anchors = [torch.from_numpy(x) for x in cav_content['all_anchors']]
            num_anchors_per_location = batch[0]['ego'][
                'num_anchors_per_location']
            output_dict['ego'].update({'all_anchors': all_anchors,
                                        'num_anchors_per_location': num_anchors_per_location})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})
    
        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict, gt_range=self.gt_range)
        gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx(data_dict, gt_range=self.gt_range)

        return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
