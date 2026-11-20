"""Opt-in loader for the raw V2V4Real release.

V2V4Real stores a 4x4 ``lidar_pose`` and object boxes in each CAV's local
LiDAR frame.  The normal UrbanIng BaseDataset is intentionally left alone;
this module supplies the raw-format boundary and reuses the existing fusion
classes through a small mixin.
"""

from __future__ import annotations

import copy
import math
import os
import random
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.datasets.lidar.early_fusion_dataset import (
    LidarEarlyFusionDataset,
)
from opencood.data_utils.datasets.lidar.intermediate_fusion_dataset import (
    LidarIntermediateFusionDataset,
)
from opencood.data_utils.datasets.lidar.late_fusion_dataset import (
    LidarLateFusionDataset,
)
from opencood.data_utils.post_processor.voxel_postprocessor import (
    VoxelPostprocessor,
)
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.transformation_utils import x_to_world
from opencood.models.sub_modules.coalign_pose import load_stage1_artifact


V2V4REAL_GT_RANGE = [-100.0, -40.0, -5.0, 100.0, 40.0, 3.0]


def _pose_matrix(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"V2V4Real lidar_pose must be finite 4x4, got {matrix.shape}")
    return matrix.copy()


def matrix_to_pose(value):
    """Compatibility [x,y,z,roll,yaw,pitch] view in degrees.

    The raw matrix remains authoritative for all projections.  This view is
    only used by legacy fusion code for distance checks and metadata.
    """

    matrix = _pose_matrix(value)
    rotation = matrix[:3, :3]
    yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    roll = math.degrees(math.atan2(-rotation[2, 1], rotation[2, 2]))
    pitch = math.degrees(
        math.atan2(rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    )
    return [*matrix[:3, 3].tolist(), roll, yaw, pitch]


def _local_box_corners(content):
    location = np.asarray(content["location"], dtype=np.float64)
    center = np.asarray(content.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    extent = np.asarray(content["extent"], dtype=np.float64)
    angle = np.asarray(content.get("angle", [0.0, 0.0, 0.0]), dtype=np.float64)
    if any(array.shape != (3,) for array in (location, center, extent, angle)):
        raise ValueError("V2V4Real location/center/extent/angle must have 3 values")
    # The official V2V4Real path passes signed extents directly to
    # ``create_bbx``.  A negative component is a recoverable annotation
    # convention and must not be normalized before corner construction,
    # otherwise the box yaw/corner ordering changes.  Zero is degenerate.
    if np.any(extent == 0) or not np.all(
        np.isfinite(np.concatenate((location, center, extent, angle)))
    ):
        raise ValueError("V2V4Real object geometry is nonfinite or degenerate")
    pose = [
        *(location + center).tolist(),
        float(angle[0]),
        float(angle[1]),
        float(angle[2]),
    ]
    corners = box_utils.create_bbx(extent).T
    homogeneous = np.vstack((corners, np.ones(corners.shape[1])))
    return (x_to_world(pose) @ homogeneous).T[:, :3]


def _corners_in_range(corners, limit_range):
    limit = np.asarray(limit_range, dtype=np.float64)
    if limit.shape != (6,):
        raise ValueError(f"V2V4Real range must have six values, got {limit.shape}")
    inside = (
        (corners >= limit[:3][None, :])
        & (corners <= limit[3:][None, :])
    ).all(axis=1)
    return int(inside.sum()) >= 2


class V2V4RealVoxelPostprocessor(VoxelPostprocessor):
    """Official raw pose/object semantics with the current label API."""

    def __init__(self, anchor_params, class_names, train):
        super().__init__(anchor_params, class_names, train)
        self.v2v4real_anchor_num = int(anchor_params["anchor_args"]["num"])
        if len(anchor_params["anchor_args"]["r"]) != self.v2v4real_anchor_num:
            raise ValueError("anchor_args.num must equal len(anchor_args.r)")

    def select_decoded_boxes(self, boxes3d, selected_mask, output):
        direction = self.params.get("dir_args")
        if direction is None:
            return super().select_decoded_boxes(boxes3d, selected_mask, output)
        from opencood.data_utils.post_processor.coalign_direction import apply_direction_bins

        logits = output.get("dir_preds", output.get("dm"))
        if logits is None:
            raise KeyError("Native direction decoding requires dir_preds or dm")
        return apply_direction_bins(
            boxes3d, logits, selected_mask,
            dir_offset=float(direction["dir_offset"]), num_bins=int(direction["num_bins"]),
        )

    def generate_anchor_box(self):
        """Official linspace grid; return the current ``([anchors], [2])`` API."""

        args = self.params["anchor_args"]
        stride = int(args.get("feature_stride", 2))
        if stride <= 0:
            raise ValueError("V2V4Real feature_stride must be positive")
        lidar_range = args["cav_lidar_range"]
        x = np.linspace(
            float(lidar_range[0]) + float(args["vw"]),
            float(lidar_range[3]) - float(args["vw"]),
            int(args["W"]) // stride,
        )
        y = np.linspace(
            float(lidar_range[1]) + float(args["vh"]),
            float(lidar_range[4]) - float(args["vh"]),
            int(args["H"]) // stride,
        )
        cx, cy = np.meshgrid(x, y)
        cx = np.repeat(cx[..., None], self.v2v4real_anchor_num, axis=-1)
        cy = np.repeat(cy[..., None], self.v2v4real_anchor_num, axis=-1)
        cz = np.full_like(cx, -1.0)
        h = np.full_like(cx, float(args["h"]))
        w = np.full_like(cx, float(args["w"]))
        length = np.full_like(cx, float(args["l"]))
        yaw = np.broadcast_to(
            np.deg2rad(np.asarray(args["r"], dtype=np.float64)).reshape(1, 1, -1),
            cx.shape,
        )
        if self.params["order"] == "hwl":
            anchors = np.stack((cx, cy, cz, h, w, length, yaw), axis=-1)
        elif self.params["order"] == "lhw":
            anchors = np.stack((cx, cy, cz, length, h, w, yaw), axis=-1)
        else:
            raise ValueError(f"Unknown V2V4Real order {self.params['order']!r}")
        return [anchors.astype(np.float32)], [self.v2v4real_anchor_num]

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _transform_for(self, cav_content, reference_pose):
        params = cav_content["params"]
        if params.pop("_v2v4real_single_label_pending", False):
            return np.eye(4)
        marker = params.get("_v2v4real_box_transform")
        if marker is not None:
            return _pose_matrix(marker)
        reference = np.asarray(reference_pose)
        return _pose_matrix(reference) if reference.shape == (4, 4) else np.eye(4)

    def generate_object_center(self, cav_contents, reference_lidar_pose, filter_range):
        objects = np.zeros((self.params["max_num"], 8), dtype=np.float32)
        mask = np.zeros(self.params["max_num"], dtype=np.float32)
        object_ids = []
        count = 0
        for cav_content in cav_contents:
            transform = self._transform_for(cav_content, reference_lidar_pose)
            cav_id = int(cav_content.get("cav_id", 0))
            for raw_id, content in cav_content["params"].get("vehicles", {}).items():
                if count >= self.params["max_num"]:
                    break
                if str(content.get("obj_type", "Car")).lower() == "pedestrian":
                    continue
                try:
                    local_corners = _local_box_corners(content)
                    homogeneous = np.hstack(
                        (local_corners, np.ones((local_corners.shape[0], 1)))
                    )
                    corners = (transform @ homogeneous.T).T[:, :3]
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "Malformed V2V4Real object "
                        f"cav={cav_id}, object={raw_id!r}"
                    ) from error
                if not _corners_in_range(corners, filter_range):
                    continue
                center = box_utils.corner_to_center(
                    corners[None], order=self.params["order"]
                )[0]
                objects[count, :7] = np.asarray(center, dtype=np.float32)
                objects[count, 7] = 1.0
                mask[count] = 1.0
                ass_id = content.get("ass_id", -1)
                try:
                    ass_id = int(ass_id)
                    local_id = int(raw_id)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"V2V4Real IDs must be integer-like: {raw_id!r}") from error
                object_ids.append(ass_id if ass_id != -1 else local_id + 100 * cav_id)
                count += 1
            if count >= self.params["max_num"]:
                break
        return objects, mask, object_ids

    def generate_gt_bbx(self, data_dict, gt_range):
        """Use native all-corners XY clipping for final evaluation GT."""

        all_corners, all_labels, all_ids = [], [], []
        for cav_content in data_dict.values():
            if "object_bbx_center" not in cav_content:
                continue
            boxes = self._numpy(cav_content["object_bbx_center"])
            masks = self._numpy(cav_content["object_bbx_mask"])
            if boxes.ndim == 3:
                boxes = boxes[0]
            if masks.ndim == 2:
                masks = masks[0]
            boxes = boxes[masks.astype(bool)]
            if boxes.size == 0:
                continue
            corners = np.asarray(box_utils.boxes_to_corners_3d(boxes[:, :7], self.params["order"]))
            transform = self._numpy(
                cav_content.get(
                    "gt_transformation_matrix",
                    cav_content.get("transformation_matrix", np.eye(4)),
                )
            )
            homogeneous = np.concatenate(
                (corners, np.ones((*corners.shape[:2], 1), dtype=corners.dtype)),
                axis=-1,
            )
            corners = np.einsum("ij,nkj->nki", transform, homogeneous)[..., :3]
            all_corners.append(corners)
            all_labels.append(boxes[:, -1])
            all_ids.extend(list(cav_content.get("object_ids", []))[: len(boxes)])
        if not all_corners:
            return torch.zeros((0, 8, 3)), torch.zeros((0,), dtype=torch.long)
        corners = np.concatenate(all_corners, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        unique = [all_ids.index(value) for value in set(all_ids)]
        corners, labels = corners[unique], labels[unique]
        limit = np.asarray(gt_range, dtype=np.float64)
        valid = (
            (corners[:, :, :2] >= limit[:2])
            & (corners[:, :, :2] <= limit[3:5])
        ).all(axis=2).all(axis=1)
        return torch.from_numpy(corners[valid]).float(), torch.from_numpy(labels[valid]).long()


class V2V4RealDatasetMixin:
    """Raw initialization/retrieval shared by existing fusion classes."""

    _v2v4real_project_boxes = True

    def _init_v2v4real(self, params, visualize, train):
        self.params, self.visualize, self.train = params, visualize, train
        self.isSim = False
        self.pre_processor = self.post_processor = None
        key = "cav_lidar_range" if train else "cav_lidar_range_validation"
        self.gt_range = list(params["preprocess"].get(key, V2V4REAL_GT_RANGE))
        self.data_augmentor = (
            DataAugmentor(params["data_augment"], train)
            if "data_augment" in params else None
        )
        wild = params.get("wild_setting", {})
        self.seed = int(wild.get("seed", 0))
        self.async_flag = bool(wild.get("async", False))
        self.async_mode = str(wild.get("async_mode", "sim"))
        self.async_overhead = float(wild.get("async_overhead", 0))
        self.loc_err_flag = bool(wild.get("loc_err", False))
        self.xyz_noise_std = float(wild.get("xyz_std", 0))
        self.ryp_noise_std = float(wild.get("ryp_std", 0))
        self.data_size = float(wild.get("data_size", 0))
        self.transmission_speed = float(wild.get("transmission_speed", 27))
        self.backbone_delay = float(wild.get("backbone_delay", 0))
        if self.loc_err_flag:
            raise ValueError("Raw V2V4Real adapter does not apply 6-D pose noise")

        root_dir = params["root_dir"] if train else params["validate_dir"]
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(root_dir)
        self.dataset_root = os.path.realpath(root_dir)
        self.dataset_split = str(params.get("dataset_split", "train" if train else "val"))
        train_params = params.get("train_params", {})
        self.max_cav = int(train_params.get("max_cav", 3))
        self.include_infrastructure = bool(train_params.get("include_infrastructure", False))
        self.only_infrastructure = bool(train_params.get("only_infrastructure", False))
        self.specific_ego_id = train_params.get("specific_ego_id")
        self.all_yaml_files = {}
        self.object_types = self._prepare_object_types(
            params["preprocess"]["anchor_generator_config"]
        )
        self.build_class_name2int_map()
        self.scenario_folders = sorted(
            os.path.realpath(os.path.join(root_dir, name))
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        )
        self.reinitialize()

    def reinitialize(self):
        self.scenario_database, self.len_record = OrderedDict(), []
        for scene_index, scene_folder in enumerate(self.scenario_folders):
            cav_list = sorted(
                name for name in os.listdir(scene_folder)
                if os.path.isdir(os.path.join(scene_folder, name))
            )
            if self.only_infrastructure:
                cav_list = [name for name in cav_list if int(name) < 0]
            elif not self.include_infrastructure:
                cav_list = [name for name in cav_list if int(name) >= 0]
            if self.train:
                random.shuffle(cav_list)
            elif cav_list and int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]
            if self.specific_ego_id is not None:
                ego = str(self.specific_ego_id)
                if ego not in cav_list:
                    raise ValueError(f"specific_ego_id {ego} missing in {scene_folder}")
                cav_list.remove(ego)
                cav_list.insert(0, ego)
            cav_list = cav_list[: self.max_cav]
            if not cav_list:
                raise ValueError(f"No CAVs in {scene_folder}")

            scene = OrderedDict()
            ego_timestamps = None
            for cav_index, cav_id in enumerate(cav_list):
                path = os.path.join(scene_folder, cav_id)
                timestamps = sorted(
                    name[:-5] for name in os.listdir(path)
                    if name.endswith(".yaml")
                    and "additional" not in name
                    and "camera_gt" not in name
                )
                if cav_index == 0:
                    ego_timestamps = timestamps
                entries = OrderedDict(
                    (timestamp, {
                        "yaml": os.path.join(path, timestamp + ".yaml"),
                        "lidar": os.path.join(path, timestamp + ".pcd"),
                        "scenario_folder": scene_folder,
                    }) for timestamp in timestamps
                )
                entries["ego"] = cav_index == 0
                scene[cav_id] = entries
            self.scenario_database[scene_index] = scene
            previous = self.len_record[-1] if self.len_record else 0
            self.len_record.append(previous + len(ego_timestamps or []))

    def _scenario_index(self, idx):
        if isinstance(idx, tuple):
            return self.scenario_database[idx[0]], int(idx[1])
        if not isinstance(idx, int):
            raise TypeError("V2V4Real index must be int or (scene, frame)")
        for scene_index, end in enumerate(self.len_record):
            if idx < end:
                start = self.len_record[scene_index - 1] if scene_index else 0
                return self.scenario_database[scene_index], idx - start
        raise IndexError(idx)

    @staticmethod
    def _timestamp(scene, index):
        first = next(iter(scene.values()))
        timestamps = [key for key, value in first.items() if isinstance(value, Mapping)]
        return timestamps[index]

    def _raw_params(self, scene, cav_id, current_key, delayed_key, cur_ego_pose_flag):
        ego_id = next(name for name, content in scene.items() if content["ego"])
        cav = scene[cav_id]
        ego = scene[ego_id]
        current = self._load_yaml(cav[current_key]["yaml"])
        delayed = copy.deepcopy(self._load_yaml(cav[delayed_key]["yaml"]))
        current_ego = self._load_yaml(ego[current_key]["yaml"])
        delayed_ego = self._load_yaml(ego[delayed_key]["yaml"])
        cav_delay = _pose_matrix(delayed["lidar_pose"])
        cav_current = _pose_matrix(current["lidar_pose"])
        ego_current = _pose_matrix(current_ego["lidar_pose"])
        ego_delay = _pose_matrix(delayed_ego["lidar_pose"])
        ego_reference = ego_current if cur_ego_pose_flag else ego_delay
        delayed["vehicles"] = copy.deepcopy(current.get("vehicles", {}))
        delayed["lidar_pose_matrix"] = cav_delay
        delayed["current_lidar_pose_matrix"] = cav_current
        delayed["reference_lidar_pose_matrix"] = ego_reference
        delayed["lidar_pose"] = matrix_to_pose(cav_delay)
        if cav_id != ego_id:
            delayed["coalign_observed_lidar_pose"] = matrix_to_pose(cav_delay)
        delayed["transformation_matrix"] = np.linalg.solve(ego_reference, cav_delay)
        delayed["gt_transformation_matrix"] = np.linalg.solve(ego_current, cav_current)
        delayed["spatial_correction_matrix"] = (
            np.eye(4) if cur_ego_pose_flag else np.linalg.solve(ego_current, ego_delay)
        )
        delayed["scenario_folder"] = cav[current_key].get("scenario_folder")
        return delayed

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True,
                           load_lidar_data=False, load_camera_data=False):
        if not load_lidar_data and not load_camera_data:
            raise ValueError("V2V4Real retrieval requires LiDAR or camera data")
        scene, frame_index = self._scenario_index(idx)
        current_key = self._timestamp(scene, frame_index)
        result = OrderedDict()
        for cav_id, content in scene.items():
            if current_key not in content:
                continue
            timestamp_keys = [key for key, value in content.items() if isinstance(value, Mapping)]
            delay = self.time_delay_calculation(bool(content["ego"]))
            delayed_index = max(0, min(frame_index - int(delay), len(timestamp_keys) - 1))
            delayed_key = timestamp_keys[delayed_index]
            params = self._raw_params(scene, cav_id, current_key, delayed_key, cur_ego_pose_flag)
            result[cav_id] = OrderedDict(
                ego=bool(content["ego"]),
                time_delay=int(delay),
                current_timestamp=str(current_key),
                history_timestamp=str(delayed_key),
                sample_idx=str(idx),
                scenario_id=os.path.basename(os.path.normpath(params["scenario_folder"])),
                dataset_root=self.dataset_root,
                dataset_split=self.dataset_split,
                cav_id=int(cav_id),
                params=params,
            )
            if load_lidar_data:
                result[cav_id]["lidar_np"] = pcd_to_np(content[delayed_key]["lidar"])
        return result

    def get_item_single_car(self, selected_cav_base, ego_pose=None):
        params = selected_cav_base["params"]
        previous = params.get("_v2v4real_box_transform")
        previous_single = params.get("_v2v4real_single_label_pending")
        params["_v2v4real_box_transform"] = (
            params["gt_transformation_matrix"] if self._v2v4real_project_boxes else np.eye(4)
        )
        if getattr(self, "supervise_single", False) and self._v2v4real_project_boxes:
            params["_v2v4real_single_label_pending"] = True
        try:
            if ego_pose is None:
                return super().get_item_single_car(selected_cav_base)
            return super().get_item_single_car(selected_cav_base, ego_pose)
        finally:
            if previous is None:
                params.pop("_v2v4real_box_transform", None)
            else:
                params["_v2v4real_box_transform"] = previous
            if previous_single is None:
                params.pop("_v2v4real_single_label_pending", None)
            else:
                params["_v2v4real_single_label_pending"] = previous_single

    def _postprocessor(self, params, train):
        if params["postprocess"]["core_method"] != "VoxelPostprocessor":
            raise ValueError("Raw V2V4Real adapter currently requires VoxelPostprocessor")
        return V2V4RealVoxelPostprocessor(params["postprocess"], self.object_types, train)


class V2V4RealLateFusionDataset(V2V4RealDatasetMixin, LidarLateFusionDataset):
    _v2v4real_project_boxes = False

    def __init__(self, params, visualize=False, train=True, **kwargs):
        self._init_v2v4real(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = self._postprocessor(params, train)


class V2V4RealEarlyFusionDataset(V2V4RealDatasetMixin, LidarEarlyFusionDataset):
    communication_range = float("inf")

    def __init__(self, params, visualize=False, train=True, **kwargs):
        self._init_v2v4real(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = self._postprocessor(params, train)


class V2V4RealIntermediateFusionDataset(
    V2V4RealDatasetMixin, LidarIntermediateFusionDataset
):
    def generate_augment(self):
        sampled = super().generate_augment()
        self._sampled_point_augmentation = None
        fusion_args = self.params.get("fusion", {}).get("args", {})
        fusion_args = fusion_args if isinstance(fusion_args, Mapping) else {}
        if (self.train and not self.proj_first
                and fusion_args.get("correct_augmented_transforms", False)):
            from opencood.data_utils.augmentor.transform_consistency import (
                sampled_augmentation_matrix,
            )
            self._sampled_point_augmentation = sampled_augmentation_matrix(
                self.data_augmentor, *sampled)
        return sampled

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        transforms = super().get_pairwise_transformation(base_data_dict, max_cav)
        augmentation = getattr(self, "_sampled_point_augmentation", None)
        if augmentation is not None:
            from opencood.data_utils.augmentor.transform_consistency import (
                conjugate_pairwise_transforms,
            )
            transforms = conjugate_pairwise_transforms(transforms, augmentation)
        return transforms

    def __init__(self, params, visualize=False, train=True, **kwargs):
        self._init_v2v4real(params, visualize, train)
        fusion_args = params.get("fusion", {}).get("args", {})
        fusion_args = fusion_args if isinstance(fusion_args, Mapping) else {}
        self.proj_first = bool(fusion_args.get("proj_first", True))
        self.cur_ego_pose_flag = bool(fusion_args.get("cur_ego_pose_flag", True))
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = self._postprocessor(params, train)
        from opencood.data_utils.datasets.lidar.intermediate_fusion_dataset import (
            _geometry_scaffold_enabled,
            _geometry_scaffold_mask_size,
            _global_bev_footprint_enabled,
            phase59_requires_anchor_grid,
            phase59_requires_ego_targets,
            phase64_requires_single_agent_targets,
            teacher_preprocess_config,
        )
        self.geometry_scaffold_enabled = bool(train) and _geometry_scaffold_enabled(params)
        self.geometry_scaffold_mask_size = _geometry_scaffold_mask_size(params)
        self.global_bev_footprint_enabled = bool(train) and _global_bev_footprint_enabled(params)
        self.coalign_stage1 = None
        self.coalign_args = {}
        self.coalign_split = str(params.get("dataset_split", "train" if train else "val"))
        protocol = params.get("v2v4real_protocol", {})
        if not train and isinstance(protocol, Mapping):
            active_split = protocol.get("active_evaluation_split")
            if active_split is not None:
                self.coalign_split = str(active_split)
        if "box_align" in params:
            box_align = params["box_align"]
            if not isinstance(box_align, Mapping):
                raise TypeError("V2V4Real box_align must be a mapping")
            result_key = "train_result" if train else "val_result"
            if result_key not in box_align:
                raise KeyError(f"box_align.{result_key} is required")
            self.coalign_stage1 = load_stage1_artifact(box_align[result_key])
            self.coalign_args = dict(box_align.get("args", {}))
            self.coalign_split = str(box_align.get("split", self.coalign_split))
        self.phase59_stage1_precalc = bool(params.get("phase59_stage1_precalc", False))
        # Stage-1 inference must produce a frame for every raw CAV, including
        # the >70 m pairs that normal V2V4Real fusion deliberately excludes.
        # This opt-in is only used by the split-specific pre-calculation
        # configs; ordinary training/evaluation retains COM_RANGE=70.
        self.communication_range = (
            float("inf")
            if self.phase59_stage1_precalc
            and bool(params.get("phase59_stage1_precalc_all_agents", False))
            else 70.0
        )
        self.kd_flag = bool(params.get("kd_flag", False))
        self.teacher_pre_processor = None
        self.teacher_lidar_range = None
        if self.kd_flag:
            teacher = teacher_preprocess_config(params)
            self.teacher_pre_processor = build_preprocessor(teacher, train)
            self.teacher_lidar_range = teacher["cav_lidar_range"]
        self.phase59_anchor_grid = phase59_requires_anchor_grid(params)
        self.phase59_ego_targets = phase59_requires_ego_targets(params)
        self.supervise_single = phase64_requires_single_agent_targets(params)


V2V4REAL_DATASETS = {
    "LidarLateFusionDataset": V2V4RealLateFusionDataset,
    "LidarEarlyFusionDataset": V2V4RealEarlyFusionDataset,
    "LidarIntermediateFusionDataset": V2V4RealIntermediateFusionDataset,
}
