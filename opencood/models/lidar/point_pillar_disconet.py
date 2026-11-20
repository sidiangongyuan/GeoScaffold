from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from opencood.models.fusion_modules.disconet_fuse import DiscoNetFusion
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.geometry_scaffold import GeometryScaffoldMixin
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


def _anchor_count(args: Mapping[str, object]) -> int:
    if "anchor_number" in args:
        return int(args["anchor_number"])
    if "anchor_num" in args:
        return int(args["anchor_num"])
    raise KeyError("Model args must define anchor_number or anchor_num.")


def _feature_channels(args: Mapping[str, object], backbone: BaseBEVBackbone) -> int:
    if "shrink_header" in args:
        shrink = args["shrink_header"]
        if not isinstance(shrink, Mapping):
            raise TypeError("shrink_header must be a mapping.")
        return int(shrink["dim"][-1])
    return int(backbone.num_bev_features)


def _load_checkpoint(module: nn.Module, checkpoint_path: str | Path, strict: bool) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break
    module.load_state_dict(checkpoint, strict=strict)


class _PointPillarDiscoTeacher(nn.Module):
    """Frozen teacher branch used by DiscoNet feature distillation."""

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__()
        anchor_num = _anchor_count(args)
        num_class = int(args["num_class"])
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)
        self.shrink_flag = "shrink_header" in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        self.out_channels = _feature_channels(args, self.backbone)
        self.cls_head = nn.Conv2d(
            self.out_channels,
            anchor_num * num_class * num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            self.out_channels,
            7 * anchor_num * num_class,
            kernel_size=1,
        )

    def _encode(self, processed_lidar: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        feature = batch_dict["spatial_features_2d"]
        if self.shrink_flag:
            feature = self.shrink_conv(feature)
        return feature

    def forward(self, data_dict: Mapping[str, object]) -> dict[str, torch.Tensor]:
        feature = self._encode(data_dict["teacher_processed_lidar"])
        teacher_psm = self.cls_head(feature)
        teacher_rm = self.reg_head(feature)
        return {
            "teacher_feature": feature,
            "teacher_psm": teacher_psm,
            "teacher_rm": teacher_rm,
            "teacher_cls_preds": teacher_psm,
            "teacher_reg_preds": teacher_rm,
        }


class PointPillarDiscoNet(GeometryScaffoldMixin, nn.Module):
    """DiscoNet intermediate fusion with optional frozen feature distillation."""

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__()
        self.max_cav = int(args["max_cav"])
        self.num_class = int(args["num_class"])
        self.anchor_num = _anchor_count(args)

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.shrink_flag = "shrink_header" in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        self.out_channels = _feature_channels(args, self.backbone)

        compression = int(args.get("compression", 0))
        self.compression = compression > 0
        if self.compression:
            self.naive_compressor = NaiveCompressor(self.out_channels, compression)

        fusion_args = dict(args["disconet_fusion"])
        fusion_args.setdefault("in_channels", self.out_channels)
        self.fusion_net = DiscoNetFusion(fusion_args)
        self.cls_head = nn.Conv2d(
            self.out_channels,
            self.anchor_num * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            self.out_channels,
            7 * self.anchor_num * self.num_class,
            kernel_size=1,
        )
        self.init_geometry_scaffold(args, in_channels=self.out_channels, num_classes=self.num_class)

        self.teacher_load_report = {
            "enabled": False,
            "checkpoint_path": None,
            "strict": False,
            "loaded": False,
        }
        self.teacher_model = self._build_teacher(args)
        teacher_channels = (
            self.out_channels
            if self.teacher_model is None
            else int(self.teacher_model.out_channels)
        )
        if teacher_channels == self.out_channels:
            self.kd_student_adapter = nn.Identity()
        else:
            self.kd_student_adapter = nn.Conv2d(
                self.out_channels,
                teacher_channels,
                kernel_size=1,
                bias=False,
            )
        if bool(args.get("backbone_fix", False)):
            self.backbone_fix()

    def _build_teacher(self, args: Mapping[str, object]) -> _PointPillarDiscoTeacher | None:
        teacher_root = args.get("teacher", {})
        if teacher_root is None:
            teacher_root = {}
        if not isinstance(teacher_root, Mapping):
            raise TypeError("model.args.teacher must be a mapping when provided.")
        enabled = bool(teacher_root.get("enabled", False))
        teacher_model_config = teacher_root.get("model_config", args)
        teacher_path = teacher_root.get("checkpoint_path")
        if teacher_path is not None:
            enabled = True
        if not enabled:
            return None
        if not isinstance(teacher_model_config, Mapping):
            raise TypeError("teacher.model_config must be a mapping.")
        teacher = _PointPillarDiscoTeacher(teacher_model_config)
        strict = bool(teacher_root.get("strict", False))
        if teacher_path is not None:
            _load_checkpoint(teacher, teacher_path, strict=strict)
        self.teacher_load_report = {
            "enabled": True,
            "checkpoint_path": None if teacher_path is None else str(teacher_path),
            "strict": strict,
            "loaded": teacher_path is not None,
        }
        if bool(teacher_root.get("freeze", True)):
            teacher.requires_grad_(False)
            teacher.eval()
        return teacher

    def train(self, mode: bool = True) -> "PointPillarDiscoNet":
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    def backbone_fix(self) -> None:
        for module in (
            self.pillar_vfe,
            self.scatter,
            self.backbone,
            self.cls_head,
            self.reg_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if self.shrink_flag:
            for parameter in self.shrink_conv.parameters():
                parameter.requires_grad = False
        if self.compression:
            for parameter in self.naive_compressor.parameters():
                parameter.requires_grad = False

    def _encode_processed_lidar(
        self,
        processed_lidar: Mapping[str, torch.Tensor],
        record_len: torch.Tensor,
    ) -> torch.Tensor:
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict["spatial_features_2d"]
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        return spatial_features_2d

    @staticmethod
    def _ensure_bchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D fused feature map, got {tuple(feature.shape)}.")
        if feature.shape[1] == expected_channels:
            return feature.contiguous()
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "Unable to align fused feature channels for DiscoNet: "
            f"expected {expected_channels}, got {tuple(feature.shape)}."
        )

    def extract_features(self, data_dict: Mapping[str, object]) -> torch.Tensor:
        return self._encode_processed_lidar(data_dict["processed_lidar"], data_dict["record_len"])

    def _decode_heads(self, fused_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        return {
            "feature": fused_feature,
            "psm": psm,
            "rm": rm,
            "cls_preds": psm,
            "reg_preds": rm,
        }

    def fuse_features(self, data: Mapping[str, object]) -> dict[str, torch.Tensor]:
        packed_bev = data["bev"]
        record_len = data["record_len"]
        pairwise_t_matrix = data["pairwise_t_matrix"]
        fused_feature = self._ensure_bchw(
            self.fusion_net(packed_bev, record_len, pairwise_t_matrix),
            self.out_channels,
        )
        output = self._decode_heads(fused_feature)
        output["student_kd_feature"] = self.kd_student_adapter(fused_feature)
        output.update(
            self.build_geometry_scaffold_output(
                packed_bev,
                fused_feature,
                record_len,
                data,
            )
        )
        return output

    def forward(self, data_dict: Mapping[str, object]) -> dict[str, torch.Tensor]:
        packed_bev = self.extract_features(data_dict)
        output = self.fuse_features(
            {
                **data_dict,
                "bev": packed_bev,
            }
        )
        if self.teacher_model is not None and "teacher_processed_lidar" in data_dict:
            with torch.no_grad():
                output.update(self.teacher_model(data_dict))
        return output
