"""Author-faithful native CoAlign PointPillar path for V2V4Real.

This module mirrors the public CoAlign implementation: a residual BEV
backbone produces three scales, each scale is warped with the non-identity
pairwise pose and fused by pixel-wise attention, and the decoded feature is
sent to the detector heads.  GeoScaffold is attached only after this native
fused feature and is therefore absent from evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from opencood.models.fusion_modules.coalign_fuse import (
    Att_w_Warp,
    normalize_pairwise_tfm,
)
from opencood.models.sub_modules.base_bev_backbone_resnet_coalign import (
    ResNetBEVBackbone,
)
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.geometry_scaffold import GeometryScaffoldMixin
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


def _as_int(value: object, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


class PointPillarCoAlignNative(GeometryScaffoldMixin, nn.Module):
    """PointPillar CoAlign with author-style multiscale residual fusion.

    The detector interface intentionally retains this repository's ``psm`` /
    ``rm`` keys.  ``cls_preds`` / ``reg_preds`` are aliases for compatibility
    with the public CoAlign loss and do not create a second computation.
    """

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__()
        if not isinstance(args, Mapping):
            raise TypeError("PointPillarCoAlignNative args must be a mapping")

        num_class = _as_int(args["num_class"], "num_class")
        self.num_class = num_class
        self.anchor_number = _as_int(
            args.get("anchor_number", args.get("anchor_num", 2)),
            "anchor_number",
        )
        self.voxel_size = tuple(float(value) for value in args["voxel_size"])

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])

        base_cfg = args.get("base_bev_backbone")
        if not isinstance(base_cfg, Mapping):
            raise ValueError("PointPillarCoAlignNative requires base_bev_backbone")
        self.backbone = ResNetBEVBackbone(base_cfg, 64)

        layer_count = len(tuple(base_cfg.get("layer_nums", ())))
        if layer_count <= 0:
            raise ValueError("CoAlign requires at least one residual BEV level")
        feature_dims = tuple(base_cfg["num_filters"])
        attention_cfg = args.get("att", {})
        if isinstance(attention_cfg, Mapping) and attention_cfg.get("feat_dim"):
            feature_dims = tuple(attention_cfg["feat_dim"])
        if len(feature_dims) != layer_count:
            raise ValueError(
                "CoAlign attention dimensions must match backbone levels: "
                f"{len(feature_dims)} versus {layer_count}"
            )
        self.fusion_net = nn.ModuleList(
            Att_w_Warp(_as_int(dim, "attention feature dimension"))
            for dim in feature_dims
        )

        self.compression = False
        compression = int(args.get("compression", 0) or 0)
        if compression > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, compression)

        out_channels = int(sum(base_cfg["num_upsample_filter"]))
        shrink_cfg = args.get("shrink_header")
        self.shrink_flag = isinstance(shrink_cfg, Mapping)
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(shrink_cfg)
            out_channels = int(tuple(shrink_cfg["dim"])[-1])
        self.out_channels = out_channels

        self.cls_head = nn.Conv2d(
            out_channels,
            self.anchor_number * num_class * num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            out_channels,
            7 * self.anchor_number * num_class,
            kernel_size=1,
        )

        direction_cfg = args.get("dir_args")
        self.use_direction = isinstance(direction_cfg, Mapping)
        if self.use_direction:
            num_bins = _as_int(direction_cfg.get("num_bins", 2), "dir_args.num_bins")
            self.direction_num_bins = num_bins
            self.dir_head = nn.Conv2d(
                out_channels,
                num_bins * self.anchor_number * num_class,
                kernel_size=1,
            )
        else:
            self.direction_num_bins = 0

        self.init_geometry_scaffold(
            args,
            in_channels=out_channels,
            num_classes=num_class,
        )
        self._phase59_last_fused_feature: Tensor | None = None
        # One-based activation; optimizer and data order continue without a reset.
        self.geometry_scaffold_start_epoch = _as_int(
            args.get("geometry_scaffold", {}).get("start_epoch", 1), "GeoScaffold start_epoch"
        )
        self.set_training_epoch(0)

    def set_training_epoch(self, completed_epochs: int) -> None:
        self.geometry_scaffold_active = completed_epochs + 1 >= self.geometry_scaffold_start_epoch

    def build_geometry_scaffold_output(self, *args, **kwargs) -> dict[str, Tensor]:
        if not self.geometry_scaffold_active:
            return {}
        return super().build_geometry_scaffold_output(*args, **kwargs)

    def _encode(self, data_dict: Mapping[str, object]) -> tuple[Tensor, Tensor]:
        processed_lidar = data_dict["processed_lidar"]
        if not isinstance(processed_lidar, Mapping):
            raise TypeError("processed_lidar must be a mapping")
        record_len = data_dict["record_len"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        return batch_dict["spatial_features"], record_len

    def forward(self, data_dict: Mapping[str, object]) -> dict[str, Tensor]:
        spatial_features, record_len = self._encode(data_dict)
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        _, _, height, width = spatial_features.shape
        pairwise = data_dict["pairwise_t_matrix"].clone()
        normalized_tfm = normalize_pairwise_tfm(
            pairwise,
            height,
            width,
            self.voxel_size[0],
        )

        # The order is intentional: every residual level is fused before its
        # upsampling block, exactly as in the public CoAlign implementation.
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = [
            fusion(feature, record_len, normalized_tfm)
            for feature, fusion in zip(feature_list, self.fusion_net)
        ]
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        self._phase59_last_fused_feature = fused_feature

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output: dict[str, Tensor] = {
            "psm": psm,
            "rm": rm,
            "cls_preds": psm,
            "reg_preds": rm,
        }
        if self.use_direction:
            direction = self.dir_head(fused_feature)
            output["dir_preds"] = direction
            output["dm"] = direction

        output.update(
            self.build_geometry_scaffold_output(
                spatial_features,
                fused_feature,
                record_len,
                data_dict,
            )
        )
        return output
