"""Homogeneous LiDAR PointPillar model with independent HEAL pyramid fusion."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.geometry_scaffold import GeometryScaffoldMixin
from opencood.models.sub_modules.heal_pyramid import HEALPyramidFusion
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarHeal(GeometryScaffoldMixin, nn.Module):
    """PointPillar detector using HEAL's homogeneous pyramid fusion path.

    The first backbone creates a packed 64-channel feature map at the normal
    PointPillar intermediate-fusion stride.  HEALPyramidFusion then keeps the
    three scales separate while agents are fused and decodes them to the
    repository's 256-channel detection feature.  GeoScaffold is attached only
    after this final fused feature and therefore remains training-only.
    """

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__()
        if not isinstance(args, Mapping):
            raise TypeError("PointPillarHeal args must be a mapping.")

        self.max_cav = int(args["max_cav"])
        self.num_class = int(args["num_class"])
        self.anchor_number = int(args.get("anchor_number", args.get("anchor_num", 2)))
        if self.anchor_number <= 0 or self.num_class <= 0:
            raise ValueError("anchor_number and num_class must be positive.")

        voxel_size = args["voxel_size"]
        lidar_range = args["lidar_range"]
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=voxel_size,
            point_cloud_range=lidar_range,
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])

        base_cfg = args.get("base_bev_backbone", args.get("backbone_args"))
        if not isinstance(base_cfg, Mapping):
            raise ValueError("PointPillarHeal requires base_bev_backbone configuration.")
        self.backbone = BaseBEVBackbone(base_cfg, 64)
        # BaseBEVBackbone reports zero when it intentionally omits decoder
        # blocks.  HEAL's first stage uses that packed last level directly.
        if base_cfg.get("upsample_strides"):
            packed_channels = int(tuple(base_cfg["num_upsample_filter"])[-1])
        else:
            packed_channels = int(tuple(base_cfg["num_filters"])[-1])

        pyramid_cfg = args.get("fusion_backbone")
        if not isinstance(pyramid_cfg, Mapping):
            raise ValueError("PointPillarHeal requires fusion_backbone configuration.")
        pyramid_input_channels = int(tuple(pyramid_cfg.get("num_filters", (64,)))[0])
        if packed_channels != pyramid_input_channels:
            raise ValueError(
                "HEAL's single-agent backbone output must match the first pyramid "
                f"level ({packed_channels} versus {pyramid_input_channels})."
            )
        self.compression = False
        if int(args.get("compression", 0)) > 0:
            raise ValueError("HEAL does not support compression before pyramid fusion.")

        in_head = int(args.get("in_head", 256))
        self.fusion_net = HEALPyramidFusion(
            dict(pyramid_cfg),
            input_channels=packed_channels,
            output_channels=in_head,
            discrete_ratio=float(voxel_size[0]),
            downsample_rate=float(args.get("fusion_downsample_rate", 2.0)),
        )
        self.fused_channels = in_head
        self.cls_head = nn.Conv2d(
            self.fused_channels,
            self.anchor_number * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            self.fused_channels,
            7 * self.anchor_number * self.num_class,
            kernel_size=1,
        )

        # The released HEAL detector has a direction head, but the historical
        # UrbanIng port pre-dates that part of the objective.  Keep the head
        # behind an explicit protocol switch so loading an old HEAL config
        # remains graph- and checkpoint-compatible.
        # ``native_detection_objective`` is the loss-side protocol switch.
        # A non-empty model ``dir_args`` is itself the explicit model-side
        # opt-in, matching the released HEAL/CoAlign configuration shape.
        native_objective = args.get(
            "native_detection_objective",
            args.get("native_objective", False),
        )
        if isinstance(native_objective, Mapping):
            native_objective = native_objective.get("enabled", True)
        direction_args = args.get("dir_args")
        has_explicit_objective = (
            "native_detection_objective" in args or "native_objective" in args
        )
        self.native_objective = bool(native_objective) if has_explicit_objective else bool(
            direction_args
        )
        self.native_detection_objective = self.native_objective
        direction_cfg = direction_args
        self.use_direction = self.native_objective and isinstance(
            direction_cfg, Mapping
        )
        if self.use_direction:
            num_bins = int(direction_cfg.get("num_bins", 2))
            if num_bins <= 0:
                raise ValueError("dir_args.num_bins must be positive.")
            self.direction_num_bins = num_bins
            self.dir_head = nn.Conv2d(
                self.fused_channels,
                num_bins * self.anchor_number * self.num_class,
                kernel_size=1,
            )
        else:
            self.direction_num_bins = 0

        self.init_geometry_scaffold(
            args,
            in_channels=self.fused_channels,
            num_classes=self.num_class,
        )
        self.geometry_scaffold_ego_adapter = None
        if self.geometry_scaffold_mode == "ego":
            self.geometry_scaffold_ego_adapter = nn.Conv2d(
                packed_channels,
                self.fused_channels,
                kernel_size=1,
            )

        if bool(args.get("backbone_fix", False)):
            self.backbone_fix()

    def backbone_fix(self) -> None:
        """Freeze the detector path for optional downstream fine-tuning."""
        for module in (
            self.pillar_vfe,
            self.scatter,
            self.backbone,
            self.fusion_net,
            self.cls_head,
            self.reg_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.use_direction:
            for parameter in self.dir_head.parameters():
                parameter.requires_grad_(False)

    def _encode_packed_features(self, data_dict: Mapping[str, object]) -> Tensor:
        processed_lidar = data_dict["processed_lidar"]
        voxel_features = processed_lidar["voxel_features"]
        voxel_coords = processed_lidar["voxel_coords"]
        voxel_num_points = processed_lidar["voxel_num_points"]
        record_len = data_dict["record_len"]
        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        return batch_dict["spatial_features_2d"]

    def extract_features(self, data_dict: Mapping[str, object]) -> Tensor:
        """Return packed pre-fusion features for the intermediate API."""
        return self._encode_packed_features(data_dict)

    def _fuse_and_decode(
        self,
        packed_features: Tensor,
        record_len: Tensor,
        pairwise_t_matrix: Tensor | None,
        data_dict: Mapping[str, object] | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        fused_feature, occupancy = self.fusion_net.forward_collab(
            packed_features,
            record_len,
            pairwise_t_matrix,
        )
        output_dict: dict[str, Tensor | list[Tensor]] = {
            "psm": self.cls_head(fused_feature),
            "rm": self.reg_head(fused_feature),
            "occ_single_list": occupancy,
            "heal_occ_logits": occupancy,
            "record_len": record_len,
        }
        if self.use_direction:
            direction = self.dir_head(fused_feature)
            # Keep both names used by the repository's native CoAlign path.
            # They are aliases, not a second head or computation.
            output_dict["dir_preds"] = direction
            output_dict["dm"] = direction
        if data_dict is not None:
            scaffold_features = packed_features
            if self.geometry_scaffold_ego_adapter is not None:
                scaffold_features = self.geometry_scaffold_ego_adapter(packed_features)
            output_dict.update(
                self.build_geometry_scaffold_output(
                    scaffold_features,
                    fused_feature,
                    record_len,
                    data_dict,
                )
            )
        return output_dict

    def fuse_features(self, data: Mapping[str, object]):
        """Fuse precomputed packed BEV features for the intermediate API."""
        return self._fuse_and_decode(
            data["bev"],
            data["record_len"],
            data.get("pairwise_t_matrix"),
            data,
        )

    def forward(self, data_dict: Mapping[str, object]):
        packed_features = self._encode_packed_features(data_dict)
        return self._fuse_and_decode(
            packed_features,
            data_dict["record_len"],
            data_dict.get("pairwise_t_matrix"),
            data_dict,
        )
