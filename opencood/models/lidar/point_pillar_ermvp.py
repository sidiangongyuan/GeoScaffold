from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from einops import repeat

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.cluster import cluster_dpc_knn, index_points, merge_tokens
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.ermvp_fusion_modules import ERMVPFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.geometry_scaffold import GeometryScaffoldMixin
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.sampler import SortSampler


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


def _ego_features(packed_bev: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Every coalition must contain an ego feature.")
    if sum(lengths) != packed_bev.shape[0]:
        raise ValueError(
            f"sum(record_len)={sum(lengths)} does not match BEV count {packed_bev.shape[0]}."
        )
    starts = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length
    return packed_bev.index_select(
        0,
        torch.tensor(starts, device=packed_bev.device, dtype=torch.long),
    )


def _agent_local_indices(record_len: torch.Tensor) -> list[int]:
    local_indices: list[int] = []
    for length in record_len.detach().cpu().tolist():
        local_indices.extend(range(int(length)))
    return local_indices


class PointPillarErmvp(GeometryScaffoldMixin, nn.Module):
    """ERMVP fusion with official auxiliary ego decoder outputs."""

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

        self.fusion_net = ERMVPFusionEncoder(args["ermvp_fusion"])
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

        comm_args = args["comm"]
        self.topk_ratio = float(comm_args["topk_ratio"])
        self.cluster_sample_ratio = float(comm_args["cluster_sample_ratio"])
        self.sampler = SortSampler(
            topk_ratio=self.topk_ratio,
            input_dim=self.out_channels,
            score_pred_net="2layer-fc-256",
        )

        if bool(args.get("backbone_fix", False)):
            self.backbone_fix()

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

    def extract_features(self, data_dict: Mapping[str, object]) -> torch.Tensor:
        return self._encode_processed_lidar(data_dict["processed_lidar"], data_dict["record_len"])

    def _apply_official_sampler(
        self,
        spatial_features_2d: torch.Tensor,
        record_len: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cav_count, channels, height, width = spatial_features_2d.shape
        device = spatial_features_2d.device
        dis_priority = torch.ones(
            (cav_count, height, width),
            device=device,
            dtype=spatial_features_2d.dtype,
        )
        pos_index = (
            torch.arange(height * width, device=device, dtype=torch.long)
            .view(height * width, 1, 1)
            .expand(height * width, cav_count, 1)
        )
        sampled_src, _, sort_confidence_topk, pos_embed = self.sampler(
            spatial_features_2d,
            pos_index,
            None,
            dis_priority,
        )
        sampled_src = sampled_src.permute(1, 0, 2)
        _, sampled_len, _ = sampled_src.shape
        cluster_num = max(int(math.ceil(sampled_len * self.cluster_sample_ratio)), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(sampled_src, cluster_num, 10)
        merged_tokens, idx = merge_tokens(
            sampled_src,
            idx_cluster,
            cluster_num,
            sort_confidence_topk.unsqueeze(2),
        )
        row_index = torch.arange(sampled_len, device=device)
        gathered_idx = torch.vstack([idx[b][row_index] for b in range(cav_count)])
        sampled_src = index_points(merged_tokens, gathered_idx).permute(0, 2, 1)
        pos_embed = pos_embed.permute(1, 2, 0)

        reconstructed = []
        local_indices = _agent_local_indices(record_len)
        for cav_idx in range(cav_count):
            spatial_feature = torch.zeros(
                channels,
                height * width,
                device=device,
                dtype=sampled_src.dtype,
            )
            spatial_feature[:, pos_embed[cav_idx][0].long()] = sampled_src[cav_idx]
            if local_indices[cav_idx] == 0:
                spatial_feature = spatial_features_2d[cav_idx].flatten(1)
            reconstructed.append(spatial_feature.view(channels, height, width))
        reconstructed_bev = torch.stack(reconstructed, dim=0)
        stats = {
            "ermvp_sparse_path_executed": torch.ones(
                (),
                device=device,
                dtype=spatial_features_2d.dtype,
            ),
            "ermvp_sampled_token_count": torch.tensor(
                sampled_len,
                device=device,
                dtype=spatial_features_2d.dtype,
            ),
            "ermvp_cluster_token_count": torch.tensor(
                cluster_num,
                device=device,
                dtype=spatial_features_2d.dtype,
            ),
        }
        return reconstructed_bev, stats

    def _decode(
        self,
        packed_bev: torch.Tensor,
        record_len: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sampled_bev, sampler_stats = self._apply_official_sampler(packed_bev, record_len)
        ego_feature = _ego_features(sampled_bev, record_len)
        regroup_feature, mask = regroup(sampled_bev, record_len, self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(
            com_mask,
            "b h w c l -> b (h new_h) (w new_w) c l",
            new_h=regroup_feature.shape[3],
            new_w=regroup_feature.shape[4],
        )
        fused_feature = self.fusion_net(regroup_feature, com_mask)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        ego_psm = self.cls_head(ego_feature)
        ego_rm = self.reg_head(ego_feature)
        output = {
            "psm": psm,
            "rm": rm,
            "ego_psm": ego_psm,
            "ego_rm": ego_rm,
            "psm_ego": ego_psm,
            "rm_ego": ego_rm,
        }
        output.update(sampler_stats)
        return output, fused_feature

    def fuse_features(self, data: Mapping[str, object]) -> dict[str, torch.Tensor]:
        output, fused_feature = self._decode(data["bev"], data["record_len"])
        output.update(
            self.build_geometry_scaffold_output(
                data["bev"],
                fused_feature,
                data["record_len"],
                data,
            )
        )
        return output

    def forward(self, data_dict: Mapping[str, object]) -> dict[str, torch.Tensor]:
        packed_bev = self.extract_features(data_dict)
        output, fused_feature = self._decode(packed_bev, data_dict["record_len"])
        output.update(
            self.build_geometry_scaffold_output(
                packed_bev,
                fused_feature,
                data_dict["record_len"],
                data_dict,
            )
        )
        return output
