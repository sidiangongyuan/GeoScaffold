# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.geometry_scaffold import (
    GeometryScaffoldHead,
    resolve_geometry_scaffold_config,
    sample_anchor_rois,
)
from opencood.models.fusion_modules.self_attn import AttFusion


class SignedConditionalEvidenceBooster(nn.Module):
    """Calibrate frozen ego/full output differences without changing their sign."""

    def __init__(
        self,
        class_channels,
        box_channels,
        hidden_channels=64,
        mode='signed',
    ):
        super().__init__()
        if mode not in {'signed', 'free'}:
            raise ValueError(f'Unknown evidence booster mode {mode!r}.')
        if class_channels <= 0 or box_channels <= 0 or hidden_channels <= 0:
            raise ValueError('Evidence booster channel counts must be positive.')
        self.mode = mode
        context_channels = 4 * class_channels + box_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.support_head = nn.Conv2d(hidden_channels, class_channels, kernel_size=1)
        self.veto_head = nn.Conv2d(hidden_channels, class_channels, kernel_size=1)
        self.box_head = nn.Conv2d(hidden_channels, box_channels, kernel_size=1)
        for head in (self.support_head, self.veto_head, self.box_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _gain(logits):
        return F.softplus(logits) / math.log(2.0)

    def forward(
        self,
        ego_psm,
        full_psm,
        ego_rm,
        full_rm,
        has_collaborator,
    ):
        if ego_psm.shape != full_psm.shape or ego_rm.shape != full_rm.shape:
            raise ValueError('Ego and full output tensors must have matching shapes.')
        if ego_psm.shape[0] != ego_rm.shape[0]:
            raise ValueError('Classification and box batches must match.')
        if has_collaborator.ndim != 1 \
                or has_collaborator.shape[0] != ego_psm.shape[0]:
            raise ValueError('has_collaborator must contain one value per sample.')

        positive_evidence = F.relu(full_psm - ego_psm)
        negative_evidence = F.relu(ego_psm - full_psm)
        box_evidence = full_rm - ego_rm
        context = torch.cat(
            (
                ego_psm,
                full_psm,
                positive_evidence,
                negative_evidence,
                box_evidence.abs(),
            ),
            dim=1,
        )
        hidden = self.trunk(context)
        support_output = self.support_head(hidden)
        veto_output = self.veto_head(hidden)
        box_output = self.box_head(hidden)
        mask = has_collaborator.to(
            device=ego_psm.device,
            dtype=torch.bool,
        ).view(-1, 1, 1, 1)

        if self.mode == 'signed':
            support_gain = self._gain(support_output)
            veto_gain = self._gain(veto_output)
            box_gain = self._gain(box_output)
            # This form is algebraically identical to ego + A+E+ - A-E-,
            # while making zero-head initialization bitwise equal to full.
            calibrated_psm = full_psm \
                + (support_gain - 1.0) * positive_evidence \
                - (veto_gain - 1.0) * negative_evidence
            calibrated_rm = full_rm + (box_gain - 1.0) * box_evidence
            return {
                'psm': torch.where(mask, calibrated_psm, ego_psm),
                'rm': torch.where(mask, calibrated_rm, ego_rm),
                'support_gain': torch.where(mask, support_gain, torch.ones_like(support_gain)),
                'veto_gain': torch.where(mask, veto_gain, torch.ones_like(veto_gain)),
                'box_gain': torch.where(mask, box_gain, torch.ones_like(box_gain)),
            }

        calibrated_psm = full_psm + support_output - veto_output
        calibrated_rm = full_rm + box_output
        return {
            'psm': torch.where(mask, calibrated_psm, ego_psm),
            'rm': torch.where(mask, calibrated_rm, ego_rm),
            'support_residual': torch.where(
                mask,
                support_output,
                torch.zeros_like(support_output),
            ),
            'veto_residual': torch.where(
                mask,
                veto_output,
                torch.zeros_like(veto_output),
            ),
            'box_residual': torch.where(
                mask,
                box_output,
                torch.zeros_like(box_output),
            ),
        }


class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()
        num_class = args['num_class']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = AttFusion(args['base_bev_backbone']['num_filters'][-1])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'] * num_class * num_class,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_num'] * num_class,
                                  kernel_size=1)

        counterfactual = args.get('counterfactual_training', {})
        self.counterfactual_mode = str(
            counterfactual.get('mode', 'disabled')
        ).lower()
        valid_modes = {
            'disabled',
            'ego_auxiliary',
            'interventional_residual',
            'signed_evidence_booster',
            'free_residual_control',
            'ego_geoscaffold',
            'fused_geoscaffold',
        }
        if self.counterfactual_mode not in valid_modes:
            raise ValueError(
                f'Unknown counterfactual training mode '
                f'{self.counterfactual_mode!r}; expected one of '
                f'{sorted(valid_modes)}.'
            )
        self.counterfactual_detach_base = bool(
            counterfactual.get('detach_base', True)
        )
        self.freeze_base_for_booster = bool(
            counterfactual.get('freeze_base', False)
        )
        self.geoscaffold_modes = {'ego_geoscaffold', 'fused_geoscaffold'}
        geometry_scaffold_config = resolve_geometry_scaffold_config(args)
        self.geometry_scaffold_auxiliary = str(
            geometry_scaffold_config['auxiliary']
        )
        self.geometry_scaffold_mask_size = int(
            geometry_scaffold_config['mask_size']
        )
        self.geometry_scaffold_roi_size = int(
            geometry_scaffold_config['roi_size']
        )
        self.lidar_range = tuple(float(value) for value in args['lidar_range'])
        if self.counterfactual_mode == 'interventional_residual':
            # A bias-free decoder makes the residual identically zero whenever
            # the collaboration intervention leaves the BEV feature unchanged.
            self.residual_cls_head = nn.Conv2d(
                self.cls_head.in_channels,
                self.cls_head.out_channels,
                kernel_size=1,
                bias=False,
            )
            self.residual_reg_head = nn.Conv2d(
                self.reg_head.in_channels,
                self.reg_head.out_channels,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(self.residual_cls_head.weight)
            nn.init.zeros_(self.residual_reg_head.weight)
        if self.counterfactual_mode in {
            'signed_evidence_booster',
            'free_residual_control',
        }:
            if not self.freeze_base_for_booster:
                raise ValueError(
                    'Phase-36 evidence booster modes require freeze_base=true.'
                )
            self.evidence_booster = SignedConditionalEvidenceBooster(
                class_channels=self.cls_head.out_channels,
                box_channels=self.reg_head.out_channels,
                hidden_channels=int(counterfactual.get('hidden_channels', 64)),
                mode=(
                    'signed'
                    if self.counterfactual_mode == 'signed_evidence_booster'
                    else 'free'
                ),
            )
        if (
            self.counterfactual_mode == 'disabled'
            and geometry_scaffold_config['mode'] != 'disabled'
        ):
            self.counterfactual_mode = f"{geometry_scaffold_config['mode']}_geoscaffold"
        if self.counterfactual_mode in self.geoscaffold_modes:
            self.geoscaffold_head = GeometryScaffoldHead(
                in_channels=256,
                num_classes=num_class,
                hidden_channels=int(geometry_scaffold_config['hidden_channels']),
                roi_size=self.geometry_scaffold_roi_size,
                mask_size=self.geometry_scaffold_mask_size,
                checkpoint_chunk_size=int(
                    geometry_scaffold_config['checkpoint_chunk_size']
                ),
                auxiliary=self.geometry_scaffold_auxiliary,
            )
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        if self.counterfactual_mode in {
            'signed_evidence_booster',
            'free_residual_control',
        }:
            self._freeze_base_for_evidence_booster()


    def _freeze_base_for_evidence_booster(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('evidence_booster.'))
        self._keep_frozen_base_in_eval()

    def _keep_frozen_base_in_eval(self):
        for name, module in self.named_children():
            if name != 'evidence_booster':
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'freeze_base_for_booster', False):
            self._keep_frozen_base_in_eval()
            self.evidence_booster.train(mode)
        return self


    def extract_features(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']


        # plt.matshow(spatial_features_2d[0, ...].abs().sum(0).detach().cpu().numpy())
        # plt.savefig(f"{agent_name}_online.png")
        # plt.show()

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(
                spatial_features_2d, "encoder")

        return spatial_features_2d
    @staticmethod
    def _ego_features(
        packed_bev: torch.Tensor,
        record_len: torch.Tensor,
    ) -> torch.Tensor:
        if record_len.ndim != 1:
            raise ValueError(
                f'record_len must be one-dimensional, got '
                f'{tuple(record_len.shape)}.'
            )
        lengths = [int(value) for value in record_len.detach().cpu().tolist()]
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError('Every coalition must contain an ego feature.')
        if sum(lengths) != packed_bev.shape[0]:
            raise ValueError(
                f'sum(record_len)={sum(lengths)} does not match packed BEV '
                f'count {packed_bev.shape[0]}.'
            )
        starts = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
        indices = torch.tensor(starts, device=packed_bev.device, dtype=torch.long)
        return packed_bev.index_select(0, indices)

    def predict_from_fused_features(
        self,
        packed_bev: torch.Tensor,
        fused_feature: torch.Tensor,
        record_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode native fusion, optionally as an ego-conditioned residual."""
        geoscaffold_modes = getattr(self, 'geoscaffold_modes', set())
        if self.counterfactual_mode == 'disabled' \
                or self.counterfactual_mode in geoscaffold_modes:
            return {
                'psm': self.cls_head(fused_feature),
                'rm': self.reg_head(fused_feature),
            }

        ego_feature = self._ego_features(packed_bev, record_len)
        if ego_feature.shape != fused_feature.shape:
            raise ValueError(
                f'Ego and fused features must have identical shapes, got '
                f'{tuple(ego_feature.shape)} and {tuple(fused_feature.shape)}.'
            )
        ego_psm = self.cls_head(ego_feature)
        ego_rm = self.reg_head(ego_feature)

        if self.counterfactual_mode == 'ego_auxiliary':
            return {
                'psm': self.cls_head(fused_feature),
                'rm': self.reg_head(fused_feature),
                'ego_psm': ego_psm,
                'ego_rm': ego_rm,
            }

        if self.counterfactual_mode in {
            'signed_evidence_booster',
            'free_residual_control',
        }:
            base_full_psm = self.cls_head(fused_feature)
            base_full_rm = self.reg_head(fused_feature)
            calibrated = self.evidence_booster(
                ego_psm,
                base_full_psm,
                ego_rm,
                base_full_rm,
                record_len > 1,
            )
            return {
                **calibrated,
                'ego_psm': ego_psm,
                'ego_rm': ego_rm,
                'base_full_psm': base_full_psm,
                'base_full_rm': base_full_rm,
            }

        delta_feature = fused_feature - ego_feature
        full_base_psm = ego_psm.detach() \
            if self.counterfactual_detach_base else ego_psm
        full_base_rm = ego_rm.detach() \
            if self.counterfactual_detach_base else ego_rm
        return {
            'psm': full_base_psm + self.residual_cls_head(delta_feature),
            'rm': full_base_rm + self.residual_reg_head(delta_feature),
            'ego_psm': ego_psm,
            'ego_rm': ego_rm,
        }

    def fuse_features(self, data):
        spatial_features_2d = data['bev']
        record_len = data['record_len']
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d, "decoder")

        fused_feature = self.fusion_net(spatial_features_2d, record_len)
        return self.predict_from_fused_features(
            spatial_features_2d,
            fused_feature,
            record_len,
        )

    def build_geoscaffold_outputs(
        self,
        data_dict,
        packed_bev: torch.Tensor,
        fused_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        geoscaffold_modes = getattr(self, 'geoscaffold_modes', set())
        if self.counterfactual_mode not in geoscaffold_modes or not self.training:
            return {}
        if 'geoscaffold' not in data_dict:
            raise KeyError(
                'GeoScaffold training requires sparse positive-anchor targets.'
            )
        if self.counterfactual_mode == 'fused_geoscaffold':
            source_bev = fused_feature
        else:
            source_bev = self._ego_features(packed_bev, data_dict['record_len'])

        sparse_targets = data_dict['geoscaffold']
        sample_indices = sparse_targets['batch_indices'].to(
            device=source_bev.device,
            dtype=torch.long,
        )
        class_ids = sparse_targets['class_ids'].to(
            device=source_bev.device,
            dtype=torch.long,
        )
        anchor_boxes = sparse_targets['anchor_boxes_hwl'].to(
            device=source_bev.device,
            dtype=source_bev.dtype,
        )
        targets = sparse_targets['field_targets'].to(
            device=source_bev.device,
            dtype=torch.bool,
        )
        box_code_targets = sparse_targets.get('box_code_targets')
        if box_code_targets is not None:
            box_code_targets = box_code_targets.to(
                device=source_bev.device,
                dtype=source_bev.dtype,
            )
        if sample_indices.numel() == 0:
            if self.geometry_scaffold_auxiliary == 'box_code':
                return {
                    'geoscaffold_box_code_preds': source_bev.new_empty((0, 7)),
                    'geoscaffold_box_code_targets': source_bev.new_empty((0, 7)),
                    'geoscaffold_positive_count': torch.zeros(
                        (), device=source_bev.device, dtype=torch.int64
                    ),
                }
            empty_logits = source_bev.new_empty(
                (0, 2, self.geometry_scaffold_mask_size, self.geometry_scaffold_mask_size)
            )
            empty_targets = torch.empty(
                (0, self.geometry_scaffold_mask_size, self.geometry_scaffold_mask_size),
                device=source_bev.device,
                dtype=torch.bool,
            )
            return {
                'geoscaffold_logits': empty_logits,
                'geoscaffold_targets': empty_targets,
                'geoscaffold_positive_count': torch.zeros(
                    (),
                    device=source_bev.device,
                    dtype=torch.int64,
                ),
            }

        rois = sample_anchor_rois(
            source_bev,
            anchor_boxes,
            sample_indices,
            x_min=self.lidar_range[0],
            y_min=self.lidar_range[1],
            cell_x=(self.lidar_range[3] - self.lidar_range[0]) / source_bev.shape[3],
            cell_y=(self.lidar_range[4] - self.lidar_range[1]) / source_bev.shape[2],
            roi_size=self.geometry_scaffold_roi_size,
        )
        if self.geometry_scaffold_auxiliary == 'box_code':
            if box_code_targets is None:
                raise KeyError(
                    'GeoScaffold box-code auxiliary requires native box-code targets.'
                )
            return {
                'geoscaffold_box_code_preds': self.geoscaffold_head.forward_box_code(
                    rois, class_ids
                ),
                'geoscaffold_box_code_targets': box_code_targets,
                'geoscaffold_positive_count': torch.tensor(
                    sample_indices.numel(), device=source_bev.device, dtype=torch.int64
                ),
            }
        return {
            'geoscaffold_logits': self.geoscaffold_head(rois, class_ids),
            'geoscaffold_targets': targets,
            'geoscaffold_positive_count': torch.tensor(
                sample_indices.numel(),
                device=source_bev.device,
                dtype=torch.int64,
            ),
        }

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        fused_feature = self.fusion_net(spatial_features_2d, record_len)
        output_dict = self.predict_from_fused_features(
            spatial_features_2d,
            fused_feature,
            record_len,
        )
        output_dict.update(
            self.build_geoscaffold_outputs(
                data_dict,
                spatial_features_2d,
                fused_feature,
            )
        )
        return output_dict
