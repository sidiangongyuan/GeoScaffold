# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.geometry_scaffold import GeometryScaffoldMixin
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion


class PointPillarV2VNet(GeometryScaffoldMixin, nn.Module):
    def __init__(self, args):
        super(PointPillarV2VNet, self).__init__()

        self.max_cav = args['max_cav']
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

        self.fusion_net = V2VNetFusion(args['v2vfusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'] * num_class * num_class,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_num'] * num_class,
                                  kernel_size=1)
        self.init_geometry_scaffold(
            args,
            in_channels=self.cls_head.in_channels,
            num_classes=num_class,
        )
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    

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
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            # Send the compressed representation; the split fusion path
            # decodes it once immediately before V2V fusion.
            spatial_features_2d = self.naive_compressor.encoder(
                spatial_features_2d
            )
        
        return spatial_features_2d

    @staticmethod
    def unpad_prior_encoding(x, record_len):
        """Remove padded CAV rows from the official ``(B, L, 3)`` input."""
        if x is None:
            raise ValueError(
                'V2VNet temporal encoding is enabled, but '
                "data_dict['prior_encoding'] is missing; expected shape "
                '(B, L, 3).'
            )
        if not torch.is_tensor(x):
            raise ValueError(
                'V2VNet temporal encoding requires '
                "data_dict['prior_encoding'] to be a tensor with shape "
                '(B, L, 3).'
            )
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(
                'V2VNet prior_encoding must have shape (B, L, 3); got '
                f'{tuple(x.shape)}.'
            )
        if not torch.is_tensor(record_len):
            record_len = torch.as_tensor(record_len)
        if record_len.ndim != 1:
            raise ValueError(
                'V2VNet record_len must be one-dimensional when unpadding '
                'prior_encoding.'
            )
        if x.shape[0] != record_len.numel():
            raise ValueError(
                'V2VNet prior_encoding batch dimension must match record_len '
                f'({x.shape[0]} != {record_len.numel()}).'
            )

        lengths = [int(length) for length in record_len.detach().cpu().tolist()]
        if any(length < 0 for length in lengths):
            raise ValueError('V2VNet record_len cannot contain negative values.')
        if any(length > x.shape[1] for length in lengths):
            raise ValueError(
                'V2VNet prior_encoding does not contain enough padded agents '
                f'for record_len={lengths}; got L={x.shape[1]}.'
            )
        chunks = [x[batch_idx, :length, :] for batch_idx, length in enumerate(lengths)]
        if not chunks:
            return x.new_empty((0, 3))
        return torch.cat(chunks, dim=0)

    def _fuse_features(self, spatial_features_2d, record_len,
                        pairwise_t_matrix, data_dict):
        """Run the legacy or temporal V2V fusion call for this model."""
        if not bool(getattr(self.fusion_net, 'use_temporal_encoding', False)):
            return self.fusion_net(
                spatial_features_2d,
                record_len,
                pairwise_t_matrix,
            )

        prior_encoding = self.unpad_prior_encoding(
            data_dict.get('prior_encoding'),
            record_len,
        )
        return self.fusion_net(
            spatial_features_2d,
            record_len,
            pairwise_t_matrix,
            prior_encoding,
        )


    def fuse_features(self, data):
        spatial_features_2d = data['bev']
        record_len = data['record_len']

        pairwise_t_matrix = data['pairwise_t_matrix']

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor.decoder(
                spatial_features_2d
            )

        fused_feature = self._fuse_features(
            spatial_features_2d,
            record_len,
            pairwise_t_matrix,
            data,
        )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

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
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        fused_feature = self._fuse_features(
            spatial_features_2d,
            record_len,
            pairwise_t_matrix,
            data_dict,
        )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}
        output_dict.update(
            self.build_geometry_scaffold_output(
                spatial_features_2d,
                fused_feature,
                record_len,
                data_dict,
            )
        )

        return output_dict
