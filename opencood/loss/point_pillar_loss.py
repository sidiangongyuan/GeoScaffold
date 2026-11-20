# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from collections.abc import Mapping
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.geometry_scaffold import (
    field_ce_dice_loss,
    native_box_code_smooth_l1_loss,
)
from opencood.models.sub_modules.global_bev_footprint import (
    resolve_global_bev_footprint_loss_config,
)


def _resolve_geoscaffold_weight(args):
    geometry_scaffold = args.get('geometry_scaffold', {})
    if geometry_scaffold is None:
        geometry_scaffold = {}
    if not isinstance(geometry_scaffold, Mapping):
        raise TypeError('loss.args.geometry_scaffold must be a mapping when provided')
    if 'weight' in geometry_scaffold:
        return float(geometry_scaffold['weight'])

    counterfactual = args.get('counterfactual_training', {})
    if counterfactual is None:
        counterfactual = {}
    if not isinstance(counterfactual, Mapping):
        raise TypeError(
            'loss.args.counterfactual_training must be a mapping when provided'
        )
    return float(counterfactual.get('geoscaffold_weight', 0.0))


def _resolve_geoscaffold_loss_config(args):
    """Resolve independent field-loss weights while preserving legacy defaults."""

    geometry_scaffold = args.get('geometry_scaffold', {})
    if geometry_scaffold is None:
        geometry_scaffold = {}
    if not isinstance(geometry_scaffold, Mapping):
        raise TypeError('loss.args.geometry_scaffold must be a mapping when provided')

    objective = str(
        geometry_scaffold.get(
            'objective',
            geometry_scaffold.get('loss_mode', 'ce_dice'),
        )
    ).lower()
    objective = objective.replace('-', '_').replace('+', '_')
    objective_aliases = {
        'ce': (1.0, 0.0),
        'ce_only': (1.0, 0.0),
        'cross_entropy': (1.0, 0.0),
        'dice': (0.0, 1.0),
        'dice_only': (0.0, 1.0),
        'ce_dice': (1.0, 1.0),
        'dice_ce': (1.0, 1.0),
        'ce_plus_dice': (1.0, 1.0),
        'cross_entropy_dice': (1.0, 1.0),
    }
    if objective not in objective_aliases:
        raise ValueError(
            'geometry_scaffold.objective must be ce, dice, or ce_dice; '
            f'got {objective!r}'
        )
    ce_weight, dice_weight = objective_aliases[objective]
    objective_weights = geometry_scaffold.get('loss', {})
    if objective_weights is None:
        objective_weights = {}
    if not isinstance(objective_weights, Mapping):
        raise TypeError('geometry_scaffold.loss must be a mapping when provided')
    if 'ce_weight' in geometry_scaffold:
        ce_weight = float(geometry_scaffold['ce_weight'])
    elif 'ce_weight' in objective_weights:
        ce_weight = float(objective_weights['ce_weight'])
    if 'dice_weight' in geometry_scaffold:
        dice_weight = float(geometry_scaffold['dice_weight'])
    elif 'dice_weight' in objective_weights:
        dice_weight = float(objective_weights['dice_weight'])
    if (
        not math.isfinite(float(ce_weight))
        or not math.isfinite(float(dice_weight))
        or ce_weight < 0.0
        or dice_weight < 0.0
    ):
        raise ValueError('GeoScaffold CE/Dice weights must be non-negative')
    if ce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError('GeoScaffold CE/Dice weights cannot both be zero')
    return ce_weight, dice_weight


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0
        # hard_coded
        self.num_class = args['num_class']

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        reweighting = args.get('receiver_relative_reweighting', {})
        self.receiver_relative_alpha = float(reweighting.get('alpha', 0.0))
        self.receiver_relative_direction = float(reweighting.get('direction', 1.0))
        self.receiver_relative_temperature = float(
            reweighting.get('temperature', 1.0)
        )
        if self.receiver_relative_alpha < 0.0:
            raise ValueError('receiver-relative alpha must be non-negative')
        if self.receiver_relative_temperature <= 0.0:
            raise ValueError('receiver-relative temperature must be positive')
        counterfactual = args.get('counterfactual_training', {})
        self.ego_auxiliary_weight = float(
            counterfactual.get('ego_weight', 0.0)
        )
        if self.ego_auxiliary_weight < 0.0:
            raise ValueError('ego auxiliary weight must be non-negative')
        self.geoscaffold_weight = _resolve_geoscaffold_weight(args)
        if self.geoscaffold_weight < 0.0:
            raise ValueError('geoscaffold weight must be non-negative')
        (
            self.geoscaffold_ce_weight,
            self.geoscaffold_dice_weight,
        ) = _resolve_geoscaffold_loss_config(args)
        (
            self.global_bev_footprint_weight,
            self.global_bev_footprint_ce_weight,
            self.global_bev_footprint_dice_weight,
        ) = resolve_global_bev_footprint_loss_config(args)
        uncertainty = args.get('uncertainty', {})
        if uncertainty is None:
            uncertainty = {}
        if not isinstance(uncertainty, Mapping):
            raise TypeError('loss.args.uncertainty must be a mapping when provided')
        self.uncertainty_weight = float(uncertainty.get('weight', 0.0))
        self.uncertainty_dim = int(uncertainty.get('dim', 3))
        if self.uncertainty_weight < 0.0:
            raise ValueError('uncertainty weight must be non-negative')
        if self.uncertainty_dim not in (2, 3):
            raise ValueError('uncertainty dim must be 2 or 3')
        self.loss_dict = {}

    def receiver_relative_positive_scale(
        self,
        positives: torch.Tensor,
        receiver_evidence_q: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-anchor scales whose positive mean is one per sample."""
        if positives.shape != receiver_evidence_q.shape:
            raise ValueError(
                "receiver_evidence_q must have the flattened positive-label shape; "
                f"got {receiver_evidence_q.shape} and {positives.shape}."
            )
        q = receiver_evidence_q.to(
            device=positives.device,
            dtype=torch.float32,
        )
        base_scale = 1.0 + self.receiver_relative_alpha * torch.sigmoid(
            self.receiver_relative_direction
            * q
            / self.receiver_relative_temperature
        )
        positive_float = positives.float()
        positive_count = positive_float.sum(dim=1, keepdim=True)
        positive_scale_sum = (base_scale * positive_float).sum(
            dim=1,
            keepdim=True,
        )
        positive_mean = positive_scale_sum / torch.clamp(
            positive_count,
            min=1.0,
        )
        normalized_scale = base_scale / torch.clamp(positive_mean, min=1e-12)
        return torch.where(
            positives,
            normalized_scale,
            torch.ones_like(normalized_scale),
        )

    def _detection_loss(
        self,
        output_dict,
        target_dict,
        *,
        apply_receiver_reweighting=True,
    ):
        rm = output_dict['rm']
        psm = output_dict['psm']
        targets = target_dict['targets']
        # (B, 100, 176, 18)
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()
        # (B, 100, 176, 4)
        box_cls_labels = target_dict['pos_equal_one']
        # (B, 100*176*4)
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        # Alpha zero deliberately bypasses every weighting operation. This
        # preserves the original arithmetic graph for exact loss/gradient parity.
        if apply_receiver_reweighting and self.receiver_relative_alpha != 0.0:
            if 'receiver_evidence_q' not in target_dict:
                raise KeyError(
                    'receiver_evidence_q is required when receiver-relative '
                    'reweighting is enabled'
                )
            receiver_evidence_q = target_dict['receiver_evidence_q'].view(
                psm.shape[0],
                -1,
            )
            positive_scale = self.receiver_relative_positive_scale(
                positives,
                receiver_evidence_q,
            ).type_as(cls_weights)
            cls_weights = negative_cls_weights.float() + \
                positives.float() * positive_scale
            reg_weights = positives.float() * positive_scale

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]  # not count the background class

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = reg_loss + conf_loss

        return total_loss, reg_loss, conf_loss

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss, reg_loss, conf_loss = self._detection_loss(
            output_dict,
            target_dict,
        )

        self.loss_dict = {'total_loss': total_loss,
                          'reg_loss': reg_loss,
                          'conf_loss': conf_loss}
        if self.uncertainty_weight != 0.0:
            if 'uncertainty_preds' not in output_dict:
                raise KeyError(
                    'uncertainty_preds are required when uncertainty loss is enabled'
                )
            uncertainty_loss = self._uncertainty_loss(output_dict, target_dict)
            total_loss = total_loss + self.uncertainty_weight * uncertainty_loss
            self.loss_dict.update({
                'total_loss': total_loss,
                'uncertainty_loss': uncertainty_loss,
            })
        if self.ego_auxiliary_weight != 0.0:
            if 'ego_psm' not in output_dict or 'ego_rm' not in output_dict:
                raise KeyError(
                    'ego_psm and ego_rm are required when ego auxiliary loss '
                    'is enabled'
                )
            ego_loss, ego_reg_loss, ego_conf_loss = self._detection_loss(
                {
                    'psm': output_dict['ego_psm'],
                    'rm': output_dict['ego_rm'],
                },
                target_dict,
                apply_receiver_reweighting=False,
            )
            total_loss = total_loss + self.ego_auxiliary_weight * ego_loss
            self.loss_dict.update({
                'total_loss': total_loss,
                'ego_loss': ego_loss,
                'ego_reg_loss': ego_reg_loss,
                'ego_conf_loss': ego_conf_loss,
            })
        if (
            self.geoscaffold_weight != 0.0
            and 'geoscaffold_logits' in output_dict
        ):
            if 'geoscaffold_targets' not in output_dict:
                raise KeyError(
                    'geoscaffold_targets are required when geoscaffold_logits '
                    'are present'
                )
            geoscaffold_loss = field_ce_dice_loss(
                # The auxiliary head is positive-anchor only and has no
                # inference-time path, so negatives remain under native cls loss.
                output_dict['geoscaffold_logits'],
                output_dict['geoscaffold_targets'],
                ce_weight=self.geoscaffold_ce_weight,
                dice_weight=self.geoscaffold_dice_weight,
            )
            total_loss = total_loss + self.geoscaffold_weight * geoscaffold_loss
            self.loss_dict.update({
                'total_loss': total_loss,
                'geoscaffold_loss': geoscaffold_loss,
            })
            if 'geoscaffold_positive_count' in output_dict:
                self.loss_dict['geoscaffold_positive_count'] = \
                    output_dict['geoscaffold_positive_count']
        if self.geoscaffold_weight != 0.0 and 'geoscaffold_box_code_preds' in output_dict:
            if 'geoscaffold_box_code_targets' not in output_dict:
                raise KeyError(
                    'geoscaffold_box_code_targets are required when box-code '
                    'auxiliary predictions are present'
                )
            box_code_loss = native_box_code_smooth_l1_loss(
                output_dict['geoscaffold_box_code_preds'],
                output_dict['geoscaffold_box_code_targets'],
            )
            total_loss = total_loss + self.geoscaffold_weight * box_code_loss
            self.loss_dict.update({
                'total_loss': total_loss,
                'geoscaffold_box_code_loss': box_code_loss,
            })
            if 'geoscaffold_positive_count' in output_dict:
                self.loss_dict['geoscaffold_positive_count'] = \
                    output_dict['geoscaffold_positive_count']
        total_loss = self._apply_global_bev_footprint_loss(total_loss, output_dict)
        if 'receiver_evidence_missing_hit_count' in target_dict:
            self.loss_dict['receiver_evidence_missing_hit_count'] = \
                target_dict['receiver_evidence_missing_hit_count'].sum()

        return total_loss

    def _apply_global_bev_footprint_loss(self, total_loss, output_dict):
        """Add the dense global-footprint objective when its training output exists."""
        if (
            self.global_bev_footprint_weight == 0.0
            or 'global_bev_footprint_logits' not in output_dict
        ):
            return total_loss
        if 'global_bev_footprint_targets' not in output_dict:
            raise KeyError(
                'global_bev_footprint_targets are required when global footprint '
                'logits are present'
            )
        footprint_loss = field_ce_dice_loss(
            output_dict['global_bev_footprint_logits'],
            output_dict['global_bev_footprint_targets'],
            ce_weight=self.global_bev_footprint_ce_weight,
            dice_weight=self.global_bev_footprint_dice_weight,
        )
        total_loss = total_loss + self.global_bev_footprint_weight * footprint_loss
        self.loss_dict.update({
            'total_loss': total_loss,
            'global_bev_footprint_loss': footprint_loss,
        })
        if 'global_bev_footprint_positive_count' in output_dict:
            self.loss_dict['global_bev_footprint_positive_count'] = output_dict[
                'global_bev_footprint_positive_count'
            ]
        return total_loss

    def _uncertainty_loss(self, output_dict, target_dict):
        """Heteroscedastic x/y/(yaw) NLL on official positive anchors."""
        rm = output_dict['rm'].permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.shape[0], -1, 7)
        targets = target_dict['targets'].view(rm.shape[0], -1, 7)
        log_variance = output_dict['uncertainty_preds'].permute(
            0, 2, 3, 1
        ).contiguous()
        log_variance = log_variance.view(
            rm.shape[0], -1, self.uncertainty_dim
        )
        if log_variance.shape[:2] != rm.shape[:2]:
            raise ValueError('uncertainty predictions must align with box anchors')
        difference = rm[..., :2] - targets[..., :2]
        if self.uncertainty_dim == 3:
            yaw_difference = torch.sin(rm[..., 6:7] - targets[..., 6:7])
            difference = torch.cat((difference, yaw_difference), dim=-1)
        stable_log_variance = torch.clamp(log_variance, min=-10.0, max=10.0)
        nll = 0.5 * (
            torch.exp(-stable_log_variance) * difference.square()
            + stable_log_variance
        )
        positives = target_dict['pos_equal_one'].view(rm.shape[0], -1) > 0
        weights = positives.float()
        weights /= torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
        return (nll * weights.unsqueeze(-1)).sum() / rm.shape[0]

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item()))


        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)
        if 'ego_loss' in self.loss_dict:
            writer.add_scalar(
                'Ego_auxiliary_loss',
                self.loss_dict['ego_loss'].item(),
                epoch * batch_len + batch_id,
            )
        if 'geoscaffold_loss' in self.loss_dict:
            writer.add_scalar(
                'GeoScaffold_loss',
                self.loss_dict['geoscaffold_loss'].item(),
                epoch * batch_len + batch_id,
            )
        if 'geoscaffold_positive_count' in self.loss_dict:
            writer.add_scalar(
                'GeoScaffold_positive_count',
                self.loss_dict['geoscaffold_positive_count'].item(),
                epoch * batch_len + batch_id,
            )
        if 'global_bev_footprint_loss' in self.loss_dict:
            writer.add_scalar(
                'GlobalBEVFootprint_loss',
                self.loss_dict['global_bev_footprint_loss'].item(),
                epoch * batch_len + batch_id,
            )
        if 'uncertainty_loss' in self.loss_dict:
            writer.add_scalar(
                'Uncertainty_loss',
                self.loss_dict['uncertainty_loss'].item(),
                epoch * batch_len + batch_id,
            )
        if 'receiver_evidence_missing_hit_count' in self.loss_dict:
            missing_hits = self.loss_dict['receiver_evidence_missing_hit_count']
            writer.add_scalar(
                'Receiver_evidence_missing_hit_count',
                missing_hits.item(),
                epoch * batch_len + batch_id,
            )
