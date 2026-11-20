"""Losses for the homogeneous-LiDAR HEAL model.

The historical HEAL path keeps the repository's PointPillar detection loss.
An opt-in ``native_detection_objective`` path mirrors the released HEAL
classification, regression, and direction terms while retaining the local
pyramid and GeoScaffold additions.  Unlike the original single-class
implementation, occupancy labels here are reduced over all UrbanIng
anchor/class channels: a cell is positive when any channel is positive and
negative only when every available channel is negative.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarHealLoss(PointPillarLoss):
    """PointPillar/HEAL loss with an opt-in released detection objective."""

    def __init__(self, args: Mapping[str, object]) -> None:
        # The shared base requires scalar ``reg``/``cls_weight`` values.  The
        # native protocol additionally accepts structured ``regression`` and
        # ``cls`` mappings, so normalize only those compatibility forms before
        # constructing the unchanged base loss.
        base_args = dict(args)
        base_reg = base_args.get("reg", 2.0)
        if isinstance(base_reg, Mapping):
            base_args["reg"] = float(base_reg.get("weight", 2.0))
        else:
            base_args.setdefault("reg", base_reg)
        base_cls_weight = base_args.get("cls_weight", 1.0)
        if isinstance(base_cls_weight, Mapping):
            base_args["cls_weight"] = float(base_cls_weight.get("weight", 1.0))
        else:
            base_args.setdefault("cls_weight", base_cls_weight)
        super().__init__(base_args)

        native_setting = args.get(
            "native_detection_objective",
            args.get("native_objective", False),
        )
        native_config: Mapping[str, object]
        if isinstance(native_setting, Mapping):
            native_config = native_setting
            native_setting = native_setting.get("enabled", True)
        else:
            native_config = {}
        self.native_detection_objective = bool(native_setting)
        # Short alias retained for callers that use the model-side spelling.
        self.native_objective = self.native_detection_objective

        def native_value(name: str, default: object) -> object:
            if name in args:
                return args[name]
            return native_config.get(name, default)

        cls_config = native_value("cls", {})
        if not isinstance(cls_config, Mapping):
            cls_config = {}
        self.native_cls_alpha = float(cls_config.get("alpha", 0.25))
        self.native_cls_gamma = float(cls_config.get("gamma", 2.0))
        cls_weight = cls_config.get(
            "weight", native_value("cls_weight", 1.0)
        )
        if isinstance(cls_weight, Mapping):
            cls_weight = cls_weight.get("weight", 1.0)
        self.native_cls_weight = float(cls_weight)
        self.pos_cls_weight = float(native_value("pos_cls_weight", 2.0))

        regression_config = native_value("regression", None)
        if regression_config is None and isinstance(args.get("reg"), Mapping):
            regression_config = args["reg"]
        if not isinstance(regression_config, Mapping):
            regression_config = {}
        self.native_reg_sigma = float(regression_config.get("sigma", 3.0))
        reg_weight = regression_config.get(
            "weight", native_value("reg", self.reg_coe)
        )
        if isinstance(reg_weight, Mapping):
            reg_weight = reg_weight.get("weight", self.reg_coe)
        self.native_reg_weight = float(reg_weight)

        direction = native_value("dir", None)
        if direction is None:
            direction = native_value("dir_args", {})
        if not isinstance(direction, Mapping):
            direction = {}
        nested_direction = direction.get("args", {})
        if not isinstance(nested_direction, Mapping):
            nested_direction = {}
        self.direction_num_bins = int(
            direction.get("num_bins", nested_direction.get("num_bins", 2))
        )
        self.direction_weight = float(direction.get("weight", 0.2))
        anchor_yaw = direction.get(
            "anchor_yaw", nested_direction.get("anchor_yaw", (0.0, 90.0))
        )
        if isinstance(anchor_yaw, (str, bytes)):
            raise TypeError("HEAL direction anchor_yaw must be a sequence.")
        try:
            self.anchor_yaw = tuple(float(value) for value in anchor_yaw)
        except TypeError as error:
            raise TypeError("HEAL direction anchor_yaw must be a sequence.") from error
        self.direction_offset = float(
            direction.get(
                "dir_offset", nested_direction.get("dir_offset", 0.7853)
            )
        )

        if self.native_detection_objective:
            if self.pos_cls_weight < 0.0 or not math.isfinite(self.pos_cls_weight):
                raise ValueError("native pos_cls_weight must be finite and non-negative")
            if not 0.0 <= self.native_cls_alpha <= 1.0:
                raise ValueError("native cls alpha must be in [0, 1]")
            if self.native_cls_gamma < 0.0 or not math.isfinite(self.native_cls_gamma):
                raise ValueError("native cls gamma must be finite and non-negative")
            if self.native_cls_weight < 0.0 or not math.isfinite(self.native_cls_weight):
                raise ValueError("native cls weight must be finite and non-negative")
            if self.native_reg_sigma <= 0.0 or not math.isfinite(self.native_reg_sigma):
                raise ValueError("native regression sigma must be finite and positive")
            if self.native_reg_weight < 0.0 or not math.isfinite(self.native_reg_weight):
                raise ValueError("native regression weight must be finite and non-negative")
            if self.direction_num_bins <= 0:
                raise ValueError("HEAL direction num_bins must be positive")
            if self.direction_weight < 0.0 or not math.isfinite(self.direction_weight):
                raise ValueError("HEAL direction weight must be finite and non-negative")
            if not self.anchor_yaw:
                raise ValueError("HEAL direction anchor_yaw must not be empty")
            if not math.isfinite(self.direction_offset):
                raise ValueError("HEAL direction offset must be finite")

        pyramid = args.get("pyramid", {})
        if pyramid is None:
            pyramid = {}
        if not isinstance(pyramid, Mapping):
            raise TypeError("loss.args.pyramid must be a mapping.")
        self.relative_downsample = tuple(
            int(value) for value in pyramid.get("relative_downsample", (1, 2, 4))
        )
        self.pyramid_weight = tuple(
            float(value) for value in pyramid.get("weight", (0.4, 0.2, 0.1))
        )
        if not self.relative_downsample or len(self.relative_downsample) != len(
            self.pyramid_weight
        ):
            raise ValueError("HEAL pyramid downsample and weight lists must align.")
        if any(value <= 0 for value in self.relative_downsample):
            raise ValueError("HEAL pyramid downsample values must be positive.")
        if any(value < 0.0 for value in self.pyramid_weight):
            raise ValueError("HEAL pyramid weights must be non-negative.")
        self.pyramid_pos_cls_weight = float(pyramid.get("pos_cls_weight", 2.0))
        if self.pyramid_pos_cls_weight < 0.0:
            raise ValueError("HEAL pyramid positive class weight must be non-negative.")

    @staticmethod
    def _native_focal_loss(
        logits: Tensor,
        targets: Tensor,
        weights: Tensor,
        *,
        alpha: float,
        gamma: float,
    ) -> Tensor:
        """Return the released HEAL sigmoid-focal tensor before reduction."""
        per_entry_cross_ent = torch.clamp(logits, min=0.0) - logits * targets
        per_entry_cross_ent = per_entry_cross_ent + torch.log1p(
            torch.exp(-torch.abs(logits))
        )
        probabilities = torch.sigmoid(logits)
        p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        modulating_factor = torch.pow(1.0 - p_t, gamma)
        alpha_weight = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        return modulating_factor * alpha_weight * per_entry_cross_ent * weights

    @staticmethod
    def _native_smooth_l1(
        predictions: Tensor,
        targets: Tensor,
        weights: Tensor,
        *,
        sigma: float,
    ) -> Tensor:
        """Return the official sigma-parameterized smooth-L1 tensor."""
        difference = predictions - targets
        absolute = torch.abs(difference)
        threshold = 1.0 / (sigma**2)
        quadratic = absolute <= threshold
        loss = torch.where(
            quadratic,
            0.5 * torch.pow(absolute * sigma, 2),
            absolute - 0.5 * threshold,
        )
        return loss * weights

    def _native_detection_loss(
        self,
        output_dict: Mapping[str, Tensor],
        target_dict: Mapping[str, Tensor],
        *,
        apply_receiver_reweighting: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Match HEAL's released cls/reg objective for all label channels."""
        psm = output_dict["psm"]
        rm = output_dict["rm"]
        labels = torch.as_tensor(target_dict["pos_equal_one"], device=psm.device)
        targets = torch.as_tensor(target_dict["targets"], device=psm.device)
        if labels.ndim != 4 or targets.ndim != 5:
            raise ValueError(
                "HEAL native targets must be [B,H,W,A] and [B,H,W,A,7]"
            )
        batch, height, width, anchor_groups = labels.shape
        if psm.ndim != 4 or psm.shape[:1] != (batch,):
            raise ValueError("HEAL native psm must be [B,C,H,W]")
        if psm.shape[2:] != (height, width):
            raise ValueError("HEAL native psm and labels have different grids")
        expected_cls_channels = anchor_groups * self.num_class
        if psm.shape[1] != expected_cls_channels:
            raise ValueError(
                "HEAL native cls output channels do not match labels: "
                f"{psm.shape[1]} versus {expected_cls_channels}"
            )
        if (
            targets.shape[:4] != (batch, height, width, anchor_groups)
            or targets.shape[-1] != 7
        ):
            raise ValueError("HEAL native box targets do not match labels")
        expected_reg_channels = anchor_groups * 7
        if rm.ndim != 4 or rm.shape != (
            batch,
            expected_reg_channels,
            height,
            width,
        ):
            raise ValueError(
                "HEAL native reg output does not match labels: "
                f"{tuple(rm.shape)} versus "
                f"{(batch, expected_reg_channels, height, width)}"
            )

        positive = labels > 0
        negative_labels = target_dict.get("neg_equal_one")
        if negative_labels is None:
            negative = labels == 0
        else:
            negative_tensor = torch.as_tensor(
                negative_labels,
                device=psm.device,
            )
            if negative_tensor.shape == labels.shape:
                negative = negative_tensor > 0
            else:
                # Older multi-class VoxelPostprocessor labels exposed only
                # the final class's negative channels.  In that case labels
                # themselves provide the complete cared-negative mask.
                negative_all = target_dict.get("neg_equal_one_all")
                if negative_all is not None:
                    negative_all_tensor = torch.as_tensor(
                        negative_all,
                        device=psm.device,
                    )
                else:
                    negative_all_tensor = None
                if (
                    negative_all_tensor is not None
                    and negative_all_tensor.shape == labels.shape
                ):
                    negative = negative_all_tensor > 0
                else:
                    negative = labels == 0
        negative = negative & ~positive & (labels >= 0)
        positive_count = positive.flatten(1).sum(dim=1, keepdim=True).float()
        normalizer = positive_count.clamp_min(1.0)

        positive_scale = torch.ones_like(labels, dtype=psm.dtype)
        if apply_receiver_reweighting and self.receiver_relative_alpha != 0.0:
            if "receiver_evidence_q" not in target_dict:
                raise KeyError(
                    "receiver_evidence_q is required when receiver-relative "
                    "reweighting is enabled"
                )
            evidence = torch.as_tensor(
                target_dict["receiver_evidence_q"],
                device=psm.device,
            )
            if evidence.shape != labels.shape:
                raise ValueError("receiver_evidence_q must match HEAL labels")
            positive_scale = self.receiver_relative_positive_scale(
                positive.flatten(1), evidence.flatten(1)
            ).view_as(labels).to(dtype=psm.dtype)

        cls_weights = (
            positive.to(dtype=psm.dtype) * self.pos_cls_weight * positive_scale
            + negative.to(dtype=psm.dtype)
        ) / normalizer.view(batch, 1, 1, 1).to(dtype=psm.dtype)
        cls_indices = torch.where(
            positive,
            labels.long() - 1,
            torch.zeros_like(labels.long()),
        )
        if bool((positive & (labels > self.num_class)).any()):
            raise ValueError("HEAL native positive labels exceed num_class")
        cls_targets = F.one_hot(cls_indices, num_classes=self.num_class).to(psm.dtype)
        cls_targets = cls_targets * positive.unsqueeze(-1).to(dtype=psm.dtype)
        cls_logits = psm.permute(0, 2, 3, 1).contiguous().view(
            batch, height, width, anchor_groups, self.num_class
        )
        cls_loss = self._native_focal_loss(
            cls_logits,
            cls_targets,
            cls_weights.unsqueeze(-1),
            alpha=self.native_cls_alpha,
            gamma=self.native_cls_gamma,
        ).sum() * self.native_cls_weight / float(batch)

        reg_weights = (
            positive.to(dtype=rm.dtype) * positive_scale.to(dtype=rm.dtype)
        ) / normalizer.view(batch, 1, 1, 1).to(dtype=rm.dtype)
        reg_preds = rm.permute(0, 2, 3, 1).contiguous().view(batch, -1, 7)
        reg_targets = targets.to(dtype=reg_preds.dtype).view(batch, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = self._native_smooth_l1(
            reg_preds,
            reg_targets,
            reg_weights.view(batch, -1, 1),
            sigma=self.native_reg_sigma,
        ).sum() * self.native_reg_weight / float(batch)
        return reg_loss + cls_loss, reg_loss, cls_loss

    def _detection_loss(
        self,
        output_dict,
        target_dict,
        *,
        apply_receiver_reweighting: bool = True,
    ):
        if not self.native_detection_objective:
            return super()._detection_loss(
                output_dict,
                target_dict,
                apply_receiver_reweighting=apply_receiver_reweighting,
            )
        return self._native_detection_loss(
            output_dict,
            target_dict,
            apply_receiver_reweighting=apply_receiver_reweighting,
        )

    def _direction_loss(
        self,
        output_dict: Mapping[str, Tensor],
        target_dict: Mapping[str, Tensor],
    ) -> Tensor:
        """Compute positive-anchor direction CE using HEAL's bin convention."""
        logits = output_dict.get("dir_preds", output_dict.get("dm"))
        if logits is None:
            # Keep the objective composable with a direction-disabled native
            # config; a model that supplies ``dir_args`` emits this tensor.
            return output_dict["psm"].sum() * 0.0
        labels = torch.as_tensor(target_dict["pos_equal_one"], device=logits.device)
        targets = torch.as_tensor(target_dict["targets"], device=logits.device)
        if labels.ndim != 4 or targets.ndim != 5:
            raise ValueError(
                "HEAL direction targets must be [B,H,W,A] and [B,H,W,A,7]"
            )
        batch, height, width, anchor_groups = labels.shape
        expected_channels = anchor_groups * self.direction_num_bins
        if logits.shape != (batch, expected_channels, height, width):
            raise ValueError(
                "HEAL direction logits do not align with labels: "
                f"{tuple(logits.shape)} versus "
                f"{(batch, expected_channels, height, width)}"
            )
        if (
            targets.shape[:4] != (batch, height, width, anchor_groups)
            or targets.shape[-1] != 7
        ):
            raise ValueError("HEAL direction box targets do not match labels")

        yaw_anchors = torch.as_tensor(
            self.anchor_yaw,
            device=targets.device,
            dtype=targets.dtype,
        )
        if float(yaw_anchors.abs().max()) > 2.0 * math.pi + 1.0e-4:
            yaw_anchors = torch.deg2rad(yaw_anchors)
        if yaw_anchors.numel() == anchor_groups:
            yaw_anchors = yaw_anchors
        elif anchor_groups % yaw_anchors.numel() == 0:
            yaw_anchors = yaw_anchors.repeat(anchor_groups // yaw_anchors.numel())
        else:
            raise ValueError(
                "HEAL direction anchor_yaw count must divide label channels: "
                f"{yaw_anchors.numel()} versus {anchor_groups}"
            )
        yaw = targets[..., 6] + yaw_anchors.view(1, 1, 1, anchor_groups)
        period = 2.0 * math.pi
        direction_targets = torch.floor(
            torch.remainder(yaw - self.direction_offset, period)
            / (period / float(self.direction_num_bins))
        ).long().clamp_(0, self.direction_num_bins - 1)
        direction_logits = logits.permute(0, 2, 3, 1).contiguous().view(
            batch, height, width, anchor_groups, self.direction_num_bins
        )
        per_anchor = F.cross_entropy(
            direction_logits.view(-1, self.direction_num_bins),
            direction_targets.view(-1),
            reduction="none",
        ).view(batch, height, width, anchor_groups)
        positive = labels > 0
        positive_count = positive.flatten(1).sum(dim=1).float().clamp_min(1.0)
        weights = positive.to(dtype=per_anchor.dtype)
        return (per_anchor * weights).flatten(1).sum(dim=1).div(positive_count).mean()

    @staticmethod
    def _label_mapping(
        target_dict: Mapping[str, object] | dict[str, Tensor],
    ) -> Mapping[str, Tensor]:
        """Accept both the direct OpenCOOD label mapping and nested test API."""
        if "label_dict" in target_dict:
            label_dict = target_dict["label_dict"]
            if not isinstance(label_dict, Mapping):
                raise TypeError("target_dict['label_dict'] must be a mapping.")
            return label_dict
        return target_dict  # type: ignore[return-value]

    @staticmethod
    def _repeat_to_batch(
        label: Tensor,
        target_batch: int,
        output_dict: Mapping[str, object],
    ) -> Tensor:
        if label.shape[0] == target_batch:
            return label
        record_len = output_dict.get("record_len")
        if isinstance(record_len, Tensor):
            lengths = [int(value) for value in record_len.detach().cpu().tolist()]
            if len(lengths) == label.shape[0] and sum(lengths) == target_batch:
                indices = torch.repeat_interleave(
                    torch.arange(label.shape[0], device=label.device),
                    torch.as_tensor(lengths, device=label.device),
                )
                return label.index_select(0, indices)
        if label.shape[0] > 0 and target_batch % label.shape[0] == 0:
            repeats = target_batch // label.shape[0]
            return label.repeat_interleave(repeats, dim=0)
        raise ValueError(
            "Pyramid occupancy target batch does not align with packed predictions: "
            f"{label.shape[0]} versus {target_batch}."
        )

    @classmethod
    def _occupancy_targets(
        cls,
        label_dict: Mapping[str, Tensor],
        *,
        target_batch: int,
        output_dict: Mapping[str, object],
    ) -> tuple[Tensor, Tensor]:
        if "pos_equal_one" not in label_dict:
            raise KeyError("HEAL occupancy supervision requires pos_equal_one.")
        positives = torch.as_tensor(label_dict["pos_equal_one"])
        if positives.ndim == 3:
            positives = positives.unsqueeze(0)
        if positives.ndim != 4:
            raise ValueError("pos_equal_one must have shape [B,H,W,A].")
        positives = cls._repeat_to_batch(positives, target_batch, output_dict)
        positive_mask = positives > 0
        positive_occupancy = positive_mask.any(dim=-1, keepdim=True)

        if "neg_equal_one_all" not in label_dict:
            raise KeyError(
                "HEAL occupancy supervision requires class-complete "
                "neg_equal_one_all targets."
            )
        negatives = torch.as_tensor(label_dict["neg_equal_one_all"])
        if negatives.ndim == 3:
            negatives = negatives.unsqueeze(0)
        if negatives.ndim != 4:
            raise ValueError("neg_equal_one_all must have shape [B,H,W,A].")
        negatives = cls._repeat_to_batch(negatives, target_batch, output_dict)
        if negatives.shape != positives.shape:
            raise ValueError(
                "neg_equal_one_all must have the same class/anchor channels "
                "as pos_equal_one."
            )
        negative_occupancy = (negatives > 0).all(dim=-1, keepdim=True)
        return positive_occupancy, negative_occupancy

    @staticmethod
    def _pool_occupancy_target(
        occupancy: Tensor,
        *,
        downsample: int,
        require_all: bool,
    ) -> Tensor:
        """Pool positive-any or negative-all labels exactly over a coarse cell."""
        if require_all:
            return 1.0 - F.max_pool2d(
                1.0 - occupancy,
                kernel_size=downsample,
                stride=downsample,
            )
        return F.max_pool2d(
            occupancy,
            kernel_size=downsample,
            stride=downsample,
        )

    def calc_pyramid_loss(
        self,
        occupancy_logits: Sequence[Tensor],
        label_dict: Mapping[str, Tensor],
        *,
        output_dict: Mapping[str, object] | None = None,
    ) -> Tensor:
        """Compute multi-resolution occupancy focal loss for packed agents."""
        if not occupancy_logits:
            raise ValueError("HEAL requires at least one occupancy prediction level.")
        context = output_dict or {}
        positive, negative = self._occupancy_targets(
            label_dict,
            target_batch=int(occupancy_logits[0].shape[0]),
            output_dict=context,
        )
        positive = positive.to(device=occupancy_logits[0].device)
        negative = negative.to(device=occupancy_logits[0].device)
        if positive.shape[1:3] != occupancy_logits[0].shape[2:]:
            raise ValueError(
                "Level-0 occupancy map and labels have different spatial shapes: "
                f"{tuple(occupancy_logits[0].shape[2:])} versus "
                f"{tuple(positive.shape[1:3])}."
            )

        total = occupancy_logits[0].sum() * 0.0
        for level, (logits, downsample, level_weight) in enumerate(
            zip(occupancy_logits, self.relative_downsample, self.pyramid_weight)
        ):
            if logits.ndim != 4 or logits.shape[1] != 1:
                raise ValueError("Each HEAL occupancy prediction must be [N,1,H,W].")
            positive_level = self._pool_occupancy_target(
                positive.permute(0, 3, 1, 2).float(),
                downsample=downsample,
                require_all=False,
            )
            negative_level = self._pool_occupancy_target(
                negative.permute(0, 3, 1, 2).float(),
                downsample=downsample,
                require_all=True,
            )
            if positive_level.shape[2:] != logits.shape[2:]:
                raise ValueError(
                    f"HEAL level {level} prediction shape {tuple(logits.shape[2:])} "
                    f"does not match pooled target {tuple(positive_level.shape[2:])}."
                )
            positive_level = positive_level.permute(0, 2, 3, 1)
            negative_level = negative_level.permute(0, 2, 3, 1)
            labels = positive_level.reshape(logits.shape[0], -1, 1).to(logits.dtype)
            weights = (
                positive_level * self.pyramid_pos_cls_weight + negative_level
            ).reshape(logits.shape[0], -1, 1).to(logits.dtype)
            positive_count = positive_level.reshape(logits.shape[0], -1, 1).sum(
                dim=1,
                keepdim=True,
            )
            weights = weights / torch.clamp(positive_count, min=1.0)
            predictions = logits.permute(0, 2, 3, 1).reshape(logits.shape[0], -1, 1)
            level_loss = self.cls_loss_func(predictions, labels, weights=weights)
            total = total + float(level_weight) * level_loss.sum() / logits.shape[0]
        return total

    def forward(self, output_dict, target_dict):
        if isinstance(target_dict, Mapping) and "label_dict" in target_dict:
            native_target = self._label_mapping(target_dict)
            single_target = target_dict.get("label_dict_single", native_target)
            if not isinstance(single_target, Mapping):
                raise TypeError("target_dict['label_dict_single'] must be a mapping.")
        else:
            native_target = self._label_mapping(target_dict)
            single_target = native_target

        total_loss = super().forward(output_dict, native_target)
        if (
            self.native_detection_objective
            and self.direction_weight != 0.0
            and ("dir_preds" in output_dict or "dm" in output_dict)
        ):
            direction_loss = self._direction_loss(output_dict, native_target)
            total_loss = total_loss + self.direction_weight * direction_loss
            self.loss_dict.update(
                {
                    "total_loss": total_loss,
                    "direction_loss": direction_loss,
                }
            )
        occupancy_logits = output_dict.get("occ_single_list")
        if occupancy_logits is None:
            occupancy_logits = output_dict.get("heal_occ_logits")
        if occupancy_logits is None:
            return total_loss
        pyramid_loss = self.calc_pyramid_loss(
            occupancy_logits,
            single_target,
            output_dict=output_dict,
        )
        total_loss = total_loss + pyramid_loss
        self.loss_dict.update(
            {
                "total_loss": total_loss,
                "pyramid_loss": pyramid_loss,
            }
        )
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        super().logging(epoch, batch_id, batch_len, writer, pbar=pbar)
        if "pyramid_loss" in self.loss_dict:
            writer.add_scalar(
                "Pyramid_loss",
                self.loss_dict["pyramid_loss"].item(),
                epoch * batch_len + batch_id,
            )
