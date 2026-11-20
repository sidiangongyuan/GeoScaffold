"""CoAlign's native classification, regression and direction objective."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarCoAlignLoss(PointPillarLoss):
    """Repository-native loss plus the author's positive-anchor direction loss.

    CoAlign's public recipe uses a positive-anchor weight of two, focal
    classification weight two, and a sigma-three regression loss.  Those
    values are resolved here rather than inherited from the repository's
    generic PointPillar defaults.  The parent ``forward`` still owns all
    optional GeoScaffold additions, so the auxiliary is never added twice.
    """

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__(args)
        direction = args.get("dir_args", args.get("dir", {}))
        if not isinstance(direction, Mapping):
            direction = {}
        nested = direction.get("args", {})
        if not isinstance(nested, Mapping):
            nested = {}
        self.direction_num_bins = int(
            direction.get("num_bins", nested.get("num_bins", 2))
        )
        self.direction_weight = float(direction.get("weight", 0.2))
        anchor_yaw = direction.get("anchor_yaw", nested.get("anchor_yaw", (0.0, 90.0)))
        self.anchor_yaw = tuple(float(value) for value in anchor_yaw)
        self.direction_offset = float(
            direction.get("dir_offset", nested.get("dir_offset", 0.7853))
        )
        self.pos_cls_weight = float(args.get("pos_cls_weight", 2.0))
        cls_config = args.get("cls", {})
        if not isinstance(cls_config, Mapping):
            cls_config = {}
        # Keep the scalar ``reg`` key required by the repository base class
        # separate from the public CoAlign mapping.  New configs may use the
        # explicit ``regression`` name; accepting a mapping under ``reg`` is
        # retained for compatibility with the upstream loss file.
        reg_config = args.get("regression", args.get("reg", {}))
        if not isinstance(reg_config, Mapping):
            reg_config = {}
        self.cls_alpha = float(cls_config.get("alpha", 0.25))
        self.cls_gamma = float(cls_config.get("gamma", 2.0))
        cls_weight_default = args.get("cls_weight", 2.0)
        if isinstance(cls_weight_default, Mapping):
            cls_weight_default = 2.0
        self.cls_weight = float(cls_config.get("weight", cls_weight_default))
        self.reg_sigma = float(reg_config.get("sigma", 3.0))
        reg_weight_default = args.get("reg", 2.0)
        if isinstance(reg_weight_default, Mapping):
            reg_weight_default = 2.0
        self.reg_weight = float(reg_config.get("weight", reg_weight_default))
        if self.direction_num_bins <= 0:
            raise ValueError("direction num_bins must be positive")
        if self.direction_weight < 0.0 or not math.isfinite(self.direction_weight):
            raise ValueError("direction weight must be finite and non-negative")
        if self.pos_cls_weight < 0.0 or not math.isfinite(self.pos_cls_weight):
            raise ValueError("pos_cls_weight must be finite and non-negative")
        if not 0.0 <= self.cls_alpha <= 1.0:
            raise ValueError("cls alpha must be in [0, 1]")
        if self.cls_gamma < 0.0 or not math.isfinite(self.cls_gamma):
            raise ValueError("cls gamma must be finite and non-negative")
        if self.cls_weight < 0.0 or not math.isfinite(self.cls_weight):
            raise ValueError("cls weight must be finite and non-negative")
        if self.reg_sigma <= 0.0 or not math.isfinite(self.reg_sigma):
            raise ValueError("reg sigma must be finite and positive")
        if self.reg_weight < 0.0 or not math.isfinite(self.reg_weight):
            raise ValueError("reg weight must be finite and non-negative")
        if not math.isfinite(self.direction_offset):
            raise ValueError("direction offset must be finite")

    @staticmethod
    def _official_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        *,
        alpha: float,
        gamma: float,
    ) -> torch.Tensor:
        """Public CoAlign ``sigmoid_focal_loss`` with explicit reduction."""

        per_entry_cross_ent = torch.clamp(logits, min=0.0) - logits * targets
        per_entry_cross_ent = per_entry_cross_ent + torch.log1p(
            torch.exp(-torch.abs(logits))
        )
        probabilities = torch.sigmoid(logits)
        p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        modulating_factor = torch.pow(1.0 - p_t, gamma)
        alpha_weight_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        return modulating_factor * alpha_weight_factor * per_entry_cross_ent * weights

    @staticmethod
    def _official_smooth_l1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        *,
        sigma: float,
    ) -> torch.Tensor:
        """Public CoAlign ``weighted_smooth_l1_loss`` before reduction."""

        diff = predictions - targets
        abs_diff = torch.abs(diff)
        threshold = 1.0 / (sigma ** 2)
        quadratic = abs_diff <= threshold
        loss = quadratic * 0.5 * torch.pow(abs_diff * sigma, 2)
        loss = loss + (abs_diff - 0.5 * threshold) * (~quadratic)
        return loss * weights

    def _detection_loss(self, output_dict, target_dict, **_kwargs):
        """Match the public single-class CoAlign cls/reg tensor formulas.

        ``pos_equal_one == -1`` is accepted as an ignore marker in this
        repository and receives zero classification and regression weight.
        ``neg_equal_one`` remains authoritative for negatives when present,
        matching the public OpenCOOD implementation.
        """

        psm = output_dict["psm"]
        rm = output_dict["rm"]
        labels = target_dict["pos_equal_one"]
        targets = target_dict["targets"]
        if labels.ndim != 4 or targets.ndim != 5:
            raise ValueError("CoAlign targets must be [B,H,W,A] and [B,H,W,A,7]")
        if self.num_class != 1:
            return self._multiclass_detection_loss(
                psm,
                rm,
                labels,
                targets,
                target_dict,
            )
        batch, height, width, anchors = labels.shape
        if psm.shape != (batch, anchors, height, width):
            raise ValueError(
                f"CoAlign cls output {tuple(psm.shape)} does not match labels "
                f"{tuple(labels.shape)}"
            )
        if rm.shape != (batch, anchors * 7, height, width):
            raise ValueError(
                f"CoAlign reg output {tuple(rm.shape)} does not match labels "
                f"{tuple(labels.shape)}"
            )

        positive = labels > 0
        negative_labels = target_dict.get("neg_equal_one")
        if negative_labels is None:
            negative = labels == 0
        else:
            negative = (negative_labels > 0) & (labels >= 0)
        # An ignore marker must not become a negative through a malformed or
        # padded negative tensor.
        negative = negative & ~positive & (labels >= 0)
        positive_count = positive.flatten(1).sum(dim=1, keepdim=True).float()
        normalizer = positive_count.clamp_min(1.0)
        cls_weights = (
            positive.float() * self.pos_cls_weight + negative.float()
        ).div(normalizer.view(batch, 1, 1, 1))
        cls_targets = positive.to(dtype=psm.dtype)
        cls_logits = psm.permute(0, 2, 3, 1).contiguous()
        cls_loss = self._official_focal_loss(
            cls_logits,
            cls_targets.to(dtype=cls_logits.dtype),
            cls_weights.to(dtype=cls_logits.dtype),
            alpha=self.cls_alpha,
            gamma=self.cls_gamma,
        ).sum() * self.cls_weight / float(batch)

        reg_weights = positive.float().div(normalizer.view(batch, 1, 1, 1))
        reg_preds = rm.permute(0, 2, 3, 1).contiguous().view(batch, -1, 7)
        reg_targets = targets.view(batch, -1, 7).to(dtype=reg_preds.dtype)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = self._official_smooth_l1(
            reg_preds,
            reg_targets,
            reg_weights.view(batch, -1, 1).to(dtype=reg_preds.dtype),
            sigma=self.reg_sigma,
        ).sum() * self.reg_weight / float(batch)
        total_loss = reg_loss + cls_loss
        return total_loss, reg_loss, cls_loss

    def _multiclass_detection_loss(
        self,
        psm: torch.Tensor,
        rm: torch.Tensor,
        labels: torch.Tensor,
        targets: torch.Tensor,
        target_dict: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CoAlign detection loss for class-aware anchor groups.

        UrbanIng's voxel target generator concatenates one anchor group per
        class.  With four classes and two rotations this gives eight anchor
        groups in ``pos_equal_one``/``targets`` while the detector emits four
        class logits for each group.  The public CoAlign loss only handles its
        one-class ``[B,H,W,A]`` logits, so this path expands the labels to
        one-hot ``[B,H,W,A,C]`` targets and keeps regression/direction anchor
        ordering at the eight-group level.
        """

        if psm.ndim != 4 or rm.ndim != 4:
            raise ValueError("CoAlign predictions must be four-dimensional maps")
        batch, height, width, anchors = labels.shape
        if psm.shape[:1] != (batch,) or psm.shape[2:] != (height, width):
            raise ValueError(
                f"CoAlign cls output {tuple(psm.shape)} does not match labels "
                f"{tuple(labels.shape)}"
            )
        expected_cls_channels = anchors * self.num_class
        if psm.shape[1] != expected_cls_channels:
            raise ValueError(
                "Class-aware CoAlign cls output must have one logit per "
                f"anchor/class pair: {tuple(psm.shape)} versus "
                f"{expected_cls_channels} channels for {anchors} anchors and "
                f"{self.num_class} classes."
            )
        expected_reg_channels = anchors * 7
        if rm.shape != (batch, expected_reg_channels, height, width):
            raise ValueError(
                "Class-aware CoAlign reg output must have seven values per "
                f"anchor: {tuple(rm.shape)} versus "
                f"{expected_reg_channels} channels for {anchors} anchors."
            )
        if targets.shape != (batch, height, width, anchors, 7):
            raise ValueError(
                "Class-aware CoAlign targets must align with labels: "
                f"{tuple(targets.shape)} versus "
                f"{(batch, height, width, anchors, 7)}"
            )

        positive = labels > 0
        negative_labels = target_dict.get("neg_equal_one_all", target_dict.get("neg_equal_one"))
        if negative_labels is None:
            negative = labels == 0
        else:
            if negative_labels.shape != labels.shape:
                raise ValueError(
                    "CoAlign negative mask shape must match all class/anchor labels; "
                    "use neg_equal_one_all from CoAlignVoxelPostprocessor, not a "
                    "broadcast or repetition of the legacy final-class mask."
                )
            negative = (negative_labels > 0) & (labels >= 0)
        # Keep the supplied negative mask authoritative while preventing
        # malformed negatives from turning positive or ignored anchors into
        # classification terms.
        negative = negative & ~positive & (labels >= 0)
        positive_class_labels = labels[positive]
        if positive_class_labels.numel() > 0 and (
            positive_class_labels.min() < 1
            or positive_class_labels.max() > self.num_class
        ):
            raise ValueError(
                "Class-aware CoAlign labels must use class IDs in the range "
                f"[1,{self.num_class}]"
            )

        positive_count = positive.flatten(1).sum(dim=1, keepdim=True).float()
        normalizer = positive_count.clamp_min(1.0)
        cls_weights = (
            positive.float() * self.pos_cls_weight + negative.float()
        ).div(normalizer.view(batch, 1, 1, 1))
        # Regression is positive-anchor only in the released CoAlign loss;
        # classification's positive reweighting must not leak into it.
        reg_weights = positive.float().div(normalizer.view(batch, 1, 1, 1))

        class_targets = torch.zeros(
            batch,
            height,
            width,
            anchors,
            self.num_class,
            dtype=psm.dtype,
            device=psm.device,
        )
        # Ignored/background labels are clamped only for the scatter index;
        # their positive mask removes the resulting one-hot value below.
        class_indices = labels.to(dtype=torch.long).clamp(
            min=1,
            max=self.num_class,
        ) - 1
        class_targets.scatter_(-1, class_indices.unsqueeze(-1), 1.0)
        class_targets.mul_(positive.unsqueeze(-1).to(dtype=class_targets.dtype))

        cls_logits = psm.permute(0, 2, 3, 1).contiguous().view(
            batch,
            height,
            width,
            anchors,
            self.num_class,
        )
        cls_loss = self._official_focal_loss(
            cls_logits,
            class_targets,
            cls_weights.unsqueeze(-1).to(dtype=cls_logits.dtype),
            alpha=self.cls_alpha,
            gamma=self.cls_gamma,
        ).sum() * self.cls_weight / float(batch)

        reg_preds = rm.permute(0, 2, 3, 1).contiguous().view(
            batch,
            height,
            width,
            anchors,
            7,
        )
        reg_targets = targets.to(dtype=reg_preds.dtype)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = self._official_smooth_l1(
            reg_preds,
            reg_targets,
            reg_weights.unsqueeze(-1).to(dtype=reg_preds.dtype),
            sigma=self.reg_sigma,
        ).sum() * self.reg_weight / float(batch)

        total_loss = reg_loss + cls_loss
        return total_loss, reg_loss, cls_loss

    def _direction_loss(self, output_dict, target_dict) -> torch.Tensor:
        logits = output_dict.get("dir_preds", output_dict.get("dm"))
        if logits is None:
            return next(iter(output_dict.values())).sum() * 0.0
        labels = target_dict["pos_equal_one"]
        targets = target_dict["targets"]
        if labels.ndim != 4 or targets.ndim != 5:
            raise ValueError("CoAlign direction targets must be [B,H,W,A,(7)]")
        batch, height, width, anchors = labels.shape
        expected_channels = anchors * self.direction_num_bins
        if logits.shape != (batch, expected_channels, height, width):
            raise ValueError(
                "Direction logits do not align with labels: "
                f"{tuple(logits.shape)} versus {(batch, expected_channels, height, width)}"
            )
        logits = logits.permute(0, 2, 3, 1).reshape(
            batch, height, width, anchors, self.direction_num_bins
        )
        anchor_yaw = torch.as_tensor(
            self.anchor_yaw[:anchors], device=targets.device, dtype=targets.dtype
        )
        if anchor_yaw.numel() != anchors:
            raise ValueError(
                f"anchor_yaw has {anchor_yaw.numel()} values for {anchors} anchors"
            )
        if float(anchor_yaw.abs().max()) > 2.0 * math.pi + 1e-4:
            anchor_yaw = torch.deg2rad(anchor_yaw)
        yaw = targets[..., 6] + anchor_yaw.view(1, 1, 1, anchors)
        offset = self.direction_offset
        period = 2.0 * math.pi
        direction_targets = torch.floor(
            torch.remainder(yaw - offset, period)
            / (period / float(self.direction_num_bins))
        ).long().clamp_(0, self.direction_num_bins - 1)
        positive = labels > 0
        positive_count = positive.flatten(1).sum(dim=1).float().clamp_min(1.0)
        per_anchor = F.cross_entropy(
            logits.reshape(-1, self.direction_num_bins),
            direction_targets.reshape(-1),
            reduction="none",
        ).reshape(batch, height, width, anchors)
        weights = positive.to(dtype=per_anchor.dtype)
        return (per_anchor * weights).flatten(1).sum(dim=1).div(positive_count).mean()

    def forward(self, output_dict, target_dict):
        total_loss = super().forward(output_dict, target_dict)
        if self.direction_weight == 0.0:
            return total_loss
        if "dir_preds" not in output_dict and "dm" not in output_dict:
            return total_loss
        direction_loss = self._direction_loss(output_dict, target_dict)
        total_loss = total_loss + self.direction_weight * direction_loss
        self.loss_dict.update(
            {
                "total_loss": total_loss,
                "direction_loss": direction_loss,
            }
        )
        return total_loss
