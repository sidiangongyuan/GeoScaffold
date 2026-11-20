from __future__ import annotations

from collections.abc import Mapping

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.sub_modules.geometry_scaffold import field_ce_dice_loss


class PointPillarErmvpLoss(PointPillarLoss):
    """ERMVP loss with optional separate ego supervision bundle."""

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__(args)
        if "ego_weight" in args:
            self.ego_auxiliary_weight = float(args["ego_weight"])
        if self.ego_auxiliary_weight < 0.0:
            raise ValueError("ERMVP ego auxiliary weight must be non-negative.")

    @staticmethod
    def _normalize_output_dict(output_dict):
        normalized = dict(output_dict)
        if "ego_psm" not in normalized and "psm_ego" in normalized:
            normalized["ego_psm"] = normalized["psm_ego"]
        if "ego_rm" not in normalized and "rm_ego" in normalized:
            normalized["ego_rm"] = normalized["rm_ego"]
        return normalized

    def forward(self, output_dict, target_dict):
        normalized_output = self._normalize_output_dict(output_dict)
        if not isinstance(target_dict, Mapping) or "label_dict" not in target_dict:
            return super().forward(normalized_output, target_dict)

        main_target = target_dict["label_dict"]
        total_loss, reg_loss, conf_loss = self._detection_loss(normalized_output, main_target)
        self.loss_dict = {
            "total_loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
        }

        if self.ego_auxiliary_weight != 0.0:
            if "ego_psm" not in normalized_output or "ego_rm" not in normalized_output:
                raise KeyError(
                    "ERMVP ego auxiliary loss requires ego_psm/ego_rm or psm_ego/rm_ego."
                )
            ego_target = target_dict.get("ego_label_dict", main_target)
            ego_loss, ego_reg_loss, ego_conf_loss = self._detection_loss(
                {
                    "psm": normalized_output["ego_psm"],
                    "rm": normalized_output["ego_rm"],
                },
                ego_target,
                apply_receiver_reweighting=False,
            )
            total_loss = total_loss + self.ego_auxiliary_weight * ego_loss
            self.loss_dict.update(
                {
                    "total_loss": total_loss,
                    "ego_loss": ego_loss,
                    "ego_reg_loss": ego_reg_loss,
                    "ego_conf_loss": ego_conf_loss,
                }
            )

        if self.geoscaffold_weight != 0.0 and "geoscaffold_logits" in normalized_output:
            if "geoscaffold_targets" not in normalized_output:
                raise KeyError(
                    "geoscaffold_targets are required when geoscaffold_logits are present"
                )
            geoscaffold_loss = field_ce_dice_loss(
                normalized_output["geoscaffold_logits"],
                normalized_output["geoscaffold_targets"],
                ce_weight=self.geoscaffold_ce_weight,
                dice_weight=self.geoscaffold_dice_weight,
            )
            total_loss = total_loss + self.geoscaffold_weight * geoscaffold_loss
            self.loss_dict.update(
                {
                    "total_loss": total_loss,
                    "geoscaffold_loss": geoscaffold_loss,
                }
            )
            if "geoscaffold_positive_count" in normalized_output:
                self.loss_dict["geoscaffold_positive_count"] = normalized_output[
                    "geoscaffold_positive_count"
                ]

        total_loss = self._apply_global_bev_footprint_loss(
            total_loss,
            normalized_output,
        )

        if "receiver_evidence_missing_hit_count" in main_target:
            self.loss_dict["receiver_evidence_missing_hit_count"] = main_target[
                "receiver_evidence_missing_hit_count"
            ].sum()
        return total_loss
