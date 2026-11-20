from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


def _lookup(output_dict: Mapping[str, torch.Tensor], *names: str) -> torch.Tensor:
    for name in names:
        if name in output_dict:
            return output_dict[name]
    raise KeyError(f"Expected one of {names} in model outputs.")


class PointPillarDiscoNetLoss(PointPillarLoss):
    """Native PointPillar detection loss plus DiscoNet feature distillation."""

    def __init__(self, args: Mapping[str, object]) -> None:
        super().__init__(args)
        kd = args.get("kd", {})
        if kd is None:
            kd = {}
        if not isinstance(kd, Mapping):
            raise TypeError("loss.args.kd must be a mapping when provided.")
        self.kd_weight = float(kd.get("weight", 0.0))
        self.kd_temperature = float(kd.get("temperature", 1.0))
        self.decoder_kd = bool(kd.get("decoder_kd", False))
        if self.kd_weight < 0.0:
            raise ValueError("DiscoNet KD weight must be non-negative.")
        if self.kd_temperature <= 0.0:
            raise ValueError("DiscoNet KD temperature must be positive.")

    def _kl_distill(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape != teacher.shape:
            raise ValueError(
                "KD tensors must share the same shape, got "
                f"{tuple(student.shape)} and {tuple(teacher.shape)}."
            )
        temperature = self.kd_temperature
        student_logits = student.reshape(-1, student.shape[-1]) / temperature
        teacher_logits = teacher.reshape(-1, teacher.shape[-1]) / temperature
        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

    @staticmethod
    def _align_teacher_tensor(
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Pool a higher-resolution frozen teacher tensor onto the student BEV grid."""
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError(f"DiscoNet {name} KD expects BCHW tensors.")
        if student.shape[:2] != teacher.shape[:2]:
            raise ValueError(
                f"Student and teacher {name} tensors must align batch/channels, got "
                f"{tuple(student.shape[:2])} and {tuple(teacher.shape[:2])}."
            )
        if student.shape[-2:] == teacher.shape[-2:]:
            return teacher
        student_h, student_w = student.shape[-2:]
        teacher_h, teacher_w = teacher.shape[-2:]
        if (
            teacher_h < student_h
            or teacher_w < student_w
            or teacher_h % student_h != 0
            or teacher_w % student_w != 0
        ):
            raise ValueError(
                "Frozen teacher BEV resolution must be an integer multiple of "
                f"the student grid, got {teacher_h}x{teacher_w} and "
                f"{student_h}x{student_w}."
            )
        return F.avg_pool2d(
            teacher,
            kernel_size=(teacher_h // student_h, teacher_w // student_w),
            stride=(teacher_h // student_h, teacher_w // student_w),
        )

    def forward(self, output_dict, target_dict):
        total_loss = super().forward(output_dict, target_dict)
        if self.kd_weight == 0.0:
            return total_loss
        if "teacher_feature" not in output_dict:
            return total_loss

        student_feature = _lookup(output_dict, "student_kd_feature", "feature")
        teacher_feature = self._align_teacher_tensor(
            student_feature,
            _lookup(output_dict, "teacher_feature"),
            name="feature",
        )
        kd_loss = self._kl_distill(
            student_feature.permute(0, 2, 3, 1).contiguous(),
            teacher_feature.permute(0, 2, 3, 1).contiguous(),
        )
        if self.decoder_kd:
            student_psm = _lookup(output_dict, "psm", "cls_preds")
            teacher_psm = self._align_teacher_tensor(
                student_psm,
                _lookup(output_dict, "teacher_psm", "teacher_cls_preds"),
                name="classification",
            )
            kd_loss = kd_loss + self._kl_distill(
                student_psm.permute(0, 2, 3, 1).contiguous(),
                teacher_psm.permute(0, 2, 3, 1).contiguous(),
            )
            student_rm = _lookup(output_dict, "rm", "reg_preds")
            teacher_rm = self._align_teacher_tensor(
                student_rm,
                _lookup(output_dict, "teacher_rm", "teacher_reg_preds"),
                name="regression",
            )
            kd_loss = kd_loss + self._kl_distill(
                student_rm.permute(0, 2, 3, 1).contiguous(),
                teacher_rm.permute(0, 2, 3, 1).contiguous(),
            )

        kd_loss = kd_loss * self.kd_weight
        total_loss = total_loss + kd_loss
        self.loss_dict.update({"total_loss": total_loss, "kd_loss": kd_loss})
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        super().logging(epoch, batch_id, batch_len, writer, pbar=pbar)
        if "kd_loss" in self.loss_dict:
            writer.add_scalar(
                "KD_loss",
                self.loss_dict["kd_loss"].item(),
                epoch * batch_len + batch_id,
            )
