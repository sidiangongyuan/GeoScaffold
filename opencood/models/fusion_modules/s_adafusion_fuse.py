# Based on the official S-AdaFusion implementation by Donghao (Jeremy) Qiao.
# Copyright (c) 2022 Donghao (Jeremy) Qiao.  Licensed under the MIT License.

"""Spatially adaptive feature fusion from Qiao and Zulkernine (WACV 2023).

This implementation follows the official MIT-licensed reference: source-wise
maximum and mean statistics are stacked and processed by a single 3-D
convolution followed by ReLU.  The module intentionally does not add spatial
warping; the UrbanIng intermediate-fusion dataset projects source BEV maps
into the receiver frame before this source-wise operator is called.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SpatialFusion(nn.Module):
    """Official S-AdaFusion source-statistics fusion operator."""

    def __init__(self) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _validate_inputs(x: Tensor, record_len: Tensor | Sequence[int]) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"S-AdaFusion expects [sum_agents, C, H, W], got {tuple(x.shape)}."
            )
        if not torch.is_tensor(record_len):
            record_len = torch.as_tensor(record_len, dtype=torch.long, device=x.device)
        else:
            record_len = record_len.to(device=x.device, dtype=torch.long)
        if record_len.ndim != 1 or record_len.numel() == 0:
            raise ValueError("record_len must be a non-empty one-dimensional tensor.")
        if torch.any(record_len <= 0):
            raise ValueError("Every coalition must contain at least one source agent.")
        if int(record_len.sum().item()) != x.shape[0]:
            raise ValueError(
                f"sum(record_len)={int(record_len.sum().item())} does not match "
                f"source count {x.shape[0]}"
            )
        return record_len

    def regroup(self, x: Tensor, record_len: Tensor | Sequence[int]) -> list[Tensor]:
        """Split packed source features into one tensor per coalition."""

        record_len = self._validate_inputs(x, record_len)
        cumulative = torch.cumsum(record_len, dim=0)
        return list(torch.tensor_split(x, cumulative[:-1].detach().cpu()))

    def forward(self, x: Tensor, record_len: Tensor | Sequence[int]) -> Tensor:
        """Fuse source features into one BEV map per coalition."""

        split_x = self.regroup(x, record_len)
        source_statistics = []
        for coalition in split_x:
            source_max = coalition.max(dim=0, keepdim=True).values
            source_mean = coalition.mean(dim=0, keepdim=True)
            source_statistics.append(torch.cat((source_max, source_mean), dim=0).unsqueeze(0))

        stacked_statistics = torch.cat(source_statistics, dim=0)
        return self.conv3d(stacked_statistics).squeeze(1)


SAdaFusion = SpatialFusion
