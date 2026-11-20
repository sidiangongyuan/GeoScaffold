from __future__ import annotations

import torch
from torch import nn


class SortSampler(nn.Module):
    """Official ERMVP top-k token sampler, kept device-agnostic."""

    def __init__(
        self,
        topk_ratio: float,
        input_dim: int,
        score_pred_net: str = "2layer-fc-256",
    ) -> None:
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        if score_pred_net == "2layer-fc-256":
            self.score_pred_net = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(input_dim, 1, kernel_size=1),
            )
        elif score_pred_net == "2layer-fc-32":
            self.score_pred_net = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )
        elif score_pred_net == "1layer-fc":
            self.score_pred_net = nn.Conv2d(input_dim, 1, kernel_size=1)
        else:
            raise ValueError(f"Unsupported score predictor: {score_pred_net}")

        self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)

    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor,
        sample_ratio: float | None,
        dis_priority: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = src.shape
        weighted_src = dis_priority.unsqueeze(1) * src.float()
        sample_weight = self.score_pred_net(weighted_src).sigmoid().view(batch_size, -1)
        detached_weight = sample_weight.detach()

        ratio = self.topk_ratio if sample_ratio is None else float(sample_ratio)
        sample_lens = torch.full(
            (batch_size, 1),
            int(height * width * ratio),
            device=src.device,
            dtype=torch.int64,
        )
        sample_lens = torch.clamp(sample_lens, min=1, max=height * width)
        max_sample_num = int(sample_lens.max().item())

        sort_order = detached_weight.sort(descending=True, dim=1)[1]
        topk_indices = sort_order[:, :max_sample_num]

        flattened_src = src.flatten(2).permute(2, 0, 1)
        flattened_src = self.norm_feature(flattened_src)

        topk_scores = sample_weight.gather(1, topk_indices)
        sample_reg_loss = topk_scores.mean()
        sampled_src = flattened_src.gather(
            0,
            topk_indices.permute(1, 0)[..., None].expand(-1, -1, channels),
        ) * topk_scores.permute(1, 0).unsqueeze(-1)
        sampled_pos = pos_embed.gather(
            0,
            topk_indices.permute(1, 0)[..., None],
        )
        return sampled_src, sample_reg_loss, topk_indices, sampled_pos
