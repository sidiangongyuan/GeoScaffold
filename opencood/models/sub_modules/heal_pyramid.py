"""Independent homogeneous-LiDAR implementation of HEAL's pyramid fusion.

The module follows the public HEAL paper and configuration semantics: each
agent is represented at several BEV scales, a learned occupancy map supplies
spatial evidence, and the evidence-weighted maps are warped into the receiver
frame before aggregation.  The implementation is written locally rather than
copying code from the HEAL repository; see Lu et al., ICLR 2024 for the
method provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _warp_affine_simple(
    source: Tensor,
    matrix: Tensor,
    dsize: tuple[int, int],
    *,
    align_corners: bool,
) -> Tensor:
    """Apply a normalized affine transform without importing another fusion module."""
    if matrix.ndim != 3 or matrix.shape[-2:] != (2, 3):
        raise ValueError("matrix must have shape [N, 2, 3].")
    grid = F.affine_grid(
        matrix,
        (source.shape[0], source.shape[1], int(dsize[0]), int(dsize[1])),
        align_corners=align_corners,
    ).to(device=source.device, dtype=source.dtype)
    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )


def normalize_pairwise_tfm(
    pairwise_t_matrix: Tensor,
    height: int,
    width: int,
    discrete_ratio: float,
    downsample_rate: float = 1.0,
) -> Tensor:
    """Convert metric pairwise transforms to ``affine_grid`` coordinates.

    ``pairwise_t_matrix[i, j]`` follows OpenCOOD's convention: it maps an
    agent ``i`` feature into agent ``j``'s frame.  The returned tensor keeps
    the ``[batch, max_agents, max_agents, 2, 3]`` layout used by HEAL.  A
    normalized ``2x3`` input is accepted as a convenience for unit tests and
    callers that already performed this conversion.
    """

    if pairwise_t_matrix.ndim != 5:
        raise ValueError("pairwise_t_matrix must be [B,L,L,4,4] or [B,L,L,2,3].")
    if height <= 0 or width <= 0:
        raise ValueError("Feature-map height and width must be positive.")
    if discrete_ratio <= 0.0 or downsample_rate <= 0.0:
        raise ValueError("discrete_ratio and downsample_rate must be positive.")

    shape = pairwise_t_matrix.shape[-2:]
    if shape == (4, 4):
        affine = pairwise_t_matrix[..., :2, :][..., [0, 1, 3]].clone()
    elif shape == (2, 3):
        return pairwise_t_matrix.to(dtype=torch.float32).clone()
    else:
        raise ValueError(
            "pairwise_t_matrix must end in [4,4] or [2,3], "
            f"got {tuple(shape)}."
        )
    affine = affine.to(dtype=torch.float32)
    affine[..., 0, 1] *= float(height) / float(width)
    affine[..., 1, 0] *= float(width) / float(height)
    scale = float(discrete_ratio) * float(downsample_rate)
    affine[..., 0, 2] *= 2.0 / (scale * float(width))
    affine[..., 1, 2] *= 2.0 / (scale * float(height))
    return affine


def _identity_affine(
    batch_size: int,
    max_agents: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    identity = torch.eye(3, device=device, dtype=dtype)[:2]
    return identity.view(1, 1, 1, 2, 3).repeat(
        batch_size,
        max_agents,
        max_agents,
        1,
        1,
    )


def weighted_agent_fusion(
    features: Tensor,
    scores: Tensor,
    record_len: Tensor,
    pairwise_t_matrix: Optional[Tensor] = None,
    *,
    discrete_ratio: float = 1.0,
    downsample_rate: float = 1.0,
    align_corners: bool = False,
) -> Tensor:
    """Warp and occupancy-weight feature maps for each coalition.

    Parameters use packed OpenCOOD layout.  Scores are logits or positive
    evidence values; they are normalized directly with a softmax over agents.
    Locations sampled outside an agent's valid footprint are represented by a
    zero score and excluded from the softmax, matching HEAL's masking rule.
    """

    if features.ndim != 4:
        raise ValueError("features must have shape [sum(record_len), C, H, W].")
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [sum(record_len), 1, H, W].")
    if features.shape[0] != scores.shape[0] or features.shape[2:] != scores.shape[2:]:
        raise ValueError("features and scores must align in packed and spatial dimensions.")
    if record_len.ndim != 1:
        raise ValueError("record_len must be one-dimensional.")
    lengths = [int(value) for value in record_len.detach().cpu().tolist()]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Every coalition must contain at least one agent.")
    if sum(lengths) != features.shape[0]:
        raise ValueError("sum(record_len) must equal the packed feature count.")

    batch_size = len(lengths)
    max_agents = max(lengths)
    if pairwise_t_matrix is None:
        affine = _identity_affine(
            batch_size,
            max_agents,
            device=features.device,
            dtype=torch.float32,
        )
    else:
        if pairwise_t_matrix.ndim != 5 or pairwise_t_matrix.shape[0] != batch_size:
            raise ValueError(
                "pairwise_t_matrix must cover the current batch."
            )
        if pairwise_t_matrix.shape[1] < max_agents or pairwise_t_matrix.shape[2] < max_agents:
            raise ValueError(
                "pairwise_t_matrix must cover [batch,max_agents,max_agents]."
            )
        if pairwise_t_matrix.shape[-2:] == (4, 4):
            affine = normalize_pairwise_tfm(
                pairwise_t_matrix,
                features.shape[2],
                features.shape[3],
                discrete_ratio,
                downsample_rate,
            )
        elif pairwise_t_matrix.shape[-2:] == (2, 3):
            affine = pairwise_t_matrix.to(dtype=torch.float32)
        else:
            raise ValueError("Pairwise transforms must end in [4,4] or [2,3].")
        affine = affine.to(device=features.device)

    outputs = []
    cursor = 0
    height, width = features.shape[2:]
    for batch_index, agent_count in enumerate(lengths):
        packed_slice = features[cursor : cursor + agent_count]
        score_slice = scores[cursor : cursor + agent_count]
        transform = affine[batch_index, 0, :agent_count]
        warped_features = _warp_affine_simple(
            packed_slice,
            transform,
            (height, width),
            align_corners=align_corners,
        )
        warped_scores = _warp_affine_simple(
            score_slice,
            transform,
            (height, width),
            align_corners=align_corners,
        )
        masked_scores = warped_scores.masked_fill(warped_scores == 0, -float("inf"))
        weights = torch.softmax(masked_scores, dim=0)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        outputs.append(torch.sum(warped_features * weights, dim=0))
        cursor += agent_count
    return torch.stack(outputs, dim=0)


class _ResNeXtBottleneck(nn.Module):
    """A compact expansion-one grouped residual bottleneck used by HEAL."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        width_per_group: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride, groups, width_per_group) <= 0:
            raise ValueError("Residual dimensions and stride must be positive.")
        base_width = max(1, int(out_channels * width_per_group / 64.0))
        inner_channels = max(groups, base_width * groups)
        if inner_channels % groups != 0:
            raise ValueError("Grouped bottleneck width must be divisible by groups.")

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(
            inner_channels,
            inner_channels,
            3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv3 = nn.Conv2d(inner_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features: Tensor) -> Tensor:
        identity = features
        output = self.relu(self.bn1(self.conv1(features)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            identity = self.downsample(features)
        return self.relu(output + identity)


class _ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        groups: int,
        width_per_group: int,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("Each pyramid level must contain at least one block.")
        layers = [
            _ResNeXtBottleneck(
                in_channels,
                out_channels,
                stride,
                groups,
                width_per_group,
            )
        ]
        layers.extend(
            _ResNeXtBottleneck(
                out_channels,
                out_channels,
                1,
                groups,
                width_per_group,
            )
            for _ in range(blocks - 1)
        )
        self.blocks = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.blocks(features)


class HEALMultiscaleBackbone(nn.Module):
    """Independent ResNeXt-like multiscale backbone for pyramid fusion."""

    def __init__(
        self,
        layer_nums: Sequence[int],
        layer_strides: Sequence[int],
        num_filters: Sequence[int],
        *,
        input_channels: int = 64,
        groups: int = 32,
        width_per_group: int = 4,
        resnext: bool = True,
    ) -> None:
        super().__init__()
        if not (len(layer_nums) == len(layer_strides) == len(num_filters)):
            raise ValueError("Pyramid layer configuration lengths must match.")
        if not layer_nums:
            raise ValueError("At least one pyramid level is required.")
        if not resnext:
            groups = 1
            width_per_group = 64
        stages = []
        in_channels = int(input_channels)
        for blocks, stride, out_channels in zip(layer_nums, layer_strides, num_filters):
            stages.append(
                _ResidualStage(
                    in_channels,
                    int(out_channels),
                    int(blocks),
                    int(stride),
                    int(groups),
                    int(width_per_group),
                )
            )
            in_channels = int(out_channels)
        self.stages = nn.ModuleList(stages)
        self.num_levels = len(stages)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: Tensor) -> list[Tensor]:
        outputs = []
        for stage in self.stages:
            features = stage(features)
            outputs.append(features)
        return outputs


class HEALPyramidFusion(nn.Module):
    """Multiscale occupancy-weighted fusion used by homogeneous HEAL."""

    def __init__(
        self,
        model_cfg: dict,
        *,
        input_channels: int = 64,
        output_channels: int = 256,
        discrete_ratio: float = 0.4,
        downsample_rate: float = 2.0,
    ) -> None:
        super().__init__()
        layer_nums = tuple(model_cfg.get("layer_nums", (3, 5, 8)))
        layer_strides = tuple(model_cfg.get("layer_strides", (1, 2, 2)))
        num_filters = tuple(model_cfg.get("num_filters", (64, 128, 256)))
        upsample_strides = tuple(model_cfg.get("upsample_strides", (1, 2, 4)))
        num_upsample = tuple(
            model_cfg.get("num_upsample_filter", (128, 128, 128))
        )
        if len(layer_nums) != len(upsample_strides):
            raise ValueError("Every pyramid level needs one upsample stride.")
        if len(num_upsample) != len(layer_nums):
            raise ValueError("Every pyramid level needs one decoder channel count.")
        if output_channels <= 0:
            raise ValueError("output_channels must be positive.")

        self.backbone = HEALMultiscaleBackbone(
            layer_nums,
            layer_strides,
            num_filters,
            input_channels=input_channels,
            groups=int(model_cfg.get("groups", 32)),
            width_per_group=int(model_cfg.get("width_per_group", 4)),
            resnext=bool(model_cfg.get("resnext", True)),
        )
        self.occupancy_heads = nn.ModuleList(
            nn.Conv2d(int(channels), 1, kernel_size=1)
            for channels in num_filters
        )
        self.deblocks = nn.ModuleList()
        for channels, decoder_channels, stride in zip(
            num_filters,
            num_upsample,
            upsample_strides,
        ):
            stride = float(stride)
            if stride <= 0.0:
                raise ValueError("Upsample strides must be positive.")
            if stride >= 1.0:
                integer_stride = int(stride)
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            int(channels),
                            int(decoder_channels),
                            integer_stride,
                            stride=integer_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(int(decoder_channels), eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                integer_stride = int(round(1.0 / stride))
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(channels),
                            int(decoder_channels),
                            integer_stride,
                            stride=integer_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(int(decoder_channels), eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )
        # HEAL's stage-1 config applies the repository's DoubleConv shrink
        # header after concatenating the three decoded pyramid levels.
        decoder_input = sum(map(int, num_upsample))
        self.projection = nn.Sequential(
            nn.Conv2d(decoder_input, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.align_corners = bool(model_cfg.get("align_corners", False))
        self.discrete_ratio = float(model_cfg.get("discrete_ratio", discrete_ratio))
        self.downsample_rate = float(model_cfg.get("downsample_rate", downsample_rate))
        self.num_levels = len(layer_nums)
        self.output_channels = int(output_channels)

    def _decode(self, feature_list: Sequence[Tensor]) -> Tensor:
        decoded = [deblock(features) for deblock, features in zip(self.deblocks, feature_list)]
        target_size = decoded[0].shape[2:]
        decoded = [
            item
            if item.shape[2:] == target_size
            else F.interpolate(
                item,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for item in decoded
        ]
        return self.projection(torch.cat(decoded, dim=1))

    def forward_single(self, spatial_features: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Decode one agent and return its occupancy logits at each level."""
        feature_list = self.backbone(spatial_features)
        occupancy = [head(features) for head, features in zip(self.occupancy_heads, feature_list)]
        return self._decode(feature_list), occupancy

    def forward_collab(
        self,
        spatial_features: Tensor,
        record_len: Tensor,
        pairwise_t_matrix: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Fuse packed agents and return final feature plus occupancy logits."""
        feature_list = self.backbone(spatial_features)
        fused_levels = []
        occupancy = []
        for level, (features, head) in enumerate(zip(feature_list, self.occupancy_heads)):
            logits = head(features)
            occupancy.append(logits)
            scores = torch.sigmoid(logits) + 1e-4
            level_stride = self.downsample_rate
            if level > 0:
                level_stride *= float(2**level)
            fused_levels.append(
                weighted_agent_fusion(
                    features,
                    scores,
                    record_len,
                    pairwise_t_matrix,
                    discrete_ratio=self.discrete_ratio,
                    downsample_rate=level_stride,
                    align_corners=self.align_corners,
                )
            )
        return self._decode(fused_levels), occupancy

    def forward(
        self,
        spatial_features: Tensor,
        record_len: Optional[Tensor] = None,
        pairwise_t_matrix: Optional[Tensor] = None,
    ):
        if record_len is None:
            return self.forward_single(spatial_features)
        return self.forward_collab(spatial_features, record_len, pairwise_t_matrix)
