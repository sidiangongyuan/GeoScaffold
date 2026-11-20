"""Training-only global BEV footprint auxiliary.

This module is intentionally independent of GeoScaffold's anchor-canonical
representation.  It predicts one dense, class-agnostic footprint field from
the complete fused BEV map and never participates in decoding or evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


def resolve_global_bev_footprint_config(
    args: Mapping[str, object],
) -> dict[str, int | bool]:
    """Resolve model-side settings while keeping the default path disabled."""

    config = args.get("global_bev_footprint", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("global_bev_footprint must be a mapping when provided.")
    enabled = bool(config.get("enabled", False))
    hidden_channels = int(config.get("hidden_channels", 128))
    if hidden_channels <= 0:
        raise ValueError("global_bev_footprint.hidden_channels must be positive.")
    return {
        "enabled": enabled,
        "hidden_channels": hidden_channels,
    }


def resolve_global_bev_footprint_loss_config(
    args: Mapping[str, object],
) -> tuple[float, float, float]:
    """Resolve total, CE, and Dice weights for the global auxiliary."""

    config = args.get("global_bev_footprint", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("loss.args.global_bev_footprint must be a mapping.")
    total_weight = float(config.get("weight", 0.0))
    objective = str(config.get("objective", "ce_dice")).lower()
    objective = objective.replace("-", "_").replace("+", "_")
    aliases = {
        "ce": (1.0, 0.0),
        "ce_only": (1.0, 0.0),
        "cross_entropy": (1.0, 0.0),
        "dice": (0.0, 1.0),
        "dice_only": (0.0, 1.0),
        "ce_dice": (1.0, 1.0),
        "dice_ce": (1.0, 1.0),
        "ce_plus_dice": (1.0, 1.0),
    }
    if objective not in aliases:
        raise ValueError(
            "global_bev_footprint.objective must be ce, dice, or ce_dice; "
            f"got {objective!r}."
        )
    ce_weight, dice_weight = aliases[objective]
    if "ce_weight" in config:
        ce_weight = float(config["ce_weight"])
    if "dice_weight" in config:
        dice_weight = float(config["dice_weight"])
    if (
        not math.isfinite(total_weight)
        or not math.isfinite(ce_weight)
        or not math.isfinite(dice_weight)
        or total_weight < 0.0
        or ce_weight < 0.0
        or dice_weight < 0.0
    ):
        raise ValueError("Global BEV footprint loss weights must be finite and non-negative.")
    if total_weight != 0.0 and ce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError("Global BEV footprint CE/Dice weights cannot both be zero.")
    return total_weight, ce_weight, dice_weight


class GlobalBEVFootprintHead(nn.Module):
    """Fully convolutional two-class footprint predictor on the fused BEV map."""

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Global BEV footprint channels must be positive.")
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.logits = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, fused_bev: torch.Tensor) -> torch.Tensor:
        if fused_bev.ndim != 4:
            raise ValueError("fused_bev must have shape [B, C, H, W].")
        return self.logits(self.trunk(self.stem(fused_bev)))

