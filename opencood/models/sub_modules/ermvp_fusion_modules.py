from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual


class SE(nn.Module):
    def __init__(self, inp: int, oup: int, expansion: float = 0.25) -> None:
        super().__init__()
        hidden = max(int(inp * expansion), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        pooled = self.avg_pool(x).view(batch_size, channels)
        scale = self.fc(pooled).view(batch_size, channels, 1, 1)
        return x * scale


class FeatExtract(nn.Module):
    def __init__(self, dim: int, strides: tuple[int, int], keep_dim: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.keep_dim = bool(keep_dim)
        if not self.keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttentionGlobal(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        dim_head: int = 32,
        dropout: float = 0.0,
        agent_size: int = 2,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if dim % dim_head != 0:
            raise ValueError("dimension should be divisible by the head dimension")
        self.head_dim = dim_head
        self.num_heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.num_heads,
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.to_q_global = nn.Sequential(
            FeatExtract(dim, (2, 2), keep_dim=False),
            FeatExtract(dim, (2, 2), keep_dim=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        (
            batch_size,
            agent_size,
            height,
            width,
            window_height,
            window_width,
            _,
        ) = x.shape
        x_global = rearrange(x, "b m x y w1 w2 d -> (b m) d (x w1) (y w2)")
        x_global = self.to_q_global(x_global)
        global_height, global_width = x_global.shape[-2:]
        pad_height = (-global_height) % window_height
        pad_width = (-global_width) % window_width
        global_valid = x_global.new_ones(
            (batch_size * agent_size, 1, global_height, global_width)
        )
        if pad_height or pad_width:
            padding = (0, pad_width, 0, pad_height)
            x_global = F.pad(x_global, padding)
            global_valid = F.pad(global_valid, padding)
        x_global = rearrange(
            x_global,
            "(b m) d (x w1) (y w2) -> b m x y w1 w2 d",
            m=agent_size,
            w1=window_height,
            w2=window_width,
        )
        global_valid = rearrange(
            global_valid,
            "(b m) d (x w1) (y w2) -> b m x y w1 w2 d",
            b=batch_size,
            m=agent_size,
            w1=window_height,
            w2=window_width,
        )
        x_global = rearrange(x_global, "b l x y w1 w2 d -> b (x y) (l w1 w2) d")
        global_valid = rearrange(
            global_valid,
            "b l x y w1 w2 d -> b (x y) (l w1 w2) d",
        )
        global_logits = x_global @ x_global.transpose(-2, -1)
        global_logits = global_logits.masked_fill(
            global_valid.transpose(-2, -1) == 0,
            -float("inf"),
        )
        global_attention = self.softmax(global_logits) * (
            x_global.shape[-1] ** -0.5
        )
        x_global = einsum("b h i j, b h j d -> b h i d", global_attention, x_global)

        x_global = x_global * global_valid
        valid_groups = global_valid.sum(dim=1).clamp_min(1.0)
        q_global = (x_global.sum(dim=1) / valid_groups).unsqueeze(1)
        q_global = q_global.reshape(
            batch_size,
            1,
            agent_size * window_height * window_width,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)

        tokens = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        batch_tokens, token_count, channels = tokens.shape
        kv = self.qkv(tokens).reshape(
            batch_tokens,
            token_count,
            2,
            self.num_heads,
            channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        tiles_per_batch = batch_tokens // batch_size
        q = q_global.repeat(1, tiles_per_batch, 1, 1, 1).reshape(
            batch_tokens,
            self.num_heads,
            token_count,
            channels // self.num_heads,
        )
        attention = (q * self.scale) @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table(self.relative_position_index)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn_mask = rearrange(mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)").unsqueeze(1)
            attention = attention.masked_fill(attn_mask == 0, -float("inf"))

        attention = self.softmax(attention)
        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(
            out,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agent_size,
            w1=window_height,
            w2=window_width,
        )
        out = self.to_out(out)
        return rearrange(
            out,
            "(b x y) l w1 w2 d -> b l x y w1 w2 d",
            b=batch_size,
            x=height,
            y=width,
        )


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        dim_head: int = 32,
        dropout: float = 0.0,
        agent_size: int = 6,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if dim % dim_head != 0:
            raise ValueError("dimension should be divisible by the head dimension")
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.heads,
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        (
            batch_size,
            agent_size,
            height,
            width,
            window_height,
            window_width,
            _,
        ) = x.shape
        tokens = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        q, k, v = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )
        sim = einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, "i j h -> h i j")

        if mask is not None:
            attn_mask = rearrange(mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)").unsqueeze(1)
            sim = sim.masked_fill(attn_mask == 0, -float("inf"))

        attention = self.attend(sim)
        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(
            out,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agent_size,
            w1=window_height,
            w2=window_width,
        )
        out = self.to_out(out)
        return rearrange(
            out,
            "(b x y) l w1 w2 d -> b l x y w1 w2 d",
            b=batch_size,
            x=height,
            y=width,
        )


class ERMVPFusionBlockMask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        dim_head: int,
        window_size: int,
        agent_size: int,
        drop_out: float,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.window_attention = PreNormResidual(
            input_dim,
            Attention(input_dim, dim_head, drop_out, agent_size, window_size),
        )
        self.window_ffd = PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out))
        self.window_attention_global = PreNormResidual(
            input_dim,
            WindowAttentionGlobal(input_dim, dim_head, drop_out, agent_size, window_size),
        )
        self.window_global_ffd = PreNormResidual(
            input_dim,
            FeedForward(input_dim, mlp_dim, drop_out),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_window = rearrange(
            mask,
            "b (x w1) (y w2) e l -> b x y w1 w2 e l",
            w1=self.window_size,
            w2=self.window_size,
        )
        x = rearrange(
            x,
            "b m d (x w1) (y w2) -> b m x y w1 w2 d",
            w1=self.window_size,
            w2=self.window_size,
        )
        x = self.window_attention(x, mask=mask_window)
        x = self.window_ffd(x)
        x = rearrange(x, "b m x y w1 w2 d -> b m d (x w1) (y w2)")

        mask_window = rearrange(
            mask,
            "b (x w1) (y w2) e l -> b x y w1 w2 e l",
            w1=self.window_size,
            w2=self.window_size,
        )
        x = rearrange(
            x,
            "b m d (x w1) (y w2) -> b m x y w1 w2 d",
            w1=self.window_size,
            w2=self.window_size,
        )
        x = self.window_attention_global(x, mask=mask_window)
        x = self.window_global_ffd(x)
        return rearrange(x, "b m x y w1 w2 d -> b m d (x w1) (y w2)")


class ERMVPFusionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        dim_head: int,
        window_size: int,
        agent_size: int,
        drop_out: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Rearrange("b m d (x w1) (y w2) -> b m x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(
                input_dim,
                Attention(input_dim, dim_head, drop_out, agent_size, window_size),
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b m x y w1 w2 d -> b m d (x w1) (y w2)"),
            Rearrange("b m d (w1 x) (w2 y) -> b m x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(
                input_dim,
                Attention(input_dim, dim_head, drop_out, agent_size, window_size),
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b m x y w1 w2 d -> b m d (w1 x) (w2 y)"),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return self.block(x)


class ERMVPFusionEncoder(nn.Module):
    def __init__(self, args: dict[str, int | float | bool]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        depth = int(args["depth"])
        input_dim = int(args["input_dim"])
        mlp_dim = int(args["mlp_dim"])
        agent_size = int(args["agent_size"])
        window_size = int(args["window_size"])
        drop_out = float(args["drop_out"])
        dim_head = int(args["dim_head"])
        use_mask = bool(args.get("mask", False))

        for _ in range(depth):
            if use_mask:
                block = ERMVPFusionBlockMask(
                    input_dim,
                    mlp_dim,
                    dim_head,
                    window_size,
                    agent_size,
                    drop_out,
                )
            else:
                block = ERMVPFusionBlock(
                    input_dim,
                    mlp_dim,
                    dim_head,
                    window_size,
                    agent_size,
                    drop_out,
                )
            self.layers.append(block)

        self.mlp_head = nn.Sequential(
            Reduce("b m d h w -> b d h w", "mean"),
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.mlp_head(x)
