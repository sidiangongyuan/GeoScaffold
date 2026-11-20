from __future__ import annotations

import torch


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index batched points with batched indices."""
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def merge_tokens(
    x: torch.Tensor,
    idx_cluster: torch.Tensor,
    cluster_num: int,
    token_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge tokens that fall into the same cluster."""
    batch_size, token_count, channels = x.shape
    idx_token = torch.arange(token_count, device=x.device)[None, :].repeat(batch_size, 1)
    agg_weight = x.new_ones(batch_size, token_count, 1)
    if token_weight is None:
        token_weight = x.new_ones(batch_size, token_count, 1)

    idx_batch = torch.arange(batch_size, device=x.device)[:, None]
    flat_cluster_index = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(batch_size * cluster_num, 1)
    all_weight.index_add_(
        dim=0,
        index=flat_cluster_index.reshape(batch_size * token_count),
        source=token_weight.reshape(batch_size * token_count, 1),
    )
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[flat_cluster_index]

    merged = x.new_zeros(batch_size * cluster_num, channels)
    merged.index_add_(
        dim=0,
        index=flat_cluster_index.reshape(batch_size * token_count),
        source=(x * norm_weight).reshape(batch_size * token_count, channels).type(x.dtype),
    )
    merged = merged.reshape(batch_size, cluster_num, channels)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]
    return merged, idx_token_new


def cluster_dpc_knn(
    x: torch.Tensor,
    cluster_num: int,
    k: int = 5,
) -> tuple[torch.Tensor, int]:
    """Cluster tokens with the DPC-KNN rule from the official ERMVP release."""
    with torch.no_grad():
        batch_size, token_count, channels = x.shape
        distance = torch.cdist(x, x) / (channels ** 0.5)
        k = min(int(k), int(token_count))
        nearest_distance, _ = torch.topk(distance, k=k, dim=-1, largest=False)
        density = (-(nearest_distance ** 2).mean(dim=-1)).exp()
        # The official random epsilon only breaks equal-density ties. A fixed
        # ordinal epsilon preserves that role while keeping inference replayable.
        tie_break = torch.arange(token_count, device=x.device, dtype=x.dtype)
        tie_break = tie_break / max(token_count - 1, 1)
        density = density + tie_break.unsqueeze(0) * 1e-6

        mask = (density[:, None, :] > density[:, :, None]).type_as(x)
        max_distance = distance.flatten(1).max(dim=-1)[0][:, None, None]
        distance_to_higher_density = (
            distance * mask + max_distance * (1 - mask)
        ).min(dim=-1)[0]
        score = distance_to_higher_density * density
        _, centers = torch.topk(score, k=cluster_num, dim=-1)
        center_distance = index_points(distance, centers)
        idx_cluster = center_distance.argmin(dim=1)

        idx_batch = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(batch_size, cluster_num)
        idx_cluster[idx_batch.reshape(-1), centers.reshape(-1)] = idx_tmp.reshape(-1)
    return idx_cluster, cluster_num
