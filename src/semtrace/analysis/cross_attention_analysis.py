from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from semtrace.models.cross_attention import SemanticTraceCrossAttention


@torch.inference_mode()
def masked_cross_attention(
    module: SemanticTraceCrossAttention,
    semantic_anchor: torch.Tensor,
    trace_tokens: torch.Tensor,
    *,
    head_keep_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce eval-time MHA and optionally remove head value outputs."""
    attention = module.attention
    weight = attention.in_proj_weight
    if weight is None:
        raise ValueError("analysis requires combined MultiheadAttention projection weights")
    bias = attention.in_proj_bias
    embed_dim = attention.embed_dim
    heads = attention.num_heads
    head_dim = embed_dim // heads
    query_token = module.semantic_query(semantic_anchor.detach()).unsqueeze(1)
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = (
        bias.chunk(3, dim=0) if bias is not None else (None, None, None)
    )
    query = F.linear(query_token, q_weight, q_bias)
    key = F.linear(trace_tokens, k_weight, k_bias)
    value = F.linear(trace_tokens, v_weight, v_bias)
    batch_size, patch_count, _ = trace_tokens.shape
    query = query.reshape(batch_size, 1, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch_size, patch_count, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch_size, patch_count, heads, head_dim).transpose(1, 2)
    weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)
    head_outputs = weights @ value
    if head_keep_mask is not None:
        if head_keep_mask.shape != (heads,):
            raise ValueError("head_keep_mask must have one value per attention head")
        head_outputs = head_outputs * head_keep_mask.to(head_outputs)[None, :, None, None]
    concatenated = head_outputs.transpose(1, 2).reshape(batch_size, 1, embed_dim)
    attended = attention.out_proj(concatenated)[:, 0]
    gate = module.semantic_gate
    evidence = module.output_norm(
        (1.0 - gate) * trace_tokens.mean(dim=1) + gate * attended
    )
    return evidence, weights


def attention_statistics(
    weights: torch.Tensor,
    *,
    top_k_fraction: float = 0.1,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    if weights.ndim != 4 or weights.shape[2] != 1:
        raise ValueError("attention weights must have shape [batch, heads, 1, patches]")
    values = weights[:, :, 0]
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    entropy = -(values * values.clamp_min(eps).log()).sum(dim=-1)
    effective = values.square().sum(dim=-1).clamp_min(eps).reciprocal()
    sorted_values = values.sort(dim=-1).values
    patch_count = values.shape[-1]
    indices = torch.arange(1, patch_count + 1, device=values.device, dtype=values.dtype)
    gini = (
        (2 * indices - patch_count - 1) * sorted_values
    ).sum(dim=-1) / (patch_count * sorted_values.sum(dim=-1).clamp_min(eps))
    top_k = max(1, round(patch_count * top_k_fraction))
    return {
        "entropy": entropy,
        "maximum": values.max(dim=-1).values,
        "effective_patches": effective,
        "gini": gini,
        "top_k_mass": values.topk(top_k, dim=-1).values.sum(dim=-1),
    }


def head_pairwise_similarity(
    weights: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample head cosine similarity and Jensen-Shannon divergence."""
    if weights.ndim != 4 or weights.shape[2] != 1:
        raise ValueError("attention weights must have shape [batch, heads, 1, patches]")
    values = weights[:, :, 0]
    first = values[:, :, None, :]
    second = values[:, None, :, :]
    cosine = F.cosine_similarity(first, second, dim=-1)
    midpoint = 0.5 * (first + second)
    first_kl = (first * (first.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(-1)
    second_kl = (
        second * (second.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    ).sum(-1)
    return cosine, 0.5 * (first_kl + second_kl)


def attention_stability(
    reference: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    top_k_fraction: float = 0.1,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Compare paired attention maps after a configured input intervention."""
    if reference.shape != perturbed.shape or reference.ndim != 4:
        raise ValueError("paired attention tensors must share [batch, heads, 1, patches]")
    first = reference[:, :, 0].float()
    second = perturbed[:, :, 0].float()
    midpoint = 0.5 * (first + second)
    js = 0.5 * (
        (first * (first.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(-1)
        + (second * (second.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(-1)
    )
    cosine = F.cosine_similarity(first, second, dim=-1)
    patch_count = first.shape[-1]
    top_k = max(1, round(patch_count * top_k_fraction))
    first_top = first.topk(top_k, dim=-1).indices
    second_top = second.topk(top_k, dim=-1).indices
    first_mask = torch.zeros_like(first, dtype=torch.bool).scatter(-1, first_top, True)
    second_mask = torch.zeros_like(second, dtype=torch.bool).scatter(-1, second_top, True)
    intersection = (first_mask & second_mask).sum(-1)
    union = (first_mask | second_mask).sum(-1).clamp_min(1)
    rank = torch.empty(first.shape[:2], dtype=torch.float32, device=first.device)
    for batch in range(first.shape[0]):
        for head in range(first.shape[1]):
            statistic = spearmanr(
                first[batch, head].cpu().numpy(),
                second[batch, head].cpu().numpy(),
            ).statistic
            rank[batch, head] = float(np.nan_to_num(statistic))
    return {
        "js_divergence": js,
        "cosine_similarity": cosine,
        "rank_correlation": rank,
        "top_k_iou": intersection.float() / union.float(),
    }
