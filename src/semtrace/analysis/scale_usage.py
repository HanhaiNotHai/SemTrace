from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ScaleUsage:
    layers: tuple[int, ...]
    activation_strength: torch.Tensor
    normalized_usage: torch.Tensor
    effective_scale_count: torch.Tensor
    scale_entropy: torch.Tensor


def compute_scale_usage(
    adapted_trace_tokens: dict[int, torch.Tensor],
    *,
    eps: float = 1.0e-12,
) -> ScaleUsage:
    """Compute activation-strength diagnostics, not causal scale contributions."""
    if not adapted_trace_tokens:
        raise ValueError("scale usage requires at least one adapted trace scale")
    layers = tuple(adapted_trace_tokens)
    strengths = torch.stack(
        [adapted_trace_tokens[layer].norm(dim=-1).mean(dim=1) for layer in layers],
        dim=1,
    )
    normalized = strengths / strengths.sum(dim=1, keepdim=True).clamp_min(eps)
    effective = normalized.square().sum(dim=1).clamp_min(eps).reciprocal()
    entropy = -(normalized * normalized.clamp_min(eps).log()).sum(dim=1)
    return ScaleUsage(layers, strengths, normalized, effective, entropy)


def fusion_weight_contributions(
    adapted_trace_tokens: dict[int, torch.Tensor],
    fusion_linear: nn.Linear,
) -> dict[int, torch.Tensor]:
    """Return per-scale projected activation magnitudes for concat-linear fusion."""
    layers = tuple(adapted_trace_tokens)
    if not layers:
        raise ValueError("fusion contribution requires adapted trace scales")
    trace_dim = adapted_trace_tokens[layers[0]].shape[-1]
    if fusion_linear.in_features != trace_dim * len(layers):
        raise ValueError("fusion linear input does not match concatenated trace scales")
    contributions: dict[int, torch.Tensor] = {}
    for position, layer in enumerate(layers):
        weight = fusion_linear.weight[:, position * trace_dim : (position + 1) * trace_dim]
        projected = torch.nn.functional.linear(adapted_trace_tokens[layer], weight)
        contributions[layer] = projected.norm(dim=-1).mean(dim=1)
    return contributions
