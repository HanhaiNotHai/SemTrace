from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResidualStrength:
    patch_l1: torch.Tensor
    patch_l2: torch.Tensor
    channel_energy: torch.Tensor
    image_mean: torch.Tensor
    image_max: torch.Tensor
    top_k_mean: torch.Tensor
    spatial_entropy: torch.Tensor
    sparsity: torch.Tensor


def residual_strength(
    prediction_error: torch.Tensor,
    *,
    top_k_fraction: float = 0.1,
    eps: float = 1.0e-12,
) -> ResidualStrength:
    """Summarize the pre-LayerNorm prediction error ``observed - predicted``."""
    if prediction_error.ndim != 3:
        raise ValueError("prediction error must have shape [batch, patches, channels]")
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    patch_l1 = prediction_error.abs().sum(dim=-1)
    patch_l2 = prediction_error.norm(dim=-1)
    energy = prediction_error.square().mean(dim=-1)
    top_k = max(1, math.ceil(prediction_error.shape[1] * top_k_fraction))
    mass = patch_l2 / patch_l2.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(mass * mass.clamp_min(eps).log()).sum(dim=1)
    return ResidualStrength(
        patch_l1=patch_l1,
        patch_l2=patch_l2,
        channel_energy=energy,
        image_mean=patch_l2.mean(dim=1),
        image_max=patch_l2.max(dim=1).values,
        top_k_mean=patch_l2.topk(top_k, dim=1).values.mean(dim=1),
        spatial_entropy=entropy,
        sparsity=(patch_l2 <= eps).float().mean(dim=1),
    )


def pooled_stage_representations(
    *,
    semantic_anchor: torch.Tensor,
    raw_patch_features: dict[int, torch.Tensor],
    candidate_trace_residuals: dict[int, torch.Tensor],
    adapted_trace_tokens: dict[int, torch.Tensor],
    fused_trace_tokens: torch.Tensor,
    trace_evidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    representations = {"semantic_anchor": semantic_anchor, "trace_evidence": trace_evidence}
    for layer, tokens in raw_patch_features.items():
        representations[f"raw_{layer}"] = tokens.mean(dim=1)
    for layer, tokens in candidate_trace_residuals.items():
        representations[f"residual_{layer}"] = tokens.mean(dim=1)
    for layer, tokens in adapted_trace_tokens.items():
        representations[f"adapted_{layer}"] = tokens.mean(dim=1)
    representations["fused_trace_mean"] = fused_trace_tokens.mean(dim=1)
    return representations
