from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def differentiable_global_batch(features: torch.Tensor) -> torch.Tensor:
    """Gather a global batch with autograd support when DDP is initialized."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return features
    return torch.cat(all_gather(features), dim=0)


def semantic_trace_separation_loss(
    semantic_anchor: torch.Tensor,
    trace_evidence: torch.Tensor,
    *,
    margin: float = 0.01,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if semantic_anchor.ndim != 2 or trace_evidence.ndim != 2:
        raise ValueError("semantic and trace features must have shape [batch, channels]")
    if semantic_anchor.shape[0] != trace_evidence.shape[0]:
        raise ValueError("semantic and trace batch sizes must match")
    if margin < 0 or eps <= 0:
        raise ValueError("margin must be non-negative and eps must be positive")

    global_semantic = differentiable_global_batch(semantic_anchor.detach())
    global_trace = differentiable_global_batch(trace_evidence)
    batch_size = global_semantic.shape[0]
    if batch_size < 2:
        return global_trace.sum() * 0.0
    normalized_semantic = _standardize(global_semantic, eps)
    normalized_trace = _standardize(global_trace, eps)
    cross_covariance = normalized_semantic.T @ normalized_trace / (batch_size - 1)
    normalized_energy = cross_covariance.square().mean()
    return F.relu(normalized_energy - margin)


def _standardize(features: torch.Tensor, eps: float) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    standard_deviation = centered.square().mean(dim=0, keepdim=True).sqrt()
    return centered / (standard_deviation + eps)
