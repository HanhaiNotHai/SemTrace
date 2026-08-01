from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def candidate_trace_residual(
    observed_patch_features: torch.Tensor,
    predicted_normal_features: torch.Tensor,
) -> torch.Tensor:
    """Return a mixed residual candidate, not a proven pure generation artifact."""
    if observed_patch_features.shape != predicted_normal_features.shape:
        raise ValueError("observed and predicted patch feature shapes must match")
    difference = observed_patch_features - predicted_normal_features
    return F.layer_norm(difference, (difference.shape[-1],))


class TraceAdapter(nn.Module):
    """Local spatial adapter that preserves the patch grid."""

    def __init__(self, input_dim: int, trace_dim: int = 256) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, trace_dim)
        self.depthwise_convolution = nn.Conv2d(
            trace_dim,
            trace_dim,
            kernel_size=3,
            padding=1,
            groups=trace_dim,
        )
        self.pointwise_mlp = nn.Sequential(
            nn.Linear(trace_dim, trace_dim * 2),
            nn.GELU(),
            nn.Linear(trace_dim * 2, trace_dim),
        )
        self.output_norm = nn.LayerNorm(trace_dim)

    def forward(
        self,
        candidate_residual: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        if candidate_residual.ndim != 3:
            raise ValueError("candidate residual must have shape [batch, patches, channels]")
        batch_size, token_count, _ = candidate_residual.shape
        height, width = patch_grid_size
        if token_count != height * width:
            raise ValueError("candidate residual token count does not match patch grid")
        projected = F.gelu(self.input_projection(self.input_norm(candidate_residual)))
        grid = projected.transpose(1, 2).reshape(batch_size, -1, height, width)
        spatial = self.depthwise_convolution(grid).flatten(2).transpose(1, 2)
        return self.output_norm(projected + self.pointwise_mlp(spatial))

