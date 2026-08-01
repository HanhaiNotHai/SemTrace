from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenSemanticAnchor(nn.Module):
    """Fixed orthogonal projection of final CLS and mean patch features."""

    def __init__(self, backbone_dim: int, semantic_dim: int = 512, seed: int = 3407) -> None:
        super().__init__()
        input_dim = 2 * backbone_dim
        if semantic_dim <= 0 or semantic_dim > input_dim:
            raise ValueError("semantic_dim must be in [1, 2 * backbone_dim]")
        self.projection = nn.Linear(input_dim, semantic_dim, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_matrix = torch.randn(input_dim, semantic_dim, generator=generator)
        orthogonal_columns, _ = torch.linalg.qr(random_matrix, mode="reduced")
        with torch.no_grad():
            self.projection.weight.copy_(orthogonal_columns.T)
        self.projection.weight.requires_grad_(False)

    def forward(
        self,
        semantic_cls: torch.Tensor,
        final_patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_cls.ndim != 2 or final_patch_tokens.ndim != 3:
            raise ValueError("semantic CLS and patch tokens must have rank 2 and 3")
        if semantic_cls.shape[0] != final_patch_tokens.shape[0]:
            raise ValueError("semantic CLS and patch token batch sizes must match")
        if semantic_cls.shape[-1] != final_patch_tokens.shape[-1]:
            raise ValueError("semantic CLS and patch feature dimensions must match")
        combined = torch.cat((semantic_cls, final_patch_tokens.mean(dim=1)), dim=-1)
        projected = self.projection(combined)
        return F.layer_norm(projected, (projected.shape[-1],))

