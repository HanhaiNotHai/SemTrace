from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


class MultiScaleTraceFusion(nn.Module):
    def __init__(self, trace_dim: int = 256, num_scales: int = 3) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales must be positive")
        self.num_scales = num_scales
        self.fusion = nn.Sequential(
            nn.Linear(num_scales * trace_dim, trace_dim),
            nn.GELU(),
            nn.LayerNorm(trace_dim),
        )
        self._logged_alignments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def forward(
        self,
        scale_tokens: Sequence[torch.Tensor],
        patch_grids: Sequence[tuple[int, int]],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if len(scale_tokens) != self.num_scales or len(patch_grids) != self.num_scales:
            raise ValueError(f"expected exactly {self.num_scales} trace scales")
        reference_grid = patch_grids[0]
        aligned = [
            self._align(tokens, grid, reference_grid)
            for tokens, grid in zip(scale_tokens, patch_grids, strict=True)
        ]
        batch_sizes = {tokens.shape[0] for tokens in aligned}
        if len(batch_sizes) != 1:
            raise ValueError("all trace scales must have the same batch size")
        return self.fusion(torch.cat(aligned, dim=-1)), reference_grid

    def _align(
        self,
        tokens: torch.Tensor,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != source_grid[0] * source_grid[1]:
            raise ValueError("trace scale token count does not match its patch grid")
        if source_grid == target_grid:
            return tokens
        alignment = (source_grid, target_grid)
        if alignment not in self._logged_alignments:
            LOGGER.info("aligning trace patch grid from %s to %s", source_grid, target_grid)
            self._logged_alignments.add(alignment)
        batch_size, _, channels = tokens.shape
        grid = tokens.transpose(1, 2).reshape(batch_size, channels, *source_grid)
        resized = F.interpolate(grid, size=target_grid, mode="bilinear", align_corners=False)
        return resized.flatten(2).transpose(1, 2)

