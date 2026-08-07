from __future__ import annotations

import math
from typing import Literal

import torch

PatchMaskStrategy = Literal[
    "random", "top", "low", "center", "edge"
]


def mask_scales(
    scale_tokens: dict[int, torch.Tensor],
    *,
    masked_layers: set[int],
    replacements: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    unknown = masked_layers - set(scale_tokens)
    if unknown:
        raise ValueError(f"cannot mask unavailable scales: {sorted(unknown)}")
    result = dict(scale_tokens)
    for layer in masked_layers:
        result[layer] = (
            replacements[layer]
            if replacements is not None and layer in replacements
            else torch.zeros_like(scale_tokens[layer])
        )
    return result


def mask_patches(
    tokens: torch.Tensor,
    *,
    ratio: float,
    strategy: PatchMaskStrategy,
    seed: int = 0,
    scores: torch.Tensor | None = None,
    patch_grid_size: tuple[int, int] | None = None,
    replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3:
        raise ValueError("patch tokens must have shape [batch, patches, channels]")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("patch mask ratio must be in (0, 1]")
    batch_size, patch_count, _ = tokens.shape
    masked_count = min(patch_count, max(1, round(patch_count * ratio)))
    if strategy in {"top", "low"}:
        if scores is None or scores.shape != (batch_size, patch_count):
            raise ValueError("top/low patch masking requires [batch, patches] scores")
        indices = scores.topk(masked_count, dim=1, largest=strategy == "top").indices
    elif strategy == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_scores = torch.rand(batch_size, patch_count, generator=generator)
        indices = random_scores.topk(masked_count, dim=1).indices.to(tokens.device)
    else:
        if patch_grid_size is None or math.prod(patch_grid_size) != patch_count:
            raise ValueError("center/edge masking requires the matching patch grid")
        height, width = patch_grid_size
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        distance = ((y - (height - 1) / 2) ** 2 + (x - (width - 1) / 2) ** 2).flatten()
        spatial = distance[None, :].expand(batch_size, -1).to(tokens.device)
        indices = spatial.topk(masked_count, dim=1, largest=strategy == "edge").indices
    mask = torch.zeros((batch_size, patch_count), dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, indices, True)
    result = tokens.clone()
    fill = torch.zeros_like(tokens) if replacement is None else replacement.expand_as(tokens)
    result[mask] = fill[mask]
    return result, mask
