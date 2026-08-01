from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class BackboneOutput:
    semantic_cls: torch.Tensor
    final_patch_tokens: torch.Tensor
    intermediate_patch_tokens: dict[int, torch.Tensor]
    patch_grid_size: tuple[int, int]


def split_backbone_tokens(
    hidden_state: torch.Tensor,
    num_register_tokens: int,
    patch_grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split CLS and patches using the register count declared by model config."""
    if hidden_state.ndim != 3:
        raise ValueError("hidden state must have shape [batch, tokens, channels]")
    if num_register_tokens < 0:
        raise ValueError("num_register_tokens must be non-negative")
    expected_patches = patch_grid_size[0] * patch_grid_size[1]
    prefix_tokens = 1 + num_register_tokens
    actual_patches = hidden_state.shape[1] - prefix_tokens
    if actual_patches != expected_patches:
        raise ValueError(
            f"patch token count {actual_patches} does not match grid {patch_grid_size} "
            f"({expected_patches})"
        )
    return hidden_state[:, 0], hidden_state[:, prefix_tokens:]


class TinyBackbone(nn.Module):
    """Small frozen backbone used only by offline smoke tests."""

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        patch_size: int = 16,
        num_layers: int = 4,
        selected_layers: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.selected_layers = selected_layers
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> TinyBackbone:
        super().train(False)
        return self

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("input height and width must be divisible by patch_size")
        with torch.no_grad():
            patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
            intermediate: dict[int, torch.Tensor] = {}
            for index, layer in enumerate(self.layers):
                patches = patches + torch.tanh(layer(patches))
                if index in self.selected_layers:
                    intermediate[index] = self.norm(patches)
            patches = self.norm(patches)
        return BackboneOutput(
            semantic_cls=patches.mean(dim=1),
            final_patch_tokens=patches,
            intermediate_patch_tokens=intermediate,
            patch_grid_size=(height // self.patch_size, width // self.patch_size),
        )
