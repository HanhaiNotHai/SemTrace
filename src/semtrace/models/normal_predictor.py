from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NormalFeaturePredictor(nn.Module):
    """Predict each real-image patch from semantics, position, and non-center neighbors."""

    def __init__(
        self,
        *,
        input_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        neighborhood_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be an odd integer of at least 3")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.neighborhood_size = neighborhood_size
        self.semantic_projection = nn.Linear(semantic_dim, hidden_dim)
        self.position_projection = nn.Linear(2, hidden_dim)
        self.neighbor_projection = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.query_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        semantic_anchor: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        if semantic_anchor.ndim != 2 or patch_tokens.ndim != 3:
            raise ValueError("semantic anchor and patch tokens must have rank 2 and 3")
        batch_size, token_count, feature_dim = patch_tokens.shape
        grid_height, grid_width = patch_grid_size
        if token_count != grid_height * grid_width:
            raise ValueError("patch token count does not match patch_grid_size")
        if grid_height < 2 or grid_width < 2:
            raise ValueError("normal prediction requires a patch grid of at least 2x2")

        neighbors, valid_neighbors = self._extract_neighbors(
            patch_tokens,
            patch_grid_size,
        )
        neighbor_features = self.neighbor_projection(neighbors)
        positions = _normalized_positions(
            patch_grid_size,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        query = self.semantic_projection(semantic_anchor.detach())[:, None, :]
        query = query + self.position_projection(positions)[None, :, :]

        query = query.reshape(batch_size * token_count, 1, -1)
        neighbor_features = neighbor_features.reshape(
            batch_size * token_count,
            neighbors.shape[2],
            -1,
        )
        padding_mask = ~valid_neighbors.reshape(
            batch_size * token_count,
            valid_neighbors.shape[2],
        )
        for attention, norm in zip(
            self.attention_layers,
            self.query_norms,
            strict=True,
        ):
            attended, _ = attention(
                query,
                neighbor_features,
                neighbor_features,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            query = norm(query + attended)
        return self.output_projection(query).reshape(batch_size, token_count, feature_dim)

    def _extract_neighbors(
        self,
        patch_tokens: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, feature_dim = patch_tokens.shape
        grid_height, grid_width = patch_grid_size
        grid = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            feature_dim,
            grid_height,
            grid_width,
        )
        padding = self.neighborhood_size // 2
        unfolded = F.unfold(
            grid,
            kernel_size=self.neighborhood_size,
            padding=padding,
        )
        neighborhood_count = self.neighborhood_size**2
        unfolded = unfolded.reshape(
            batch_size,
            feature_dim,
            neighborhood_count,
            grid_height * grid_width,
        ).permute(0, 3, 2, 1)
        validity_grid = torch.ones(
            (batch_size, 1, grid_height, grid_width),
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        validity = F.unfold(
            validity_grid,
            kernel_size=self.neighborhood_size,
            padding=padding,
        )
        validity = validity.reshape(
            batch_size,
            neighborhood_count,
            grid_height * grid_width,
        ).permute(0, 2, 1)
        center = neighborhood_count // 2
        keep = [index for index in range(neighborhood_count) if index != center]
        return unfolded[:, :, keep, :], validity[:, :, keep].bool()


class MultiScaleNormalPredictors(nn.Module):
    """DDP-compatible owner for independent per-layer normal predictors."""

    def __init__(self, normal_predictors: nn.ModuleDict) -> None:
        super().__init__()
        self.normal_predictors = normal_predictors

    def forward(
        self,
        semantic_anchor: torch.Tensor,
        intermediate_patch_tokens: dict[int, torch.Tensor],
        patch_grid_size: tuple[int, int],
    ) -> dict[int, torch.Tensor]:
        predictions: dict[int, torch.Tensor] = {}
        for layer, observed in intermediate_patch_tokens.items():
            predictor = self.normal_predictors[str(layer)]
            if not isinstance(predictor, NormalFeaturePredictor):
                raise TypeError("normal predictor has an unexpected module type")
            predictions[layer] = predictor(semantic_anchor, observed, patch_grid_size)
        return predictions


def _normalized_positions(
    patch_grid_size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = patch_grid_size
    y_coordinates = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x_coordinates = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    return torch.stack((x_grid, y_grid), dim=-1).reshape(height * width, 2)
