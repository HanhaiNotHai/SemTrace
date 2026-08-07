from __future__ import annotations

import torch
from torch.nn import functional as F

from semtrace.models.normal_predictor import NormalFeaturePredictor, _normalized_positions


@torch.inference_mode()
def normal_prediction_intervention(
    predictor: NormalFeaturePredictor,
    semantic_anchor: torch.Tensor,
    patch_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    *,
    use_semantic: bool,
    use_neighbors: bool,
) -> torch.Tensor:
    """Reproduce the predictor while allowing semantic or neighborhood removal."""
    batch_size, token_count, feature_dim = patch_tokens.shape
    neighbors, valid_neighbors = predictor._extract_neighbors(patch_tokens, patch_grid_size)
    neighbor_features = predictor.neighbor_projection(neighbors)
    if not use_neighbors:
        neighbor_features = torch.zeros_like(neighbor_features)
    positions = _normalized_positions(
        patch_grid_size,
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    )
    semantic = semantic_anchor if use_semantic else torch.zeros_like(semantic_anchor)
    query = predictor.semantic_projection(semantic)[:, None, :]
    query = query + predictor.position_projection(positions)[None, :, :]
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
        predictor.attention_layers,
        predictor.query_norms,
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
    return predictor.output_projection(query).reshape(batch_size, token_count, feature_dim)


def prediction_error_metrics(
    observed_features: dict[int, torch.Tensor],
    predicted_features: dict[int, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    if set(observed_features) != set(predicted_features):
        raise ValueError("observed and predicted layers must match")
    metrics: dict[int, dict[str, torch.Tensor]] = {}
    for layer, observed in observed_features.items():
        predicted = predicted_features[layer]
        if observed.shape != predicted.shape or observed.ndim != 3:
            raise ValueError("normal prediction tensors must share [batch, patches, channels]")
        smooth = F.smooth_l1_loss(predicted, observed, reduction="none").mean(dim=(1, 2))
        cosine_error = (1.0 - F.cosine_similarity(predicted, observed, dim=-1)).mean(dim=1)
        l2_error = (observed - predicted).norm(dim=-1).mean(dim=1)
        metrics[layer] = {
            "smooth_l1": smooth,
            "cosine_error": cosine_error,
            "l2_error": l2_error,
        }
    return metrics
