from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from semtrace.losses.normal_prediction import normal_prediction_loss
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semantic_anchor import FrozenSemanticAnchor


def train_normal_batch(
    *,
    backbone: nn.Module,
    semantic_anchor: FrozenSemanticAnchor,
    predictors: nn.ModuleDict,
    images: torch.Tensor,
    labels: torch.Tensor,
    selected_layers: tuple[int, ...],
    optimizer: torch.optim.Optimizer,
    lambda_smooth: float = 1.0,
    lambda_cos: float = 0.5,
) -> float:
    if not torch.all(labels == 0):
        raise ValueError("stage 2 normal predictors may only train on real images (label 0)")
    backbone.eval()
    semantic_anchor.eval()
    optimizer.zero_grad(set_to_none=True)
    output: Any = backbone(images)
    semantic = semantic_anchor(output.semantic_cls, output.final_patch_tokens).detach()
    loss = torch.zeros((), device=images.device)
    for layer in selected_layers:
        predictor = predictors[str(layer)]
        if not isinstance(predictor, NormalFeaturePredictor):
            raise TypeError("normal predictor has an unexpected module type")
        observed = output.intermediate_patch_tokens[layer]
        prediction = predictor(semantic, observed, output.patch_grid_size)
        loss = loss + normal_prediction_loss(
            prediction,
            observed,
            lambda_smooth=lambda_smooth,
            lambda_cos=lambda_cos,
        )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def freeze_normal_predictors(predictors: Mapping[str, nn.Module]) -> None:
    for predictor in predictors.values():
        predictor.eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)

