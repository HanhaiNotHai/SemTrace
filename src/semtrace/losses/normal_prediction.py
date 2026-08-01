from __future__ import annotations

import torch
from torch.nn import functional as F


def normal_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_smooth: float = 1.0,
    lambda_cos: float = 0.5,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("normal prediction and target shapes must match")
    detached_target = target.detach()
    prediction = F.layer_norm(prediction, (prediction.shape[-1],))
    detached_target = F.layer_norm(detached_target, (detached_target.shape[-1],))
    smooth = F.smooth_l1_loss(prediction, detached_target)
    cosine = 1.0 - F.cosine_similarity(prediction, detached_target, dim=-1).mean()
    return lambda_smooth * smooth + lambda_cos * cosine
