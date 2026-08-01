from __future__ import annotations

import torch

from semtrace.losses.normal_prediction import normal_prediction_loss


def test_normal_prediction_loss_detaches_targets() -> None:
    prediction = torch.randn(2, 4, 6, requires_grad=True)
    target = torch.randn(2, 4, 6, requires_grad=True)

    loss = normal_prediction_loss(prediction, target, lambda_smooth=1.0, lambda_cos=0.5)
    loss.backward()

    assert prediction.grad is not None
    assert target.grad is None
