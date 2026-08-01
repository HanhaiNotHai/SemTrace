from __future__ import annotations

import torch

import semtrace.losses.separation as separation
from semtrace.losses.separation import (
    differentiable_global_batch,
    semantic_trace_separation_loss,
)


def test_separation_loss_penalizes_correlated_features_more_than_independent_ones() -> None:
    torch.manual_seed(11)
    semantic = torch.randn(512, 8)
    correlated_trace = semantic.clone().requires_grad_()
    independent_trace = torch.randn(512, 8, requires_grad=True)

    correlated = semantic_trace_separation_loss(
        semantic,
        correlated_trace,
        margin=0.01,
    )
    independent = semantic_trace_separation_loss(
        semantic,
        independent_trace,
        margin=0.01,
    )
    correlated.backward()

    assert correlated > independent
    assert correlated_trace.grad is not None
    assert semantic.grad is None


def test_separation_margin_can_zero_the_loss() -> None:
    semantic = torch.randn(64, 4)
    trace = torch.randn(64, 4, requires_grad=True)

    loss = semantic_trace_separation_loss(semantic, trace, margin=100.0)

    assert loss.item() == 0.0


def test_differentiable_global_batch_uses_all_gather_branch(monkeypatch) -> None:
    features = torch.tensor([[1.0, 2.0]], requires_grad=True)
    monkeypatch.setattr(separation.dist, "is_available", lambda: True)
    monkeypatch.setattr(separation.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(separation.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        separation,
        "all_gather",
        lambda tensor: (tensor, tensor * 2.0),
    )

    gathered = differentiable_global_batch(features)
    gathered.sum().backward()

    assert gathered.shape == (2, 2)
    torch.testing.assert_close(features.grad, torch.full_like(features, 3.0))
