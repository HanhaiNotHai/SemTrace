from __future__ import annotations

import inspect

import torch

from semtrace.models.classifier import TraceClassifier
from semtrace.models.cross_attention import SemanticTraceCrossAttention


def test_cross_attention_uses_semantic_query_without_query_residual() -> None:
    module = SemanticTraceCrossAttention(
        semantic_dim=6,
        trace_dim=8,
        num_heads=2,
        dropout=0.0,
        max_semantic_gate=0.5,
    ).eval()
    repeated_trace = torch.randn(2, 1, 8).expand(-1, 5, -1).clone()

    first, first_attention = module(torch.randn(2, 6), repeated_trace)
    second, second_attention = module(torch.randn(2, 6) * 100.0, repeated_trace)

    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
    assert first.shape == (2, 8)
    assert first_attention.shape == (2, 2, 1, 5)
    assert second_attention.shape == (2, 2, 1, 5)
    assert 0.0 <= module.semantic_gate.item() <= 0.5


def test_classifier_public_interface_only_accepts_trace_evidence() -> None:
    signature = inspect.signature(TraceClassifier.forward)

    assert list(signature.parameters) == ["self", "trace_evidence"]
    assert TraceClassifier(trace_dim=8)(torch.randn(3, 8)).shape == (3,)

