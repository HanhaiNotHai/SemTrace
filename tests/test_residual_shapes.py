from __future__ import annotations

import torch

from semtrace.models.trace_adapter import TraceAdapter, candidate_trace_residual


def test_candidate_trace_residual_and_adapter_preserve_patch_layout() -> None:
    observed = torch.randn(2, 12, 16)
    predicted = torch.randn(2, 12, 16)

    residual = candidate_trace_residual(observed, predicted)
    adapted = TraceAdapter(input_dim=16, trace_dim=8)(residual, (3, 4))

    assert residual.shape == observed.shape
    assert adapted.shape == (2, 12, 8)
    torch.testing.assert_close(residual.mean(dim=-1), torch.zeros(2, 12), atol=1e-5, rtol=0)

