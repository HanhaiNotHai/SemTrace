from __future__ import annotations

import torch

from semtrace.models.multiscale_fusion import MultiScaleTraceFusion


def test_multiscale_fusion_concatenates_three_aligned_scales() -> None:
    fusion = MultiScaleTraceFusion(trace_dim=8, num_scales=3)
    scales = [torch.randn(2, 12, 8) for _ in range(3)]

    output, grid = fusion(scales, [(3, 4), (3, 4), (3, 4)])

    assert output.shape == (2, 12, 8)
    assert grid == (3, 4)


def test_multiscale_fusion_explicitly_aligns_different_grids() -> None:
    fusion = MultiScaleTraceFusion(trace_dim=8, num_scales=2)
    scales = [torch.randn(1, 12, 8), torch.randn(1, 4, 8)]

    output, grid = fusion(scales, [(3, 4), (2, 2)])

    assert output.shape == (1, 12, 8)
    assert grid == (3, 4)

