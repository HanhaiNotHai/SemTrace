import pytest
import torch

from semtrace.analysis.scale_usage import compute_scale_usage


def test_scale_usage_is_normalized_and_effective_count_is_bounded() -> None:
    adapted = {
        2: torch.ones(2, 4, 3),
        6: torch.ones(2, 4, 3) * 2,
        8: torch.ones(2, 4, 3) * 3,
    }

    result = compute_scale_usage(adapted)

    torch.testing.assert_close(result.normalized_usage.sum(dim=1), torch.ones(2))
    assert torch.all(result.effective_scale_count >= 1)
    assert torch.all(result.effective_scale_count <= 3)
    assert result.layers == (2, 6, 8)


def test_equal_scale_activation_has_three_effective_scales() -> None:
    adapted = {layer: torch.ones(1, 4, 2) for layer in (2, 4, 8)}

    result = compute_scale_usage(adapted)

    assert result.effective_scale_count.item() == pytest.approx(3.0)
