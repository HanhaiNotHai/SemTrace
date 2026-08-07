import torch

from semtrace.analysis.cross_attention_analysis import (
    attention_stability,
    attention_statistics,
)


def test_attention_statistics_have_valid_effective_patch_counts() -> None:
    weights = torch.tensor(
        [[[[0.5, 0.5, 0.0, 0.0]], [[0.25, 0.25, 0.25, 0.25]]]]
    )

    result = attention_statistics(weights, top_k_fraction=0.5)

    assert result["effective_patches"].shape == (1, 2)
    torch.testing.assert_close(result["effective_patches"], torch.tensor([[2.0, 4.0]]))
    assert torch.all(result["gini"] >= 0)
    assert torch.all(result["gini"] <= 1)
    torch.testing.assert_close(result["top_k_mass"], torch.tensor([[1.0, 0.5]]))


def test_identical_attention_is_stable() -> None:
    weights = torch.softmax(torch.randn(2, 3, 1, 8), dim=-1)
    stability = attention_stability(weights, weights, top_k_fraction=0.25)

    assert torch.allclose(stability["js_divergence"], torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(stability["cosine_similarity"], torch.ones(2, 3), atol=1e-6)
    assert torch.equal(stability["top_k_iou"], torch.ones(2, 3))
