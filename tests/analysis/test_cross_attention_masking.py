import torch

from semtrace.analysis.cross_attention_analysis import masked_cross_attention
from semtrace.models.cross_attention import SemanticTraceCrossAttention


def test_manual_cross_attention_matches_module_and_can_mask_heads() -> None:
    module = SemanticTraceCrossAttention(
        semantic_dim=4,
        trace_dim=8,
        num_heads=2,
        dropout=0.0,
    ).eval()
    semantic = torch.randn(3, 4)
    trace = torch.randn(3, 5, 8)

    expected, expected_weights = module(semantic, trace)
    actual, weights = masked_cross_attention(module, semantic, trace)
    masked, _ = masked_cross_attention(
        module,
        semantic,
        trace,
        head_keep_mask=torch.tensor([1.0, 0.0]),
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(weights, expected_weights, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, 2, 1))
    assert not torch.allclose(masked, actual)
