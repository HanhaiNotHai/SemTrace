import torch

from semtrace.analysis.normal_predictor_analysis import (
    normal_prediction_intervention,
    prediction_error_metrics,
)
from semtrace.models.normal_predictor import NormalFeaturePredictor


def test_prediction_error_metrics_are_per_sample_and_per_layer() -> None:
    observed = {2: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    predicted = {2: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])}

    metrics = prediction_error_metrics(observed, predicted)

    assert set(metrics) == {2}
    assert set(metrics[2]) == {"smooth_l1", "cosine_error", "l2_error"}
    assert metrics[2]["smooth_l1"].shape == (1,)
    assert metrics[2]["cosine_error"].shape == (1,)
    assert metrics[2]["l2_error"].shape == (1,)
    assert metrics[2]["l2_error"].item() > 0


def test_unmodified_normal_intervention_matches_predictor() -> None:
    predictor = NormalFeaturePredictor(
        input_dim=8,
        semantic_dim=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    ).eval()
    semantic = torch.randn(2, 4)
    patches = torch.randn(2, 9, 8)

    expected = predictor(semantic, patches, (3, 3))
    actual = normal_prediction_intervention(
        predictor,
        semantic,
        patches,
        (3, 3),
        use_semantic=True,
        use_neighbors=True,
    )

    torch.testing.assert_close(actual, expected)
