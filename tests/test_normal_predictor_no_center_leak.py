from __future__ import annotations

import torch

from semtrace.models.normal_predictor import NormalFeaturePredictor


def test_normal_predictor_does_not_read_the_target_center_token() -> None:
    torch.manual_seed(3)
    predictor = NormalFeaturePredictor(
        input_dim=12,
        semantic_dim=8,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        neighborhood_size=3,
        dropout=0.0,
    ).eval()
    semantic = torch.randn(1, 8)
    patches = torch.randn(1, 9, 12)
    changed = patches.clone()
    changed[:, 4] = changed[:, 4] + 1000.0

    prediction = predictor(semantic, patches, (3, 3))
    changed_prediction = predictor(semantic, changed, (3, 3))

    torch.testing.assert_close(prediction[:, 4], changed_prediction[:, 4])


def test_normal_predictor_supports_rectangular_patch_grids() -> None:
    predictor = NormalFeaturePredictor(
        input_dim=12,
        semantic_dim=8,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        neighborhood_size=3,
        dropout=0.0,
    )

    output = predictor(torch.randn(2, 8), torch.randn(2, 12, 12), (3, 4))

    assert output.shape == (2, 12, 12)

