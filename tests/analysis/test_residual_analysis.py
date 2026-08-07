import torch

from semtrace.analysis.residual_analysis import residual_strength


def test_residual_strength_reports_patch_and_image_statistics() -> None:
    errors = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        ]
    )

    result = residual_strength(errors, top_k_fraction=0.5)

    assert result.patch_l1.shape == (1, 4)
    assert result.patch_l2.shape == (1, 4)
    assert result.channel_energy.shape == (1, 4)
    assert result.image_mean.shape == (1,)
    assert result.image_max.item() == 2.0
    assert result.top_k_mean.item() > result.image_mean.item()
    assert torch.isfinite(result.spatial_entropy).all()
    assert 0 <= result.sparsity.item() <= 1
