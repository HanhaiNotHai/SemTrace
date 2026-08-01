from __future__ import annotations

import torch

from semtrace.backbones.base import TinyBackbone
from semtrace.models.baselines import FrozenFeatureLinearBaseline


def test_final_and_intermediate_linear_baselines_are_separate_from_semtrace() -> None:
    images = torch.randn(2, 3, 16, 16)
    final = FrozenFeatureLinearBaseline(
        TinyBackbone(
            hidden_size=16,
            patch_size=4,
            num_layers=4,
            selected_layers=(1,),
        ),
        feature_dim=16,
        mode="final_cls",
    )
    intermediate = FrozenFeatureLinearBaseline(
        TinyBackbone(
            hidden_size=16,
            patch_size=4,
            num_layers=4,
            selected_layers=(1,),
        ),
        feature_dim=16,
        mode="intermediate_patch_mean",
        layer=1,
    )

    assert final(images).logits.shape == (2,)
    assert intermediate(images).logits.shape == (2,)
    assert all(parameter.grad is None for parameter in final.backbone.parameters())
