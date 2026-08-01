from __future__ import annotations

import torch
from torch import nn

from semtrace.backbones.base import TinyBackbone
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.semtrace import SemTrace


def test_stage3_backward_keeps_backbone_and_normal_predictors_frozen() -> None:
    backbone = TinyBackbone(
        hidden_size=16,
        patch_size=4,
        num_layers=4,
        selected_layers=(0, 1, 2),
    )
    predictors = nn.ModuleDict(
        {
            str(layer): NormalFeaturePredictor(
                input_dim=16,
                semantic_dim=8,
                hidden_dim=16,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
            )
            for layer in (0, 1, 2)
        }
    )
    model = SemTrace(
        backbone=backbone,
        semantic_anchor=FrozenSemanticAnchor(16, 8, seed=1),
        selected_layers=(0, 1, 2),
        feature_dim=16,
        semantic_dim=8,
        trace_dim=8,
        normal_predictors=predictors,
        cross_attention_heads=2,
        cross_attention_dropout=0.0,
    )

    output = model(torch.randn(2, 3, 16, 16))
    output.logits.sum().backward()

    assert output.logits.shape == (2,)
    assert output.trace_evidence.shape == (2, 8)
    assert set(output.candidate_trace_residuals) == {0, 1, 2}
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert all(
        parameter.grad is None
        for predictor in predictors.values()
        for parameter in predictor.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.trace_adapters.parameters()
        if parameter.requires_grad
    )


def test_detector_can_disable_normal_prediction_and_cross_attention() -> None:
    model = SemTrace(
        backbone=TinyBackbone(
            hidden_size=16,
            patch_size=4,
            num_layers=4,
            selected_layers=(0,),
        ),
        semantic_anchor=FrozenSemanticAnchor(16, 8, seed=1),
        selected_layers=(0,),
        feature_dim=16,
        semantic_dim=8,
        trace_dim=8,
        normal_predictors=None,
        use_normal_predictor=False,
        use_cross_attention=False,
        cross_attention_heads=2,
    )

    output = model(torch.randn(1, 3, 16, 16))

    assert output.attention_map is None
    assert output.logits.shape == (1,)
