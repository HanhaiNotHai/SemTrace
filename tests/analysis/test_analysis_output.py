import torch
from torch import nn

from semtrace.analysis.semantic_counterfactual import recompute_analysis_path
from semtrace.backbones.base import TinyBackbone
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.semtrace import SemTrace, SemTraceAnalysisOutput, SemTraceOutput


def _model() -> SemTrace:
    layers = (0, 1, 2)
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
            for layer in layers
        }
    )
    return SemTrace(
        backbone=TinyBackbone(
            hidden_size=16,
            patch_size=4,
            num_layers=4,
            selected_layers=layers,
        ),
        semantic_anchor=FrozenSemanticAnchor(16, 8),
        selected_layers=layers,
        feature_dim=16,
        semantic_dim=8,
        trace_dim=8,
        normal_predictors=predictors,
        cross_attention_heads=2,
        cross_attention_dropout=0.0,
    ).eval()


def test_analysis_returns_all_stages_without_changing_default_forward() -> None:
    model = _model()
    images = torch.randn(2, 3, 16, 16)

    standard = model(images)
    analysis = model.analyze(images)

    assert isinstance(standard, SemTraceOutput)
    assert isinstance(analysis, SemTraceAnalysisOutput)
    torch.testing.assert_close(analysis.logits, standard.logits)
    assert analysis.selected_layers == (0, 1, 2)
    assert analysis.patch_grid_size == (4, 4)
    for layer in analysis.selected_layers:
        assert analysis.raw_patch_features[layer].shape == (2, 16, 16)
        assert analysis.predicted_normal_features[layer].shape == (2, 16, 16)
        torch.testing.assert_close(
            analysis.prediction_errors[layer],
            analysis.raw_patch_features[layer] - analysis.predicted_normal_features[layer],
        )
        assert analysis.candidate_trace_residuals[layer].shape == (2, 16, 16)
        assert analysis.adapted_trace_tokens[layer].shape == (2, 16, 8)
    assert analysis.fused_trace_tokens.shape == (2, 16, 8)
    assert analysis.attention_weights is not None
    assert analysis.attention_weights.shape == (2, 2, 1, 16)
    assert analysis.trace_evidence.shape == (2, 8)


def test_analysis_interface_adds_no_checkpoint_parameters() -> None:
    source = _model()
    restored = _model()

    restored.load_state_dict(source.state_dict(), strict=True)

    assert source.state_dict().keys() == restored.state_dict().keys()


def test_unmodified_analysis_recomputation_matches_baseline() -> None:
    model = _model()
    baseline = model.analyze(torch.randn(2, 3, 16, 16))

    recomputed = recompute_analysis_path(model, baseline)

    torch.testing.assert_close(recomputed.logits, baseline.logits)
