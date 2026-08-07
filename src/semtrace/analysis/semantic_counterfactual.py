from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from semtrace.analysis.cross_attention_analysis import masked_cross_attention
from semtrace.analysis.normal_predictor_analysis import normal_prediction_intervention
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semtrace import SemTrace, SemTraceAnalysisOutput
from semtrace.models.trace_adapter import candidate_trace_residual


@dataclass(frozen=True, slots=True)
class MechanismState:
    semantic_anchor: torch.Tensor
    trace_tokens: torch.Tensor


@torch.inference_mode()
def recompute_analysis_path(
    model: SemTrace,
    baseline: SemTraceAnalysisOutput,
    *,
    normal_semantic: torch.Tensor | None = None,
    cross_attention_semantic: torch.Tensor | None = None,
    use_normal_semantic: bool = True,
    use_neighbors: bool = True,
    residual_replacements: dict[int, torch.Tensor] | None = None,
    adapted_replacements: dict[int, torch.Tensor] | None = None,
    fused_trace_replacement: torch.Tensor | None = None,
    head_keep_mask: torch.Tensor | None = None,
) -> SemTraceAnalysisOutput:
    """Recompute only the downstream analysis path under inference interventions."""
    normal_condition = (
        baseline.semantic_anchor if normal_semantic is None else normal_semantic
    )
    cross_condition = (
        baseline.semantic_anchor
        if cross_attention_semantic is None
        else cross_attention_semantic
    )
    predictions: dict[int, torch.Tensor] = {}
    errors: dict[int, torch.Tensor] = {}
    residuals: dict[int, torch.Tensor] = {}
    adapted: dict[int, torch.Tensor] = {}
    for layer in baseline.selected_layers:
        observed = baseline.raw_patch_features[layer]
        if model.use_normal_predictor:
            predictor = model.normal_predictors[str(layer)]
            if not isinstance(predictor, NormalFeaturePredictor):
                raise TypeError("normal predictor has an unexpected module type")
            predicted = normal_prediction_intervention(
                predictor,
                normal_condition,
                observed,
                baseline.patch_grid_size,
                use_semantic=use_normal_semantic,
                use_neighbors=use_neighbors,
            )
        else:
            predicted = torch.zeros_like(observed)
        residual = candidate_trace_residual(observed, predicted)
        if residual_replacements is not None and layer in residual_replacements:
            residual = residual_replacements[layer]
        predictions[layer] = predicted
        errors[layer] = observed - predicted
        residuals[layer] = residual
        tokens = model.trace_adapters[str(layer)](residual, baseline.patch_grid_size)
        if adapted_replacements is not None and layer in adapted_replacements:
            tokens = adapted_replacements[layer]
        adapted[layer] = tokens
    fused, _ = model.trace_fusion(
        [adapted[layer] for layer in baseline.selected_layers],
        [baseline.patch_grid_size] * len(baseline.selected_layers),
    )
    if fused_trace_replacement is not None:
        fused = fused_trace_replacement
    if model.use_cross_attention:
        if model.cross_attention is None:
            raise RuntimeError("cross-attention was enabled but not constructed")
        evidence, attention = masked_cross_attention(
            model.cross_attention,
            cross_condition,
            fused,
            head_keep_mask=head_keep_mask,
        )
    else:
        evidence, attention = fused.mean(dim=1), None
    logits = model.classifier(evidence)
    return SemTraceAnalysisOutput(
        semantic_anchor=baseline.semantic_anchor,
        selected_layers=baseline.selected_layers,
        raw_patch_features=baseline.raw_patch_features,
        predicted_normal_features=predictions,
        prediction_errors=errors,
        candidate_trace_residuals=residuals,
        adapted_trace_tokens=adapted,
        fused_trace_tokens=fused,
        attention_weights=attention,
        trace_evidence=evidence,
        logits=logits,
        patch_grid_size=baseline.patch_grid_size,
    )


def swap_semantic(state: MechanismState, permutation: torch.Tensor) -> MechanismState:
    _validate_permutation(permutation, state.semantic_anchor.shape[0])
    return MechanismState(state.semantic_anchor[permutation], state.trace_tokens)


def swap_trace(state: MechanismState, permutation: torch.Tensor) -> MechanismState:
    _validate_permutation(permutation, state.trace_tokens.shape[0])
    return MechanismState(state.semantic_anchor, state.trace_tokens[permutation])


def grouped_permutation(
    groups: list[object],
    *,
    same_group: bool,
    seed: int,
) -> torch.Tensor:
    """Create deterministic donors within or outside a metadata group."""
    rng = np.random.default_rng(seed)
    donors: list[int] = []
    for index, group in enumerate(groups):
        candidates = [
            candidate
            for candidate, candidate_group in enumerate(groups)
            if candidate != index and ((candidate_group == group) == same_group)
        ]
        donors.append(int(rng.choice(candidates)) if candidates else index)
    return torch.tensor(donors, dtype=torch.long)


def semantic_sensitivity_rate(
    baseline_probabilities: torch.Tensor,
    intervened_probabilities: torch.Tensor,
    *,
    threshold: float,
) -> float:
    baseline = baseline_probabilities >= threshold
    intervened = intervened_probabilities >= threshold
    return float((baseline != intervened).float().mean())


def trace_following_rate(
    intervened_probabilities: torch.Tensor,
    donor_labels: torch.Tensor,
    *,
    threshold: float,
) -> float:
    predictions = (intervened_probabilities >= threshold).to(donor_labels.dtype)
    return float((predictions == donor_labels).float().mean())


def _validate_permutation(permutation: torch.Tensor, batch_size: int) -> None:
    if permutation.ndim != 1 or permutation.shape[0] != batch_size:
        raise ValueError("swap permutation must contain one donor per sample")
    if torch.any(permutation < 0) or torch.any(permutation >= batch_size):
        raise ValueError("swap permutation index is out of range")
