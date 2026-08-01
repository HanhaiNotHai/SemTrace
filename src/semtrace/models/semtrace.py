from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from semtrace.models.classifier import TraceClassifier
from semtrace.models.cross_attention import SemanticTraceCrossAttention
from semtrace.models.multiscale_fusion import MultiScaleTraceFusion
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.trace_adapter import TraceAdapter, candidate_trace_residual


@dataclass(frozen=True, slots=True)
class SemTraceOutput:
    logits: torch.Tensor
    trace_evidence: torch.Tensor
    semantic_anchor: torch.Tensor
    candidate_trace_residuals: dict[int, torch.Tensor]
    residual_statistics: dict[int, dict[str, torch.Tensor]]
    attention_map: torch.Tensor | None


class SemTrace(nn.Module):
    """Semantic-conditioned detector whose classifier receives only trace evidence."""

    backbone: Any

    def __init__(
        self,
        *,
        backbone: nn.Module,
        semantic_anchor: FrozenSemanticAnchor,
        selected_layers: tuple[int, ...],
        feature_dim: int,
        semantic_dim: int = 512,
        trace_dim: int = 256,
        normal_predictors: nn.ModuleDict | None,
        use_normal_predictor: bool = True,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        max_semantic_gate: float = 0.5,
    ) -> None:
        super().__init__()
        if not selected_layers:
            raise ValueError("at least one selected trace layer is required")
        if use_normal_predictor and normal_predictors is None:
            raise ValueError("normal predictors are required when normal prediction is enabled")
        self.backbone = backbone
        self.semantic_anchor = semantic_anchor
        self.selected_layers = selected_layers
        self.use_normal_predictor = use_normal_predictor
        self.use_cross_attention = use_cross_attention
        self.normal_predictors = (
            normal_predictors if normal_predictors is not None else nn.ModuleDict()
        )
        if use_normal_predictor:
            missing = {
                str(layer)
                for layer in selected_layers
                if str(layer) not in self.normal_predictors
            }
            if missing:
                raise ValueError(f"missing normal predictors for layers: {sorted(missing)}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.semantic_anchor.parameters():
            parameter.requires_grad_(False)
        for predictor in self.normal_predictors.values():
            for parameter in predictor.parameters():
                parameter.requires_grad_(False)
            predictor.eval()

        self.trace_adapters = nn.ModuleDict(
            {
                str(layer): TraceAdapter(input_dim=feature_dim, trace_dim=trace_dim)
                for layer in selected_layers
            }
        )
        self.trace_fusion = MultiScaleTraceFusion(
            trace_dim=trace_dim,
            num_scales=len(selected_layers),
        )
        self.cross_attention = (
            SemanticTraceCrossAttention(
                semantic_dim=semantic_dim,
                trace_dim=trace_dim,
                num_heads=cross_attention_heads,
                dropout=cross_attention_dropout,
                max_semantic_gate=max_semantic_gate,
            )
            if use_cross_attention
            else None
        )
        self.classifier = TraceClassifier(trace_dim=trace_dim)
        self.train()

    def train(self, mode: bool = True) -> SemTrace:
        super().train(mode)
        self.backbone.eval()
        self.semantic_anchor.eval()
        for predictor in self.normal_predictors.values():
            predictor.eval()
        return self

    def forward(self, images: torch.Tensor) -> SemTraceOutput:
        backbone_output = self.backbone(images)
        semantic = self.semantic_anchor(
            backbone_output.semantic_cls,
            backbone_output.final_patch_tokens,
        )
        residuals: dict[int, torch.Tensor] = {}
        statistics: dict[int, dict[str, torch.Tensor]] = {}
        adapted_scales: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []
        for layer in self.selected_layers:
            observed = backbone_output.intermediate_patch_tokens[layer]
            if self.use_normal_predictor:
                predictor = self.normal_predictors[str(layer)]
                if not isinstance(predictor, NormalFeaturePredictor):
                    raise TypeError("normal predictor has an unexpected module type")
                with torch.no_grad():
                    predicted = predictor(
                        semantic,
                        observed,
                        backbone_output.patch_grid_size,
                    )
            else:
                predicted = torch.zeros_like(observed)
            residual = candidate_trace_residual(observed, predicted)
            residuals[layer] = residual
            statistics[layer] = {
                "mean": residual.mean().detach(),
                "norm": residual.norm(dim=-1).mean().detach(),
            }
            adapted_scales.append(
                self.trace_adapters[str(layer)](
                    residual,
                    backbone_output.patch_grid_size,
                )
            )
            grids.append(backbone_output.patch_grid_size)

        trace_tokens, _ = self.trace_fusion(adapted_scales, grids)
        if self.use_cross_attention:
            if self.cross_attention is None:
                raise RuntimeError("cross-attention was enabled but not constructed")
            trace_evidence, attention_map = self.cross_attention(semantic, trace_tokens)
        else:
            trace_evidence = trace_tokens.mean(dim=1)
            attention_map = None
        logits = self.classifier(trace_evidence)
        return SemTraceOutput(
            logits=logits,
            trace_evidence=trace_evidence,
            semantic_anchor=semantic,
            candidate_trace_residuals=residuals,
            residual_statistics=statistics,
            attention_map=attention_map,
        )
