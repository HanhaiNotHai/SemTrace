from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from semtrace.models.semtrace import SemTraceOutput

BaselineMode = Literal["final_cls", "intermediate_patch_mean"]


class FrozenFeatureLinearBaseline(nn.Module):
    """Diagnostic linear baseline kept outside the SemTrace information path."""

    backbone: Any

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        mode: BaselineMode,
        layer: int | None = None,
    ) -> None:
        super().__init__()
        if mode == "intermediate_patch_mean" and layer is None:
            raise ValueError("an intermediate layer is required for patch-mean baseline")
        self.backbone = backbone
        self.mode = mode
        self.layer = layer
        self.classifier = nn.Linear(feature_dim, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> FrozenFeatureLinearBaseline:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> SemTraceOutput:
        output = self.backbone(images)
        if self.mode == "final_cls":
            features = output.semantic_cls
        else:
            if self.layer is None:
                raise RuntimeError("intermediate baseline layer is missing")
            features = output.intermediate_patch_tokens[self.layer].mean(dim=1)
        return SemTraceOutput(
            logits=self.classifier(features).squeeze(-1),
            trace_evidence=features,
            semantic_anchor=features.detach(),
            candidate_trace_residuals={},
            residual_statistics={},
            attention_map=None,
        )
