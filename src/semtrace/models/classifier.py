from __future__ import annotations

import torch
from torch import nn


class TraceClassifier(nn.Module):
    def __init__(self, trace_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Linear(trace_dim, 1)

    def forward(self, trace_evidence: torch.Tensor) -> torch.Tensor:
        if trace_evidence.ndim != 2:
            raise ValueError("trace evidence must have shape [batch, channels]")
        return self.classifier(trace_evidence).squeeze(-1)

