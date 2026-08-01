from __future__ import annotations

import torch
from torch import nn


class SemanticTraceCrossAttention(nn.Module):
    """Semantic query selects trace values but is never added to the evidence."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        trace_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_semantic_gate: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= max_semantic_gate <= 1.0:
            raise ValueError("max_semantic_gate must be in [0, 1]")
        self.semantic_query = nn.Linear(semantic_dim, trace_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=trace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.max_semantic_gate = max_semantic_gate
        self.output_norm = nn.LayerNorm(trace_dim)

    @property
    def semantic_gate(self) -> torch.Tensor:
        return self.max_semantic_gate * self.gate_logit.sigmoid()

    def forward(
        self,
        semantic_anchor: torch.Tensor,
        trace_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_anchor.ndim != 2 or trace_tokens.ndim != 3:
            raise ValueError("semantic anchor and trace tokens must have rank 2 and 3")
        query = self.semantic_query(semantic_anchor.detach()).unsqueeze(1)
        attended, weights = self.attention(
            query,
            trace_tokens,
            trace_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        if weights is None:
            raise RuntimeError("cross-attention weights were not returned")
        mean_trace = trace_tokens.mean(dim=1)
        gate = self.semantic_gate
        evidence = self.output_norm((1.0 - gate) * mean_trace + gate * attended[:, 0])
        return evidence, weights

