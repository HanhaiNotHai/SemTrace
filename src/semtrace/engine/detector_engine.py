from __future__ import annotations

import torch

from semtrace.losses.detection import detection_loss
from semtrace.losses.separation import semantic_trace_separation_loss
from semtrace.models.semtrace import SemTrace


def train_detector_batch(
    *,
    model: SemTrace,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    separation_weight: float = 0.05,
    separation_margin: float = 0.01,
    separation_eps: float = 1.0e-6,
    use_separation_loss: bool = True,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = detection_loss(output.logits, labels)
    if use_separation_loss:
        loss = loss + separation_weight * semantic_trace_separation_loss(
            output.semantic_anchor,
            output.trace_evidence,
            margin=separation_margin,
            eps=separation_eps,
        )
    loss.backward()
    optimizer.step()
    return float(loss.detach())

