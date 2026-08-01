from __future__ import annotations

import torch
from torch.nn import functional as F


def detection_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("detection logits and labels must have matching [batch] shapes")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("detection labels must follow real=0, fake=1")
    return F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))

