from __future__ import annotations

import torch

from semtrace.losses.detection import detection_loss


def test_detection_loss_uses_real_zero_fake_one_logits() -> None:
    logits = torch.tensor([-10.0, 10.0])
    labels = torch.tensor([0, 1])

    assert detection_loss(logits, labels).item() < 0.001
