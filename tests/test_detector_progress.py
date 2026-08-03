from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from semtrace.cli import train_detector
from semtrace.data.collate import collate_image_samples
from semtrace.data.sample import ImageSample
from semtrace.engine import evaluator


class RecordingProgress:
    instances: list[RecordingProgress] = []
    messages: list[str] = []

    def __init__(self, iterable, **options) -> None:
        self.iterable = iterable
        self.options = options
        self.postfixes: list[dict[str, object]] = []
        self.instances.append(self)

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **values) -> None:
        self.postfixes.append(values)

    @classmethod
    def write(cls, message: str) -> None:
        cls.messages.append(message)


class TinyDetector(nn.Module):
    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        batch_size = images.shape[0]
        logits = torch.linspace(-1.0, 1.0, batch_size, device=images.device)
        residual = torch.zeros(batch_size, 4, 2, device=images.device)
        return SimpleNamespace(
            logits=logits,
            candidate_trace_residuals={1: residual},
            attention_map=None,
        )


def test_stage3_training_progress_is_rank_safe_and_reports_running_values(
    monkeypatch,
) -> None:
    RecordingProgress.instances.clear()
    monkeypatch.setattr(train_detector, "tqdm", RecordingProgress)

    progress = train_detector._progress_bar(
        [1, 2],
        description="Stage 3 train 1/200",
        show_progress=True,
    )
    train_detector._update_progress(
        progress,
        samples=64,
        running_loss=0.75,
        learning_rate=2.0e-4,
    )

    assert progress.options["desc"] == "Stage 3 train 1/200"
    assert progress.options["disable"] is False
    assert progress.postfixes[-1] == {
        "samples": 64,
        "loss": "0.7500",
        "lr": "2.00e-04",
    }
    hidden = train_detector._progress_bar(
        [1],
        description="hidden rank",
        show_progress=False,
    )
    assert hidden.options["disable"] is True


def test_stage3_epoch_summary_uses_epochs_completed_since_resume(monkeypatch) -> None:
    RecordingProgress.messages.clear()
    monkeypatch.setattr(train_detector, "tqdm", RecordingProgress)

    train_detector._write_epoch_summary(
        epoch=102,
        epochs=200,
        completed_this_run=2,
        train_loss=0.75,
        validation_accuracy=0.8,
        validation_ap=0.9,
        epoch_seconds=90.0,
        elapsed_seconds=180.0,
        show_progress=True,
    )

    message = RecordingProgress.messages[-1]
    assert "Epoch 102/200 complete" in message
    assert "accuracy=0.8000" in message
    assert "AP=0.9000" in message
    assert "epoch_time=1m 30s" in message
    assert "stage_eta=2h 27m" in message


def test_detector_validation_reports_rank_local_sample_progress(monkeypatch) -> None:
    RecordingProgress.instances.clear()
    monkeypatch.setattr(evaluator, "tqdm", RecordingProgress)
    samples = [
        ImageSample(
            image=torch.randn(3, 8, 8),
            label=index,
            semantic_class=index,
            generator="synthetic",
            source="synthetic",
            degradation=None,
            path=f"{index}.png",
        )
        for index in range(2)
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_image_samples)

    metrics, predictions = evaluator.evaluate_loader(
        TinyDetector(),
        loader,
        torch.device("cpu"),
        description="Stage 3 validation 1/200",
        show_progress=True,
    )

    progress = RecordingProgress.instances[0]
    assert progress.options["desc"] == "Stage 3 validation 1/200"
    assert progress.options["disable"] is False
    assert progress.postfixes[-1] == {"samples": 2}
    assert len(predictions) == 2
    assert metrics["accuracy"] == 1.0
