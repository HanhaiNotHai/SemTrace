from __future__ import annotations

from semtrace.cli import train_normal


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


def test_stage2_progress_is_rank_safe_and_reports_running_values(monkeypatch) -> None:
    RecordingProgress.instances.clear()
    monkeypatch.setattr(train_normal, "tqdm", RecordingProgress)

    progress = train_normal._progress_bar(
        [1, 2],
        description="Stage 2 train 1/2",
        show_progress=True,
    )
    train_normal._update_progress(
        progress,
        samples=64,
        running_loss=1.25,
        learning_rate=2.0e-4,
    )

    assert progress.options == {
        "total": 2,
        "desc": "Stage 2 train 1/2",
        "unit": "batch",
        "dynamic_ncols": True,
        "disable": False,
    }
    assert progress.postfixes[-1] == {
        "samples": 64,
        "loss": "1.2500",
        "lr": "2.00e-04",
    }

    hidden = train_normal._progress_bar(
        [1],
        description="hidden rank",
        show_progress=False,
    )
    assert hidden.options["disable"] is True


def test_stage2_epoch_summary_reports_duration_and_remaining_eta(monkeypatch) -> None:
    RecordingProgress.messages.clear()
    monkeypatch.setattr(train_normal, "tqdm", RecordingProgress)

    train_normal._write_epoch_summary(
        epoch=2,
        epochs=5,
        train_loss=0.75,
        validation_loss=0.5,
        epoch_seconds=90.0,
        elapsed_seconds=180.0,
        show_progress=True,
    )

    message = RecordingProgress.messages[-1]
    assert "Epoch 2/5 complete" in message
    assert "train_loss=0.7500" in message
    assert "validation_loss=0.5000" in message
    assert "epoch_time=1m 30s" in message
    assert "stage_eta=4m 30s" in message
