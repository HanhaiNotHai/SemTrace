from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

import semtrace.engine.probe_engine as probe_engine
import semtrace.models.probes as probes
from semtrace.backbones.base import TinyBackbone
from semtrace.cli import probe_layers
from semtrace.data.collate import collate_image_samples
from semtrace.data.sample import ImageSample
from semtrace.models.probes import ProbeSplit


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


def test_feature_extraction_reports_batches_samples_throughput_and_completion(
    monkeypatch,
) -> None:
    RecordingProgress.instances.clear()
    RecordingProgress.messages.clear()
    monkeypatch.setattr(probe_engine, "tqdm", RecordingProgress)
    samples = [
        ImageSample(
            image=torch.randn(3, 16, 16),
            label=index % 2,
            semantic_class=index % 2,
            generator="synthetic",
            source="synthetic",
            degradation=None,
            path=f"{index}.png",
        )
        for index in range(4)
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_image_samples)
    backbone = TinyBackbone(
        hidden_size=8,
        patch_size=4,
        num_layers=4,
        selected_layers=(0, 1, 2),
    )

    output = probe_engine.extract_probe_features(
        backbone,
        loader,
        torch.device("cpu"),
        description="Train features",
        show_progress=True,
    )

    progress = RecordingProgress.instances[0]
    assert progress.options["desc"] == "Train features"
    assert progress.options["disable"] is False
    assert progress.postfixes[-1]["samples"] == 4
    assert "samples/s" in progress.postfixes[-1]
    assert "complete" in RecordingProgress.messages[-1]
    assert len(output.authenticity) == 4


def test_linear_probe_fitting_reports_each_candidate_layer(monkeypatch) -> None:
    RecordingProgress.instances.clear()
    monkeypatch.setattr(probes, "tqdm", RecordingProgress)
    authenticity = np.asarray([0, 1] * 4)
    semantic = np.asarray([0, 0, 1, 1] * 2)
    nuisance = np.asarray([0, 1, 1, 0] * 2)
    train = ProbeSplit(
        features={
            layer: np.random.default_rng(layer).normal(size=(8, 4))
            for layer in (0, 1, 2)
        },
        authenticity=authenticity,
        semantic=semantic,
        nuisance=nuisance,
    )
    validation = ProbeSplit(
        features={layer: values.copy() for layer, values in train.features.items()},
        authenticity=authenticity,
        semantic=semantic,
        nuisance=nuisance,
    )

    metrics = probes.fit_layer_probes(
        train,
        validation,
        seed=1,
        max_iter=20,
        show_progress=True,
    )

    progress = RecordingProgress.instances[0]
    assert progress.options["desc"] == "Fitting layer probes"
    assert progress.options["unit"] == "layer"
    assert progress.options["disable"] is False
    assert progress.postfixes[-1]["layer"] == 2
    assert set(metrics) == {0, 1, 2}


def test_cli_stage_timer_reports_start_and_completion(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(probe_layers.tqdm, "write", messages.append)

    with probe_layers._timed_stage("test stage", enabled=True):
        pass

    assert messages[0] == "[Stage 1] Starting test stage..."
    assert messages[1].startswith("[Stage 1] Completed test stage in ")
