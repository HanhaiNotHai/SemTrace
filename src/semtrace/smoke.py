from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from semtrace.backbones.base import TinyBackbone
from semtrace.engine.checkpoint import load_training_checkpoint, save_training_checkpoint
from semtrace.engine.detector_engine import train_detector_batch
from semtrace.engine.evaluator import evaluate_batch
from semtrace.engine.normal_engine import train_normal_batch
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.probes import (
    ProbeSplit,
    fit_layer_probes,
    save_probe_artifacts,
    select_probe_layers,
)
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.semtrace import SemTrace


@dataclass(frozen=True, slots=True)
class SmokeResult:
    selected_layers: tuple[int, int, int]
    normal_loss: float
    detector_loss: float
    evaluation: dict[str, object]
    checkpoint_path: Path


def run_synthetic_smoke(output_dir: str | Path, *, seed: int = 3407) -> SmokeResult:
    destination = Path(output_dir)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_split, validation_split = _synthetic_probe_splits(rng)
    probe_metrics = fit_layer_probes(train_split, validation_split, seed=seed, max_iter=100)
    selection = select_probe_layers(
        probe_metrics,
        num_hidden_layers=12,
        alpha=0.5,
        beta=0.5,
    )
    save_probe_artifacts(
        selection,
        destination / "probes",
        model_id="synthetic/tiny-backbone",
        model_revision="test",
        semantic_label_coverage=1.0,
        nuisance_label="source",
        generator_probe_enabled=False,
    )

    selected_layers = selection.selected_layers
    backbone = TinyBackbone(
        hidden_size=16,
        patch_size=4,
        num_layers=12,
        selected_layers=selected_layers,
    )
    semantic_anchor = FrozenSemanticAnchor(16, 8, seed=seed)
    predictors = nn.ModuleDict(
        {
            str(layer): NormalFeaturePredictor(
                input_dim=16,
                semantic_dim=8,
                hidden_dim=16,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
            )
            for layer in selected_layers
        }
    )
    normal_optimizer = torch.optim.Adam(predictors.parameters(), lr=1.0e-3)
    normal_loss = train_normal_batch(
        backbone=backbone,
        semantic_anchor=semantic_anchor,
        predictors=predictors,
        images=torch.randn(2, 3, 16, 16),
        labels=torch.zeros(2, dtype=torch.long),
        selected_layers=selected_layers,
        optimizer=normal_optimizer,
    )

    detector = SemTrace(
        backbone=backbone,
        semantic_anchor=semantic_anchor,
        selected_layers=selected_layers,
        feature_dim=16,
        semantic_dim=8,
        trace_dim=8,
        normal_predictors=predictors,
        cross_attention_heads=2,
        cross_attention_dropout=0.0,
    )
    trainable_parameters = [
        parameter for parameter in detector.parameters() if parameter.requires_grad
    ]
    detector_optimizer = torch.optim.Adam(trainable_parameters, lr=1.0e-3)
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    detector_loss = train_detector_batch(
        model=detector,
        images=images,
        labels=labels,
        optimizer=detector_optimizer,
    )
    checkpoint_path = destination / "checkpoints" / "smoke.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=detector,
        optimizer=detector_optimizer,
        epoch=1,
        global_step=1,
        config={"experiment": "synthetic_smoke"},
        selected_layers=selected_layers,
        manifest_hash="synthetic",
        best_validation_metric=0.0,
    )
    load_training_checkpoint(
        checkpoint_path,
        model=detector,
        optimizer=detector_optimizer,
        restore_random_state=True,
    )
    evaluation = evaluate_batch(
        detector,
        images,
        labels,
        ["synthetic_a", "synthetic_a", "synthetic_b", "synthetic_b"],
    )
    return SmokeResult(
        selected_layers=selected_layers,
        normal_loss=normal_loss,
        detector_loss=detector_loss,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path,
    )


def _synthetic_probe_splits(
    rng: np.random.Generator,
) -> tuple[ProbeSplit, ProbeSplit]:
    def make_split(sample_count: int) -> ProbeSplit:
        authenticity = np.arange(sample_count) % 2
        semantic = (np.arange(sample_count) // 2) % 2
        nuisance = (np.arange(sample_count) // 4) % 2
        features: dict[int, np.ndarray] = {}
        for layer in range(11):
            noise = rng.normal(size=(sample_count, 8)).astype(np.float32)
            noise[:, 0] += authenticity * (0.2 + (layer % 4) * 0.2)
            noise[:, 1] += semantic * (1.0 - layer * 0.05)
            noise[:, 2] += nuisance * 0.3
            features[layer] = noise
        return ProbeSplit(features, authenticity, semantic, nuisance)

    return make_split(64), make_split(32)
