from pathlib import Path

import torch
from torch import nn

from semtrace.analysis.diagnostics import LiveDiagnosticCollector
from semtrace.analysis.feature_cache import (
    AnalysisSampleMetadata,
    CacheFingerprint,
    FeatureCacheWriter,
)
from semtrace.analysis.report import generate_mechanism_report
from semtrace.backbones.base import TinyBackbone
from semtrace.data.collate import ImageBatch
from semtrace.models.normal_predictor import NormalFeaturePredictor
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.semtrace import SemTrace


def test_synthetic_cache_generates_mechanism_report(tmp_path: Path) -> None:
    cache = tmp_path / "feature_cache"
    fingerprint = CacheFingerprint("checkpoint", "config", "manifest")
    writer = FeatureCacheWriter(cache, fingerprint, dtype=torch.float32, rank=0)
    generator = torch.Generator().manual_seed(3)
    for split in ("train", "test"):
        batch = 8
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        signal = labels.float()[:, None, None]
        raw = {layer: torch.randn(batch, 4, 6, generator=generator) for layer in (2, 6, 8)}
        predicted = {layer: value * 0.8 for layer, value in raw.items()}
        errors = {layer: raw[layer] - predicted[layer] + signal for layer in raw}
        residuals = {
            layer: torch.nn.functional.layer_norm(value, (value.shape[-1],))
            for layer, value in errors.items()
        }
        adapted = {
            layer: torch.randn(batch, 4, 4, generator=generator) + signal
            for layer in raw
        }
        fused = torch.stack(list(adapted.values())).mean(dim=0)
        metadata = [
            AnalysisSampleMetadata(
                sample_id=f"{split}-{index}",
                path=f"/{split}/{index}.png",
                label=int(labels[index]),
                generator="real" if labels[index] == 0 else f"fake-{index % 2}",
                semantic_class=index % 2,
                content_env=f"env-{index % 2}",
                real_source="camera" if labels[index] == 0 else None,
                source_dataset="synthetic",
                degradation=None,
                split=split,
            )
            for index in range(batch)
        ]
        writer.write_shard(
            metadata,
            {
                "semantic_anchor": torch.randn(batch, 5, generator=generator),
                "raw_patch_features": raw,
                "predicted_normal_features": predicted,
                "prediction_errors": errors,
                "candidate_trace_residuals": residuals,
                "adapted_trace_tokens": adapted,
                "fused_trace_tokens": fused,
                "attention_weights": torch.softmax(
                    torch.randn(batch, 2, 1, 4, generator=generator), dim=-1
                ),
                "trace_evidence": fused.mean(dim=1),
                "logits": labels.float() * 3.0 - 1.0,
                "probability": (labels.float() * 3.0 - 1.0).sigmoid(),
                "patch_grid_size": torch.tensor([[2, 2]]).repeat(batch, 1),
            },
        )
    writer.finalize()

    output = tmp_path / "report"
    summary = generate_mechanism_report(
        cache,
        output,
        bootstrap_iterations=20,
        random_seeds=(0, 1),
        prototype_counts=(2,),
        top_r=2,
    )

    assert summary["sample_count"] == 16
    assert (output / "mechanism_report.md").is_file()
    assert (output / "mechanism_summary.json").is_file()
    assert (output / "tables" / "normal_predictor.csv").is_file()
    assert (output / "tables" / "residuals.csv").is_file()
    assert (output / "tables" / "scale_usage_overall.csv").is_file()
    assert (output / "tables" / "representation_mi.csv").is_file()
    assert (output / "tables" / "linear_probes.csv").is_file()
    assert (output / "tables" / "trace_coverage.csv").is_file()


def test_synthetic_live_masking_diagnostics(tmp_path: Path) -> None:
    layers = (0, 1, 2)
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
            for layer in layers
        }
    )
    model = SemTrace(
        backbone=TinyBackbone(
            hidden_size=16,
            patch_size=4,
            num_layers=4,
            selected_layers=layers,
        ),
        semantic_anchor=FrozenSemanticAnchor(16, 8),
        selected_layers=layers,
        feature_dim=16,
        semantic_dim=8,
        trace_dim=8,
        normal_predictors=predictors,
        cross_attention_heads=2,
        cross_attention_dropout=0.0,
    ).eval()
    batch = ImageBatch(
        images=torch.randn(4, 3, 16, 16),
        labels=torch.tensor([0, 0, 1, 1]),
        semantic_classes=torch.tensor([0, 1, 0, 1]),
        generators=["real", "real", "fake-a", "fake-b"],
        sources=["synthetic"] * 4,
        degradations=[None] * 4,
        file_formats=["png"] * 4,
        paths=[f"sample-{index}.png" for index in range(4)],
        content_envs=["a", "b", "a", "b"],
        real_sources=["camera", "camera", None, None],
        source_datasets=["synthetic"] * 4,
        splits=["test"] * 4,
    )
    baseline = model.analyze(batch.images)
    collector = LiveDiagnosticCollector(
        tmp_path,
        rank=0,
        tasks=frozenset({"masking", "cross_attention"}),
        patch_mask_ratios=(0.25,),
        random_seeds=(0, 1, 2),
        visualization_limit=0,
    )

    collector.process(model, batch, baseline)
    collector.save_rank()
    result = collector.finalize(bootstrap_iterations=20)

    assert result["condition_count"] > 1
    assert (tmp_path / "tables" / "masking.csv").is_file()
    assert (tmp_path / "tables" / "cross_attention_stability.csv").is_file()
    assert (tmp_path / "predictions" / "intervention_predictions.csv").is_file()
