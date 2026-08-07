from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semtrace.analysis.feature_cache import (
    AnalysisSampleMetadata,
    CacheFingerprint,
    FeatureCacheWriter,
    FeatureValue,
)
from semtrace.analysis.scale_usage import fusion_weight_contributions
from semtrace.data.collate import ImageBatch
from semtrace.models.semtrace import SemTrace, SemTraceAnalysisOutput
from semtrace.utils.environment import file_sha256


def cache_fingerprint(
    checkpoint: str | Path,
    config: DictConfig,
    manifests: Iterable[str | Path],
) -> CacheFingerprint:
    config_payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_payload, dict):
        raise TypeError("resolved analysis configuration must be a mapping")
    analysis = config_payload.get("analysis")
    if isinstance(analysis, dict):
        config_payload["analysis"] = {
            "cache_dtype": analysis.get("cache_dtype"),
            "max_samples_per_group": analysis.get("max_samples_per_group"),
        }
    config_text = OmegaConf.to_yaml(OmegaConf.create(config_payload), resolve=True)
    manifest_digest = hashlib.sha256()
    for manifest in sorted(Path(path) for path in manifests):
        manifest_digest.update(file_sha256(manifest).encode("ascii"))
    return CacheFingerprint(
        checkpoint_sha256=file_sha256(checkpoint),
        config_sha256=hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        manifest_sha256=manifest_digest.hexdigest(),
    )


def analysis_features(
    output: SemTraceAnalysisOutput,
    model: SemTrace | None = None,
) -> dict[str, FeatureValue]:
    batch_size = output.logits.shape[0]
    features: dict[str, FeatureValue] = {
        "semantic_anchor": output.semantic_anchor,
        "raw_patch_features": output.raw_patch_features,
        "predicted_normal_features": output.predicted_normal_features,
        "prediction_errors": output.prediction_errors,
        "candidate_trace_residuals": output.candidate_trace_residuals,
        "adapted_trace_tokens": output.adapted_trace_tokens,
        "fused_trace_tokens": output.fused_trace_tokens,
        "trace_evidence": output.trace_evidence,
        "logits": output.logits,
        "probability": output.logits.sigmoid(),
        "patch_grid_size": torch.tensor(
            output.patch_grid_size,
            device=output.logits.device,
            dtype=output.logits.dtype,
        )[None].repeat(batch_size, 1),
    }
    if output.attention_weights is not None:
        features["attention_weights"] = output.attention_weights
    if model is not None:
        fusion_linear = model.trace_fusion.fusion[0]
        if not isinstance(fusion_linear, torch.nn.Linear):
            raise TypeError("trace fusion must begin with the configured linear projection")
        features["fusion_contributions"] = fusion_weight_contributions(
            output.adapted_trace_tokens,
            fusion_linear,
        )
    return features


@torch.inference_mode()
def extract_feature_cache(
    model: SemTrace,
    loader: DataLoader[ImageBatch],
    device: torch.device,
    writer: FeatureCacheWriter,
    *,
    amp_mode: str = "none",
    show_progress: bool = True,
    max_samples_per_group: int | None = None,
    batch_callback: (
        Callable[[SemTrace, ImageBatch, SemTraceAnalysisOutput], None] | None
    ) = None,
) -> int:
    """Extract one CPU shard per batch and resume by stable path-derived sample ID."""
    model.eval()
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    group_counts: dict[tuple[int, str], int] = {}
    written = 0
    progress = tqdm(
        loader,
        total=len(loader),
        desc="Extracting mechanism features",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in progress:
        sample_ids = [_sample_id(path) for path in batch.paths]
        keep: list[int] = []
        for index, sample_id in enumerate(sample_ids):
            group = (int(batch.labels[index]), batch.generators[index])
            if sample_id in writer.completed_sample_ids:
                continue
            if (
                max_samples_per_group is not None
                and group_counts.get(group, 0) >= max_samples_per_group
            ):
                continue
            keep.append(index)
            group_counts[group] = group_counts.get(group, 0) + 1
        if not keep:
            continue
        selected = _select_batch(batch, keep).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_mode != "none",
        ):
            output = model.analyze(selected.images)
        if batch_callback is not None:
            batch_callback(model, selected, output)
        writer.write_shard(_metadata(selected), analysis_features(output, model))
        written += len(keep)
        if show_progress:
            progress.set_postfix(samples=written)
    return written


def _metadata(batch: ImageBatch) -> list[AnalysisSampleMetadata]:
    return [
        AnalysisSampleMetadata(
            sample_id=_sample_id(path),
            path=path,
            label=int(batch.labels[index].cpu()),
            generator=batch.generators[index],
            semantic_class=(
                int(batch.semantic_classes[index].cpu())
                if int(batch.semantic_classes[index].cpu()) >= 0
                else None
            ),
            content_env=batch.content_envs[index],
            real_source=batch.real_sources[index],
            source_dataset=batch.source_datasets[index],
            degradation=batch.degradations[index],
            split=batch.splits[index],
        )
        for index, path in enumerate(batch.paths)
    ]


def _select_batch(batch: ImageBatch, indices: list[int]) -> ImageBatch:
    return ImageBatch(
        images=batch.images[indices],
        labels=batch.labels[indices],
        semantic_classes=batch.semantic_classes[indices],
        generators=[batch.generators[index] for index in indices],
        sources=[batch.sources[index] for index in indices],
        degradations=[batch.degradations[index] for index in indices],
        file_formats=[batch.file_formats[index] for index in indices],
        paths=[batch.paths[index] for index in indices],
        content_envs=[batch.content_envs[index] for index in indices],
        real_sources=[batch.real_sources[index] for index in indices],
        source_datasets=[batch.source_datasets[index] for index in indices],
        splits=[batch.splits[index] for index in indices],
    )


def _sample_id(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()
