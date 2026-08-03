from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semtrace.data.collate import ImageBatch
from semtrace.models.probes import ProbeSplit, choose_nuisance_label


@dataclass(frozen=True, slots=True)
class ExtractedProbeFeatures:
    features: dict[int, np.ndarray]
    authenticity: np.ndarray
    semantic: np.ndarray
    metadata: dict[str, list[str | None]]


@torch.no_grad()
def extract_probe_features(
    backbone: nn.Module,
    loader: DataLoader[ImageBatch],
    device: torch.device,
    *,
    description: str = "Extracting probe features",
    show_progress: bool = False,
) -> ExtractedProbeFeatures:
    backbone.eval()
    features: dict[int, list[np.ndarray]] = {}
    authenticity: list[np.ndarray] = []
    semantic: list[np.ndarray] = []
    metadata: dict[str, list[str | None]] = {
        "source": [],
        "degradation": [],
        "file_format": [],
        "generator": [],
    }
    sample_count = 0
    extraction_started = time.perf_counter()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in progress:
        batch = batch.to(device)
        output = backbone(batch.images)
        for layer, patch_tokens in output.intermediate_patch_tokens.items():
            features.setdefault(layer, []).append(
                patch_tokens.mean(dim=1).float().cpu().numpy()
            )
        authenticity.append(batch.labels.cpu().numpy())
        semantic.append(batch.semantic_classes.cpu().numpy())
        metadata["source"].extend(batch.sources)
        metadata["degradation"].extend(batch.degradations)
        metadata["file_format"].extend(batch.file_formats)
        metadata["generator"].extend(batch.generators)
        sample_count += int(batch.labels.shape[0])
        elapsed = max(time.perf_counter() - extraction_started, 1.0e-9)
        progress.set_postfix(
            samples=sample_count,
            **{"samples/s": f"{sample_count / elapsed:.1f}"},
        )
    if not authenticity:
        raise ValueError("probe feature loader is empty")
    local = ExtractedProbeFeatures(
        features={
            layer: np.concatenate(chunks, axis=0) for layer, chunks in features.items()
        },
        authenticity=np.concatenate(authenticity),
        semantic=np.concatenate(semantic),
        metadata=metadata,
    )
    extraction_elapsed = time.perf_counter() - extraction_started
    if show_progress:
        tqdm.write(
            f"[Stage 1] {description} complete: {sample_count} local samples "
            f"in {_format_duration(extraction_elapsed)} "
            f"({sample_count / max(extraction_elapsed, 1.0e-9):.1f} samples/s)"
        )
    return gather_probe_features(local, show_progress=show_progress)


def gather_probe_features(
    local: ExtractedProbeFeatures,
    *,
    show_progress: bool = False,
) -> ExtractedProbeFeatures:
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if show_progress:
        tqdm.write(f"[Stage 1] Gathering features across {world_size} rank(s)...")
    gather_started = time.perf_counter()
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        if show_progress:
            elapsed = time.perf_counter() - gather_started
            tqdm.write(
                f"[Stage 1] Feature gather complete: "
                f"{len(local.authenticity)} global samples in {_format_duration(elapsed)}"
            )
        return local
    gathered: list[ExtractedProbeFeatures | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    complete = [part for part in gathered if part is not None]
    combined = ExtractedProbeFeatures(
        features={
            layer: np.concatenate([part.features[layer] for part in complete], axis=0)
            for layer in local.features
        },
        authenticity=np.concatenate([part.authenticity for part in complete]),
        semantic=np.concatenate([part.semantic for part in complete]),
        metadata={
            name: [value for part in complete for value in part.metadata[name]]
            for name in local.metadata
        },
    )
    if show_progress:
        elapsed = time.perf_counter() - gather_started
        tqdm.write(
            f"[Stage 1] Feature gather complete: "
            f"{len(combined.authenticity)} global samples in {_format_duration(elapsed)}"
        )
    return combined


def prepare_probe_splits(
    train: ExtractedProbeFeatures,
    validation: ExtractedProbeFeatures,
) -> tuple[ProbeSplit, ProbeSplit, str, bool, float]:
    nuisance_name, train_nuisance, generator_enabled = choose_nuisance_label(
        train.metadata
    )
    validation_nuisance = (
        validation.metadata[nuisance_name]
        if nuisance_name in validation.metadata
        else [None] * len(validation.authenticity)
    )
    nuisance_mapping = {
        value: index
        for index, value in enumerate(
            sorted({value for value in train_nuisance if value is not None})
        )
    }

    def encode(values: list[str | None] | tuple[str | None, ...]) -> np.ndarray:
        return np.asarray(
            [
                nuisance_mapping.get(value, -1) if value is not None else -1
                for value in values
            ],
            dtype=np.int64,
        )

    coverage = float(np.mean(train.semantic != -1))
    return (
        ProbeSplit(
            train.features,
            train.authenticity,
            train.semantic,
            encode(list(train_nuisance)),
        ),
        ProbeSplit(
            validation.features,
            validation.authenticity,
            validation.semantic,
            encode(list(validation_nuisance)),
        ),
        nuisance_name,
        generator_enabled,
        coverage,
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m"
