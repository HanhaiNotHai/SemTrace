from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from semtrace.data.collate import ImageBatch
from semtrace.metrics.binary import grouped_binary_metrics
from semtrace.models.semtrace import SemTrace


@torch.no_grad()
def evaluate_batch(
    model: SemTrace,
    images: torch.Tensor,
    labels: torch.Tensor,
    domains: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    model.eval()
    probabilities = model(images).logits.sigmoid().cpu().numpy()
    grouped = grouped_binary_metrics(
        labels.cpu().numpy(),
        probabilities,
        domains,
        threshold=threshold,
    )
    result: dict[str, object] = asdict(grouped.overall)
    result["per_generator"] = {
        name: asdict(metrics) for name, metrics in grouped.per_domain.items()
    }
    result["mAcc"] = grouped.mean_accuracy
    result["mAP"] = grouped.mean_average_precision
    return result


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[ImageBatch],
    device: torch.device,
    *,
    threshold: float = 0.5,
    amp_mode: str = "none",
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    model.eval()
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    local_predictions: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_mode != "none",
        ):
            output = model(batch.images)
        probabilities = output.logits.float().sigmoid().cpu().tolist()
        for index, probability in enumerate(probabilities):
            residuals = {
                str(layer): {
                    "mean": float(residual[index].mean().cpu()),
                    "norm": float(residual[index].norm(dim=-1).mean().cpu()),
                }
                for layer, residual in output.candidate_trace_residuals.items()
            }
            local_predictions.append(
                {
                    "path": batch.paths[index],
                    "label": int(batch.labels[index].cpu()),
                    "fake_probability": probability,
                    "generator": batch.generators[index],
                    "semantic_class": int(batch.semantic_classes[index].cpu()),
                    "candidate_trace_residual": residuals,
                    "cross_attention": (
                        output.attention_map[index].float().cpu().tolist()
                        if output.attention_map is not None
                        else None
                    ),
                }
            )
    predictions = local_predictions
    if dist.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_predictions)
        predictions = deduplicate_predictions(
            [
                prediction
                for rank_predictions in gathered
                if rank_predictions is not None
                for prediction in rank_predictions
            ]
        )
    if not predictions:
        raise ValueError("evaluation loader is empty")
    grouped = grouped_binary_metrics(
        [prediction["label"] for prediction in predictions],
        [prediction["fake_probability"] for prediction in predictions],
        [prediction["generator"] for prediction in predictions],
        threshold=threshold,
    )
    result: dict[str, object] = asdict(grouped.overall)
    result["per_generator"] = {
        name: asdict(metrics) for name, metrics in grouped.per_domain.items()
    }
    result["mAcc"] = grouped.mean_accuracy
    result["mAP"] = grouped.mean_average_precision
    result["residual_distributions"] = _residual_distributions(predictions)
    return result, predictions


def deduplicate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove exact dataset-path repeats introduced by sampler padding."""
    unique: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for prediction in predictions:
        path = prediction.get("path")
        if not isinstance(path, str):
            raise TypeError("each prediction must contain a string path")
        if path not in seen_paths:
            seen_paths.add(path)
            unique.append(prediction)
    return unique


def _residual_distributions(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        keys = (
            f"label:{prediction['label']}",
            f"semantic:{prediction['semantic_class']}",
            f"generator:{prediction['generator']}",
        )
        for key in keys:
            groups.setdefault(key, []).append(prediction)
    distributions: dict[str, dict[str, dict[str, float]]] = {}
    for group, members in groups.items():
        layers = members[0]["candidate_trace_residual"]
        distributions[group] = {}
        for layer in layers:
            distributions[group][layer] = {
                "mean": float(
                    sum(
                        member["candidate_trace_residual"][layer]["mean"]
                        for member in members
                    )
                    / len(members)
                ),
                "norm": float(
                    sum(
                        member["candidate_trace_residual"][layer]["norm"]
                        for member in members
                    )
                    / len(members)
                ),
            }
    return distributions
