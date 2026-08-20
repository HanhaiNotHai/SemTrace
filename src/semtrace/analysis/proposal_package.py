from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from semtrace.analysis.feature_cache import FeatureCacheReader
from semtrace.analysis.proposal_mechanisms import (
    MASK_CONDITION,
    SCALE_NAMES,
    build_core1_statistics,
    build_core2_statistics,
    build_core3_statistics,
    build_intervention_plan,
    create_package_archive,
    write_file_manifest,
)
from semtrace.analysis.proposal_plotting import generate_proposal_figures
from semtrace.analysis.proposal_reporting import write_proposal_documents
from semtrace.models.classifier import TraceClassifier
from semtrace.models.cross_attention import SemanticTraceCrossAttention
from semtrace.utils.environment import file_sha256


def build_proposal_package(
    *,
    mechanism_root: str | Path,
    eval_root: str | Path,
    checkpoint: str | Path,
    selected_layers_path: str | Path,
    dataset: str,
    protocol: str,
    output_root: str | Path,
    output_dir: str | Path | None,
    random_seeds: Sequence[int],
    bootstrap_iterations: int,
    dpi: int,
    image_formats: Sequence[str],
    head_batch_size: int,
    device: str,
) -> Path:
    mechanism = Path(mechanism_root).resolve()
    evaluation = Path(eval_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    selected_path = Path(selected_layers_path).resolve()
    _validate_inputs(mechanism, evaluation, checkpoint_path, selected_path)
    if not random_seeds:
        raise ValueError("at least one random seed is required")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    seed = int(random_seeds[0])
    selected_payload = _read_json(selected_path)
    selected_layers = tuple(int(layer) for layer in selected_payload["selected_layers"])
    if len(selected_layers) != 3:
        raise ValueError("selected_layers.json must contain exactly three layers")
    mechanism_config = _read_yaml(mechanism / "config_resolved.yaml")
    eval_config = _read_yaml(evaluation / "config_resolved.yaml")
    generators = tuple(str(value) for value in eval_config["evaluation"]["domains"])
    threshold = _mechanism_threshold(mechanism)
    package = _new_output_directory(output_root, output_dir)
    for name in ("data", "figures", "descriptions", "reports", "metadata"):
        (package / name).mkdir(parents=True, exist_ok=True)

    print("[1/7] Reading existing baseline and leave-one-scale-out predictions")
    masking_predictions = load_masking_predictions(
        mechanism / "predictions" / "intervention_predictions.csv",
        selected_layers=selected_layers,
    )
    baseline_predictions = masking_predictions[
        masking_predictions["condition"] == "baseline"
    ].copy()

    print("[2/7] Building three-scale candidate residual statistics")
    core1_samples = load_core1_samples(
        mechanism / "tables" / "residuals.csv",
        baseline_predictions,
        selected_layers=selected_layers,
        threshold=threshold,
    )
    core1_summary, core1_statistics = build_core1_statistics(
        core1_samples,
        selected_layers=selected_layers,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    _write_frame(package / "data" / "core1_residual_samples.csv", core1_samples)
    _write_frame(package / "data" / "core1_residual_summary.csv", core1_summary)
    _write_json(package / "data" / "core1_residual_statistics.json", core1_statistics)

    print("[3/7] Building generator-by-scale masking statistics")
    core2_table, core2_summary, core2_statistics = build_core2_statistics(
        masking_predictions,
        selected_layers=selected_layers,
        generators=generators,
        threshold=threshold,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    _write_frame(package / "data" / "core2_scale_masking.csv", core2_table)
    _write_frame(package / "data" / "core2_scale_masking_summary.csv", core2_summary)
    _write_json(package / "data" / "core2_scale_masking_statistics.json", core2_statistics)

    print("[4/7] Recomputing global semantic/trace swaps from cached features only")
    core3_samples, cache_context = run_cached_head_interventions(
        mechanism / "feature_cache",
        checkpoint_path,
        expected_selected_layers=selected_layers,
        threshold=threshold,
        seed=seed,
        batch_size=head_batch_size,
        device=device,
    )
    core3_summary, core3_statistics = build_core3_statistics(
        core3_samples,
        threshold=threshold,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    core3_statistics["cache_context"] = cache_context
    _write_frame(package / "data" / "core3_semantic_trace_swap.csv", core3_samples)
    _write_frame(package / "data" / "core3_semantic_trace_swap_summary.csv", core3_summary)
    _write_json(
        package / "data" / "core3_semantic_trace_swap_statistics.json",
        core3_statistics,
    )

    print("[5/7] Rendering independent figures and triptych")
    figure_paths = generate_proposal_figures(
        core1_samples,
        core2_table,
        core3_summary,
        package / "figures",
        selected_layers=selected_layers,
        dpi=dpi,
        formats=image_formats,
    )
    eval_metrics = _read_json(evaluation / "metrics.jsonl")
    status = _package_status(core1_statistics, core2_statistics, core3_statistics)
    provenance = {
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "mechanism_root": str(mechanism),
        "eval_root": str(evaluation),
        "dataset": dataset,
        "protocol": protocol,
        "selected_layers": list(selected_layers),
        "sample_counts": {
            "core1_unique_samples": int(core1_samples["sample_id"].nunique()),
            "core2_unique_samples": int(masking_predictions["sample_id"].nunique()),
            "core3_unique_samples": int(core3_samples["sample_id"].nunique()),
        },
        "random_seeds": list(random_seeds),
        "bootstrap_iterations": bootstrap_iterations,
        "confidence_level": 0.95,
        "generated_at": datetime.now(UTC).isoformat(),
        "uv_lock_hash": file_sha256(Path("uv.lock").resolve()),
        "mechanism_cache_fingerprint": cache_context["cache_fingerprint"],
    }
    experiment_context = {
        "dataset": dataset,
        "protocol": protocol,
        "generators": list(generators),
        "mechanism_threshold": threshold,
        "main_evaluation": eval_metrics,
        "mechanism_config": mechanism_config,
        "evaluation_config": eval_config,
    }
    model_info = {
        "model": mechanism_config["model"],
        "cross_attention": mechanism_config["cross_attention"],
        "selected_layers": list(selected_layers),
        "checkpoint_epoch": cache_context["checkpoint_epoch"],
        "checkpoint_global_step": cache_context["checkpoint_global_step"],
        "semantic_gate": cache_context["semantic_gate"],
    }
    _write_json(package / "metadata" / "experiment_context.json", experiment_context)
    _write_json(package / "metadata" / "model_info.json", model_info)
    _write_json(package / "metadata" / "selected_layers.json", selected_payload)
    _write_json(package / "metadata" / "package_status.json", status)
    _write_json(package / "metadata" / "provenance.json", provenance)

    print("[6/7] Writing descriptions, report, talking points, and ChatGPT handoff")
    write_proposal_documents(
        package,
        core1_statistics=core1_statistics,
        core2_table=core2_table,
        core2_summary=core2_summary,
        core2_statistics=core2_statistics,
        core3_summary=core3_summary,
        core3_statistics=core3_statistics,
        eval_metrics=eval_metrics,
        provenance=provenance,
        package_status=status,
    )

    print("[7/7] Validating figures, writing manifest, and building ZIP")
    validate_package_data(
        core1_samples,
        core1_summary,
        core2_table,
        core3_summary,
        generators=generators,
        selected_layers=selected_layers,
    )
    validate_figure_files(figure_paths, dpi=dpi)
    write_file_manifest(package)
    create_package_archive(package)
    return package


def load_masking_predictions(
    path: str | Path,
    *,
    selected_layers: Sequence[int],
) -> pd.DataFrame:
    conditions = {"baseline"} | {MASK_CONDITION.format(layer=layer) for layer in selected_layers}
    usecols = ["sample_id", "generator", "label", "condition", "logit", "probability"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        selected = chunk[chunk["condition"].isin(conditions)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("no required baseline/masking predictions were found")
    frame = pd.concat(chunks, ignore_index=True)
    if set(frame["condition"]) != conditions:
        missing = conditions - set(frame["condition"])
        raise ValueError(f"missing masking predictions: {sorted(missing)}")
    return frame


def load_core1_samples(
    path: str | Path,
    baseline_predictions: pd.DataFrame,
    *,
    selected_layers: Sequence[int],
    threshold: float,
) -> pd.DataFrame:
    metadata_columns = [
        "sample_id",
        "path",
        "label",
        "generator",
        "semantic_class",
        "content_env",
        "layer",
        "metric",
        "value",
    ]
    chunks = []
    for chunk in pd.read_csv(path, usecols=metadata_columns, chunksize=200_000):
        selected = chunk[chunk["metric"].isin(("patch_l1_mean", "patch_l2_mean"))]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("candidate residual sample statistics are absent")
    residuals = pd.concat(chunks, ignore_index=True)
    l2 = (
        residuals[residuals["metric"] == "patch_l2_mean"]
        .drop(columns=["metric"])
        .rename(columns={"value": "residual_l2_mean", "layer": "selected_layer"})
    )
    l1 = residuals[residuals["metric"] == "patch_l1_mean"][["sample_id", "layer", "value"]].rename(
        columns={"layer": "selected_layer", "value": "residual_l1_mean"}
    )
    samples = l2.merge(l1, on=["sample_id", "selected_layer"], validate="one_to_one")
    baseline = (
        baseline_predictions[baseline_predictions["condition"] == "baseline"][
            ["sample_id", "probability"]
        ]
        .drop_duplicates("sample_id")
        .rename(columns={"sample_id": "diagnostic_path"})
    )
    samples = samples.merge(
        baseline,
        left_on="path",
        right_on="diagnostic_path",
        validate="many_to_one",
    ).drop(columns=["diagnostic_path"])
    scale_by_layer = dict(zip(selected_layers, SCALE_NAMES, strict=True))
    samples["scale"] = samples["selected_layer"].map(scale_by_layer)
    samples["prediction"] = (samples["probability"] >= threshold).astype(int)
    columns = [
        "sample_id",
        "path",
        "label",
        "generator",
        "semantic_class",
        "content_env",
        "scale",
        "selected_layer",
        "residual_l2_mean",
        "residual_l1_mean",
        "prediction",
        "probability",
    ]
    return samples[columns].sort_values(["selected_layer", "sample_id"]).reset_index(drop=True)


@torch.inference_mode()
def run_cached_head_interventions(
    cache_root: str | Path,
    checkpoint: str | Path,
    *,
    expected_selected_layers: Sequence[int],
    threshold: float,
    seed: int,
    batch_size: int,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("head_batch_size must be positive")
    reader = FeatureCacheReader(cache_root)
    checkpoint_hash = file_sha256(checkpoint)
    validate_cache_checkpoint(
        reader.index["fingerprint"],
        checkpoint_sha256=checkpoint_hash,
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_layers = tuple(int(layer) for layer in checkpoint_payload["selected_layers"])
    if checkpoint_layers != tuple(expected_selected_layers):
        raise ValueError(
            "checkpoint selected layers do not match selected_layers.json: "
            f"{checkpoint_layers} != {tuple(expected_selected_layers)}"
        )
    cross_attention, classifier = _load_detection_head(checkpoint_payload)
    metadata: list[dict[str, Any]] = []
    semantic_parts: list[torch.Tensor] = []
    trace_parts: list[torch.Tensor] = []
    cached_logits: list[torch.Tensor] = []
    total_shards = len(reader.index["shards"])
    for shard in tqdm(reader.iter_shards(), total=total_shards, desc="Loading cached heads"):
        semantic = shard.features.get("semantic_anchor")
        trace = shard.features.get("fused_trace_tokens")
        logits = shard.features.get("logits")
        if not isinstance(semantic, torch.Tensor) or not isinstance(trace, torch.Tensor):
            raise TypeError(
                "feature cache must contain tensor semantic_anchor and fused_trace_tokens"
            )
        if not isinstance(logits, torch.Tensor):
            raise TypeError("feature cache must contain tensor logits")
        semantic_parts.append(semantic)
        trace_parts.append(trace)
        cached_logits.append(logits)
        metadata.extend(
            {
                "sample_id": sample.sample_id,
                "path": sample.path,
                "label": sample.label,
                "generator": sample.generator,
                "semantic_class": sample.semantic_class,
                "content_env": sample.content_env,
                "source_dataset": sample.source_dataset,
                "degradation": sample.degradation,
            }
            for sample in shard.metadata
        )
    semantic_tensor = torch.cat(semantic_parts)
    trace_tensor = torch.cat(trace_parts)
    stored_logits = torch.cat(cached_logits).float().numpy()
    if semantic_tensor.shape[0] != len(metadata) or trace_tensor.shape[0] != len(metadata):
        raise ValueError("cache metadata and tensor counts differ")
    labels = np.asarray([row["label"] for row in metadata], dtype=np.int64)
    generators = np.asarray([row["generator"] for row in metadata], dtype=str)
    plan = build_intervention_plan(labels, generators, seed=seed)
    target_device = _resolve_device(device)
    cross_attention.to(target_device).eval()
    classifier.to(target_device).eval()
    condition_outputs: dict[str, np.ndarray] = {}
    for condition, indices in plan.items():
        logits_parts = []
        for start in tqdm(
            range(0, len(metadata), batch_size),
            desc=f"Head intervention: {condition}",
            leave=False,
        ):
            stop = min(start + batch_size, len(metadata))
            semantic_batch = semantic_tensor[indices["semantic_indices"][start:stop]].to(
                device=target_device, dtype=torch.float32
            )
            trace_batch = trace_tensor[indices["trace_indices"][start:stop]].to(
                device=target_device, dtype=torch.float32
            )
            evidence, _ = cross_attention(semantic_batch, trace_batch)
            logits_parts.append(classifier(evidence).cpu())
        condition_outputs[condition] = torch.cat(logits_parts).numpy()
    rows: list[dict[str, Any]] = []
    for condition, logits in condition_outputs.items():
        donor_indices = plan[condition]["donor_indices"]
        probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        for index, base in enumerate(metadata):
            donor_index = int(donor_indices[index])
            rows.append(
                {
                    **base,
                    "condition": condition,
                    "logit": float(logits[index]),
                    "probability": float(probabilities[index]),
                    "prediction": int(probabilities[index] >= threshold),
                    "donor_sample_id": metadata[donor_index]["sample_id"],
                    "donor_label": int(labels[donor_index]),
                    "donor_generator": str(generators[donor_index]),
                    "donor_matched": bool(plan[condition]["donor_matched"][index]),
                }
            )
    recomputed_baseline = condition_outputs["baseline"]
    return pd.DataFrame(rows), {
        "cache_fingerprint": reader.index["fingerprint"],
        "cache_dtype": reader.index["dtype"],
        "sample_count": len(metadata),
        "device": str(target_device),
        "stored_vs_recomputed_baseline_mean_absolute_logit_difference": float(
            np.mean(np.abs(stored_logits - recomputed_baseline))
        ),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "checkpoint_global_step": int(checkpoint_payload["global_step"]),
        "checkpoint_selected_layers": list(checkpoint_layers),
        "semantic_gate": float(cross_attention.semantic_gate.cpu()),
    }


def validate_cache_checkpoint(
    cache_fingerprint: dict[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    cached_hash = cache_fingerprint.get("checkpoint_sha256")
    if cached_hash != checkpoint_sha256:
        raise ValueError(
            "checkpoint fingerprint mismatch: "
            f"cache={cached_hash!r}, requested={checkpoint_sha256!r}"
        )


def validate_package_data(
    core1_samples: pd.DataFrame,
    core1_summary: pd.DataFrame,
    core2_table: pd.DataFrame,
    core3_summary: pd.DataFrame,
    *,
    generators: Sequence[str],
    selected_layers: Sequence[int],
) -> None:
    if set(core1_samples["selected_layer"].astype(int)) != set(selected_layers):
        raise ValueError("core1 validation failed: selected layers are incomplete")
    if set(core1_samples["label"].astype(int)) != {0, 1}:
        raise ValueError("core1 validation failed: real/fake samples are incomplete")
    if len(core1_summary[core1_summary["group_type"] == "authenticity"]) != 6:
        raise ValueError("core1 validation failed: summary has the wrong size")
    if list(core2_table["generator"]) != list(generators):
        raise ValueError("core2 validation failed: generator order or coverage differs")
    if set(core3_summary["condition"]) != {
        "matched_semantic_swap",
        "real_fake_trace_swap",
    }:
        raise ValueError("core3 validation failed: intervention coverage differs")
    numeric_frames = (
        core1_samples.select_dtypes(include=[np.number]),
        core1_summary.select_dtypes(include=[np.number]),
        core2_table.select_dtypes(include=[np.number]),
    )
    if any(np.isinf(frame.to_numpy()).any() for frame in numeric_frames):
        raise ValueError("package data contain Inf")


def validate_figure_files(paths: Sequence[Path], *, dpi: int) -> None:
    if len(paths) < 8:
        raise ValueError("all four PNG/PDF figures are required")
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"figure was not generated: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                if min(image.size) < 600:
                    raise ValueError(f"PNG resolution is too small: {path} {image.size}")
                recorded_dpi = image.info.get("dpi")
                if recorded_dpi is not None and min(recorded_dpi) + 1 < dpi:
                    raise ValueError(f"PNG DPI is lower than requested: {path} {recorded_dpi}")


def _load_detection_head(
    checkpoint: dict[str, Any],
) -> tuple[SemanticTraceCrossAttention, TraceClassifier]:
    config = checkpoint["config"]
    model_config = config["model"]
    attention_config = config["cross_attention"]
    semantic_dim = int(model_config["semantic_dim"])
    trace_dim = int(model_config["trace_dim"])
    cross_attention = SemanticTraceCrossAttention(
        semantic_dim=semantic_dim,
        trace_dim=trace_dim,
        num_heads=int(attention_config["num_heads"]),
        dropout=float(attention_config["dropout"]),
        max_semantic_gate=float(attention_config["max_semantic_gate"]),
    )
    classifier = TraceClassifier(trace_dim)
    state = cast(dict[str, torch.Tensor], checkpoint["model"])
    cross_attention.load_state_dict(
        {
            name.removeprefix("cross_attention."): value
            for name, value in state.items()
            if name.startswith("cross_attention.")
        }
    )
    classifier.load_state_dict(
        {
            name.removeprefix("classifier."): value
            for name, value in state.items()
            if name.startswith("classifier.")
        }
    )
    return cross_attention, classifier


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _mechanism_threshold(mechanism_root: Path) -> float:
    masking = pd.read_csv(
        mechanism_root / "tables" / "masking.csv",
        usecols=["global_baseline_threshold"],
    )
    values = masking["global_baseline_threshold"].dropna().unique()
    if values.size != 1:
        raise ValueError("mechanism masking output must contain one global baseline threshold")
    return float(values[0])


def _package_status(
    core1: dict[str, Any],
    core2: dict[str, Any],
    core3: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "core1_normal_pattern_deviation": bool(core1["supports_normal_pattern_deviation"]),
        "core2_scale_complementarity": bool(core2["supports_scale_complementarity"]),
        "core3_trace_as_evidence": bool(core3["supports_trace_as_evidence"]),
    }
    needs_review = not all(checks.values())
    reasons = [name for name, supported in checks.items() if not supported]
    return {
        "proposal_material_status": "NEEDS_REVIEW" if needs_review else "READY",
        "needs_review": needs_review,
        "reason": "; ".join(reasons) if reasons else None,
        "mechanism_checks": checks,
        "known_counterexample": (
            "A negative masking drop means that masking improved that generator; it is retained "
            "in all tables and figures."
        ),
    }


def _new_output_directory(output_root: str | Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = Path(output_root) / timestamp
    else:
        destination = Path(output_dir)
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _validate_inputs(
    mechanism: Path,
    evaluation: Path,
    checkpoint: Path,
    selected_layers: Path,
) -> None:
    required = (
        mechanism / "tables" / "residuals.csv",
        mechanism / "tables" / "masking.csv",
        mechanism / "predictions" / "intervention_predictions.csv",
        mechanism / "feature_cache" / "index.json",
        evaluation / "metrics.jsonl",
        checkpoint,
        selected_layers,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"proposal mechanism inputs are missing: {missing}")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
