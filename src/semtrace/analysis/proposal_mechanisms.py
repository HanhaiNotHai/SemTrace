from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from semtrace.analysis.statistics import (
    real_fake_comparison,
    summarize_distribution,
    summary_dict,
)

SCALE_NAMES = ("shallow", "middle", "deep")
MASK_CONDITION = "mask_scale_L{layer}_after_adapter"


def build_core1_statistics(
    samples: pd.DataFrame,
    *,
    selected_layers: Sequence[int],
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize pre-LayerNorm prediction errors without relabeling them as pure traces."""
    _require_columns(
        samples,
        {"label", "generator", "selected_layer", "scale", "residual_l2_mean"},
    )
    _validate_layers(samples, selected_layers)
    values = samples["residual_l2_mean"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("core1 residual values contain NaN or Inf")
    if set(samples["label"].astype(int)) != {0, 1}:
        raise ValueError("core1 requires both real=0 and fake=1 samples")

    rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "metric": "mean patch L2 norm of pre-LayerNorm prediction error h-h_hat",
        "layers": {},
    }
    for layer in selected_layers:
        layer_frame = samples[samples["selected_layer"] == layer]
        layer_stats: dict[str, Any] = {}
        for label, label_name in ((0, "real"), (1, "fake")):
            group = layer_frame[layer_frame["label"] == label]["residual_l2_mean"].to_numpy()
            distribution = summary_dict(
                summarize_distribution(
                    group,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                )
            )
            rows.append(
                {
                    "group_type": "authenticity",
                    "group": label_name,
                    "label": label,
                    "selected_layer": layer,
                    "scale": _scale_for_layer(layer, selected_layers),
                    **distribution,
                }
            )
            layer_stats[label_name] = distribution
        comparison = real_fake_comparison(
            layer_frame[layer_frame["label"] == 0]["residual_l2_mean"].to_numpy(),
            layer_frame[layer_frame["label"] == 1]["residual_l2_mean"].to_numpy(),
        )
        layer_stats["comparison"] = comparison
        fake_frame = layer_frame[layer_frame["label"] == 1]
        by_generator: dict[str, Any] = {}
        for generator, group_frame in fake_frame.groupby("generator", sort=True):
            distribution = summary_dict(
                summarize_distribution(
                    group_frame["residual_l2_mean"].to_numpy(),
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                )
            )
            rows.append(
                {
                    "group_type": "fake_generator",
                    "group": str(generator),
                    "label": 1,
                    "selected_layer": layer,
                    "scale": _scale_for_layer(layer, selected_layers),
                    **distribution,
                }
            )
            by_generator[str(generator)] = distribution
        layer_stats["fake_by_generator"] = by_generator
        statistics["layers"][str(layer)] = layer_stats
    statistics["supports_normal_pattern_deviation"] = all(
        statistics["layers"][str(layer)]["fake"]["mean"]
        > statistics["layers"][str(layer)]["real"]["mean"]
        and statistics["layers"][str(layer)]["comparison"]["mann_whitney_p"] < 0.05
        for layer in selected_layers
    )
    return pd.DataFrame(rows), statistics


def build_core2_statistics(
    predictions: pd.DataFrame,
    *,
    selected_layers: Sequence[int],
    generators: Sequence[str],
    threshold: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute paired full-minus-mask drops for after-adapter scale interventions."""
    _require_columns(
        predictions,
        {"sample_id", "generator", "label", "condition", "probability"},
    )
    required_conditions = {"baseline"} | {
        MASK_CONDITION.format(layer=layer) for layer in selected_layers
    }
    missing_conditions = required_conditions - set(predictions["condition"])
    if missing_conditions:
        raise ValueError(f"core2 missing masking conditions: {sorted(missing_conditions)}")
    rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "metric_units": "AP/Acc in [0,1]; deltas in absolute percentage points",
        "threshold": threshold,
        "generators": {},
    }
    for generator in generators:
        generator_frame = predictions[predictions["generator"] == generator]
        if generator_frame.empty:
            raise ValueError(f"core2 generator is absent: {generator}")
        baseline = _condition_frame(generator_frame, "baseline")
        labels = baseline["label"].to_numpy(dtype=np.int64)
        baseline_scores = baseline["probability"].to_numpy(dtype=np.float64)
        baseline_ap = _average_precision(labels, baseline_scores)
        baseline_acc = float(np.mean((baseline_scores >= threshold) == labels))
        row: dict[str, Any] = {
            "generator": generator,
            "sample_count": len(baseline),
            "baseline_ap": baseline_ap,
            "baseline_acc": baseline_acc,
        }
        generator_stats: dict[str, Any] = {}
        for scale, layer in zip(SCALE_NAMES, selected_layers, strict=True):
            masked = _condition_frame(
                generator_frame, MASK_CONDITION.format(layer=layer)
            ).set_index("sample_id")
            if set(masked.index) != set(baseline["sample_id"]):
                raise ValueError(f"core2 sample mismatch for {generator}, layer {layer}")
            masked = masked.loc[baseline["sample_id"]]
            if not np.array_equal(labels, masked["label"].to_numpy(dtype=np.int64)):
                raise ValueError(f"core2 label mismatch for {generator}, layer {layer}")
            masked_scores = masked["probability"].to_numpy(dtype=np.float64)
            masked_ap = _average_precision(labels, masked_scores)
            masked_acc = float(np.mean((masked_scores >= threshold) == labels))
            ap_ci = _paired_delta_ci(
                labels,
                baseline_scores,
                masked_scores,
                metric="ap",
                threshold=threshold,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            acc_ci = _paired_delta_ci(
                labels,
                baseline_scores,
                masked_scores,
                metric="accuracy",
                threshold=threshold,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            row.update(
                {
                    f"mask_{scale}_ap": masked_ap,
                    f"mask_{scale}_acc": masked_acc,
                    f"delta_ap_{scale}": baseline_ap - masked_ap,
                    f"delta_acc_{scale}": baseline_acc - masked_acc,
                    f"delta_ap_{scale}_pp": (baseline_ap - masked_ap) * 100.0,
                    f"delta_acc_{scale}_pp": (baseline_acc - masked_acc) * 100.0,
                    f"delta_ap_{scale}_ci_low_pp": ap_ci[0] * 100.0,
                    f"delta_ap_{scale}_ci_high_pp": ap_ci[1] * 100.0,
                    f"delta_acc_{scale}_ci_low_pp": acc_ci[0] * 100.0,
                    f"delta_acc_{scale}_ci_high_pp": acc_ci[1] * 100.0,
                }
            )
            generator_stats[str(layer)] = {
                "scale": scale,
                "baseline_ap": baseline_ap,
                "masked_ap": masked_ap,
                "delta_ap_percentage_points": (baseline_ap - masked_ap) * 100.0,
                "delta_ap_ci_percentage_points": [ap_ci[0] * 100.0, ap_ci[1] * 100.0],
                "baseline_accuracy": baseline_acc,
                "masked_accuracy": masked_acc,
                "delta_accuracy_percentage_points": (baseline_acc - masked_acc) * 100.0,
                "delta_accuracy_ci_percentage_points": [
                    acc_ci[0] * 100.0,
                    acc_ci[1] * 100.0,
                ],
            }
        rows.append(row)
        statistics["generators"][generator] = generator_stats
    table = pd.DataFrame(rows)
    summary_rows = []
    for scale, layer in zip(SCALE_NAMES, selected_layers, strict=True):
        delta = table[f"delta_ap_{scale}_pp"]
        max_index = int(delta.idxmax())
        summary_rows.append(
            {
                "scale": scale,
                "selected_layer": layer,
                "mean_delta_ap_pp": float(delta.mean()),
                "standard_deviation_delta_ap_pp": float(delta.std(ddof=1)),
                "max_affected_generator": str(table.loc[max_index, "generator"]),
                "max_delta_ap_pp": float(delta.loc[max_index]),
                "positive_drop_generator_count": int((delta > 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    dominant_scales = {
        str(row["generator"]): max(
            SCALE_NAMES, key=lambda scale: float(row[f"delta_ap_{scale}_pp"])
        )
        for _, row in table.iterrows()
    }
    statistics["dominant_scale_by_generator"] = dominant_scales
    statistics["supports_scale_complementarity"] = bool(
        all(summary["positive_drop_generator_count"] > 0) and len(set(dominant_scales.values())) > 1
    )
    return table, summary, statistics


def build_donor_indices(
    labels: np.ndarray,
    generators: np.ndarray,
    *,
    opposite_authenticity: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic global donors matched by generator and authenticity rule."""
    targets = np.asarray(labels, dtype=np.int64)
    domains = np.asarray(generators, dtype=str)
    if targets.ndim != 1 or domains.shape != targets.shape:
        raise ValueError("labels and generators must be equally sized vectors")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("labels must follow real=0, fake=1")
    rng = np.random.default_rng(seed)
    donors = np.arange(targets.size, dtype=np.int64)
    matched = np.zeros(targets.size, dtype=bool)
    for generator in np.unique(domains):
        for label in (0, 1):
            target_indices = np.flatnonzero((domains == generator) & (targets == label))
            donor_label = 1 - label if opposite_authenticity else label
            candidates = np.flatnonzero((domains == generator) & (targets == donor_label))
            if target_indices.size == 0 or candidates.size == 0:
                continue
            shuffled = rng.permutation(candidates)
            if not opposite_authenticity:
                if candidates.size < 2:
                    continue
                donor_lookup = np.empty(target_indices.size, dtype=np.int64)
                order = rng.permutation(target_indices.size)
                donor_lookup[order] = np.roll(target_indices[order], 1)
            else:
                donor_lookup = np.resize(shuffled, target_indices.size)
                rng.shuffle(donor_lookup)
            donors[target_indices] = donor_lookup
            matched[target_indices] = True
    return donors, matched


def build_intervention_plan(
    labels: np.ndarray,
    generators: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Declare which sample supplies semantic and trace inputs for each intervention."""
    size = np.asarray(labels).size
    original = np.arange(size, dtype=np.int64)
    semantic_donors, semantic_matched = build_donor_indices(
        labels,
        generators,
        opposite_authenticity=False,
        seed=seed,
    )
    trace_donors, trace_matched = build_donor_indices(
        labels,
        generators,
        opposite_authenticity=True,
        seed=seed,
    )
    return {
        "baseline": {
            "semantic_indices": original,
            "trace_indices": original,
            "donor_indices": original,
            "donor_matched": np.ones(size, dtype=bool),
        },
        "matched_semantic_swap": {
            "semantic_indices": semantic_donors,
            "trace_indices": original,
            "donor_indices": semantic_donors,
            "donor_matched": semantic_matched,
        },
        "real_fake_trace_swap": {
            "semantic_indices": original,
            "trace_indices": trace_donors,
            "donor_indices": trace_donors,
            "donor_matched": trace_matched,
        },
    }


def build_core3_statistics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize offline head interventions against each sample's baseline."""
    _require_columns(
        predictions,
        {"sample_id", "label", "condition", "logit", "probability", "donor_label"},
    )
    baseline = _condition_frame(predictions, "baseline").set_index("sample_id")
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for condition in ("matched_semantic_swap", "real_fake_trace_swap"):
        changed = _condition_frame(predictions, condition).set_index("sample_id")
        common = baseline.index.intersection(changed.index)
        if common.empty:
            raise ValueError(f"core3 condition has no matched baseline: {condition}")
        before = baseline.loc[common]
        after = changed.loc[common]
        before_predictions = before["probability"].to_numpy() >= threshold
        after_predictions = after["probability"].to_numpy() >= threshold
        flips = (before_predictions != after_predictions).astype(np.float64)
        probability_change = np.abs(
            after["probability"].to_numpy(dtype=np.float64)
            - before["probability"].to_numpy(dtype=np.float64)
        )
        logit_change = np.abs(
            after["logit"].to_numpy(dtype=np.float64) - before["logit"].to_numpy(dtype=np.float64)
        )
        flip_summary = summary_dict(
            summarize_distribution(
                flips,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
        probability_summary = summary_dict(
            summarize_distribution(
                probability_change,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
        logit_summary = summary_dict(
            summarize_distribution(
                logit_change,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
        following_rate: float | None = None
        following_ci: list[float] | None = None
        if condition == "real_fake_trace_swap":
            donor_labels = after["donor_label"].to_numpy(dtype=np.int64)
            following = (after_predictions.astype(np.int64) == donor_labels).astype(np.float64)
            following_summary = summary_dict(
                summarize_distribution(
                    following,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                )
            )
            following_rate = float(following_summary["mean"])
            following_ci = [
                float(following_summary["confidence_interval_low"]),
                float(following_summary["confidence_interval_high"]),
            ]
        coverage = (
            float(after["donor_matched"].astype(bool).mean()) if "donor_matched" in after else 1.0
        )
        rows.append(
            {
                "condition": condition,
                "sample_count": len(common),
                "donor_match_coverage": coverage,
                "prediction_flip_rate": float(flip_summary["mean"]),
                "prediction_flip_ci_low": float(flip_summary["confidence_interval_low"]),
                "prediction_flip_ci_high": float(flip_summary["confidence_interval_high"]),
                "mean_absolute_probability_change": float(probability_summary["mean"]),
                "mean_absolute_logit_change": float(logit_summary["mean"]),
                "trace_following_rate": following_rate,
                "trace_following_ci_low": following_ci[0] if following_ci else None,
                "trace_following_ci_high": following_ci[1] if following_ci else None,
            }
        )
        details[condition] = {
            "prediction_flip_rate": flip_summary,
            "absolute_probability_change": probability_summary,
            "absolute_logit_change": logit_summary,
            "trace_following_rate": following_rate,
            "trace_following_confidence_interval": following_ci,
            "donor_match_coverage": coverage,
        }
    summary = pd.DataFrame(rows)
    semantic_rate = float(
        summary.loc[summary["condition"] == "matched_semantic_swap", "prediction_flip_rate"].iloc[0]
    )
    trace_row = summary[summary["condition"] == "real_fake_trace_swap"].iloc[0]
    trace_rate = float(trace_row["prediction_flip_rate"])
    following_rate = float(trace_row["trace_following_rate"])
    details.update(
        {
            "threshold": threshold,
            "supports_trace_as_evidence": bool(trace_rate > semantic_rate and following_rate > 0.5),
            "interpretation_limit": (
                "Inference-time interventions diagnose sensitivity; they are not strict "
                "causal proof."
            ),
        }
    )
    return summary, details


def write_file_manifest(root: str | Path) -> Path:
    """Write hashes for package payload files, excluding the recursive manifest and ZIP."""
    package_root = Path(root)
    metadata = package_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    destination = metadata / "file_manifest.json"
    archive_name = "semtrace_proposal_mechanism_package.zip"
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == destination or path.name == archive_name:
            continue
        relative = path.relative_to(package_root).as_posix()
        files.append(
            {
                "path": relative,
                "type": _artifact_type(relative),
                "description": _artifact_description(relative),
                "sha256": _sha256(path),
            }
        )
    destination.write_text(
        json.dumps({"files": files}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def create_package_archive(root: str | Path) -> Path:
    package_root = Path(root)
    destination = package_root / "semtrace_proposal_mechanism_package.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file() and path != destination:
                archive.write(path, path.relative_to(package_root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad_file = archive.testzip()
    if bad_file is not None:
        raise RuntimeError(f"package ZIP validation failed at {bad_file}")
    return destination


def _condition_frame(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    selected = frame[frame["condition"] == condition].copy()
    if selected.empty:
        raise ValueError(f"condition is absent: {condition}")
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"condition contains duplicate sample IDs: {condition}")
    return selected.sort_values("sample_id").reset_index(drop=True)


def _paired_delta_ci(
    labels: np.ndarray,
    baseline: np.ndarray,
    changed: np.ndarray,
    *,
    metric: str,
    threshold: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, labels.size, labels.size)
        sampled_labels = labels[indices]
        if metric == "ap":
            if np.unique(sampled_labels).size < 2:
                continue
            before = _average_precision(sampled_labels, baseline[indices])
            after = _average_precision(sampled_labels, changed[indices])
        elif metric == "accuracy":
            before = float(np.mean((baseline[indices] >= threshold) == sampled_labels))
            after = float(np.mean((changed[indices] >= threshold) == sampled_labels))
        else:
            raise ValueError(f"unsupported paired metric: {metric}")
        deltas.append(before - after)
    if not deltas:
        raise ValueError("paired bootstrap produced no valid resamples")
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        raise ValueError("average precision requires both real and fake samples")
    return float(average_precision_score(labels, scores))


def _scale_for_layer(layer: int, selected_layers: Sequence[int]) -> str:
    try:
        return SCALE_NAMES[list(selected_layers).index(layer)]
    except ValueError as error:
        raise ValueError(f"unexpected selected layer: {layer}") from error


def _validate_layers(frame: pd.DataFrame, selected_layers: Sequence[int]) -> None:
    expected = set(selected_layers)
    actual = set(frame["selected_layer"].astype(int))
    if actual != expected:
        raise ValueError(
            f"selected layers mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    if len(selected_layers) != 3:
        raise ValueError("proposal mechanism analysis requires exactly three selected layers")


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_type(relative: str) -> str:
    top = relative.split("/", maxsplit=1)[0]
    return {
        "figures": "figure",
        "data": "data",
        "reports": "report",
        "descriptions": "report",
        "metadata": "metadata",
    }.get(top, "report")


def _artifact_description(relative: str) -> str:
    if relative.endswith(".png") or relative.endswith(".pdf"):
        return "Reproducible proposal mechanism figure"
    if relative.endswith(".csv") or relative.endswith(".json"):
        return "Machine-readable mechanism result or provenance"
    return "Proposal mechanism documentation"
