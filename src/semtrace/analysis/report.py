from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from semtrace.analysis.cross_attention_analysis import attention_statistics
from semtrace.analysis.feature_cache import (
    AnalysisSampleMetadata,
    FeatureCacheReader,
    FeatureValue,
)
from semtrace.analysis.linear_probe import ProbeTask, fit_linear_probe
from semtrace.analysis.normal_predictor_analysis import prediction_error_metrics
from semtrace.analysis.representation_label_mi import (
    linear_cka,
    linear_hsic,
    mutual_information_with_permutation,
)
from semtrace.analysis.residual_analysis import pooled_stage_representations, residual_strength
from semtrace.analysis.scale_usage import compute_scale_usage
from semtrace.analysis.statistics import (
    real_fake_comparison,
    summarize_distribution,
    summary_dict,
)
from semtrace.analysis.trace_pattern_coverage import (
    fit_trace_prototypes,
    prototype_combination_novelty,
    prototype_coverage,
)


def generate_mechanism_report(
    cache_root: str | Path,
    output_root: str | Path,
    *,
    bootstrap_iterations: int = 1000,
    random_seeds: tuple[int, ...] = (0, 1, 2),
    prototype_counts: tuple[int, ...] = (64, 128, 256),
    top_r: int = 10,
) -> dict[str, Any]:
    """Stream cached shards and write reproducible offline analysis artifacts."""
    output = Path(output_root)
    tables = output / "tables"
    plots = output / "plots"
    heatmaps = output / "heatmaps"
    predictions = output / "predictions"
    for directory in (tables, plots, heatmaps, predictions):
        directory.mkdir(parents=True, exist_ok=True)

    normal_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    metadata: list[AnalysisSampleMetadata] = []
    representation_parts: dict[str, list[np.ndarray]] = {}
    prototype_parts: list[np.ndarray] = []

    for shard in FeatureCacheReader(cache_root).iter_shards():
        features = shard.features
        metadata.extend(shard.metadata)
        _append_normal_rows(normal_rows, shard.metadata, features)
        _append_residual_rows(residual_rows, shard.metadata, features)
        _append_usage_rows(usage_rows, shard.metadata, features)
        _append_attention_rows(attention_rows, shard.metadata, features)
        pooled = pooled_stage_representations(
            semantic_anchor=_tensor(features, "semantic_anchor"),
            raw_patch_features=_layer_tensors(features, "raw_patch_features"),
            candidate_trace_residuals=_layer_tensors(
                features, "candidate_trace_residuals"
            ),
            adapted_trace_tokens=_layer_tensors(features, "adapted_trace_tokens"),
            fused_trace_tokens=_tensor(features, "fused_trace_tokens"),
            trace_evidence=_tensor(features, "trace_evidence"),
        )
        for name, values in pooled.items():
            representation_parts.setdefault(name, []).append(_numpy(values))
        prototype_parts.append(
            _numpy(_tensor(features, "fused_trace_tokens")).reshape(
                -1, _tensor(features, "fused_trace_tokens").shape[-1]
            )
        )

    if not metadata:
        raise ValueError("feature cache contains no samples")
    normal_frame = pd.DataFrame(normal_rows)
    residual_frame = pd.DataFrame(residual_rows)
    usage_frame = pd.DataFrame(usage_rows)
    attention_frame = pd.DataFrame(attention_rows)
    normal_frame.to_csv(tables / "normal_predictor.csv", index=False)
    residual_frame.to_csv(tables / "residuals.csv", index=False)
    _write_group_summaries(normal_frame, tables / "normal_predictor_grouped.csv")
    _write_group_summaries(residual_frame, tables / "residuals_grouped.csv")
    usage_frame.to_csv(tables / "scale_usage_samples.csv", index=False)
    attention_frame.to_csv(tables / "attention_statistics.csv", index=False)
    _write_usage_tables(usage_frame, tables)
    _write_basic_plots(normal_frame, residual_frame, plots)

    representations = {
        name: np.concatenate(parts, axis=0) for name, parts in representation_parts.items()
    }
    mi_frame = _representation_mi(representations, metadata, random_seeds)
    mi_frame.to_csv(tables / "representation_mi.csv", index=False)
    probe_frame = _linear_probes(representations, metadata, random_seeds)
    probe_frame.to_csv(tables / "linear_probes.csv", index=False)
    metric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "macro_f1", "average_precision", "auroc")
        if column in probe_frame
    ]
    if metric_columns:
        probe_summary = (
            probe_frame.groupby(["representation", "label"])[metric_columns]
            .agg(["mean", "std"])
            .reset_index()
        )
        probe_summary.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in probe_summary.columns
        ]
        probe_summary.to_csv(tables / "linear_probes_summary.csv", index=False)
    coverage_frame = _trace_coverage(
        np.concatenate(prototype_parts, axis=0),
        metadata,
        prototype_counts,
        top_r,
        random_seeds[0],
    )
    coverage_frame.to_csv(tables / "trace_coverage.csv", index=False)
    summaries = _distribution_tables(
        normal_frame,
        residual_frame,
        tables,
        bootstrap_iterations,
        random_seeds[0],
    )
    for filename in ("masking.csv", "semantic_counterfactual.csv"):
        pd.DataFrame(
            [
                {
                    "skipped_reason": (
                        "requires live inference; use the corresponding CLI without analysis.cache"
                    )
                }
            ]
        ).to_csv(tables / filename, index=False)
    summary: dict[str, Any] = {
        "sample_count": len(metadata),
        "selected_layers": sorted(int(value) for value in usage_frame["layer"].unique()),
        "cache_fingerprint": FeatureCacheReader(cache_root).index["fingerprint"],
        "normal_prediction_summary_rows": summaries[0],
        "residual_summary_rows": summaries[1],
        "finite_sample_mi_notice": (
            "MI values are finite-sample estimates for relative comparison, not exact mutual "
            "information."
        ),
        "diagnostic_notice": (
            "Masking and counterfactual results are inference-time interventions, not causal "
            "proof."
        ),
    }
    (output / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "mechanism_report.md").write_text(
        _report_markdown(summary), encoding="utf-8"
    )
    return summary


def _append_normal_rows(
    rows: list[dict[str, Any]],
    metadata: list[AnalysisSampleMetadata],
    features: Mapping[str, FeatureValue],
) -> None:
    metrics = prediction_error_metrics(
        _layer_tensors(features, "raw_patch_features"),
        _layer_tensors(features, "predicted_normal_features"),
    )
    for layer, layer_metrics in metrics.items():
        for metric, values in layer_metrics.items():
            for sample, value in zip(metadata, _numpy(values), strict=True):
                rows.append(_row(sample, layer=layer, metric=metric, value=float(value)))


def _append_residual_rows(
    rows: list[dict[str, Any]],
    metadata: list[AnalysisSampleMetadata],
    features: Mapping[str, FeatureValue],
) -> None:
    for layer, error in _layer_tensors(features, "prediction_errors").items():
        strength = residual_strength(error)
        values = {
            "patch_l1_mean": strength.patch_l1.mean(dim=1),
            "patch_l2_mean": strength.patch_l2.mean(dim=1),
            "channel_energy": strength.channel_energy.mean(dim=1),
            "image_mean": strength.image_mean,
            "image_max": strength.image_max,
            "top_k_mean": strength.top_k_mean,
            "spatial_entropy": strength.spatial_entropy,
            "sparsity": strength.sparsity,
        }
        for metric, tensor in values.items():
            for sample, value in zip(metadata, _numpy(tensor), strict=True):
                rows.append(_row(sample, layer=layer, metric=metric, value=float(value)))


def _append_usage_rows(
    rows: list[dict[str, Any]],
    metadata: list[AnalysisSampleMetadata],
    features: Mapping[str, FeatureValue],
) -> None:
    usage = compute_scale_usage(_layer_tensors(features, "adapted_trace_tokens"))
    contributions = (
        _layer_tensors(features, "fusion_contributions")
        if "fusion_contributions" in features
        else {}
    )
    activation = _numpy(usage.activation_strength)
    normalized = _numpy(usage.normalized_usage)
    effective = _numpy(usage.effective_scale_count)
    entropy = _numpy(usage.scale_entropy)
    for index, sample in enumerate(metadata):
        for position, layer in enumerate(usage.layers):
            rows.append(
                _row(
                    sample,
                    layer=layer,
                    activation_strength=float(activation[index, position]),
                    normalized_usage=float(normalized[index, position]),
                    effective_scale_count=float(effective[index]),
                    scale_entropy=float(entropy[index]),
                    approximate_fusion_contribution=(
                        float(_numpy(contributions[layer])[index])
                        if layer in contributions
                        else float("nan")
                    ),
                )
            )


def _append_attention_rows(
    rows: list[dict[str, Any]],
    metadata: list[AnalysisSampleMetadata],
    features: Mapping[str, FeatureValue],
) -> None:
    value = features.get("attention_weights")
    if not isinstance(value, torch.Tensor):
        return
    stats = attention_statistics(value.float())
    arrays = {name: _numpy(tensor) for name, tensor in stats.items()}
    residual_norm = torch.stack(
        [
            tensor.float().norm(dim=-1)
            for tensor in _layer_tensors(features, "candidate_trace_residuals").values()
        ]
    ).mean(dim=0)
    adapted_norm = torch.stack(
        [
            tensor.float().norm(dim=-1)
            for tensor in _layer_tensors(features, "adapted_trace_tokens").values()
        ]
    ).mean(dim=0)
    fused_norm = _tensor(features, "fused_trace_tokens").norm(dim=-1)
    weight_array = _numpy(value[:, :, 0])
    score_arrays = {
        "residual": _numpy(residual_norm),
        "adapted": _numpy(adapted_norm),
        "fused": _numpy(fused_norm),
    }
    for sample_index, sample in enumerate(metadata):
        for head in range(value.shape[1]):
            correlations = {
                f"{kind}_{metric}": result
                for kind, scores in score_arrays.items()
                for metric, result in _correlations(
                    weight_array[sample_index, head], scores[sample_index]
                ).items()
            }
            rows.append(
                _row(
                    sample,
                    head=head,
                    **{
                        name: float(array[sample_index, head])
                        for name, array in arrays.items()
                    },
                    **correlations,
                )
            )


def _write_usage_tables(frame: pd.DataFrame, tables: Path) -> None:
    frame.groupby("layer", dropna=False)[
        ["activation_strength", "normalized_usage", "effective_scale_count", "scale_entropy"]
    ].mean().reset_index().to_csv(tables / "scale_usage_overall.csv", index=False)
    for column, filename in (
        ("generator", "scale_usage_by_generator.csv"),
        ("content_env", "scale_usage_by_content.csv"),
        ("real_source", "scale_usage_by_real_source.csv"),
        ("degradation", "scale_usage_by_degradation.csv"),
    ):
        valid = frame[frame[column].notna()]
        result = valid.groupby([column, "layer"])["normalized_usage"].mean().reset_index()
        result.to_csv(tables / filename, index=False)


def _distribution_tables(
    normal: pd.DataFrame,
    residual: pd.DataFrame,
    tables: Path,
    iterations: int,
    seed: int,
) -> tuple[int, int]:
    counts: list[int] = []
    for source, filename in (
        (normal, "normal_predictor_summary.csv"),
        (residual, "residual_summary.csv"),
    ):
        rows: list[dict[str, Any]] = []
        for keys, group in source.groupby(["label", "layer", "metric"]):
            summary = summarize_distribution(
                group["value"].to_numpy(), bootstrap_iterations=iterations, seed=seed
            )
            rows.append(
                {
                    "label": int(keys[0]),
                    "layer": int(keys[1]),
                    "metric": str(keys[2]),
                    **summary_dict(summary),
                }
            )
        pd.DataFrame(rows).to_csv(tables / filename, index=False)
        comparisons: list[dict[str, Any]] = []
        for keys, group in source.groupby(["layer", "metric"]):
            real = group.loc[group["label"] == 0, "value"].to_numpy()
            fake = group.loc[group["label"] == 1, "value"].to_numpy()
            if real.size and fake.size:
                comparisons.append(
                    {
                        "layer": int(keys[0]),
                        "metric": str(keys[1]),
                        **real_fake_comparison(real, fake),
                    }
                )
        pd.DataFrame(comparisons).to_csv(
            tables / filename.replace("_summary", "_real_fake"), index=False
        )
        counts.append(len(rows))
    return counts[0], counts[1]


def _representation_mi(
    representations: dict[str, np.ndarray],
    metadata: list[AnalysisSampleMetadata],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    labels = _label_arrays(metadata)
    rows: list[dict[str, Any]] = []
    for representation, features in representations.items():
        for label_name, values in labels.items():
            mask = _task_mask(label_name, values, metadata)
            encoded, valid = _encode_labels(values[mask])
            if len(np.unique(encoded)) < 2:
                rows.append(
                    {"representation": representation, "label": label_name, "skipped": True}
                )
                continue
            estimate = mutual_information_with_permutation(
                features[mask], encoded, pca_dimensions=32, seeds=seeds
            )
            one_hot = np.eye(int(encoded.max()) + 1, dtype=np.float32)[encoded]
            rows.append(
                {
                    "representation": representation,
                    "label": label_name,
                    "skipped": False,
                    **asdict(estimate),
                    "linear_hsic": linear_hsic(features[mask], one_hot),
                    "linear_cka": linear_cka(features[mask], one_hot),
                    "sample_count": int(valid),
                }
            )
    return pd.DataFrame(rows)


def _linear_probes(
    representations: dict[str, np.ndarray],
    metadata: list[AnalysisSampleMetadata],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    splits = np.asarray([sample.split for sample in metadata], dtype=object)
    labels = _label_arrays(metadata)
    train_mask = splits == "train"
    test_mask = np.isin(splits, ["test", "validation", "val"])
    rows: list[dict[str, Any]] = []
    for representation, features in representations.items():
        for label_name, values in labels.items():
            available = _task_mask(label_name, values, metadata)
            task = "binary" if label_name == "authenticity" else "multiclass"
            if not (np.any(train_mask & available) and np.any(test_mask & available)):
                rows.append(
                    {
                        "representation": representation,
                        "label": label_name,
                        "skipped_reason": "requires disjoint train and test/validation samples",
                    }
                )
                continue
            encoded, _ = _encode_labels(values[available])
            full = np.full(values.shape, -1, dtype=np.int64)
            full[available] = encoded
            for seed in seeds:
                result = fit_linear_probe(
                    features[train_mask & available],
                    full[train_mask & available],
                    features[test_mask & available],
                    full[test_mask & available],
                    task=cast(ProbeTask, task),
                    seed=seed,
                    pca_dimensions=32,
                )
                rows.append(
                    {
                        "representation": representation,
                        "label": label_name,
                        "seed": seed,
                        "skipped_reason": result.skipped_reason,
                        **result.metrics,
                    }
                )
    return pd.DataFrame(rows)


def _trace_coverage(
    tokens: np.ndarray,
    metadata: list[AnalysisSampleMetadata],
    counts: tuple[int, ...],
    top_r: int,
    seed: int,
) -> pd.DataFrame:
    patch_count = tokens.shape[0] // len(metadata)
    image_tokens = tokens.reshape(len(metadata), patch_count, -1)
    train_images = np.asarray([sample.split == "train" for sample in metadata])
    rows: list[dict[str, Any]] = []
    if not train_images.any() or train_images.all():
        return pd.DataFrame([{"skipped_reason": "coverage requires train and target splits"}])
    training = image_tokens[train_images].reshape(-1, tokens.shape[1])
    target_generators = np.asarray(
        [sample.generator for sample in metadata], dtype=object
    )[~train_images]
    for count in counts:
        if training.shape[0] < count:
            rows.append(
                {"prototype_count": count, "skipped_reason": "insufficient training tokens"}
            )
            continue
        model = fit_trace_prototypes(
            training,
            prototype_count=count,
            pca_dimensions=min(32, training.shape[1]),
            seed=seed,
        )
        training_assignments = np.stack(
            [model.assign(values)[0] for values in image_tokens[train_images]]
        )
        for generator in sorted(np.unique(target_generators)):
            target_images = image_tokens[~train_images][target_generators == generator]
            target = target_images.reshape(-1, tokens.shape[1])
            target_assignments = np.stack(
                [model.assign(values)[0] for values in target_images]
            )
            rows.append(
                {
                    "prototype_count": count,
                    "generator": str(generator),
                    "skipped_reason": None,
                    **asdict(prototype_coverage(model, training, target, top_r=top_r)),
                    "combination_novelty": prototype_combination_novelty(
                        training_assignments, target_assignments, top_r=top_r
                    ),
                }
            )
    return pd.DataFrame(rows)


def _label_arrays(metadata: list[AnalysisSampleMetadata]) -> dict[str, np.ndarray]:
    return {
        "authenticity": np.asarray([sample.label for sample in metadata], dtype=object),
        "semantic_class": np.asarray(
            [sample.semantic_class for sample in metadata], dtype=object
        ),
        "content_env": np.asarray([sample.content_env for sample in metadata], dtype=object),
        "generator": np.asarray([sample.generator for sample in metadata], dtype=object),
        "real_source": np.asarray([sample.real_source for sample in metadata], dtype=object),
        "degradation": np.asarray([sample.degradation for sample in metadata], dtype=object),
        "source_dataset": np.asarray(
            [sample.source_dataset for sample in metadata], dtype=object
        ),
    }


def _task_mask(
    label_name: str,
    values: np.ndarray,
    metadata: list[AnalysisSampleMetadata],
) -> np.ndarray:
    mask = values != None  # noqa: E711
    labels = np.asarray([sample.label for sample in metadata])
    if label_name == "generator":
        mask &= labels == 1
    elif label_name == "real_source":
        mask &= labels == 0
    return mask


def _write_basic_plots(normal: pd.DataFrame, residual: pd.DataFrame, plots: Path) -> None:
    from semtrace.analysis.plotting import save_boxplot, save_matrix

    normal_l2 = normal[normal["metric"] == "l2_error"]
    groups = {
        f"{'real' if label == 0 else 'fake'}-L{layer}": group["value"].to_numpy()
        for (label, layer), group in normal_l2.groupby(["label", "layer"])
    }
    if groups:
        save_boxplot(
            groups,
            plots / "normal_prediction_real_fake.png",
            title="Normal prediction error by scale",
            ylabel="L2 error",
        )
    image_mean = residual[residual["metric"] == "image_mean"]
    pivot = image_mean.pivot_table(index="sample_id", columns="layer", values="value")
    if pivot.shape[1] >= 1:
        correlation = pivot.corr().fillna(0.0).to_numpy()
        labels = [f"L{int(layer)}" for layer in pivot.columns]
        save_matrix(
            correlation,
            labels,
            plots / "residual_scale_correlation.png",
            title="Candidate residual strength correlation",
        )


def _write_group_summaries(frame: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for column in (
        "label",
        "generator",
        "semantic_class",
        "content_env",
        "real_source",
        "degradation",
        "source_dataset",
    ):
        valid = frame[frame[column].notna()]
        for keys, group in valid.groupby([column, "layer", "metric"]):
            rows.append(
                {
                    "group_type": column,
                    "group": keys[0],
                    "layer": keys[1],
                    "metric": keys[2],
                    "count": len(group),
                    "mean": group["value"].mean(),
                    "standard_deviation": group["value"].std(ddof=1),
                    "median": group["value"].median(),
                    "q1": group["value"].quantile(0.25),
                    "q3": group["value"].quantile(0.75),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _encode_labels(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, encoded = np.unique(values.astype(str), return_inverse=True)
    return encoded, int(values.size)


def _row(sample: AnalysisSampleMetadata, **values: Any) -> dict[str, Any]:
    return {**asdict(sample), **values}


def _tensor(features: Mapping[str, FeatureValue], name: str) -> torch.Tensor:
    value = features.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"cached feature '{name}' must be a tensor")
    return value.float()


def _layer_tensors(
    features: Mapping[str, FeatureValue], name: str
) -> dict[int, torch.Tensor]:
    value = features.get(name)
    if not isinstance(value, dict) or not all(
        isinstance(layer, int) and isinstance(tensor, torch.Tensor)
        for layer, tensor in value.items()
    ):
        raise TypeError(f"cached feature '{name}' must be a layer-to-tensor mapping")
    return {int(layer): tensor.float() for layer, tensor in value.items()}


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy()


def _report_markdown(summary: dict[str, Any]) -> str:
    return f"""# SemTrace Mechanism Analysis Report

## Scope

- Samples: {summary['sample_count']}
- Selected layers: {summary['selected_layers']}
- Statistics, plots, and probe results are stored under `tables/`, `plots/`, and
  `heatmaps/`.

## Interpretation boundaries

- Candidate trace residuals may contain generation traces, remaining semantics,
  post-processing interference, and normal-prediction error. They are not pure traces.
- Mutual information values are finite-sample estimates for relative comparison, not exact
  mutual information.
- Masking and counterfactual experiments are inference-time mechanism diagnostics, not causal
  proof.
- Heatmaps show internal candidate-residual intensity or attention. They are not automatically
  human-interpretable generation artifacts and do not establish causality.
- Post-hoc trace prototypes cluster learned continuous tokens; they are not internal discrete
  forensic primitives.

## Automatically supportable conclusions

Use the generated tables to report measured scale activation, relative representation-label
dependence, prediction changes under interventions, and prototype coverage with uncertainty.

## Conclusions not automatically supported

The analysis cannot prove semantic independence, causal sufficiency, pure generation traces,
or that attention is a causal explanation.
"""


def _correlations(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if np.std(first) == 0 or np.std(second) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(first, second).statistic),
        "spearman": float(spearmanr(first, second).statistic),
    }
