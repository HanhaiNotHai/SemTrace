from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True, slots=True)
class LayerProbeMetrics:
    authenticity_ap: float
    authenticity_balanced_accuracy: float
    semantic_accuracy: float
    nuisance_accuracy: float


@dataclass(frozen=True, slots=True)
class ProbeSelectionResult:
    selected_layers: tuple[int, int, int]
    layer_scores: dict[int, float]
    probe_metrics: dict[int, LayerProbeMetrics]
    score_formula: str


@dataclass(frozen=True, slots=True)
class ProbeSplit:
    features: Mapping[int, np.ndarray]
    authenticity: np.ndarray
    semantic: np.ndarray
    nuisance: np.ndarray


def choose_nuisance_label(
    labels: Mapping[str, Sequence[str | None]],
) -> tuple[str, Sequence[str | None], bool]:
    """Choose a non-degenerate nuisance target without inventing generators."""
    priority = ("source", "degradation", "file_format", "generator")
    for name in priority:
        values = labels.get(name)
        if values is None:
            continue
        categories = {value for value in values if value is not None}
        if len(categories) >= 2:
            return name, values, name == "generator"
    fallback = labels.get("source", ())
    return "unavailable", fallback, False


def fit_layer_probes(
    train: ProbeSplit,
    validation: ProbeSplit,
    *,
    seed: int = 3407,
    max_iter: int = 1000,
    show_progress: bool = False,
) -> dict[int, LayerProbeMetrics]:
    """Fit independent frozen-feature linear probes for every candidate layer."""
    if set(train.features) != set(validation.features):
        raise ValueError("train and validation layers must match")
    metrics: dict[int, LayerProbeMetrics] = {}
    layers = sorted(train.features)
    progress = tqdm(
        layers,
        total=len(layers),
        desc="Fitting layer probes",
        unit="layer",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for layer in progress:
        x_train = _as_feature_matrix(train.features[layer])
        x_validation = _as_feature_matrix(validation.features[layer])
        if x_train.shape[1] != x_validation.shape[1]:
            raise ValueError(f"feature dimension mismatch at layer {layer}")

        auth_predictions, auth_scores = _fit_predict(
            x_train,
            train.authenticity,
            x_validation,
            seed=seed,
            max_iter=max_iter,
            require_binary_score=True,
        )
        auth_targets = np.asarray(validation.authenticity)
        authenticity_ap = float(average_precision_score(auth_targets, auth_scores))
        authenticity_bacc = float(
            balanced_accuracy_score(auth_targets, auth_predictions)
        )
        semantic_accuracy = _optional_probe_accuracy(
            x_train,
            train.semantic,
            x_validation,
            validation.semantic,
            seed=seed,
            max_iter=max_iter,
        )
        nuisance_accuracy = _optional_probe_accuracy(
            x_train,
            train.nuisance,
            x_validation,
            validation.nuisance,
            seed=seed,
            max_iter=max_iter,
        )
        metrics[layer] = LayerProbeMetrics(
            authenticity_ap=authenticity_ap,
            authenticity_balanced_accuracy=authenticity_bacc,
            semantic_accuracy=semantic_accuracy,
            nuisance_accuracy=nuisance_accuracy,
        )
        progress.set_postfix(
            layer=layer,
            auth_ap=f"{authenticity_ap:.3f}",
            semantic_acc=f"{semantic_accuracy:.3f}",
            nuisance_acc=f"{nuisance_accuracy:.3f}",
        )
    return metrics


def select_probe_layers(
    probe_metrics: Mapping[int, LayerProbeMetrics],
    *,
    num_hidden_layers: int,
    alpha: float,
    beta: float,
) -> ProbeSelectionResult:
    """Select one maximum-score layer from each fixed depth third."""
    if num_hidden_layers < 4:
        raise ValueError("at least four hidden layers are required for three-scale selection")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    candidate_layers = list(range(num_hidden_layers - 1))
    missing = set(candidate_layers) - set(probe_metrics)
    if missing:
        raise ValueError(f"missing probe metrics for layers: {sorted(missing)}")

    authenticity = _zscore(
        np.asarray(
            [probe_metrics[layer].authenticity_ap for layer in candidate_layers],
            dtype=np.float64,
        )
    )
    semantic = _zscore(
        np.asarray(
            [probe_metrics[layer].semantic_accuracy for layer in candidate_layers],
            dtype=np.float64,
        )
    )
    nuisance = _zscore(
        np.asarray(
            [probe_metrics[layer].nuisance_accuracy for layer in candidate_layers],
            dtype=np.float64,
        )
    )
    scores = authenticity - alpha * semantic - beta * nuisance
    layer_scores = {
        layer: float(score) for layer, score in zip(candidate_layers, scores, strict=True)
    }

    full_depth_bands = np.array_split(np.arange(num_hidden_layers), 3)
    depth_bands = [
        [int(layer) for layer in band if int(layer) != num_hidden_layers - 1]
        for band in full_depth_bands
    ]
    if any(not band for band in depth_bands):
        raise ValueError("a depth band has no candidate layers")
    selected = tuple(
        max(band, key=lambda layer: (layer_scores[layer], -layer))
        for band in depth_bands
    )
    return ProbeSelectionResult(
        selected_layers=(selected[0], selected[1], selected[2]),
        layer_scores=layer_scores,
        probe_metrics={layer: probe_metrics[layer] for layer in candidate_layers},
        score_formula="z(AP_auth) - alpha*z(Acc_sem) - beta*z(Acc_nuis)",
    )


def save_probe_artifacts(
    result: ProbeSelectionResult,
    output_dir: str | Path,
    *,
    model_id: str,
    model_revision: str,
    semantic_label_coverage: float,
    nuisance_label: str,
    generator_probe_enabled: bool,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, metrics in result.probe_metrics.items():
        rows.append(
            {
                "layer": layer,
                **asdict(metrics),
                "score": result.layer_scores[layer],
                "selected": layer in result.selected_layers,
            }
        )
    with (destination / "probe_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "selected_layers": list(result.selected_layers),
        "score_formula": result.score_formula,
        "probe_metrics": {
            str(layer): {**asdict(metrics), "score": result.layer_scores[layer]}
            for layer, metrics in result.probe_metrics.items()
        },
        "semantic_label_coverage": semantic_label_coverage,
        "nuisance_label": nuisance_label,
        "generator_probe_enabled": generator_probe_enabled,
    }
    (destination / "selected_layers.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    layers = list(result.layer_scores)
    scores = [result.layer_scores[layer] for layer in layers]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(layers, scores, marker="o")
    axis.scatter(
        result.selected_layers,
        [result.layer_scores[layer] for layer in result.selected_layers],
        color="tab:red",
        label="selected",
        zorder=3,
    )
    axis.set(xlabel="Transformer block", ylabel="Probe score", title="Layer probe scores")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "layer_score_plot.png", dpi=160)
    plt.close(figure)


def _zscore(values: np.ndarray) -> np.ndarray:
    standard_deviation = values.std()
    if standard_deviation <= np.finfo(values.dtype).eps:
        return np.zeros_like(values)
    return (values - values.mean()) / standard_deviation


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("probe features must have shape [samples, channels]")
    return array


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    require_binary_score: bool,
) -> tuple[np.ndarray, np.ndarray]:
    encoder = LabelEncoder().fit(y_train)
    encoded_train = encoder.transform(y_train)
    if len(encoder.classes_) < 2:
        raise ValueError("a linear probe requires at least two training classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=seed,
            max_iter=max_iter,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, encoded_train)
    predictions = model.predict(x_validation)
    decision = model.decision_function(x_validation)
    if require_binary_score and len(encoder.classes_) != 2:
        raise ValueError("authenticity probe must be binary")
    scores = np.asarray(decision)
    if scores.ndim != 1:
        scores = scores[:, 1]
    return predictions, scores


def _optional_probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> float:
    train_targets = np.asarray(y_train)
    validation_targets = np.asarray(y_validation)
    train_mask = train_targets != -1
    validation_mask = validation_targets != -1
    if train_mask.sum() == 0 or validation_mask.sum() == 0:
        return 0.0
    if len(np.unique(train_targets[train_mask])) < 2:
        return 0.0
    predictions, _ = _fit_predict(
        x_train[train_mask],
        train_targets[train_mask],
        x_validation[validation_mask],
        seed=seed,
        max_iter=max_iter,
        require_binary_score=False,
    )
    return float(np.mean(predictions == validation_targets[validation_mask]))
