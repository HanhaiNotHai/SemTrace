from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ProbeTask = Literal["binary", "multiclass"]


@dataclass(frozen=True, slots=True)
class LinearProbeResult:
    metrics: dict[str, float]
    preprocessing_mean: np.ndarray
    skipped_reason: str | None = None


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: ProbeTask,
    seed: int,
    pca_dimensions: int | None = None,
) -> LinearProbeResult:
    train_features = np.asarray(x_train, dtype=np.float32)
    test_features = np.asarray(x_test, dtype=np.float32)
    train_targets = np.asarray(y_train)
    test_targets = np.asarray(y_test)
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("probe features must have shape [samples, dimensions]")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("probe train/test feature dimensions must match")
    if len(np.unique(train_targets)) < 2:
        return LinearProbeResult(
            metrics={},
            preprocessing_mean=train_features.mean(axis=0),
            skipped_reason="fewer than two training classes",
        )

    steps: list[tuple[str, object]] = [("standardize", StandardScaler())]
    if pca_dimensions is not None and train_features.shape[1] > pca_dimensions:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=min(
                        pca_dimensions,
                        train_features.shape[0] - 1,
                        train_features.shape[1],
                    ),
                    random_state=seed,
                ),
            )
        )
    steps.append(
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    )
    pipeline = Pipeline(steps)
    pipeline.fit(train_features, train_targets)
    predictions = pipeline.predict(test_features)
    metrics = {
        "accuracy": float(accuracy_score(test_targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_targets, predictions)),
        "macro_f1": float(f1_score(test_targets, predictions, average="macro")),
    }
    if task == "binary":
        scores = pipeline.decision_function(test_features)
        if len(np.unique(test_targets)) >= 2:
            metrics["average_precision"] = float(average_precision_score(test_targets, scores))
            metrics["auroc"] = float(roc_auc_score(test_targets, scores))
        else:
            metrics["average_precision"] = float("nan")
            metrics["auroc"] = float("nan")
    scaler = pipeline.named_steps["standardize"]
    if not isinstance(scaler, StandardScaler) or scaler.mean_ is None:
        raise RuntimeError("linear probe standardization was not fitted")
    return LinearProbeResult(metrics, np.asarray(scaler.mean_), None)
