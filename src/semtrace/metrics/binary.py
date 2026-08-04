from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    accuracy: float
    average_precision: float
    auroc: float
    false_positive_rate: float
    false_positive_rate_at_95_tpr: float
    real_accuracy: float
    fake_accuracy: float
    count: int


@dataclass(frozen=True, slots=True)
class GroupedBinaryMetrics:
    overall: BinaryMetrics
    per_domain: dict[str, BinaryMetrics]
    mean_accuracy: float
    mean_average_precision: float


def optimal_accuracy_threshold(
    labels: list[int] | np.ndarray,
    fake_scores: list[float] | np.ndarray,
    *,
    reference_threshold: float = 0.5,
) -> float:
    """Return the global accuracy-maximizing threshold for ``score >= threshold``."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fake_scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("labels and fake_scores must be equally sized one-dimensional arrays")
    if targets.size == 0:
        raise ValueError("threshold selection requires at least one sample")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("labels must follow real=0, fake=1")
    if not np.isfinite(scores).all() or not np.isfinite(reference_threshold):
        raise ValueError("fake_scores and reference_threshold must be finite")

    unique_scores, inverse = np.unique(scores, return_inverse=True)
    real_counts = np.bincount(inverse[targets == 0], minlength=unique_scores.size)
    fake_counts = np.bincount(inverse[targets == 1], minlength=unique_scores.size)
    real_below = np.concatenate(([0], np.cumsum(real_counts)[:-1]))
    fake_at_or_above = fake_counts.sum() - np.concatenate(
        ([0], np.cumsum(fake_counts)[:-1])
    )
    candidates = np.concatenate(
        (
            unique_scores,
            [np.nextafter(unique_scores[-1], np.inf), reference_threshold],
        )
    )
    correct = np.concatenate(
        (
            real_below + fake_at_or_above,
            [real_counts.sum(), np.sum((scores >= reference_threshold) == targets)],
        )
    )
    best = candidates[correct == correct.max()]
    distances = np.abs(best - reference_threshold)
    closest = best[np.isclose(distances, distances.min(), rtol=0.0, atol=1.0e-12)]
    return float(closest.max())


def binary_metrics(
    labels: list[int] | np.ndarray,
    fake_scores: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute metrics from continuous fake probabilities (real=0, fake=1)."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fake_scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("labels and fake_scores must be equally sized one-dimensional arrays")
    if targets.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("labels must follow real=0, fake=1")
    if not np.isfinite(scores).all():
        raise ValueError("fake_scores must be finite")

    predictions = (scores >= threshold).astype(np.int64)
    real_mask = targets == 0
    fake_mask = targets == 1
    accuracy = float(np.mean(predictions == targets))
    real_accuracy = float(np.mean(predictions[real_mask] == 0)) if real_mask.any() else float("nan")
    fake_accuracy = float(np.mean(predictions[fake_mask] == 1)) if fake_mask.any() else float("nan")
    false_positive_rate = (
        float(np.mean(predictions[real_mask] == 1)) if real_mask.any() else float("nan")
    )

    average_precision = (
        float(average_precision_score(targets, scores)) if fake_mask.any() else float("nan")
    )
    if real_mask.any() and fake_mask.any():
        auroc = float(roc_auc_score(targets, scores))
        fpr, tpr, _ = roc_curve(targets, scores)
        eligible = fpr[tpr >= 0.95]
        fpr_at_95 = float(eligible.min()) if eligible.size else float("nan")
    else:
        auroc = float("nan")
        fpr_at_95 = float("nan")

    return BinaryMetrics(
        accuracy=accuracy,
        average_precision=average_precision,
        auroc=auroc,
        false_positive_rate=false_positive_rate,
        false_positive_rate_at_95_tpr=fpr_at_95,
        real_accuracy=real_accuracy,
        fake_accuracy=fake_accuracy,
        count=int(targets.size),
    )


def grouped_binary_metrics(
    labels: list[int] | np.ndarray,
    fake_scores: list[float] | np.ndarray,
    domains: list[str] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> GroupedBinaryMetrics:
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fake_scores, dtype=np.float64)
    groups = np.asarray(domains, dtype=str)
    if targets.shape != scores.shape or targets.shape != groups.shape:
        raise ValueError("labels, fake_scores, and domains must have identical shapes")

    per_domain: dict[str, BinaryMetrics] = {}
    for domain in sorted(np.unique(groups)):
        mask = groups == domain
        per_domain[str(domain)] = binary_metrics(targets[mask], scores[mask], threshold=threshold)
    mean_accuracy = float(np.mean([metrics.accuracy for metrics in per_domain.values()]))
    mean_ap = float(np.nanmean([metrics.average_precision for metrics in per_domain.values()]))
    return GroupedBinaryMetrics(
        overall=binary_metrics(targets, scores, threshold=threshold),
        per_domain=per_domain,
        mean_accuracy=mean_accuracy,
        mean_average_precision=mean_ap,
    )
