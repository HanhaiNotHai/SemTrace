from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


@dataclass(frozen=True, slots=True)
class MutualInformationEstimate:
    estimate_mean: float
    estimate_standard_deviation: float
    permutation_mean: float
    permutation_standard_deviation: float


def mutual_information_with_permutation(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_dimensions: int | None = 32,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> MutualInformationEstimate:
    """Estimate relative finite-sample MI; this is not exact mutual information."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("MI features and labels must have shapes [samples, dims] and [samples]")
    if len(np.unique(y)) < 2:
        raise ValueError("MI requires at least two label classes")
    if pca_dimensions is not None and x.shape[1] > pca_dimensions:
        components = min(pca_dimensions, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=components, random_state=seeds[0]).fit_transform(x)
    observed: list[float] = []
    permuted: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        observed.append(
            float(mutual_info_classif(x, y, random_state=seed, n_jobs=1).mean())
        )
        permuted.append(
            float(
                mutual_info_classif(
                    x,
                    rng.permutation(y),
                    random_state=seed,
                    n_jobs=1,
                ).mean()
            )
        )
    return MutualInformationEstimate(
        estimate_mean=float(np.mean(observed)),
        estimate_standard_deviation=float(np.std(observed)),
        permutation_mean=float(np.mean(permuted)),
        permutation_standard_deviation=float(np.std(permuted)),
    )


def linear_hsic(x: np.ndarray, y: np.ndarray) -> float:
    x_centered, y_centered = _paired_centered(x, y)
    cross = x_centered.T @ y_centered
    return float(np.square(cross).sum() / max((x_centered.shape[0] - 1) ** 2, 1))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_centered, y_centered = _paired_centered(x, y)
    cross = np.square(x_centered.T @ y_centered).sum()
    x_norm = np.square(x_centered.T @ x_centered).sum()
    y_norm = np.square(y_centered.T @ y_centered).sum()
    denominator = np.sqrt(x_norm * y_norm)
    return float(cross / max(float(denominator), np.finfo(np.float64).eps))


def _paired_centered(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("HSIC/CKA inputs must be paired [samples, dims] matrices")
    return first - first.mean(axis=0), second - second.mean(axis=0)
