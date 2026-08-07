from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class TracePrototypeModel:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    centers: np.ndarray

    def assign(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(tokens, dtype=np.float32)
        reduced = (values - self.pca_mean) @ self.pca_components.T
        distances = np.linalg.norm(reduced[:, None, :] - self.centers[None, :, :], axis=-1)
        assignments = distances.argmin(axis=1)
        return assignments, distances[np.arange(values.shape[0]), assignments]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            centers=self.centers,
        )

    @classmethod
    def load(cls, path: str | Path) -> TracePrototypeModel:
        payload = np.load(path)
        return cls(payload["pca_mean"], payload["pca_components"], payload["centers"])


@dataclass(frozen=True, slots=True)
class PrototypeCoverage:
    coverage: float
    top_r_jaccard: float
    mean_nearest_distance: float
    ood_token_ratio: float


def fit_trace_prototypes(
    training_tokens: np.ndarray,
    *,
    prototype_count: int,
    pca_dimensions: int,
    seed: int,
) -> TracePrototypeModel:
    tokens = np.asarray(training_tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[0] < prototype_count:
        raise ValueError("prototype fitting requires [tokens, dims] with enough tokens")
    components = min(pca_dimensions, tokens.shape[0] - 1, tokens.shape[1])
    pca = PCA(n_components=components, random_state=seed)
    reduced = pca.fit_transform(tokens)
    clusters = MiniBatchKMeans(
        n_clusters=prototype_count,
        random_state=seed,
        batch_size=min(1024, tokens.shape[0]),
        n_init="auto",
    ).fit(reduced)
    return TracePrototypeModel(
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        centers=np.asarray(clusters.cluster_centers_, dtype=np.float32),
    )


def prototype_coverage(
    model: TracePrototypeModel,
    training_tokens: np.ndarray,
    target_tokens: np.ndarray,
    *,
    top_r: int,
) -> PrototypeCoverage:
    train_assignments, train_distances = model.assign(training_tokens)
    target_assignments, target_distances = model.assign(target_tokens)
    prototype_count = model.centers.shape[0]
    count = min(top_r, prototype_count)
    train_top = set(np.argsort(np.bincount(train_assignments, minlength=prototype_count))[-count:])
    target_top = set(
        np.argsort(np.bincount(target_assignments, minlength=prototype_count))[-count:]
    )
    intersection = train_top & target_top
    union = train_top | target_top
    threshold = float(np.quantile(train_distances, 0.95))
    return PrototypeCoverage(
        coverage=len(intersection) / max(len(target_top), 1),
        top_r_jaccard=len(intersection) / max(len(union), 1),
        mean_nearest_distance=float(target_distances.mean()),
        ood_token_ratio=float(np.mean(target_distances > threshold)),
    )


def prototype_combination_novelty(
    training_assignments: np.ndarray,
    target_assignments: np.ndarray,
    *,
    top_r: int,
) -> float:
    """Fraction of target image top-r prototype sets unseen in training images."""
    training = {_top_combination(row, top_r) for row in training_assignments}
    target = [_top_combination(row, top_r) for row in target_assignments]
    return float(np.mean([combination not in training for combination in target]))


def _top_combination(assignments: np.ndarray, top_r: int) -> tuple[int, ...]:
    values, counts = np.unique(assignments, return_counts=True)
    order = np.argsort(counts)[-min(top_r, len(values)) :]
    return tuple(sorted(int(value) for value in values[order]))
