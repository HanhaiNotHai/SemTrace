from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_boxplot(
    groups: dict[str, np.ndarray],
    path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(max(6, len(groups) * 0.8), 4))
    axis.boxplot(list(groups.values()), tick_labels=list(groups), showfliers=False)
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_matrix(matrix: np.ndarray, labels: list[str], path: str | Path, *, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set(title=title, xticks=range(len(labels)), yticks=range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticklabels(labels)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
