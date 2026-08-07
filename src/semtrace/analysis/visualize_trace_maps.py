from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def patch_map(values: torch.Tensor, patch_grid_size: tuple[int, int]) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != patch_grid_size[0] * patch_grid_size[1]:
        raise ValueError("patch values must have shape [batch, grid_height * grid_width]")
    return values.reshape(values.shape[0], *patch_grid_size)


def resize_patch_map(grid: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if grid.ndim != 3:
        raise ValueError("patch heatmap grid must have shape [batch, height, width]")
    return F.interpolate(
        grid[:, None],
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def save_trace_map(
    image: np.ndarray,
    heatmap: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> None:
    if image.ndim != 3 or heatmap.shape != image.shape[:2]:
        raise ValueError("image and heatmap spatial dimensions must match")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Cropped input")
    axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title("Candidate trace intensity")
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="magma", alpha=0.5)
    axes[2].set_title("Overlay (diagnostic only)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
