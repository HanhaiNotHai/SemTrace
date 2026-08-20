from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class FullImageResizeTransform:
    """Resize the complete image for standalone visualization, then normalize."""

    def __init__(
        self,
        size: int,
        *,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if size <= 0:
            raise ValueError("visualization resize size must be positive")
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Pillow resize uses the complete source when box is omitted:
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
        resized = image.convert("RGB").resize(
            (self.size, self.size),
            resample=Image.Resampling.LANCZOS,
        )
        array = np.array(resized, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
        return (tensor - self.mean) / self.std


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
    axes[0].set_title("Model input")
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
