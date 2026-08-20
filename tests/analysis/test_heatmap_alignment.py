import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from semtrace.analysis.runner import _build_analysis_dataset
from semtrace.analysis.visualize_trace_maps import patch_map, resize_patch_map
from semtrace.config import compose_config


def test_heatmap_uses_patch_grid_row_major_alignment() -> None:
    values = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    grid = patch_map(values, (2, 3))
    resized = resize_patch_map(grid, (8, 12))

    torch.testing.assert_close(grid, torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]))
    assert resized.shape == (1, 8, 12)


def test_standalone_visualization_resizes_the_full_image_without_cropping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "quadrants.png"
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    pixels[:256, :256] = (255, 0, 0)
    pixels[:256, 256:] = (0, 255, 0)
    pixels[256:, :256] = (0, 0, 255)
    pixels[256:, 256:] = (255, 255, 255)
    Image.fromarray(pixels).save(image_path)
    manifest = tmp_path / "sample.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": str(image_path),
                "label": 1,
                "semantic_class": None,
                "generator": "synthetic",
                "source": "test",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    def forbidden_crop(*args: object, **kwargs: object) -> None:
        raise AssertionError("standalone visualization must not crop the source image")

    monkeypatch.setattr(Image.Image, "crop", forbidden_crop)
    config = compose_config("analysis/visualization")

    dataset = _build_analysis_dataset(config, str(manifest), task="visualization")
    output = dataset[0].image

    assert output.shape == (3, 128, 128)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    restored = (output * std + mean).clamp(0, 1)
    torch.testing.assert_close(restored[:, 8, 8], torch.tensor([1.0, 0.0, 0.0]), atol=0.03, rtol=0)
    torch.testing.assert_close(restored[:, 8, -9], torch.tensor([0.0, 1.0, 0.0]), atol=0.03, rtol=0)
    torch.testing.assert_close(restored[:, -9, 8], torch.tensor([0.0, 0.0, 1.0]), atol=0.03, rtol=0)
    torch.testing.assert_close(restored[:, -9, -9], torch.ones(3), atol=0.03, rtol=0)
