from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
from PIL import Image

# Pillow crop semantics: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop

SmallImagePolicy = Literal["skip", "reflect"]


class SmallImageError(ValueError):
    """Raised when the strict protocol encounters an image smaller than its crop."""


class ProtocolTransform:
    """Crop and normalize without any resize operation."""

    def __init__(
        self,
        crop_size: int = 128,
        *,
        training: bool,
        small_image_policy: SmallImagePolicy = "skip",
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        rng: random.Random | None = None,
    ) -> None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if small_image_policy not in ("skip", "reflect"):
            raise ValueError("small_image_policy must be 'skip' or 'reflect'")
        if any(value <= 0 for value in std):
            raise ValueError("normalization standard deviations must be positive")
        self.crop_size = crop_size
        self.training = training
        self.small_image_policy = small_image_policy
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.rng = rng

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < self.crop_size:
            if self.small_image_policy == "skip":
                raise SmallImageError(
                    f"image size {width}x{height} is smaller than {self.crop_size}x{self.crop_size}"
                )
            image = self._reflect_pad(image)
            width, height = image.size

        max_left = width - self.crop_size
        max_top = height - self.crop_size
        if self.training:
            generator = self.rng if self.rng is not None else random
            left = generator.randint(0, max_left)
            top = generator.randint(0, max_top)
        else:
            left = max_left // 2
            top = max_top // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))

        array = np.array(image, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
        return (tensor - self.mean) / self.std

    def _reflect_pad(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        missing_width = max(0, self.crop_size - width)
        missing_height = max(0, self.crop_size - height)
        left = missing_width // 2
        right = missing_width - left
        top = missing_height // 2
        bottom = missing_height - top
        array = np.array(image, copy=True)
        padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        return Image.fromarray(padded)
