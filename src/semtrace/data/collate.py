from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from semtrace.data.sample import ImageSample


@dataclass(frozen=True, slots=True)
class ImageBatch:
    images: torch.Tensor
    labels: torch.Tensor
    semantic_classes: torch.Tensor
    generators: list[str]
    sources: list[str | None]
    degradations: list[str | None]
    file_formats: list[str]
    paths: list[str]
    content_envs: list[str | None]
    real_sources: list[str | None]
    source_datasets: list[str | None]
    splits: list[str | None]

    def to(self, device: torch.device) -> ImageBatch:
        return ImageBatch(
            images=self.images.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            semantic_classes=self.semantic_classes.to(device, non_blocking=True),
            generators=self.generators,
            sources=self.sources,
            degradations=self.degradations,
            file_formats=self.file_formats,
            paths=self.paths,
            content_envs=self.content_envs,
            real_sources=self.real_sources,
            source_datasets=self.source_datasets,
            splits=self.splits,
        )


def collate_image_samples(samples: list[ImageSample]) -> ImageBatch:
    if not samples:
        raise ValueError("cannot collate an empty image sample batch")
    return ImageBatch(
        images=torch.stack([sample.image for sample in samples]),
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        semantic_classes=torch.tensor(
            [
                sample.semantic_class if sample.semantic_class is not None else -1
                for sample in samples
            ],
            dtype=torch.long,
        ),
        generators=[sample.generator for sample in samples],
        sources=[sample.source for sample in samples],
        degradations=[sample.degradation for sample in samples],
        file_formats=[Path(sample.path).suffix.lower().lstrip(".") for sample in samples],
        paths=[sample.path for sample in samples],
        content_envs=[sample.content_env for sample in samples],
        real_sources=[sample.real_source for sample in samples],
        source_datasets=[sample.source_dataset for sample in samples],
        splits=[sample.split for sample in samples],
    )
