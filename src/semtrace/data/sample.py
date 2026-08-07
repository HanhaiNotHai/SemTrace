from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ImageSample:
    """Unified sample returned by every SemTrace dataset adapter."""

    image: torch.Tensor
    label: int
    semantic_class: int | None
    generator: str
    source: str | None
    degradation: str | None
    path: str
    content_env: str | None = None
    real_source: str | None = None
    source_dataset: str | None = None
    split: str | None = None
