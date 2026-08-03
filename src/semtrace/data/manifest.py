from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from semtrace.data.sample import ImageSample

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    path: str
    label: int
    semantic_class: int | None
    generator: str
    source: str | None
    split: str
    degradation: str | None = None
    file_format: str | None = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must follow the binary convention real=0, fake=1")
        if not self.path:
            raise ValueError("manifest path must not be empty")
        if not self.generator:
            raise ValueError("generator must not be empty; use 'real' for authentic images")


@dataclass(frozen=True, slots=True)
class ScanRule:
    glob: str
    label: int
    generator: str
    source: str | None
    split: str
    semantic_class: int | None = None
    degradation: str | None = None
    infer_semantic_from_parent: bool = False

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("scan rule label must be real=0 or fake=1")


@dataclass(frozen=True, slots=True)
class ScanAudit:
    matched: int
    accepted: int
    skipped_small: int
    invalid: int
    duplicates: int


def scan_manifest(
    root: str | Path,
    rules: Iterable[ScanRule],
    minimum_size: int,
    small_image_policy: str = "skip",
) -> tuple[list[ManifestRecord], ScanAudit]:
    """Scan only explicit glob rules; no dataset folder semantics are guessed."""
    if small_image_policy not in {"skip", "reflect"}:
        raise ValueError("small_image_policy must be 'skip' or 'reflect'")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root_path}")

    records: list[ManifestRecord] = []
    seen: set[Path] = set()
    matched = skipped_small = invalid = duplicates = 0
    inferred_semantic_classes: dict[str, int] = {}
    for rule in rules:
        for path in sorted(root_path.glob(rule.glob)):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            matched += 1
            resolved = path.resolve()
            if resolved in seen:
                duplicates += 1
                continue
            seen.add(resolved)
            try:
                with Image.open(resolved) as image:
                    width, height = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError):
                invalid += 1
                continue
            if min(width, height) < minimum_size and small_image_policy == "skip":
                skipped_small += 1
                continue
            semantic_class = rule.semantic_class
            if semantic_class is None and rule.infer_semantic_from_parent:
                base = Path(rule.glob.split("/**", maxsplit=1)[0])
                relative_parts = resolved.relative_to(root_path / base).parts
                if len(relative_parts) > 1:
                    category = relative_parts[0]
                    semantic_class = inferred_semantic_classes.setdefault(
                        category,
                        len(inferred_semantic_classes),
                    )
            records.append(
                ManifestRecord(
                    path=str(resolved),
                    label=rule.label,
                    semantic_class=semantic_class,
                    generator=rule.generator,
                    source=rule.source,
                    split=rule.split,
                    degradation=rule.degradation,
                    file_format=resolved.suffix.lower().lstrip("."),
                )
            )
    audit = ScanAudit(
        matched=matched,
        accepted=len(records),
        skipped_small=skipped_small,
        invalid=invalid,
        duplicates=duplicates,
    )
    return sorted(records, key=lambda record: record.path), audit


def protocol_scan_rules(protocol_name: str) -> list[ScanRule]:
    """Return explicit rules for each supported installed dataset layout."""
    if protocol_name == "forensynths_progan4":
        rules: list[ScanRule] = []
        for semantic_class, class_name in enumerate(("car", "cat", "chair", "horse")):
            for split in ("train", "val"):
                for label, directory in ((0, "0_real"), (1, "1_fake")):
                    rules.append(
                        ScanRule(
                            glob=f"{split}/{class_name}/{directory}/**/*",
                            label=label,
                            generator="progan",
                            source="forensynths",
                            split="validation" if split == "val" else split,
                            semantic_class=semantic_class,
                        )
                    )
        return rules
    if protocol_name == "self_synthesis":
        generator_directories = {
            "AttGAN": "attgan",
            "BEGAN": "began",
            "CramerGAN": "cramergan",
            "InfoMaxGAN": "infomaxgan",
            "MMDGAN": "mmdgan",
            "RelGAN": "relgan",
            "S3GAN": "s3gan",
            "SNGAN": "sngan",
            "STGAN": "stgan",
        }
        rules = []
        for directory, generator in generator_directories.items():
            for label, label_directory in ((0, "0_real"), (1, "1_fake")):
                rules.append(
                    ScanRule(
                        glob=f"{directory}/{label_directory}/**/*",
                        label=label,
                        generator=generator,
                        source="self_synthesis",
                        split="test",
                    )
                )
        return rules
    if protocol_name == "genimage_sdv14":
        generator_directories = {
            "ADM": "adm",
            "BigGAN": "biggan",
            "glide": "glide",
            "Midjourney": "midjourney",
            "stable_diffusion_v_1_4": "sdv1.4",
            "stable_diffusion_v_1_5": "sdv1.5",
            "VQDM": "vqdm",
            "wukong": "wukong",
        }
        rules = []
        for directory, generator in generator_directories.items():
            splits = ("train", "val") if generator == "sdv1.4" else ("val",)
            for split in splits:
                for label, kind in ((0, "nature"), (1, "ai")):
                    rules.append(
                        ScanRule(
                            glob=f"{directory}/{split}/{kind}/**/*",
                            label=label,
                            generator=generator,
                            source="genimage",
                            split="validation" if split == "val" else split,
                            infer_semantic_from_parent=True,
                        )
                    )
        return rules
    raise ValueError(f"no official scan preset for protocol: {protocol_name}")


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (record.path, record.label, record.split))
    with destination.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(ManifestRecord(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid manifest record at line {line_number}") from error
    return records


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestImageDataset(Dataset[ImageSample]):
    """Common manifest-backed implementation used by all protocol adapters."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        split: str | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        records = load_manifest(self.manifest_path)
        self.records = [record for record in records if split is None or record.split == split]
        self.transform = transform
        self.data_root = Path(data_root) if data_root is not None else self.manifest_path.parent

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageSample:
        record = self.records[index]
        path = Path(record.path)
        if not path.is_absolute():
            path = self.data_root / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return ImageSample(
            image=tensor,
            label=record.label,
            semantic_class=record.semantic_class,
            generator=record.generator,
            source=record.source,
            degradation=record.degradation,
            path=str(path),
        )

    def __iter__(self) -> Iterator[ImageSample]:
        for index in range(len(self)):
            yield self[index]
