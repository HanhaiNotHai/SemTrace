from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

import torch

FeatureValue: TypeAlias = torch.Tensor | dict[int, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AnalysisSampleMetadata:
    sample_id: str
    path: str
    label: int
    generator: str
    semantic_class: int | None
    content_env: str | None
    real_source: str | None
    source_dataset: str | None
    degradation: str | None
    split: str | None


@dataclass(frozen=True, slots=True)
class CacheFingerprint:
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class FeatureShard:
    metadata: list[AnalysisSampleMetadata]
    features: dict[str, FeatureValue]


class FeatureCacheWriter:
    def __init__(
        self,
        root: str | Path,
        fingerprint: CacheFingerprint,
        *,
        dtype: torch.dtype,
        rank: int,
    ) -> None:
        if dtype not in {torch.float16, torch.float32}:
            raise ValueError("cache dtype must be float16 or float32")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.dtype = dtype
        self.rank = rank
        self.rank_index = self.root / f"index_rank{rank:03d}.json"
        self._entries = self._load_rank_entries()
        self.completed_sample_ids = {
            sample_id
            for entry in self._entries
            for sample_id in cast(list[str], entry["sample_ids"])
        }

    def write_shard(
        self,
        metadata: list[AnalysisSampleMetadata],
        features: dict[str, FeatureValue],
    ) -> Path:
        if not metadata:
            raise ValueError("cannot write an empty feature shard")
        _validate_batch_size(features, len(metadata))
        shard_name = f"rank{self.rank:03d}_shard{len(self._entries):06d}.pt"
        destination = self.root / shard_name
        temporary = destination.with_suffix(".pt.tmp")
        payload = {
            "fingerprint": asdict(self.fingerprint),
            "metadata": [asdict(sample) for sample in metadata],
            "features": _to_cpu(features, self.dtype),
        }
        torch.save(payload, temporary)
        temporary.replace(destination)
        self._entries.append(
            {"path": shard_name, "sample_ids": [sample.sample_id for sample in metadata]}
        )
        self._write_rank_index()
        self.completed_sample_ids.update(sample.sample_id for sample in metadata)
        return destination

    def finalize(self) -> Path:
        entries: list[dict[str, object]] = []
        for path in sorted(self.root.glob("index_rank*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["fingerprint"] != asdict(self.fingerprint):
                raise ValueError(f"cache fingerprint mismatch in {path}")
            entries.extend(payload["shards"])
        index = {
            "version": 1,
            "fingerprint": asdict(self.fingerprint),
            "dtype": str(self.dtype).removeprefix("torch."),
            "shards": entries,
        }
        destination = self.root / "index.json"
        destination.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    def _load_rank_entries(self) -> list[dict[str, object]]:
        if not self.rank_index.exists():
            return []
        payload = json.loads(self.rank_index.read_text(encoding="utf-8"))
        if payload["fingerprint"] != asdict(self.fingerprint):
            raise ValueError("cache fingerprint does not match the requested extraction")
        entries = payload["shards"]
        if not isinstance(entries, list):
            raise ValueError("cache rank index shards must be a list")
        return entries

    def _write_rank_index(self) -> None:
        payload = {"fingerprint": asdict(self.fingerprint), "shards": self._entries}
        self.rank_index.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class FeatureCacheReader:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_fingerprint: CacheFingerprint | None = None,
    ) -> None:
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if (
            expected_fingerprint is not None
            and self.index["fingerprint"] != asdict(expected_fingerprint)
        ):
            raise ValueError(
                "cache fingerprint does not match the requested checkpoint/config/manifest"
            )

    def iter_shards(
        self,
        *,
        generator: str | None = None,
        split: str | None = None,
        source_dataset: str | None = None,
    ) -> Iterator[FeatureShard]:
        seen_sample_ids: set[str] = set()
        for entry in self.index["shards"]:
            payload = torch.load(
                self.root / entry["path"],
                map_location="cpu",
                weights_only=False,
            )
            metadata = [AnalysisSampleMetadata(**sample) for sample in payload["metadata"]]
            indices = [
                index
                for index, sample in enumerate(metadata)
                if (generator is None or sample.generator == generator)
                and (split is None or sample.split == split)
                and (source_dataset is None or sample.source_dataset == source_dataset)
                and sample.sample_id not in seen_sample_ids
            ]
            if indices:
                seen_sample_ids.update(metadata[index].sample_id for index in indices)
                yield FeatureShard(
                    metadata=[metadata[index] for index in indices],
                    features=_select_rows(payload["features"], indices),
                )


def _to_cpu(features: dict[str, FeatureValue], dtype: torch.dtype) -> dict[str, FeatureValue]:
    converted: dict[str, FeatureValue] = {}
    for name, value in features.items():
        if isinstance(value, torch.Tensor):
            converted[name] = value.detach().to(device="cpu", dtype=dtype)
        else:
            converted[name] = {
                layer: tensor.detach().to(device="cpu", dtype=dtype)
                for layer, tensor in value.items()
            }
    return converted


def _select_rows(
    features: dict[str, FeatureValue], indices: list[int]
) -> dict[str, FeatureValue]:
    selected: dict[str, FeatureValue] = {}
    for name, value in features.items():
        if isinstance(value, torch.Tensor):
            selected[name] = value[indices]
        else:
            selected[name] = {layer: tensor[indices] for layer, tensor in value.items()}
    return selected


def _validate_batch_size(features: dict[str, FeatureValue], expected: int) -> None:
    for value in features.values():
        tensors = [value] if isinstance(value, torch.Tensor) else list(value.values())
        if any(tensor.ndim == 0 or tensor.shape[0] != expected for tensor in tensors):
            raise ValueError("every cached tensor must share the metadata batch dimension")
