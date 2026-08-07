from pathlib import Path

import pytest
import torch

from semtrace.analysis.feature_cache import (
    AnalysisSampleMetadata,
    CacheFingerprint,
    FeatureCacheReader,
    FeatureCacheWriter,
)


def _fingerprint(checkpoint: str = "checkpoint-a") -> CacheFingerprint:
    return CacheFingerprint(
        checkpoint_sha256=checkpoint,
        config_sha256="config-a",
        manifest_sha256="manifest-a",
    )


def test_feature_cache_writes_cpu_shards_resumes_and_filters(tmp_path: Path) -> None:
    writer = FeatureCacheWriter(tmp_path, _fingerprint(), dtype=torch.float16, rank=0)
    metadata = [
        AnalysisSampleMetadata(
            sample_id="a",
            path="a.png",
            label=0,
            generator="real",
            semantic_class=None,
            content_env=None,
            real_source="lsun",
            source_dataset="forensynths",
            degradation=None,
            split="test",
        ),
        AnalysisSampleMetadata(
            sample_id="b",
            path="b.png",
            label=1,
            generator="progan",
            semantic_class=2,
            content_env="indoor",
            real_source=None,
            source_dataset="forensynths",
            degradation="jpeg",
            split="test",
        ),
    ]
    writer.write_shard(
        metadata,
        {
            "semantic_anchor": torch.randn(2, 8, device="cpu"),
            "raw_patch_features": {2: torch.randn(2, 4, 16)},
        },
    )
    writer.finalize()

    resumed = FeatureCacheWriter(tmp_path, _fingerprint(), dtype=torch.float16, rank=0)
    assert resumed.completed_sample_ids == {"a", "b"}
    reader = FeatureCacheReader(tmp_path, expected_fingerprint=_fingerprint())
    shards = list(reader.iter_shards(generator="progan", split="test"))

    assert len(shards) == 1
    assert [sample.sample_id for sample in shards[0].metadata] == ["b"]
    assert shards[0].features["semantic_anchor"].dtype == torch.float16
    assert shards[0].features["semantic_anchor"].device.type == "cpu"


def test_feature_cache_rejects_stale_checkpoint(tmp_path: Path) -> None:
    FeatureCacheWriter(tmp_path, _fingerprint(), dtype=torch.float32, rank=0).finalize()

    with pytest.raises(ValueError, match="fingerprint"):
        FeatureCacheReader(tmp_path, expected_fingerprint=_fingerprint("checkpoint-b"))


def test_feature_cache_deduplicates_samples_without_dropping_unique_rows(
    tmp_path: Path,
) -> None:
    metadata = [
        AnalysisSampleMetadata(
            sample_id=sample_id,
            path=f"{sample_id}.png",
            label=0,
            generator="real",
            semantic_class=None,
            content_env=None,
            real_source=None,
            source_dataset="test",
            degradation=None,
            split="test",
        )
        for sample_id in ("shared", "unique")
    ]
    first = FeatureCacheWriter(tmp_path, _fingerprint(), dtype=torch.float32, rank=0)
    first.write_shard(metadata[:1], {"semantic_anchor": torch.ones(1, 2)})
    second = FeatureCacheWriter(tmp_path, _fingerprint(), dtype=torch.float32, rank=1)
    second.write_shard(metadata, {"semantic_anchor": torch.ones(2, 2)})
    first.finalize()

    samples = [
        sample.sample_id
        for shard in FeatureCacheReader(tmp_path).iter_shards()
        for sample in shard.metadata
    ]

    assert samples == ["shared", "unique"]
