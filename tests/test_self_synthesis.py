from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from semtrace.config import compose_config
from semtrace.data.manifest import (
    ManifestRecord,
    protocol_scan_rules,
    scan_manifest,
    write_manifest,
)
from semtrace.data.self_synthesis import SelfSynthesisDataset
from semtrace.metrics.binary import grouped_binary_metrics
from semtrace.runtime import build_dataset

GENERATOR_DIRECTORIES = {
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


def test_self_synthesis_rules_map_all_generators_and_binary_labels(
    tmp_path: Path,
) -> None:
    for directory in GENERATOR_DIRECTORIES:
        Image.new("RGB", (128, 128)).save(
            _make_parent(tmp_path / directory / "0_real" / "real.png")
        )
        Image.new("RGB", (128, 128)).save(
            _make_parent(tmp_path / directory / "1_fake" / "fake.png")
        )

    records, audit = scan_manifest(
        tmp_path,
        protocol_scan_rules("self_synthesis"),
        minimum_size=128,
    )

    counts = Counter((record.generator, record.label) for record in records)
    assert counts == Counter(
        {
            (generator, label): 1
            for generator in GENERATOR_DIRECTORIES.values()
            for label in (0, 1)
        }
    )
    assert len(records) == 18
    assert audit.accepted == 18
    assert {record.source for record in records} == {"self_synthesis"}
    assert {record.split for record in records} == {"test"}
    assert {record.semantic_class for record in records} == {None}


def test_self_synthesis_protocol_uses_center_crop_without_resize_and_adapter(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "AttGAN" / "0_real" / "real.png"
    Image.new("RGB", (128, 128)).save(_make_parent(image_path))
    manifest_path = tmp_path / "self_synthesis.jsonl"
    write_manifest(
        [
            ManifestRecord(
                path=str(image_path),
                label=0,
                semantic_class=None,
                generator="attgan",
                source="self_synthesis",
                split="test",
            )
        ],
        manifest_path,
    )
    config = compose_config(
        "eval",
        [
            "protocol=self_synthesis",
            f"data.root={tmp_path}",
            f"data.validation_manifest={manifest_path}",
        ],
    )

    dataset = build_dataset(
        config,
        split=None,
        training=False,
        manifest_path=manifest_path,
    )

    assert isinstance(dataset, SelfSynthesisDataset)
    assert config.preprocessing.eval_crop == "center"
    assert config.preprocessing.allow_resize is False
    assert config.preprocessing.crop_size == 128
    assert (
        config.data.test_manifests.self_synthesis
        == "artifacts/manifests/self_synthesis.jsonl"
    )
    assert dataset[0].label == 0
    assert dataset[0].generator == "attgan"


def test_self_synthesis_metrics_report_all_generators_and_unweighted_means() -> None:
    generators = list(GENERATOR_DIRECTORIES.values())
    result = grouped_binary_metrics(
        labels=[label for _ in generators for label in (0, 1)],
        fake_scores=[score for _ in generators for score in (0.1, 0.9)],
        domains=[generator for generator in generators for _ in range(2)],
    )

    assert set(result.per_domain) == set(generators)
    assert all(metrics.accuracy == 1.0 for metrics in result.per_domain.values())
    assert all(
        metrics.average_precision == 1.0 for metrics in result.per_domain.values()
    )
    assert result.mean_accuracy == 1.0
    assert result.mean_average_precision == 1.0


def _make_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
