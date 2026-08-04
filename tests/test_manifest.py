from pathlib import Path

from PIL import Image

from semtrace.cli.build_manifest import main as build_manifest_main
from semtrace.data.manifest import (
    ManifestRecord,
    ScanRule,
    load_manifest,
    manifest_sha256,
    protocol_scan_rules,
    scan_manifest,
    write_manifest,
)


def test_manifest_scan_skips_small_images_and_records_counts(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (128, 128)).save(image_root / "valid.png")
    Image.new("RGB", (127, 128)).save(image_root / "small.png")

    records, audit = scan_manifest(
        root=image_root,
        rules=[
            ScanRule(
                glob="*.png",
                label=1,
                generator="sdv1.4",
                source="genimage",
                split="train",
            )
        ],
        minimum_size=128,
        small_image_policy="skip",
    )

    assert [Path(record.path).name for record in records] == ["valid.png"]
    assert audit.accepted == 1
    assert audit.skipped_small == 1
    assert audit.invalid == 0


def test_manifest_round_trip_and_hash_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    records = [
        ManifestRecord(
            path="b.png",
            label=1,
            semantic_class=4,
            generator="progan",
            source="forensynths",
            split="train",
        ),
        ManifestRecord(
            path="a.png",
            label=0,
            semantic_class=4,
            generator="real",
            source="forensynths",
            split="train",
        ),
    ]

    write_manifest(records, path)
    first_hash = manifest_sha256(path)
    write_manifest(reversed(records), path)

    assert [record.path for record in load_manifest(path)] == ["a.png", "b.png"]
    assert manifest_sha256(path) == first_hash


def test_forensynths_official_rules_map_real_and_fake(tmp_path: Path) -> None:
    Image.new("RGB", (128, 128)).save(_make_parent(tmp_path / "train/car/0_real/a.png"))
    Image.new("RGB", (128, 128)).save(_make_parent(tmp_path / "train/car/1_fake/b.png"))

    records, _ = scan_manifest(
        tmp_path,
        protocol_scan_rules("forensynths_progan4"),
        minimum_size=128,
    )

    assert {(record.label, record.generator, record.semantic_class) for record in records} == {
        (0, "progan", 0),
        (1, "progan", 0),
    }


def test_genimage_official_rules_use_sdv14_for_training(tmp_path: Path) -> None:
    Image.new("RGB", (128, 128)).save(
        _make_parent(tmp_path / "stable_diffusion_v_1_4/train/ai/n01440764/a.png")
    )
    Image.new("RGB", (128, 128)).save(
        _make_parent(tmp_path / "stable_diffusion_v_1_4/train/nature/n01440764/b.png")
    )

    records, _ = scan_manifest(
        tmp_path,
        protocol_scan_rules("genimage_sdv14"),
        minimum_size=128,
    )

    train_records = [record for record in records if record.split == "train"]
    assert {(record.label, record.generator) for record in train_records} == {
        (0, "sdv1.4"),
        (1, "sdv1.4"),
    }
    assert {record.semantic_class for record in train_records} == {0}


def test_build_manifest_cli_writes_protocol_manifest(tmp_path: Path) -> None:
    Image.new("RGB", (128, 128)).save(_make_parent(tmp_path / "train/car/0_real/a.png"))
    Image.new("RGB", (128, 128)).save(_make_parent(tmp_path / "train/car/1_fake/b.png"))
    output = tmp_path / "manifest.jsonl"

    exit_code = build_manifest_main(
        [
            "--config-name",
            "manifest",
            "protocol=forensynths_progan4",
            f"data.root={tmp_path}",
            f"manifest.output={output}",
        ]
    )

    assert exit_code == 0
    assert len(load_manifest(output)) == 2


def _make_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
