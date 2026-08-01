from pathlib import Path

import torch
from PIL import Image

from semtrace.data.forensynths import ForenSynthsDataset
from semtrace.data.manifest import ManifestRecord, write_manifest
from semtrace.data.sample import ImageSample


def test_dataset_preserves_real_zero_fake_one(tmp_path: Path) -> None:
    real_path = tmp_path / "real.png"
    fake_path = tmp_path / "fake.png"
    Image.new("RGB", (128, 128), "white").save(real_path)
    Image.new("RGB", (128, 128), "black").save(fake_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                path=str(real_path),
                label=0,
                semantic_class=0,
                generator="real",
                source="forensynths",
                split="train",
            ),
            ManifestRecord(
                path=str(fake_path),
                label=1,
                semantic_class=0,
                generator="progan",
                source="forensynths",
                split="train",
            ),
        ],
        manifest_path,
    )

    dataset = ForenSynthsDataset(manifest_path, transform=lambda _: torch.zeros(3, 128, 128))

    assert isinstance(dataset[0], ImageSample)
    labels_by_name = {
        Path(dataset[index].path).name: dataset[index].label for index in range(len(dataset))
    }
    assert labels_by_name == {"real.png": 0, "fake.png": 1}


def test_manifest_rejects_reversed_or_invalid_binary_label() -> None:
    try:
        ManifestRecord(
            path="image.png",
            label=2,
            semantic_class=None,
            generator="unknown",
            source=None,
            split="train",
        )
    except ValueError as error:
        assert "real=0" in str(error)
    else:
        raise AssertionError("ManifestRecord accepted a non-binary label")
