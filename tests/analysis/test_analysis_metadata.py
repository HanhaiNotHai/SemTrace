import torch

from semtrace.data.collate import collate_image_samples
from semtrace.data.manifest import ManifestRecord
from semtrace.data.sample import ImageSample


def test_optional_analysis_metadata_flows_from_manifest_to_batch() -> None:
    record = ManifestRecord(
        path="image.png",
        label=0,
        semantic_class=3,
        generator="real",
        source="forensynths",
        split="test",
        content_env="outdoor",
        real_source="lsun",
        source_dataset="forensynths",
    )
    batch = collate_image_samples(
        [
            ImageSample(
                image=torch.zeros(3, 8, 8),
                label=record.label,
                semantic_class=record.semantic_class,
                generator=record.generator,
                source=record.source,
                degradation=record.degradation,
                path=record.path,
                content_env=record.content_env,
                real_source=record.real_source,
                source_dataset=record.source_dataset,
                split=record.split,
            )
        ]
    )

    assert batch.content_envs == ["outdoor"]
    assert batch.real_sources == ["lsun"]
    assert batch.source_datasets == ["forensynths"]
    assert batch.splits == ["test"]
