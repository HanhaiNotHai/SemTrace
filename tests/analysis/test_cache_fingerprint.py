from pathlib import Path

from semtrace.analysis.common import cache_fingerprint
from semtrace.config import compose_config


def test_cache_runtime_path_and_task_do_not_change_feature_fingerprint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    manifest = tmp_path / "manifest.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n", encoding="utf-8")
    extraction = compose_config("analysis/mechanism_base")
    reuse = compose_config(
        "analysis/mechanism_base",
        [
            "analysis.cache=/tmp/cache",
            "analysis.output_dir=/tmp/output",
            "+analysis.task=residuals",
        ],
    )

    assert cache_fingerprint(checkpoint, extraction, [manifest]) == cache_fingerprint(
        checkpoint, reuse, [manifest]
    )
