from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import transformers


def git_commit(repository: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record(
    *,
    model_id: str,
    model_revision: str,
    seed: int,
    global_batch: int,
    amp_mode: str,
    lock_path: str | Path = "uv.lock",
) -> dict[str, Any]:
    lock = Path(lock_path)
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": len(gpu_names),
        "gpu_models": gpu_names,
        "transformers_version": transformers.__version__,
        "dinov3_model_id": model_id,
        "model_revision": model_revision,
        "uv_lock_sha256": file_sha256(lock) if lock.is_file() else "unavailable",
        "git_commit": git_commit(),
        "seed": seed,
        "global_batch": global_batch,
        "amp_mode": amp_mode,
    }


def write_environment(path: str | Path, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

