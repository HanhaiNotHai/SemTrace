from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def create_run_directory(output_root: str | Path, experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path(output_root) / experiment / timestamp
    for child in ("checkpoints", "tensorboard", "predictions"):
        (run_directory / child).mkdir(parents=True, exist_ok=True)
    return run_directory


def write_resolved_config(config: DictConfig, run_directory: Path) -> None:
    (run_directory / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )


def append_metrics(run_directory: Path, metrics: dict[str, Any]) -> None:
    with (run_directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

