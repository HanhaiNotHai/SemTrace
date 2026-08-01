from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def compose_config(
    config_name: str,
    overrides: Sequence[str] = (),
    *,
    config_dir: str | Path = CONFIG_DIR,
) -> DictConfig:
    """Compose a repository config without changing the process working directory."""
    directory = str(Path(config_dir).resolve())
    with initialize_config_dir(version_base=None, config_dir=directory):
        return compose(config_name=config_name, overrides=list(overrides))


def parse_config_args(
    argv: Sequence[str] | None,
    *,
    description: str,
    default_config_name: str,
) -> DictConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-name", default=default_config_name)
    arguments, overrides = parser.parse_known_args(argv)
    invalid = [item for item in overrides if item.startswith("-") or "=" not in item]
    if invalid:
        parser.error(f"invalid Hydra overrides: {invalid}")
    return compose_config(arguments.config_name, overrides)
