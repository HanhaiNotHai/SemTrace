from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    selected_layers: tuple[int, ...],
    manifest_hash: str,
    best_validation_metric: float,
    scaler: Any = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": dict(config),
        "selected_layers": list(selected_layers),
        "manifest_hash": manifest_hash,
        "best_validation_metric": best_validation_metric,
    }
    torch.save(payload, destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    restore_random_state: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_random_state:
        state = payload["random_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    return payload

