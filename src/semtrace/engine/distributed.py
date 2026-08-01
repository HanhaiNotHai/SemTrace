from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class BatchProtocol:
    actual_global_batch_size: int
    target_global_batch_size: int
    strict_protocol: bool
    learning_rate_scale: float


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def validate_global_batch(
    *,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    target_global_batch_size: int,
) -> BatchProtocol:
    values = (
        world_size,
        per_device_batch_size,
        gradient_accumulation_steps,
        target_global_batch_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("world size and all batch sizes must be positive")
    actual = world_size * per_device_batch_size * gradient_accumulation_steps
    if world_size == 6 and actual not in {120, 144}:
        raise ValueError("six-GPU non-strict mode requires global batch 120 or 144")
    return BatchProtocol(
        actual_global_batch_size=actual,
        target_global_batch_size=target_global_batch_size,
        strict_protocol=actual == target_global_batch_size,
        learning_rate_scale=actual / target_global_batch_size,
    )


def initialize_distributed(backend: str = "nccl") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        selected_backend = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=selected_backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, device)


def select_amp_mode(requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "none"
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return "bf16"
    if requested in {"bf16", "fp16"}:
        return "fp16"
    if requested in {"none", "fp32"}:
        return "none"
    raise ValueError(f"unsupported AMP mode: {requested}")
