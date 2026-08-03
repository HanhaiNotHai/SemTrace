from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from semtrace.config import parse_config_args
from semtrace.data.collate import ImageBatch
from semtrace.data.manifest import manifest_sha256
from semtrace.engine.checkpoint import save_training_checkpoint
from semtrace.engine.distributed import (
    initialize_distributed,
    select_amp_mode,
    validate_global_batch,
)
from semtrace.losses.normal_prediction import normal_prediction_loss
from semtrace.runtime import (
    build_backbone,
    build_dataset,
    build_loader,
    build_normal_predictor_collection,
    load_semantic_anchor,
    read_selected_layers,
)
from semtrace.utils.environment import environment_record, git_commit, write_environment
from semtrace.utils.logging import append_metrics, create_run_directory, write_resolved_config
from semtrace.utils.seed import seed_everything


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config_args(
        argv,
        description="Train real-image multi-scale normal feature predictors.",
        default_config_name="stage2_normal",
    )
    context = initialize_distributed(str(config.distributed.backend))
    seed_everything(int(config.seed) + context.rank)
    selected_layers = read_selected_layers(config.probe.selected_layers_path)
    batch_protocol = validate_global_batch(
        world_size=context.world_size,
        per_device_batch_size=int(config.training.per_device_batch_size),
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        target_global_batch_size=int(config.training.target_global_batch_size),
    )
    if not batch_protocol.strict_protocol and context.is_main_process:
        print(
            f"NON-STRICT PROTOCOL: actual global batch "
            f"{batch_protocol.actual_global_batch_size}, "
            f"learning-rate scale {batch_protocol.learning_rate_scale:.6f}"
        )
    amp_mode = select_amp_mode(str(config.training.amp), context.device)
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")

    backbone = build_backbone(config, selected_layers).to(context.device)
    semantic_anchor = load_semantic_anchor(config, backbone.hidden_size).to(context.device)
    train_manifest_hash = manifest_sha256(config.data.train_manifest)
    collection = build_normal_predictor_collection(
        config,
        selected_layers,
        backbone.hidden_size,
    ).to(context.device)
    trainable: nn.Module = collection
    if context.world_size > 1:
        trainable = DistributedDataParallel(
            collection,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=bool(config.distributed.find_unused_parameters),
        )
    optimizer = torch.optim.Adam(
        collection.parameters(),
        lr=float(config.training.learning_rate) * batch_protocol.learning_rate_scale,
        betas=(
            float(config.training.betas[0]),
            float(config.training.betas[1]),
        ),
        weight_decay=float(config.training.weight_decay),
    )
    train_dataset = build_dataset(
        config,
        split="train",
        training=True,
        real_only=True,
    )
    validation_dataset = build_dataset(
        config,
        split="validation",
        training=False,
        real_only=True,
    )
    train_loader, train_sampler = build_loader(
        train_dataset,
        batch_size=int(config.training.per_device_batch_size),
        num_workers=int(config.training.num_workers),
        training=True,
        world_size=context.world_size,
        rank=context.rank,
    )
    validation_loader, _ = build_loader(
        validation_dataset,
        batch_size=int(config.training.per_device_batch_size),
        num_workers=int(config.training.num_workers),
        training=False,
        world_size=context.world_size,
        rank=context.rank,
    )

    run_directory = None
    writer = None
    if context.is_main_process:
        run_directory = create_run_directory(config.output_root, config.experiment)
        write_resolved_config(config, run_directory)
        shutil.copy2(config.probe.selected_layers_path, run_directory / "selected_layers.json")
        (run_directory / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
        write_environment(
            run_directory / "environment.json",
            environment_record(
                model_id=str(config.model.model_id),
                model_revision=str(config.model.revision),
                seed=int(config.seed),
                global_batch=batch_protocol.actual_global_batch_size,
                amp_mode=amp_mode,
            ),
        )
        writer = SummaryWriter(run_directory / "tensorboard")

    best_loss = float("inf")
    global_step = 0
    accumulation = int(config.training.gradient_accumulation_steps)
    epochs = int(config.training.epochs)
    show_progress = context.is_main_process
    stage_started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        trainable.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=context.device)
        batch_count = 0
        sample_count = 0
        displayed_loss = 0.0
        train_progress = _progress_bar(
            train_loader,
            description=f"Stage 2 train {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        for batch_index, batch in enumerate(train_progress):
            batch = batch.to(context.device)
            if not torch.all(batch.labels == 0):
                raise ValueError("stage 2 loader yielded a fake sample")
            with torch.autocast(
                device_type=context.device.type,
                dtype=autocast_dtype,
                enabled=amp_mode != "none",
            ):
                backbone_output = backbone(batch.images)
                semantic = semantic_anchor(
                    backbone_output.semantic_cls,
                    backbone_output.final_patch_tokens,
                ).detach()
                predictions = trainable(
                    semantic,
                    backbone_output.intermediate_patch_tokens,
                    backbone_output.patch_grid_size,
                )
                loss = _normal_loss(
                    predictions,
                    backbone_output.intermediate_patch_tokens,
                    selected_layers,
                    lambda_smooth=float(config.loss.lambda_smooth),
                    lambda_cos=float(config.loss.lambda_cos),
                )
            scaler.scale(loss / accumulation).backward()
            total_loss += loss.detach()
            batch_count += 1
            if show_progress:
                sample_count += int(batch.labels.shape[0])
                displayed_loss += float(loss.detach())
                _update_progress(
                    train_progress,
                    samples=sample_count,
                    running_loss=displayed_loss / batch_count,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    collection.parameters(),
                    float(config.training.gradient_clip_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        validation_loss = _validation_loss(
            collection,
            backbone,
            semantic_anchor,
            validation_loader,
            selected_layers,
            config,
            context.device,
            amp_mode,
            autocast_dtype,
            epoch=epoch + 1,
            epochs=epochs,
            show_progress=show_progress,
        )
        train_loss = _distributed_mean(total_loss / max(batch_count, 1))
        validation_loss = _distributed_mean(validation_loss)
        if context.is_main_process and run_directory is not None:
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_normal_loss": float(train_loss),
                "validation_normal_loss": float(validation_loss),
            }
            append_metrics(run_directory, metrics)
            if writer is not None:
                writer.add_scalars(
                    "normal_loss",
                    {"train": float(train_loss), "validation": float(validation_loss)},
                    epoch + 1,
                )
            config_payload = OmegaConf.to_container(config, resolve=True)
            if not isinstance(config_payload, dict):
                raise TypeError("resolved training config must be a mapping")
            checkpoint_config = cast(dict[str, Any], config_payload)
            save_training_checkpoint(
                run_directory / "checkpoints" / "normal_last.pt",
                model=collection,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=checkpoint_config,
                selected_layers=selected_layers,
                manifest_hash=train_manifest_hash,
                best_validation_metric=min(best_loss, float(validation_loss)),
                scaler=scaler,
            )
            if float(validation_loss) < best_loss:
                best_loss = float(validation_loss)
                save_training_checkpoint(
                    run_directory / "checkpoints" / "normal_best.pt",
                    model=collection,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    config=checkpoint_config,
                    selected_layers=selected_layers,
                    manifest_hash=train_manifest_hash,
                    best_validation_metric=best_loss,
                    scaler=scaler,
                )
            _write_epoch_summary(
                epoch=epoch + 1,
                epochs=epochs,
                train_loss=float(train_loss),
                validation_loss=float(validation_loss),
                epoch_seconds=time.perf_counter() - epoch_started,
                elapsed_seconds=time.perf_counter() - stage_started,
                show_progress=show_progress,
            )
    if writer is not None:
        writer.close()
    return 0


@torch.no_grad()
def _validation_loss(
    collection: nn.Module,
    backbone: nn.Module,
    semantic_anchor: nn.Module,
    loader: DataLoader[ImageBatch],
    selected_layers: tuple[int, ...],
    config: DictConfig,
    device: torch.device,
    amp_mode: str,
    autocast_dtype: torch.dtype,
    *,
    epoch: int,
    epochs: int,
    show_progress: bool,
) -> torch.Tensor:
    collection.eval()
    total = torch.zeros((), device=device)
    count = 0
    sample_count = 0
    displayed_loss = 0.0
    validation_progress = _progress_bar(
        loader,
        description=f"Stage 2 validation {epoch}/{epochs}",
        show_progress=show_progress,
    )
    for batch in validation_progress:
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_mode != "none",
        ):
            output = backbone(batch.images)
            semantic = semantic_anchor(output.semantic_cls, output.final_patch_tokens)
            predictions = collection(
                semantic,
                output.intermediate_patch_tokens,
                output.patch_grid_size,
            )
            loss = _normal_loss(
                predictions,
                output.intermediate_patch_tokens,
                selected_layers,
                lambda_smooth=float(config.loss.lambda_smooth),
                lambda_cos=float(config.loss.lambda_cos),
            )
        total += loss
        count += 1
        if show_progress:
            sample_count += int(batch.labels.shape[0])
            displayed_loss += float(loss)
            _update_progress(
                validation_progress,
                samples=sample_count,
                running_loss=displayed_loss / count,
            )
    return total / max(count, 1)


def _progress_bar(
    iterable: Any,
    *,
    description: str,
    show_progress: bool,
) -> Any:
    return tqdm(
        iterable,
        total=len(iterable),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )


def _update_progress(
    progress: Any,
    *,
    samples: int,
    running_loss: float,
    learning_rate: float | None = None,
) -> None:
    values: dict[str, object] = {
        "samples": samples,
        "loss": f"{running_loss:.4f}",
    }
    if learning_rate is not None:
        values["lr"] = f"{learning_rate:.2e}"
    progress.set_postfix(**values)


def _write_epoch_summary(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    validation_loss: float,
    epoch_seconds: float,
    elapsed_seconds: float,
    show_progress: bool,
) -> None:
    if not show_progress:
        return
    remaining_seconds = elapsed_seconds / epoch * (epochs - epoch)
    tqdm.write(
        f"[Stage 2] Epoch {epoch}/{epochs} complete: "
        f"train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, "
        f"epoch_time={_format_duration(epoch_seconds)}, "
        f"stage_eta={_format_duration(remaining_seconds)}"
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m"


def _normal_loss(
    predictions: dict[int, torch.Tensor],
    targets: dict[int, torch.Tensor],
    selected_layers: tuple[int, ...],
    *,
    lambda_smooth: float,
    lambda_cos: float,
) -> torch.Tensor:
    losses = [
        normal_prediction_loss(
            predictions[layer],
            targets[layer],
            lambda_smooth=lambda_smooth,
            lambda_cos=lambda_cos,
        )
        for layer in selected_layers
    ]
    return torch.stack(losses).sum()


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


if __name__ == "__main__":
    sys.exit(main())
