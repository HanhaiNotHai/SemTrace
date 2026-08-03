from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from semtrace.config import parse_config_args
from semtrace.data.manifest import manifest_sha256
from semtrace.engine.checkpoint import load_training_checkpoint, save_training_checkpoint
from semtrace.engine.distributed import (
    initialize_distributed,
    select_amp_mode,
    validate_global_batch,
)
from semtrace.engine.evaluator import evaluate_loader
from semtrace.losses.detection import detection_loss
from semtrace.losses.separation import semantic_trace_separation_loss
from semtrace.runtime import (
    build_backbone,
    build_dataset,
    build_detection_model,
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
        description="Train the SemTrace candidate-trace detector.",
        default_config_name="stage3_detector",
    )
    baseline = str(config.model_options.baseline)
    if baseline != "semtrace" and bool(config.model_options.use_separation_loss):
        raise ValueError("diagnostic baselines require use_separation_loss=false")
    context = initialize_distributed(str(config.distributed.backend))
    seed_everything(int(config.seed) + context.rank)
    all_selected_layers = read_selected_layers(config.probe.selected_layers_path)
    number_of_scales = int(config.model_options.num_trace_scales)
    if number_of_scales not in {1, 3}:
        raise ValueError("num_trace_scales must be 1 or 3")
    selected_layers = all_selected_layers[:number_of_scales]
    protocol_batch = validate_global_batch(
        world_size=context.world_size,
        per_device_batch_size=int(config.training.per_device_batch_size),
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        target_global_batch_size=int(config.training.target_global_batch_size),
    )
    if not protocol_batch.strict_protocol and context.is_main_process:
        print(
            f"NON-STRICT PROTOCOL: actual global batch "
            f"{protocol_batch.actual_global_batch_size}; "
            f"linear LR scale={protocol_batch.learning_rate_scale:.6f}"
        )
    amp_mode = select_amp_mode(str(config.training.amp), context.device)
    autocast_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")

    backbone = build_backbone(config, selected_layers).to(context.device)
    semantic_anchor = load_semantic_anchor(config, backbone.hidden_size).to(context.device)
    train_manifest_hash = manifest_sha256(config.data.train_manifest)
    use_normal = baseline == "semtrace" and bool(
        config.model_options.use_normal_predictor
    )
    collection = None
    predictors = None
    if use_normal:
        if config.normal.checkpoint is None:
            raise ValueError("normal.checkpoint is required when normal prediction is enabled")
        collection = build_normal_predictor_collection(
            config,
            all_selected_layers,
            backbone.hidden_size,
        )
        load_training_checkpoint(
            config.normal.checkpoint,
            model=collection,
            optimizer=None,
            restore_random_state=False,
        )
        predictors = collection.normal_predictors

    detector = build_detection_model(
        config,
        backbone=backbone,
        semantic_anchor=semantic_anchor,
        selected_layers=selected_layers,
        normal_predictors=predictors,
    ).to(context.device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in detector.parameters() if parameter.requires_grad],
        lr=float(config.training.learning_rate) * protocol_batch.learning_rate_scale,
        betas=(
            float(config.training.betas[0]),
            float(config.training.betas[1]),
        ),
        weight_decay=float(config.training.weight_decay),
    )
    start_epoch = 0
    global_step = 0
    best_ap = float("-inf")
    if config.training.resume is not None:
        resume_payload = load_training_checkpoint(
            config.training.resume,
            model=detector,
            optimizer=optimizer,
            scaler=scaler,
            restore_random_state=True,
        )
        start_epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload["global_step"])
        best_ap = float(resume_payload["best_validation_metric"])

    training_model: nn.Module = detector
    if context.world_size > 1:
        training_model = DistributedDataParallel(
            detector,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=bool(config.distributed.find_unused_parameters),
        )
    train_dataset = build_dataset(config, split="train", training=True)
    validation_dataset = build_dataset(config, split="validation", training=False)
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
                global_batch=protocol_batch.actual_global_batch_size,
                amp_mode=amp_mode,
            ),
        )
        writer = SummaryWriter(run_directory / "tensorboard")

    accumulation = int(config.training.gradient_accumulation_steps)
    epochs = int(config.training.epochs)
    show_progress = context.is_main_process
    stage_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        sample_count = 0
        train_progress = _progress_bar(
            train_loader,
            description=f"Stage 3 train {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        for batch_index, batch in enumerate(train_progress):
            batch = batch.to(context.device)
            with torch.autocast(
                device_type=context.device.type,
                dtype=autocast_dtype,
                enabled=amp_mode != "none",
            ):
                output = training_model(batch.images)
                loss = detection_loss(output.logits, batch.labels)
                if baseline == "semtrace" and bool(
                    config.model_options.use_separation_loss
                ):
                    loss = loss + float(
                        config.separation_loss.weight
                    ) * semantic_trace_separation_loss(
                        output.semantic_anchor,
                        output.trace_evidence,
                        margin=float(config.separation_loss.margin),
                        eps=float(config.separation_loss.eps),
                    )
            scaler.scale(loss / accumulation).backward()
            batch_count += 1
            if show_progress:
                running_loss += float(loss.detach())
                sample_count += int(batch.labels.shape[0])
                _update_progress(
                    train_progress,
                    samples=sample_count,
                    running_loss=running_loss / batch_count,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in detector.parameters() if parameter.requires_grad],
                    float(config.training.gradient_clip_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        validation_metrics, predictions = evaluate_loader(
            training_model,
            validation_loader,
            context.device,
            threshold=float(config.evaluation.threshold),
            amp_mode=amp_mode,
            description=f"Stage 3 validation {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        validation_accuracy_value = validation_metrics["accuracy"]
        validation_ap_value = validation_metrics["average_precision"]
        if not isinstance(validation_accuracy_value, (int, float)):
            raise TypeError("accuracy metric must be numeric")
        if not isinstance(validation_ap_value, (int, float)):
            raise TypeError("average_precision metric must be numeric")
        validation_accuracy = float(validation_accuracy_value)
        validation_ap = float(validation_ap_value)
        if context.is_main_process and run_directory is not None:
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": running_loss / max(batch_count, 1),
                **validation_metrics,
            }
            append_metrics(run_directory, metrics)
            if writer is not None:
                writer.add_scalar(
                    "loss/train",
                    running_loss / max(batch_count, 1),
                    epoch + 1,
                )
                writer.add_scalar("validation/AP", validation_ap, epoch + 1)
            _write_jsonl(
                run_directory / "predictions" / f"validation_epoch_{epoch + 1}.jsonl",
                predictions,
            )
            config_payload = OmegaConf.to_container(config, resolve=True)
            if not isinstance(config_payload, dict):
                raise TypeError("resolved training config must be a mapping")
            checkpoint_config = cast(dict[str, Any], config_payload)
            save_training_checkpoint(
                run_directory / "checkpoints" / "semtrace_last.pt",
                model=detector,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=checkpoint_config,
                selected_layers=selected_layers,
                manifest_hash=train_manifest_hash,
                best_validation_metric=max(best_ap, validation_ap),
                scaler=scaler,
            )
            if validation_ap > best_ap:
                best_ap = validation_ap
                save_training_checkpoint(
                    run_directory / "checkpoints" / "semtrace_best.pt",
                    model=detector,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    config=checkpoint_config,
                    selected_layers=selected_layers,
                    manifest_hash=train_manifest_hash,
                    best_validation_metric=best_ap,
                    scaler=scaler,
                )
            _write_epoch_summary(
                epoch=epoch + 1,
                epochs=epochs,
                completed_this_run=epoch - start_epoch + 1,
                train_loss=running_loss / max(batch_count, 1),
                validation_accuracy=validation_accuracy,
                validation_ap=validation_ap,
                epoch_seconds=time.perf_counter() - epoch_started,
                elapsed_seconds=time.perf_counter() - stage_started,
                show_progress=show_progress,
            )
    if writer is not None:
        writer.close()
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


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
    learning_rate: float,
) -> None:
    progress.set_postfix(
        samples=samples,
        loss=f"{running_loss:.4f}",
        lr=f"{learning_rate:.2e}",
    )


def _write_epoch_summary(
    *,
    epoch: int,
    epochs: int,
    completed_this_run: int,
    train_loss: float,
    validation_accuracy: float,
    validation_ap: float,
    epoch_seconds: float,
    elapsed_seconds: float,
    show_progress: bool,
) -> None:
    if not show_progress:
        return
    remaining_seconds = elapsed_seconds / completed_this_run * (epochs - epoch)
    tqdm.write(
        f"[Stage 3] Epoch {epoch}/{epochs} complete: "
        f"train_loss={train_loss:.4f}, accuracy={validation_accuracy:.4f}, "
        f"AP={validation_ap:.4f}, epoch_time={_format_duration(epoch_seconds)}, "
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


if __name__ == "__main__":
    sys.exit(main())
