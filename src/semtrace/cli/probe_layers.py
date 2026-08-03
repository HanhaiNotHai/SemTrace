from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator, Sequence, Sized
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoConfig

from semtrace.config import parse_config_args
from semtrace.engine.distributed import initialize_distributed
from semtrace.engine.probe_engine import extract_probe_features, prepare_probe_splits
from semtrace.models.probes import fit_layer_probes, save_probe_artifacts, select_probe_layers
from semtrace.runtime import build_backbone, build_dataset, build_loader, build_semantic_anchor
from semtrace.utils.environment import environment_record, git_commit, write_environment
from semtrace.utils.logging import create_run_directory, write_resolved_config
from semtrace.utils.seed import seed_everything


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config_args(
        argv,
        description="Fit frozen-feature probes and select three DINOv3 layers.",
        default_config_name="stage1_probe",
    )
    context = initialize_distributed(str(config.distributed.backend))
    seed_everything(int(config.seed) + context.rank)
    show_progress = context.is_main_process
    total_started = perf_counter()
    model_source = str(config.model.model_path or config.model.model_id)
    with _timed_stage("1/6 setup and frozen DINOv3 loading", show_progress):
        model_config = AutoConfig.from_pretrained(
            model_source,
            local_files_only=bool(config.model.local_files_only),
        )
        num_layers = int(model_config.num_hidden_layers)
        candidates = (
            tuple(int(layer) for layer in config.probe.candidate_layers)
            if config.probe.candidate_layers is not None
            else tuple(range(num_layers - 1))
        )
        backbone = build_backbone(config, candidates).to(context.device)
        train_dataset = build_dataset(config, split="train", training=False)
        validation_dataset = build_dataset(config, split="validation", training=False)
        train_loader, _ = build_loader(
            train_dataset,
            batch_size=int(config.probe.batch_size),
            num_workers=int(config.training.num_workers),
            training=False,
            world_size=context.world_size,
            rank=context.rank,
        )
        validation_loader, _ = build_loader(
            validation_dataset,
            batch_size=int(config.probe.batch_size),
            num_workers=int(config.training.num_workers),
            training=False,
            world_size=context.world_size,
            rank=context.rank,
        )
        if show_progress:
            tqdm.write(
                f"[Stage 1] world_size={context.world_size}, "
                f"per_device_batch={int(config.probe.batch_size)}, "
                f"train_samples={_dataset_size(train_dataset)}, "
                f"validation_samples={_dataset_size(validation_dataset)}, "
                f"candidate_layers={list(candidates)}"
            )
    with _timed_stage("2/6 training feature extraction", show_progress):
        train_features = extract_probe_features(
            backbone,
            train_loader,
            context.device,
            description="Train features",
            show_progress=show_progress,
        )
    with _timed_stage("3/6 validation feature extraction", show_progress):
        validation_features = extract_probe_features(
            backbone,
            validation_loader,
            context.device,
            description="Validation features",
            show_progress=show_progress,
        )
    if context.is_main_process:
        with _timed_stage("4/6 linear probe fitting", show_progress):
            train_split, validation_split, nuisance_name, generator_enabled, coverage = (
                prepare_probe_splits(train_features, validation_features)
            )
            tqdm.write(
                f"[Stage 1] semantic_label_coverage={coverage:.1%}, "
                f"nuisance_probe={nuisance_name}, "
                f"generator_probe_enabled={generator_enabled}"
            )
            metrics = fit_layer_probes(
                train_split,
                validation_split,
                seed=int(config.seed),
                max_iter=max(100, int(config.probe.epochs) * 10),
                show_progress=show_progress,
            )
        with _timed_stage("5/6 layer scoring and selection", show_progress):
            selection = select_probe_layers(
                metrics,
                num_hidden_layers=num_layers,
                alpha=float(config.probe.alpha),
                beta=float(config.probe.beta),
            )
            tqdm.write(f"[Stage 1] selected_layers={list(selection.selected_layers)}")
        with _timed_stage("6/6 artifact and run-record writing", show_progress):
            output_dir = Path(str(config.probe.output_dir))
            save_probe_artifacts(
                selection,
                output_dir,
                model_id=str(config.model.model_id),
                model_revision=str(config.model.revision),
                semantic_label_coverage=coverage,
                nuisance_label=nuisance_name,
                generator_probe_enabled=generator_enabled,
            )
            anchor = build_semantic_anchor(config, backbone.hidden_size)
            semantic_anchor_path = Path(str(config.probe.semantic_anchor_path))
            semantic_anchor_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(anchor.state_dict(), semantic_anchor_path)
            run_directory = create_run_directory(config.output_root, config.experiment)
            write_resolved_config(config, run_directory)
            shutil.copy2(output_dir / "selected_layers.json", run_directory)
            (run_directory / "git_commit.txt").write_text(
                git_commit() + "\n",
                encoding="utf-8",
            )
            write_environment(
                run_directory / "environment.json",
                environment_record(
                    model_id=str(config.model.model_id),
                    model_revision=str(config.model.revision),
                    seed=int(config.seed),
                    global_batch=int(config.probe.batch_size) * context.world_size,
                    amp_mode=str(config.training.amp),
                ),
            )
        print(
            json.dumps(
                {
                    "selected_layers": selection.selected_layers,
                    "nuisance_probe": nuisance_name,
                    "generator_probe_enabled": generator_enabled,
                }
            )
        )
        tqdm.write(
            f"[Stage 1] All stages complete in "
            f"{_format_duration(perf_counter() - total_started)}"
        )
    if dist.is_initialized():
        dist.barrier()
    return 0


@contextmanager
def _timed_stage(name: str, enabled: bool) -> Iterator[None]:
    started = perf_counter()
    if enabled:
        tqdm.write(f"[Stage 1] Starting {name}...")
    try:
        yield
    except Exception:
        if enabled:
            tqdm.write(
                f"[Stage 1] Failed {name} after "
                f"{_format_duration(perf_counter() - started)}"
            )
        raise
    if enabled:
        tqdm.write(
            f"[Stage 1] Completed {name} in "
            f"{_format_duration(perf_counter() - started)}"
        )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m"


def _dataset_size(dataset: object) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("Stage 1 datasets must expose a finite length")
    return len(dataset)


if __name__ == "__main__":
    sys.exit(main())
