from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
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
    model_source = str(config.model.model_path or config.model.model_id)
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
    train_features = extract_probe_features(backbone, train_loader, context.device)
    validation_features = extract_probe_features(backbone, validation_loader, context.device)
    if context.is_main_process:
        train_split, validation_split, nuisance_name, generator_enabled, coverage = (
            prepare_probe_splits(train_features, validation_features)
        )
        metrics = fit_layer_probes(
            train_split,
            validation_split,
            seed=int(config.seed),
            max_iter=max(100, int(config.probe.epochs) * 10),
        )
        selection = select_probe_layers(
            metrics,
            num_hidden_layers=num_layers,
            alpha=float(config.probe.alpha),
            beta=float(config.probe.beta),
        )
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
        (run_directory / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
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
    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
