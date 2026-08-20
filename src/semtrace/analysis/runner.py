from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from semtrace.analysis.common import cache_fingerprint, extract_feature_cache
from semtrace.analysis.diagnostics import LiveDiagnosticCollector
from semtrace.analysis.feature_cache import FeatureCacheReader, FeatureCacheWriter
from semtrace.analysis.report import generate_mechanism_report
from semtrace.analysis.visualize_trace_maps import FullImageResizeTransform
from semtrace.config import parse_config_args
from semtrace.data.manifest import ManifestImageDataset
from semtrace.engine.checkpoint import load_training_checkpoint
from semtrace.engine.distributed import initialize_distributed, select_amp_mode
from semtrace.models.semtrace import SemTrace
from semtrace.runtime import (
    build_backbone,
    build_dataset,
    build_detection_model,
    build_loader,
    build_normal_predictor_collection,
    load_semantic_anchor,
    read_selected_layers,
)
from semtrace.utils.environment import environment_record, write_environment
from semtrace.utils.logging import write_resolved_config
from semtrace.utils.seed import seed_everything


def run_analysis_cli(
    task: str,
    argv: Sequence[str] | None = None,
    *,
    default_config_name: str = "analysis/mechanism_base",
) -> int:
    config = parse_config_args(
        argv,
        description=f"Run SemTrace mechanism analysis: {task}.",
        default_config_name=default_config_name,
    )
    context = initialize_distributed(str(config.distributed.backend))
    seed_everything(int(config.seed) + context.rank)
    manifests = _configured_manifests(config)
    if config.checkpoint is None:
        raise ValueError("checkpoint must point to a trained SemTrace checkpoint")
    fingerprint = cache_fingerprint(str(config.checkpoint), config, manifests.values())
    run_directory = _run_directory(config, task, context.rank)
    cache_setting = config.analysis.cache
    cache_root = Path(str(cache_setting)) if cache_setting else run_directory / "feature_cache"

    if cache_setting:
        FeatureCacheReader(cache_root, expected_fingerprint=fingerprint)
        collector = None
    else:
        model, amp_mode = _load_model(config, context.device)
        collector = LiveDiagnosticCollector(
            output_root=run_directory,
            rank=context.rank,
            tasks=_live_tasks(task),
            patch_mask_ratios=tuple(
                float(ratio) for ratio in config.analysis.patch_mask_ratios
            ),
            random_seeds=tuple(int(seed) for seed in config.analysis.random_seeds),
            visualization_limit=int(config.analysis.visualization_limit),
        )
        writer = FeatureCacheWriter(
            cache_root,
            fingerprint,
            dtype=_cache_dtype(str(config.analysis.cache_dtype)),
            rank=context.rank,
        )
        for manifest in manifests.values():
            dataset = _build_analysis_dataset(config, manifest, task=task)
            loader, _ = build_loader(
                dataset,
                batch_size=int(config.analysis.feature_batch_size),
                num_workers=int(config.analysis.num_workers),
                training=False,
                world_size=context.world_size,
                rank=context.rank,
            )
            extract_feature_cache(
                model,
                loader,
                context.device,
                writer,
                amp_mode=amp_mode,
                show_progress=context.is_main_process,
                max_samples_per_group=_max_samples(config),
                batch_callback=collector.process,
            )
        collector.save_rank()
        if dist.is_initialized():
            dist.barrier()
        if context.is_main_process:
            writer.finalize()
        if dist.is_initialized():
            dist.barrier()

    if context.is_main_process:
        write_resolved_config(config, run_directory)
        amp_mode = select_amp_mode(str(config.analysis.amp), context.device)
        write_environment(
            run_directory / "environment.json",
            environment_record(
                model_id=str(config.model.model_id),
                model_revision=str(config.model.revision),
                seed=int(config.seed),
                global_batch=int(config.analysis.feature_batch_size) * context.world_size,
                amp_mode=amp_mode,
            ),
        )
        summary = generate_mechanism_report(
            cache_root,
            run_directory,
            bootstrap_iterations=_bootstrap_iterations(config),
            random_seeds=tuple(int(seed) for seed in config.analysis.random_seeds),
            prototype_counts=_prototype_counts(config),
            top_r=int(config.analysis.top_r),
        )
        if collector is not None:
            summary["live_diagnostics"] = collector.finalize(
                bootstrap_iterations=_bootstrap_iterations(config)
            )
            _append_live_report(run_directory, summary["live_diagnostics"])
        summary["requested_task"] = task
        (run_directory / "mechanism_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(run_directory), **summary}, indent=2))
    return 0


def _load_model(config: Any, device: torch.device) -> tuple[SemTrace, str]:
    selected = read_selected_layers(str(config.probe.selected_layers_path))
    selected_layers = selected[: int(config.model_options.num_trace_scales)]
    backbone = build_backbone(config, selected_layers).to(device)
    semantic_anchor = load_semantic_anchor(config, backbone.hidden_size).to(device)
    predictors = None
    if bool(config.model_options.use_normal_predictor):
        if config.normal.checkpoint is None:
            raise ValueError("normal.checkpoint is required for SemTrace analysis")
        collection = build_normal_predictor_collection(config, selected, backbone.hidden_size)
        load_training_checkpoint(
            str(config.normal.checkpoint), model=collection, restore_random_state=False
        )
        predictors = collection.normal_predictors
    built = build_detection_model(
        config,
        backbone=backbone,
        semantic_anchor=semantic_anchor,
        selected_layers=selected_layers,
        normal_predictors=predictors,
    ).to(device)
    if not isinstance(built, SemTrace):
        raise TypeError("mechanism analysis requires model_options.baseline=semtrace")
    load_training_checkpoint(str(config.checkpoint), model=built, restore_random_state=False)
    built.eval()
    return built, select_amp_mode(str(config.analysis.amp), device)


def _configured_manifests(config: Any) -> dict[str, str]:
    configured = OmegaConf.to_container(config.data.test_manifests, resolve=True)
    if not isinstance(configured, dict) or not configured:
        configured = {"validation": config.data.validation_manifest}
    manifests: dict[str, str] = {}
    for name, path in configured.items():
        if path is None:
            raise ValueError(f"analysis manifest '{str(name)}' is not configured")
        manifests[str(name)] = str(path)
    return manifests


def _build_analysis_dataset(
    config: Any,
    manifest: str,
    *,
    task: str,
) -> torch.utils.data.Dataset[Any]:
    if task == "visualization":
        return ManifestImageDataset(
            manifest,
            FullImageResizeTransform(int(config.visualization.resize_size)),
            split=None,
            data_root=config.data.root,
        )
    return build_dataset(
        config,
        split=None,
        training=False,
        manifest_path=manifest,
    )


def _run_directory(config: Any, task: str, rank: int) -> Path:
    configured = config.analysis.output_dir
    if configured:
        path = Path(str(configured))
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        local = str(
            Path(str(config.analysis.output_root))
            / str(config.analysis.experiment_name)
            / Path(str(config.checkpoint)).stem
            / str(config.protocol.name)
            / timestamp
        )
        values = [local]
        if dist.is_initialized():
            dist.broadcast_object_list(values, src=0)
        path = Path(values[0])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError("analysis.cache_dtype must be float16 or float32")


def _live_tasks(task: str) -> frozenset[str]:
    if task == "mechanism_suite":
        return frozenset(
            {"masking", "semantic_counterfactual", "cross_attention", "visualization"}
        )
    if task == "normal_predictor":
        return frozenset({"semantic_counterfactual"})
    if task in {
        "masking",
        "semantic_counterfactual",
        "cross_attention",
        "visualization",
    }:
        return frozenset({task})
    return frozenset()


def _max_samples(config: Any) -> int:
    configured = int(config.analysis.max_samples_per_group)
    return min(configured, 32) if bool(config.analysis.quick) else configured


def _bootstrap_iterations(config: Any) -> int:
    configured = int(config.analysis.bootstrap_iterations)
    return min(configured, 100) if bool(config.analysis.quick) else configured


def _prototype_counts(config: Any) -> tuple[int, ...]:
    if bool(config.analysis.quick):
        return (16,)
    return tuple(int(count) for count in config.analysis.prototype_counts)


def _append_live_report(run_directory: Path, diagnostics: object) -> None:
    report = run_directory / "mechanism_report.md"
    with report.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Live inference diagnostics\n\n"
            f"Summary: `{json.dumps(diagnostics, sort_keys=True)}`. Detailed masking, semantic "
            "counterfactual, normal-predictor intervention, attention-stability, and per-sample "
            "prediction records are stored under `tables/` and `predictions/`. All conditions "
            "reuse the reported baseline-global threshold.\n"
        )
