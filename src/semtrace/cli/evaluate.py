from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf

from semtrace.config import parse_config_args
from semtrace.engine.checkpoint import load_training_checkpoint
from semtrace.engine.distributed import initialize_distributed, select_amp_mode
from semtrace.engine.evaluator import evaluate_loader
from semtrace.metrics.binary import grouped_binary_metrics
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
        description="Evaluate a SemTrace checkpoint with per-generator metrics.",
        default_config_name="eval",
    )
    if config.checkpoint is None:
        raise ValueError("checkpoint must point to a trained SemTrace checkpoint")
    baseline = str(config.model_options.baseline)
    context = initialize_distributed(str(config.distributed.backend))
    seed_everything(int(config.seed) + context.rank)
    selected = read_selected_layers(config.probe.selected_layers_path)
    selected_layers = selected[: int(config.model_options.num_trace_scales)]
    amp_mode = select_amp_mode(str(config.evaluation.amp), context.device)
    backbone = build_backbone(config, selected_layers).to(context.device)
    semantic_anchor = load_semantic_anchor(config, backbone.hidden_size).to(context.device)
    use_normal = baseline == "semtrace" and bool(
        config.model_options.use_normal_predictor
    )
    predictors = None
    if use_normal:
        if config.normal.checkpoint is None:
            raise ValueError("normal.checkpoint is required when normal prediction is enabled")
        collection = build_normal_predictor_collection(
            config,
            selected,
            backbone.hidden_size,
        )
        load_training_checkpoint(
            config.normal.checkpoint,
            model=collection,
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
    load_training_checkpoint(
        config.checkpoint,
        model=detector,
        restore_random_state=False,
    )

    configured_manifests = OmegaConf.to_container(config.data.test_manifests, resolve=True)
    if not isinstance(configured_manifests, dict) or not configured_manifests:
        configured_manifests = {"validation": config.data.validation_manifest}
    all_predictions: list[dict[str, Any]] = []
    residual_distributions: dict[str, object] = {}
    for domain, manifest in configured_manifests.items():
        if not isinstance(domain, str):
            raise TypeError("test manifest names must be strings")
        if manifest is None:
            raise ValueError(f"test manifest for '{domain}' is not configured")
        if not isinstance(manifest, str):
            raise TypeError(f"test manifest path for '{domain}' must be a string")
        dataset = build_dataset(
            config,
            split=None,
            training=False,
            manifest_path=manifest,
        )
        loader, _ = build_loader(
            dataset,
            batch_size=int(config.evaluation.per_device_batch_size),
            num_workers=int(config.evaluation.num_workers),
            training=False,
            world_size=context.world_size,
            rank=context.rank,
        )
        metrics, predictions = evaluate_loader(
            detector,
            loader,
            context.device,
            threshold=float(config.evaluation.threshold),
            amp_mode=amp_mode,
        )
        residual_distributions[str(domain)] = metrics["residual_distributions"]
        all_predictions.extend(predictions)

    grouped = grouped_binary_metrics(
        [int(row["label"]) for row in all_predictions],
        [float(row["fake_probability"]) for row in all_predictions],
        [str(row["generator"]) for row in all_predictions],
        threshold=float(config.evaluation.threshold),
    )
    results = {
        "accuracy": grouped.overall.accuracy,
        "average_precision": grouped.overall.average_precision,
        "auroc": grouped.overall.auroc,
        "false_positive_rate": grouped.overall.false_positive_rate,
        "false_positive_rate_at_95_tpr": grouped.overall.false_positive_rate_at_95_tpr,
        "real_accuracy": grouped.overall.real_accuracy,
        "fake_accuracy": grouped.overall.fake_accuracy,
        "per_generator": {
            name: {
                "accuracy": metrics.accuracy,
                "average_precision": metrics.average_precision,
            }
            for name, metrics in grouped.per_domain.items()
        },
        "mAcc": grouped.mean_accuracy,
        "mAP": grouped.mean_average_precision,
        "residual_distributions": residual_distributions,
    }
    if context.is_main_process:
        run_directory = create_run_directory(config.output_root, config.experiment)
        write_resolved_config(config, run_directory)
        (run_directory / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
        write_environment(
            run_directory / "environment.json",
            environment_record(
                model_id=str(config.model.model_id),
                model_revision=str(config.model.revision),
                seed=int(config.seed),
                global_batch=int(config.evaluation.per_device_batch_size)
                * context.world_size,
                amp_mode=amp_mode,
            ),
        )
        append_metrics(run_directory, results)
        with (run_directory / "predictions" / "predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as file:
            for prediction in all_predictions:
                file.write(json.dumps(prediction, sort_keys=True) + "\n")
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
