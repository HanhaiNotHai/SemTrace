from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from semtrace.analysis.cross_attention_analysis import attention_stability
from semtrace.analysis.scale_masking import PatchMaskStrategy, mask_patches, mask_scales
from semtrace.analysis.semantic_counterfactual import (
    grouped_permutation,
    recompute_analysis_path,
)
from semtrace.analysis.statistics import summarize_distribution
from semtrace.analysis.visualize_trace_maps import patch_map, resize_patch_map, save_trace_map
from semtrace.data.collate import ImageBatch
from semtrace.metrics.binary import grouped_binary_metrics, optimal_accuracy_threshold
from semtrace.models.semtrace import SemTrace, SemTraceAnalysisOutput


@dataclass(slots=True)
class LiveDiagnosticCollector:
    output_root: Path
    rank: int
    tasks: frozenset[str]
    patch_mask_ratios: tuple[float, ...]
    random_seeds: tuple[int, ...]
    visualization_limit: int = 50
    rows: list[dict[str, Any]] = field(init=False, default_factory=list)
    attention_rows: list[dict[str, Any]] = field(init=False, default_factory=list)
    visualized: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def process(
        self,
        model: SemTrace,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
    ) -> None:
        self._record("baseline", "baseline", batch, baseline, baseline)
        if "masking" in self.tasks:
            self._masking(model, batch, baseline)
        if "semantic_counterfactual" in self.tasks:
            self._counterfactuals(model, batch, baseline)
        if "cross_attention" in self.tasks:
            self._attention_stability(model, batch, baseline)
        if "visualization" in self.tasks and self.rank == 0:
            self._visualize(batch, baseline)

    def save_rank(self) -> Path:
        path = self.output_root / f"diagnostic_rank{self.rank:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        attention_path = self.output_root / f"attention_stability_rank{self.rank:03d}.jsonl"
        with attention_path.open("w", encoding="utf-8") as handle:
            for row in self.attention_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    def finalize(self, *, bootstrap_iterations: int) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.output_root.glob("diagnostic_rank*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        baseline_rows = [row for row in rows if row["condition"] == "baseline"]
        if not baseline_rows:
            return {"skipped_reason": "no live diagnostic samples"}
        labels = np.asarray([row["label"] for row in baseline_rows])
        probabilities = np.asarray([row["probability"] for row in baseline_rows])
        threshold = optimal_accuracy_threshold(labels, probabilities, reference_threshold=0.5)
        result_rows: list[dict[str, Any]] = []
        baseline_by_id = {row["sample_id"]: row for row in baseline_rows}
        for (kind, condition), members in pd.DataFrame(rows).groupby(["kind", "condition"]):
            records = members.to_dict("records")
            targets = [int(row["label"]) for row in records]
            scores = [float(row["probability"]) for row in records]
            generators = [str(row["generator"]) for row in records]
            metrics = grouped_binary_metrics(targets, scores, generators, threshold=threshold)
            changes = np.asarray(
                [
                    abs(float(row["logit"]) - float(baseline_by_id[row["sample_id"]]["logit"]))
                    for row in records
                ]
            )
            flips = np.asarray(
                [
                    (float(row["probability"]) >= threshold)
                    != (float(baseline_by_id[row["sample_id"]]["probability"]) >= threshold)
                    for row in records
                ],
                dtype=np.float64,
            )
            interval = summarize_distribution(
                changes,
                bootstrap_iterations=bootstrap_iterations,
                seed=self.random_seeds[0],
            )
            donor_rows = [row for row in records if row.get("donor_label") is not None]
            follow = (
                float(
                    np.mean(
                        [
                            (float(row["probability"]) >= threshold)
                            == bool(row["donor_label"])
                            for row in donor_rows
                        ]
                    )
                )
                if donor_rows
                else float("nan")
            )
            result_rows.append(
                {
                    "kind": kind,
                    "condition": condition,
                    "global_baseline_threshold": threshold,
                    "accuracy": metrics.overall.accuracy,
                    "average_precision": metrics.overall.average_precision,
                    "auroc": metrics.overall.auroc,
                    "real_accuracy": metrics.overall.real_accuracy,
                    "fake_accuracy": metrics.overall.fake_accuracy,
                    "mAcc": metrics.mean_accuracy,
                    "mAP": metrics.mean_average_precision,
                    "per_generator": json.dumps(
                        {
                            name: {
                                "accuracy": value.accuracy,
                                "average_precision": value.average_precision,
                                "auroc": value.auroc,
                            }
                            for name, value in metrics.per_domain.items()
                        },
                        sort_keys=True,
                    ),
                    "prediction_flip_rate": float(flips.mean()),
                    "mean_absolute_logit_change": float(changes.mean()),
                    "logit_change_ci_low": interval.confidence_interval_low,
                    "logit_change_ci_high": interval.confidence_interval_high,
                    "trace_following_rate": follow,
                    "sample_count": len(records),
                }
            )
        frame = pd.DataFrame(result_rows)
        baseline_metrics = frame[frame["condition"] == "baseline"].iloc[0]
        for metric in ("accuracy", "average_precision", "auroc"):
            frame[f"{metric}_drop"] = float(baseline_metrics[metric]) - frame[metric]
        tables = self.output_root / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        prediction_directory = self.output_root / "predictions"
        prediction_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            prediction_directory / "intervention_predictions.csv", index=False
        )
        attention_rows: list[dict[str, Any]] = []
        for path in sorted(self.output_root.glob("attention_stability_rank*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                attention_rows.extend(json.loads(line) for line in handle if line.strip())
        pd.DataFrame(attention_rows).to_csv(
            tables / "cross_attention_stability.csv", index=False
        )
        frame[frame["kind"] == "masking"].to_csv(tables / "masking.csv", index=False)
        frame[frame["kind"] == "semantic_counterfactual"].to_csv(
            tables / "semantic_counterfactual.csv", index=False
        )
        frame[frame["kind"] == "normal_predictor_intervention"].to_csv(
            tables / "normal_predictor_interventions.csv", index=False
        )
        return {"global_baseline_threshold": threshold, "condition_count": len(result_rows)}

    def _attention_stability(
        self,
        model: SemTrace,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
    ) -> None:
        if baseline.attention_weights is None:
            return
        image_variants = {
            "jpeg": _jpeg_perturbation(batch.images),
            "scale": _scale_perturbation(batch.images),
            "blur": _blur_perturbation(batch.images),
        }
        for name, images in image_variants.items():
            output = model.analyze(images)
            if output.attention_weights is not None:
                self._record_attention_stability(name, batch, baseline, output)
        for same_group in (True, False):
            permutation = grouped_permutation(
                [int(value) for value in batch.semantic_classes.cpu()],
                same_group=same_group,
                seed=self.random_seeds[0],
            ).to(baseline.semantic_anchor.device)
            output = recompute_analysis_path(
                model,
                baseline,
                cross_attention_semantic=baseline.semantic_anchor[permutation],
            )
            if output.attention_weights is not None:
                self._record_attention_stability(
                    f"{'same' if same_group else 'different'}_semantic_swap",
                    batch,
                    baseline,
                    output,
                )

    def _record_attention_stability(
        self,
        condition: str,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
        output: SemTraceAnalysisOutput,
    ) -> None:
        if baseline.attention_weights is None or output.attention_weights is None:
            return
        metrics = attention_stability(
            baseline.attention_weights,
            output.attention_weights,
        )
        arrays = {name: values.float().cpu().numpy() for name, values in metrics.items()}
        for sample_index, path in enumerate(batch.paths):
            for head in range(baseline.attention_weights.shape[1]):
                self.attention_rows.append(
                    {
                        "path": path,
                        "condition": condition,
                        "head": head,
                        "label": int(batch.labels[sample_index].cpu()),
                        "generator": batch.generators[sample_index],
                        "content_env": batch.content_envs[sample_index],
                        **{
                            name: float(values[sample_index, head])
                            for name, values in arrays.items()
                        },
                    }
                )

    def _masking(
        self,
        model: SemTrace,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
    ) -> None:
        layers = baseline.selected_layers
        for layer in layers:
            residual_replacements = mask_scales(
                baseline.candidate_trace_residuals, masked_layers={layer}
            )
            output = recompute_analysis_path(
                model, baseline, residual_replacements=residual_replacements
            )
            self._record(
                "masking", f"mask_scale_L{layer}_after_residual", batch, baseline, output
            )
            replacements = mask_scales(
                baseline.adapted_trace_tokens, masked_layers={layer}
            )
            output = recompute_analysis_path(
                model, baseline, adapted_replacements=replacements
            )
            self._record("masking", f"mask_scale_L{layer}_after_adapter", batch, baseline, output)
            keep = set(layers) - {layer}
            keep_replacements = mask_scales(
                baseline.adapted_trace_tokens, masked_layers=keep
            )
            output = recompute_analysis_path(
                model, baseline, adapted_replacements=keep_replacements
            )
            self._record("masking", f"keep_only_scale_L{layer}", batch, baseline, output)

        strengths = torch.stack(
            [
                baseline.adapted_trace_tokens[layer].norm(dim=-1).mean(dim=1)
                for layer in layers
            ],
            dim=1,
        )
        for name, positions in (
            ("strongest", strengths.argmax(dim=1)),
            ("weakest", strengths.argmin(dim=1)),
        ):
            replacements = {
                layer: baseline.adapted_trace_tokens[layer].clone() for layer in layers
            }
            for position, layer in enumerate(layers):
                replacements[layer][positions == position] = 0
            output = recompute_analysis_path(
                model, baseline, adapted_replacements=replacements
            )
            self._record("masking", f"mask_{name}_sample_scale", batch, baseline, output)
        for seed in self.random_seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            positions = torch.randint(0, len(layers), (batch.images.shape[0],), generator=generator)
            replacements = {
                layer: baseline.adapted_trace_tokens[layer].clone() for layer in layers
            }
            for position, layer in enumerate(layers):
                replacements[layer][positions.to(replacements[layer].device) == position] = 0
            output = recompute_analysis_path(
                model, baseline, adapted_replacements=replacements
            )
            self._record("masking", "mask_random_sample_scale", batch, baseline, output)

        residual_scores = torch.stack(
            [value.norm(dim=-1) for value in baseline.candidate_trace_residuals.values()]
        ).mean(dim=0)
        attention_scores = (
            baseline.attention_weights.mean(dim=1)[:, 0]
            if baseline.attention_weights is not None
            else residual_scores
        )
        for ratio in self.patch_mask_ratios:
            strategies = (
                ("random", None),
                ("top_residual", residual_scores),
                ("top_attention", attention_scores),
                ("low_attention", attention_scores),
                ("center", None),
                ("edge", None),
            )
            for name, scores in strategies:
                strategy = (
                    "top" if name.startswith("top") else "low" if name.startswith("low") else name
                )
                seeds = self.random_seeds if name == "random" else self.random_seeds[:1]
                for seed in seeds:
                    masked, _ = mask_patches(
                        baseline.fused_trace_tokens,
                        ratio=ratio,
                        strategy=cast(PatchMaskStrategy, strategy),
                        seed=seed,
                        scores=scores,
                        patch_grid_size=baseline.patch_grid_size,
                    )
                    output = recompute_analysis_path(
                        model, baseline, fused_trace_replacement=masked
                    )
                    self._record(
                        "masking", f"{name}_{ratio:.2f}", batch, baseline, output
                    )
        if baseline.attention_weights is not None and model.cross_attention is not None:
            for head in range(baseline.attention_weights.shape[1]):
                head_keep = torch.ones(
                    baseline.attention_weights.shape[1], device=baseline.logits.device
                )
                head_keep[head] = 0
                output = recompute_analysis_path(
                    model, baseline, head_keep_mask=head_keep
                )
                self._record("masking", f"mask_attention_head_{head}", batch, baseline, output)
            entropy = -(
                baseline.attention_weights[:, :, 0]
                * baseline.attention_weights[:, :, 0].clamp_min(1.0e-12).log()
            ).sum(dim=-1).mean(dim=0)
            concentrated = int(entropy.argmin())
            for seed in self.random_seeds:
                random_head = int(
                    torch.randint(
                        0,
                        baseline.attention_weights.shape[1],
                        (1,),
                        generator=torch.Generator().manual_seed(seed),
                    )
                )
                for name, head in (
                    ("top_concentrated", concentrated),
                    ("random", random_head),
                ):
                    head_keep = torch.ones(
                        baseline.attention_weights.shape[1],
                        device=baseline.logits.device,
                    )
                    head_keep[head] = 0
                    output = recompute_analysis_path(
                        model, baseline, head_keep_mask=head_keep
                    )
                    self._record(
                        "masking", f"mask_{name}_attention_head", batch, baseline, output
                    )

    def _counterfactuals(
        self,
        model: SemTrace,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
    ) -> None:
        semantic = baseline.semantic_anchor
        variants = {
            "zero": torch.zeros_like(semantic),
            "gaussian": torch.randn_like(semantic),
            "batch_permutation": semantic[torch.arange(semantic.shape[0] - 1, -1, -1)],
        }
        groupings: dict[str, list[object]] = {
            "authenticity": [int(value) for value in batch.labels.cpu()],
            "semantic_class": [int(value) for value in batch.semantic_classes.cpu()],
            "content_env": cast(list[object], batch.content_envs),
        }
        for name, groups in groupings.items():
            for same_group in (True, False):
                permutation = grouped_permutation(
                    list(groups), same_group=same_group, seed=self.random_seeds[0]
                ).to(semantic.device)
                variants[f"{'same' if same_group else 'different'}_{name}"] = semantic[
                    permutation
                ]
        for name, replacement in variants.items():
            for target in ("normal", "cross_attention", "both"):
                output = recompute_analysis_path(
                    model,
                    baseline,
                    normal_semantic=replacement if target in {"normal", "both"} else None,
                    cross_attention_semantic=(
                        replacement if target in {"cross_attention", "both"} else None
                    ),
                )
                self._record(
                    "semantic_counterfactual",
                    f"{name}_{target}",
                    batch,
                    baseline,
                    output,
                )
        for name, use_semantic, use_neighbors in (
            ("zero_semantic", False, True),
            ("remove_neighborhood", True, False),
            ("local_only", False, True),
            ("semantic_only", True, False),
        ):
            output = recompute_analysis_path(
                model,
                baseline,
                use_normal_semantic=use_semantic,
                use_neighbors=use_neighbors,
            )
            self._record(
                "normal_predictor_intervention", name, batch, baseline, output
            )
        labels = [int(value) for value in batch.labels.cpu()]
        swap_permutations = {
            "real_fake": _matching_permutation(
                labels,
                [0] * len(labels),
                same_primary=False,
                same_secondary=True,
                seed=self.random_seeds[0],
            ),
            "same_semantic_real_fake": _matching_permutation(
                [int(value) for value in batch.semantic_classes.cpu()],
                labels,
                same_primary=True,
                same_secondary=False,
                seed=self.random_seeds[0],
            ),
            "same_generator": grouped_permutation(
                list(batch.generators), same_group=True, seed=self.random_seeds[0]
            ),
            "different_generator": grouped_permutation(
                list(batch.generators), same_group=False, seed=self.random_seeds[0]
            ),
            "same_content_env": grouped_permutation(
                cast(list[object], batch.content_envs),
                same_group=True,
                seed=self.random_seeds[0],
            ),
        }
        for name, cpu_permutation in swap_permutations.items():
            permutation = cpu_permutation.to(semantic.device)
            donor_labels = batch.labels[permutation]
            for level in ("residual", "adapted", "fused"):
                output = recompute_analysis_path(
                    model,
                    baseline,
                    residual_replacements=(
                        {
                            layer: values[permutation]
                            for layer, values in baseline.candidate_trace_residuals.items()
                        }
                        if level == "residual"
                        else None
                    ),
                    adapted_replacements=(
                        {
                            layer: values[permutation]
                            for layer, values in baseline.adapted_trace_tokens.items()
                        }
                        if level == "adapted"
                        else None
                    ),
                    fused_trace_replacement=(
                        baseline.fused_trace_tokens[permutation]
                        if level == "fused"
                        else None
                    ),
                )
                self._record(
                    "semantic_counterfactual",
                    f"{name}_{level}_trace_swap",
                    batch,
                    baseline,
                    output,
                    donor_labels=donor_labels,
                )

    def _visualize(
        self,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
    ) -> None:
        if self.visualized >= self.visualization_limit:
            return
        mean = torch.tensor((0.485, 0.456, 0.406), device=batch.images.device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=batch.images.device)[:, None, None]
        images = (batch.images * std + mean).clamp(0, 1)
        for index in range(images.shape[0]):
            if self.visualized >= self.visualization_limit:
                break
            maps: dict[str, torch.Tensor] = {}
            for layer in baseline.selected_layers:
                maps[f"residual_L{layer}"] = baseline.candidate_trace_residuals[
                    layer
                ].norm(dim=-1)
                maps[f"adapted_L{layer}"] = baseline.adapted_trace_tokens[layer].norm(
                    dim=-1
                )
            fused_strength = baseline.fused_trace_tokens.norm(dim=-1)
            maps["fused_trace"] = fused_strength
            if baseline.attention_weights is not None:
                attention = baseline.attention_weights[:, :, 0]
                for head in range(attention.shape[1]):
                    maps[f"attention_head{head}"] = attention[:, head]
                average_attention = attention.mean(dim=1)
                maps["attention_mean"] = average_attention
                maps["attention_times_trace"] = average_attention * fused_strength
                top_k = max(1, round(fused_strength.shape[1] * 0.1))
                residual_top = torch.zeros_like(fused_strength).scatter(
                    1, fused_strength.topk(top_k, dim=1).indices, 1.0
                )
                attention_top = torch.zeros_like(fused_strength).scatter(
                    1, average_attention.topk(top_k, dim=1).indices, 1.0
                )
                maps["top_residual_patches"] = residual_top
                maps["top_attention_patches"] = attention_top
            image = images[index].permute(1, 2, 0).float().cpu().numpy()
            for name, values in maps.items():
                grid = patch_map(values[index : index + 1], baseline.patch_grid_size)
                resized = resize_patch_map(
                    grid, (int(images.shape[-2]), int(images.shape[-1]))
                )[0]
                heatmap = resized.float().cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / max(
                    float(np.ptp(heatmap)), 1.0e-12
                )
                save_trace_map(
                    image,
                    heatmap,
                    self.output_root
                    / "heatmaps"
                    / f"sample_{self.visualized:05d}_{name}.png",
                    title=(
                        f"label={int(batch.labels[index])} "
                        f"prediction={float(baseline.logits[index].sigmoid()):.3f} "
                        f"generator={batch.generators[index]} "
                        f"semantic={int(batch.semantic_classes[index])} "
                        f"content_env={batch.content_envs[index]} map={name}"
                    ),
                )
            self.visualized += 1

    def _record(
        self,
        kind: str,
        condition: str,
        batch: ImageBatch,
        baseline: SemTraceAnalysisOutput,
        output: SemTraceAnalysisOutput,
        donor_labels: torch.Tensor | None = None,
    ) -> None:
        probabilities = output.logits.float().sigmoid().cpu().tolist()
        logits = output.logits.float().cpu().tolist()
        baseline_logits = baseline.logits.float().cpu().tolist()
        for index, path in enumerate(batch.paths):
            self.rows.append(
                {
                    "sample_id": path,
                    "kind": kind,
                    "condition": condition,
                    "label": int(batch.labels[index].cpu()),
                    "generator": batch.generators[index],
                    "semantic_class": int(batch.semantic_classes[index].cpu()),
                    "content_env": batch.content_envs[index],
                    "real_source": batch.real_sources[index],
                    "source_dataset": batch.source_datasets[index],
                    "degradation": batch.degradations[index],
                    "probability": float(probabilities[index]),
                    "logit": float(logits[index]),
                    "baseline_logit": float(baseline_logits[index]),
                    "donor_label": (
                        int(donor_labels[index].cpu()) if donor_labels is not None else None
                    ),
                }
            )


def _matching_permutation(
    primary: Sequence[object],
    secondary: Sequence[object],
    *,
    same_primary: bool,
    same_secondary: bool,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    donors: list[int] = []
    for index in range(len(primary)):
        candidates = [
            candidate
            for candidate in range(len(primary))
            if candidate != index
            and ((primary[candidate] == primary[index]) == same_primary)
            and ((secondary[candidate] == secondary[index]) == same_secondary)
        ]
        donors.append(int(rng.choice(candidates)) if candidates else index)
    return torch.tensor(donors, dtype=torch.long)


def _pixels(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device)[:, None, None]
    return (images * std + mean).clamp(0, 1)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device)[:, None, None]
    return (images - mean) / std


def _jpeg_perturbation(images: torch.Tensor) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    device = images.device
    for pixels in _pixels(images).cpu():
        array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            values = np.asarray(decoded.convert("RGB"), dtype=np.float32).copy() / 255.0
        encoded.append(torch.from_numpy(values).permute(2, 0, 1))
    return _normalize(torch.stack(encoded).to(device))


def _scale_perturbation(images: torch.Tensor) -> torch.Tensor:
    pixels = _pixels(images)
    height, width = pixels.shape[-2:]
    smaller = F.interpolate(
        pixels,
        size=(max(1, height * 7 // 8), max(1, width * 7 // 8)),
        mode="bilinear",
        align_corners=False,
    )
    restored = F.interpolate(smaller, size=(height, width), mode="bilinear", align_corners=False)
    return _normalize(restored)


def _blur_perturbation(images: torch.Tensor) -> torch.Tensor:
    pixels = _pixels(images)
    blurred = F.avg_pool2d(F.pad(pixels, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    return _normalize(blurred)
