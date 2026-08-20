from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semtrace.analysis.proposal_mechanisms import (
    build_core1_statistics,
    build_core2_statistics,
    build_core3_statistics,
    build_donor_indices,
    build_intervention_plan,
    create_package_archive,
    write_file_manifest,
)
from semtrace.analysis.proposal_package import load_core1_samples, validate_cache_checkpoint
from semtrace.analysis.proposal_plotting import generate_proposal_figures
from semtrace.config import compose_config

LAYERS = (2, 6, 8)


def test_proposal_config_composes_at_global_scope() -> None:
    config = compose_config("analysis/proposal_mechanisms")

    assert config.dataset == "genimage_sdv14"
    assert config.bootstrap_iterations == 1000


def test_mismatched_cache_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="checkpoint fingerprint mismatch"):
        validate_cache_checkpoint(
            {"checkpoint_sha256": "cached-checkpoint"},
            checkpoint_sha256="requested-checkpoint",
        )


def _core1_samples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scale, layer in zip(("shallow", "middle", "deep"), LAYERS, strict=True):
        for label in (0, 1):
            for index in range(8):
                rows.append(
                    {
                        "sample_id": f"{layer}-{label}-{index}",
                        "path": f"/{label}/{index}.png",
                        "label": label,
                        "generator": "real" if label == 0 else "fake-a",
                        "semantic_class": None,
                        "content_env": None,
                        "scale": scale,
                        "selected_layer": layer,
                        "residual_l2_mean": float(layer + label * 2 + index / 10),
                        "residual_l1_mean": float(layer / 2 + label + index / 20),
                        "prediction": label,
                        "probability": 0.1 if label == 0 else 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _core2_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generators = ("fake-a", "fake-b")
    conditions = (
        "baseline",
        "mask_scale_L2_after_adapter",
        "mask_scale_L6_after_adapter",
        "mask_scale_L8_after_adapter",
    )
    for generator in generators:
        for index in range(20):
            label = index % 2
            baseline_logit = -2.0 if label == 0 else 2.0
            for condition in conditions:
                penalty = {
                    "baseline": 0.0,
                    "mask_scale_L2_after_adapter": 0.4,
                    "mask_scale_L6_after_adapter": 0.8,
                    "mask_scale_L8_after_adapter": 1.2,
                }[condition]
                logit = baseline_logit - penalty if label == 1 else baseline_logit + penalty
                rows.append(
                    {
                        "sample_id": f"{generator}-{index}",
                        "generator": generator,
                        "label": label,
                        "condition": condition,
                        "logit": logit,
                        "probability": float(1.0 / (1.0 + np.exp(-logit))),
                    }
                )
    return pd.DataFrame(rows)


def _core3_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, label in enumerate((0, 0, 1, 1)):
        baseline_logit = -2.0 if label == 0 else 2.0
        for condition, logit, donor_label in (
            ("baseline", baseline_logit, label),
            ("matched_semantic_swap", baseline_logit + 0.1, label),
            ("real_fake_trace_swap", -baseline_logit, 1 - label),
        ):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "generator": "g",
                    "label": label,
                    "condition": condition,
                    "logit": logit,
                    "probability": float(1.0 / (1.0 + np.exp(-logit))),
                    "donor_label": donor_label,
                    "donor_matched": True,
                }
            )
    return pd.DataFrame(rows)


def test_core1_has_three_scales_and_summary_matches_samples() -> None:
    samples = _core1_samples()
    summary, statistics = build_core1_statistics(
        samples,
        selected_layers=LAYERS,
        bootstrap_iterations=20,
        seed=0,
    )

    assert set(samples["selected_layer"]) == set(LAYERS)
    assert set(samples["label"]) == {0, 1}
    assert np.isfinite(samples["residual_l2_mean"]).all()
    shallow_fake = summary.query(
        "group_type == 'authenticity' and selected_layer == 2 and label == 1"
    ).iloc[0]
    expected = samples.query("selected_layer == 2 and label == 1")["residual_l2_mean"].mean()
    assert shallow_fake["mean"] == expected
    assert statistics["layers"]["2"]["comparison"]["cohens_d"] > 0


def test_core1_joins_hashed_cache_id_to_diagnostic_path(tmp_path: Path) -> None:
    residual_rows = []
    for metric, value in (("patch_l2_mean", 2.0), ("patch_l1_mean", 1.0)):
        residual_rows.append(
            {
                "sample_id": "cache-hash",
                "path": "/data/sample.png",
                "label": 1,
                "generator": "fake-a",
                "semantic_class": None,
                "content_env": None,
                "layer": 2,
                "metric": metric,
                "value": value,
            }
        )
    residual_path = tmp_path / "residuals.csv"
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)
    baseline = pd.DataFrame(
        [
            {
                "sample_id": "/data/sample.png",
                "condition": "baseline",
                "probability": 0.9,
            }
        ]
    )

    samples = load_core1_samples(
        residual_path,
        baseline,
        selected_layers=(2, 6, 8),
        threshold=0.5,
    )

    assert samples.iloc[0]["sample_id"] == "cache-hash"
    assert samples.iloc[0]["prediction"] == 1


def test_core2_contains_every_generator_and_computes_mask_delta() -> None:
    table, summary, statistics = build_core2_statistics(
        _core2_predictions(),
        selected_layers=LAYERS,
        generators=("fake-a", "fake-b"),
        threshold=0.5,
        bootstrap_iterations=20,
        seed=0,
    )

    assert set(table["generator"]) == {"fake-a", "fake-b"}
    assert set(summary["selected_layer"]) == set(LAYERS)
    first = table.iloc[0]
    assert first["delta_ap_shallow_pp"] == (first["baseline_ap"] - first["mask_shallow_ap"]) * 100.0
    assert set(statistics["generators"]) == {"fake-a", "fake-b"}


def test_donor_indices_preserve_semantic_trace_boundaries() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    generators = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])

    semantic_donors, semantic_matched = build_donor_indices(
        labels, generators, opposite_authenticity=False, seed=0
    )
    trace_donors, trace_matched = build_donor_indices(
        labels, generators, opposite_authenticity=True, seed=0
    )

    assert semantic_matched.all()
    assert trace_matched.all()
    assert np.all(generators[semantic_donors] == generators)
    assert np.all(labels[semantic_donors] == labels)
    assert np.all(semantic_donors != np.arange(labels.size))
    assert np.all(generators[trace_donors] == generators)
    assert np.all(labels[trace_donors] == 1 - labels)

    plan = build_intervention_plan(labels, generators, seed=0)
    original = np.arange(labels.size)
    assert np.array_equal(plan["matched_semantic_swap"]["trace_indices"], original)
    assert np.array_equal(plan["real_fake_trace_swap"]["semantic_indices"], original)


def test_core3_flip_and_trace_following_rates_are_correct() -> None:
    summary, statistics = build_core3_statistics(
        _core3_predictions(), threshold=0.5, bootstrap_iterations=20, seed=0
    )

    semantic = summary.query("condition == 'matched_semantic_swap'").iloc[0]
    trace = summary.query("condition == 'real_fake_trace_swap'").iloc[0]
    assert semantic["prediction_flip_rate"] == 0.0
    assert trace["prediction_flip_rate"] == 1.0
    assert trace["trace_following_rate"] == 1.0
    assert statistics["supports_trace_as_evidence"] is True


def test_figures_manifest_and_archive_are_reproducible(tmp_path: Path) -> None:
    core1 = _core1_samples()
    core2, _, _ = build_core2_statistics(
        _core2_predictions(),
        selected_layers=LAYERS,
        generators=("fake-a", "fake-b"),
        threshold=0.5,
        bootstrap_iterations=20,
        seed=0,
    )
    core3, _ = build_core3_statistics(
        _core3_predictions(), threshold=0.5, bootstrap_iterations=20, seed=0
    )
    figure_dir = tmp_path / "figures"
    paths = generate_proposal_figures(
        core1,
        core2,
        core3,
        figure_dir,
        selected_layers=LAYERS,
        dpi=72,
        formats=("png", "pdf"),
    )

    assert len(paths) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    (tmp_path / "README.md").write_text("package\n", encoding="utf-8")
    manifest_path = write_file_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {item["path"] for item in manifest["files"]}
    expected = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "file_manifest.json"
    }
    assert listed == expected

    archive = create_package_archive(tmp_path)
    assert archive.is_file()
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        names = set(package.namelist())
        assert "README.md" in names
        assert "metadata/file_manifest.json" in names
        assert archive.name not in names
