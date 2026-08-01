from __future__ import annotations

import json

import numpy as np

from semtrace.models.probes import (
    LayerProbeMetrics,
    choose_nuisance_label,
    save_probe_artifacts,
    select_probe_layers,
)


def test_probe_selection_takes_best_score_from_each_depth_band(tmp_path) -> None:
    metrics = {
        layer: LayerProbeMetrics(
            authenticity_ap=float(layer in {2, 5, 9}),
            authenticity_balanced_accuracy=0.5,
            semantic_accuracy=0.2,
            nuisance_accuracy=0.2,
        )
        for layer in range(11)
    }

    result = select_probe_layers(metrics, num_hidden_layers=12, alpha=0.5, beta=0.5)

    assert result.selected_layers == (2, 5, 9)
    assert 11 not in result.layer_scores


def test_probe_selection_is_stable_when_a_metric_is_constant() -> None:
    metrics = {
        layer: LayerProbeMetrics(0.5, 0.5, 0.5, 0.5)
        for layer in range(11)
    }

    result = select_probe_layers(metrics, num_hidden_layers=12, alpha=0.5, beta=0.5)

    assert result.selected_layers == (0, 4, 8)
    assert all(np.isfinite(score) for score in result.layer_scores.values())


def test_single_generator_uses_source_instead_of_generator_probe() -> None:
    labels = {
        "source": ["camera_a", "camera_b", "camera_a"],
        "degradation": [None, None, None],
        "file_format": ["jpg", "png", "jpg"],
        "generator": ["sdv1.4", "sdv1.4", "sdv1.4"],
    }

    name, values, generator_enabled = choose_nuisance_label(labels)

    assert name == "source"
    assert values == labels["source"]
    assert generator_enabled is False


def test_probe_artifacts_contain_required_reproducibility_fields(tmp_path) -> None:
    metrics = {
        layer: LayerProbeMetrics(0.5 + layer * 0.01, 0.5, 0.2, 0.3)
        for layer in range(11)
    }
    result = select_probe_layers(metrics, num_hidden_layers=12, alpha=0.5, beta=0.5)

    save_probe_artifacts(
        result,
        tmp_path,
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        model_revision="master",
        semantic_label_coverage=0.75,
        nuisance_label="source",
        generator_probe_enabled=False,
    )

    payload = json.loads((tmp_path / "selected_layers.json").read_text())
    assert payload["selected_layers"] == list(result.selected_layers)
    assert payload["semantic_label_coverage"] == 0.75
    assert payload["generator_probe_enabled"] is False
    assert (tmp_path / "probe_results.csv").is_file()
    assert (tmp_path / "layer_score_plot.png").is_file()
