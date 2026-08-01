from __future__ import annotations

import json

import pytest

from semtrace.smoke import run_synthetic_smoke


@pytest.mark.smoke
def test_complete_synthetic_research_workflow(tmp_path) -> None:
    result = run_synthetic_smoke(tmp_path, seed=17)

    assert len(result.selected_layers) == 3
    assert result.normal_loss > 0
    assert result.detector_loss > 0
    assert 0.0 <= result.evaluation["accuracy"] <= 1.0
    assert 0.0 <= result.evaluation["average_precision"] <= 1.0
    assert result.checkpoint_path.is_file()
    payload = json.loads((tmp_path / "probes" / "selected_layers.json").read_text())
    assert payload["selected_layers"] == list(result.selected_layers)

