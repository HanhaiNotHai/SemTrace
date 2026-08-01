from __future__ import annotations

from semtrace.engine.evaluator import deduplicate_predictions


def test_distributed_evaluation_removes_sampler_padding_duplicates() -> None:
    predictions = [
        {"path": "a.png", "label": 0, "fake_probability": 0.1},
        {"path": "b.png", "label": 1, "fake_probability": 0.9},
        {"path": "a.png", "label": 0, "fake_probability": 0.1},
    ]

    unique = deduplicate_predictions(predictions)

    assert [prediction["path"] for prediction in unique] == ["a.png", "b.png"]
