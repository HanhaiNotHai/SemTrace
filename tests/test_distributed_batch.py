from __future__ import annotations

import pytest

from semtrace.engine.distributed import validate_global_batch


def test_four_gpu_protocol_batch_is_exactly_128() -> None:
    result = validate_global_batch(
        world_size=4,
        per_device_batch_size=16,
        gradient_accumulation_steps=2,
        target_global_batch_size=128,
    )

    assert result.actual_global_batch_size == 128
    assert result.strict_protocol is True
    assert result.learning_rate_scale == 1.0


def test_six_gpu_mode_is_explicitly_non_strict() -> None:
    result = validate_global_batch(
        world_size=6,
        per_device_batch_size=20,
        gradient_accumulation_steps=1,
        target_global_batch_size=128,
    )

    assert result.actual_global_batch_size == 120
    assert result.strict_protocol is False
    assert result.learning_rate_scale == pytest.approx(120 / 128)


def test_six_gpu_mode_rejects_unapproved_global_batch() -> None:
    with pytest.raises(ValueError, match="120 or 144"):
        validate_global_batch(
            world_size=6,
            per_device_batch_size=16,
            gradient_accumulation_steps=2,
            target_global_batch_size=128,
        )
