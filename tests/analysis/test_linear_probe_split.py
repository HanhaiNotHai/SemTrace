import numpy as np
import pytest

from semtrace.analysis.linear_probe import fit_linear_probe


def test_linear_probe_preprocessing_is_fit_only_on_training_data() -> None:
    x_train = np.asarray([[0.0], [1.0], [0.1], [1.1]])
    y_train = np.asarray([0, 1, 0, 1])
    x_test = np.asarray([[1000.0], [1001.0]])
    y_test = np.asarray([0, 1])

    result = fit_linear_probe(
        x_train,
        y_train,
        x_test,
        y_test,
        task="binary",
        seed=0,
    )

    assert result.skipped_reason is None
    assert result.preprocessing_mean[0] == pytest.approx(x_train.mean())
    assert result.preprocessing_mean[0] != pytest.approx(np.vstack((x_train, x_test)).mean())


def test_probe_skips_targets_with_fewer_than_two_training_classes() -> None:
    result = fit_linear_probe(
        np.ones((3, 2)),
        np.zeros(3, dtype=int),
        np.ones((2, 2)),
        np.zeros(2, dtype=int),
        task="multiclass",
        seed=0,
    )

    assert result.skipped_reason == "fewer than two training classes"
