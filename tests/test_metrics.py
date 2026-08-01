import pytest

from semtrace.metrics.binary import binary_metrics, grouped_binary_metrics


def test_binary_metrics_use_fake_probability_and_half_threshold() -> None:
    metrics = binary_metrics(labels=[0, 1, 0, 1], fake_scores=[0.1, 0.9, 0.6, 0.4])

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.average_precision == pytest.approx(5 / 6)
    assert metrics.real_accuracy == pytest.approx(0.5)
    assert metrics.fake_accuracy == pytest.approx(0.5)


def test_grouped_metrics_report_domains_and_unweighted_means() -> None:
    result = grouped_binary_metrics(
        labels=[0, 1, 0, 1],
        fake_scores=[0.1, 0.9, 0.6, 0.4],
        domains=["a", "a", "b", "b"],
    )

    assert result.per_domain["a"].accuracy == pytest.approx(1.0)
    assert result.per_domain["b"].accuracy == pytest.approx(0.0)
    assert result.mean_accuracy == pytest.approx(0.5)
    assert result.mean_average_precision == pytest.approx(0.75)
