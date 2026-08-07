from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import bootstrap, mannwhitneyu


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    count: int
    mean: float
    standard_deviation: float
    median: float
    q1: float
    q3: float
    confidence_interval_low: float
    confidence_interval_high: float


def summarize_distribution(
    values: np.ndarray,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> DistributionSummary:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("distribution summary requires a finite observation")
    if array.size == 1 or np.all(array == array[0]):
        low = high = float(array.mean())
    else:
        interval = bootstrap(
            (array,),
            np.mean,
            n_resamples=bootstrap_iterations,
            method="percentile",
            rng=np.random.default_rng(seed),
        ).confidence_interval
        low, high = float(interval.low), float(interval.high)
    return DistributionSummary(
        count=int(array.size),
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        median=float(np.median(array)),
        q1=float(np.quantile(array, 0.25)),
        q3=float(np.quantile(array, 0.75)),
        confidence_interval_low=low,
        confidence_interval_high=high,
    )


def real_fake_comparison(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    real_values = np.asarray(real, dtype=np.float64)
    fake_values = np.asarray(fake, dtype=np.float64)
    if real_values.size == 0 or fake_values.size == 0:
        raise ValueError("real/fake comparison requires both groups")
    pooled_variance = (
        ((real_values.size - 1) * real_values.var(ddof=1))
        + ((fake_values.size - 1) * fake_values.var(ddof=1))
    ) / max(real_values.size + fake_values.size - 2, 1)
    effect = (fake_values.mean() - real_values.mean()) / max(
        float(np.sqrt(pooled_variance)),
        np.finfo(np.float64).eps,
    )
    test = mannwhitneyu(real_values, fake_values, alternative="two-sided", method="auto")
    return {
        "cohens_d": float(effect),
        "mann_whitney_u": float(test.statistic),
        "mann_whitney_p": float(test.pvalue),
    }


def summary_dict(summary: DistributionSummary) -> dict[str, float | int]:
    return asdict(summary)
