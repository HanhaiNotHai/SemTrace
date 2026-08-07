import numpy as np

from semtrace.analysis.representation_label_mi import (
    linear_cka,
    linear_hsic,
    mutual_information_with_permutation,
)


def test_correlated_features_have_mi_above_permutation_baseline() -> None:
    rng = np.random.default_rng(7)
    labels = np.arange(200) % 2
    features = rng.normal(size=(200, 4))
    features[:, 0] += labels * 4.0

    result = mutual_information_with_permutation(
        features,
        labels,
        pca_dimensions=None,
        seeds=(0, 1, 2),
    )

    assert result.estimate_mean > result.permutation_mean


def test_independent_features_are_close_to_permutation_baseline() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(400, 3))
    labels = rng.integers(0, 2, size=400)

    result = mutual_information_with_permutation(
        features,
        labels,
        pca_dimensions=None,
        seeds=(0, 1, 2),
    )

    assert abs(result.estimate_mean - result.permutation_mean) < 0.05


def test_hsic_and_cka_increase_for_identical_representations() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(64, 5))
    independent = rng.normal(size=(64, 5))

    assert linear_hsic(x, x) > linear_hsic(x, independent)
    assert linear_cka(x, x) > linear_cka(x, independent)
