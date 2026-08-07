import numpy as np

from semtrace.analysis.trace_pattern_coverage import (
    fit_trace_prototypes,
    prototype_coverage,
)


def test_trace_prototypes_assign_training_patterns_and_measure_coverage() -> None:
    rng = np.random.default_rng(5)
    train = np.vstack(
        (rng.normal(0, 0.05, size=(100, 4)), rng.normal(3, 0.05, size=(100, 4)))
    )
    model = fit_trace_prototypes(train, prototype_count=2, pca_dimensions=2, seed=0)
    known = rng.normal(0, 0.05, size=(40, 4))
    unknown = rng.normal(10, 0.05, size=(40, 4))

    known_result = prototype_coverage(model, train, known, top_r=1)
    unknown_result = prototype_coverage(model, train, unknown, top_r=1)

    assert known_result.coverage == 1.0
    assert known_result.mean_nearest_distance < unknown_result.mean_nearest_distance
