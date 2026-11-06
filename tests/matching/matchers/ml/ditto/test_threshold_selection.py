import numpy as np

from matchescu.matching.matchers.ml.ditto.training._evaluator import (
    TrainingEvaluator,
)

_best_threshold = TrainingEvaluator._TrainingEvaluator__best_threshold


def test_selects_plateau_midpoint():
    probabilities = np.array([0.05, 0.15, 0.35, 0.75, 0.95])
    labels = np.array([0, 0, 0, 1, 1])

    f1, threshold, metadata = _best_threshold(probabilities, labels)

    assert f1 > 0.9
    assert 0.4 <= threshold <= 0.6
    assert metadata["threshold_lo"] <= threshold <= metadata["threshold_hi"]
    assert metadata["threshold_plateau_width"] > 0.0
    assert 0.0 <= metadata["dev_auc"] <= 1.0


def test_single_class_dev_falls_back_to_half():
    probabilities = np.array([0.1, 0.2, 0.3])
    labels = np.zeros(3, dtype=int)

    _, threshold, metadata = _best_threshold(probabilities, labels)

    assert threshold == 0.5
    assert metadata["threshold_fallback"] == "single_class"


def test_empty_dev_falls_back_to_half():
    _, threshold, metadata = _best_threshold(
        np.array([], dtype=float), np.array([], dtype=int)
    )

    assert threshold == 0.5
    assert metadata["threshold_fallback"] == "empty"
