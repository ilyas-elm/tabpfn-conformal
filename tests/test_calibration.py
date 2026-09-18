"""The quantile formula, which everything else rests on."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_conformal import conformal_quantile, mondrian_thresholds
from tabpfn_conformal.calibration import InsufficientCalibrationWarning


def test_quantile_matches_hand_computation():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # n=5, alpha=0.2 -> k = ceil(6 * 0.8) = 5 -> the 5th smallest.
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.5)
    # n=5, alpha=0.5 -> k = ceil(6 * 0.5) = 3 -> the 3rd smallest.
    assert conformal_quantile(scores, 0.5) == pytest.approx(0.3)


def test_quantile_is_not_the_plain_empirical_quantile():
    """The finite-sample correction must actually bite for small n."""
    scores = np.arange(1, 11, dtype=float)
    naive = float(np.quantile(scores, 0.9))
    assert conformal_quantile(scores, 0.1) != pytest.approx(naive)


def test_quantile_is_conservative_when_n_too_small():
    scores = np.array([0.1, 0.2, 0.3, 0.4])  # n=4 cannot certify alpha=0.1
    with pytest.warns(InsufficientCalibrationWarning, match="cannot certify"):
        assert conformal_quantile(scores, 0.1) == np.inf


def test_empty_calibration_set_warns_and_is_trivial():
    with pytest.warns(InsufficientCalibrationWarning, match="No calibration points"):
        assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_alpha_one_gives_empty_prediction_set():
    """alpha=1 must yield -inf (empty set), not the maximum score."""
    assert conformal_quantile(np.array([0.1, 0.9]), 1.0) == -np.inf


def test_alpha_zero_cannot_be_certified():
    with pytest.warns(InsufficientCalibrationWarning):
        assert conformal_quantile(np.array([0.1, 0.9]), 0.0) == np.inf


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_out_of_range_raises(alpha):
    with pytest.raises(ValueError, match="alpha must lie"):
        conformal_quantile(np.array([0.1]), alpha)


def test_mondrian_splits_by_class():
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    # n=3 per class, alpha=0.5 -> k = ceil(4 * 0.5) = 2, achievable in both.
    q = mondrian_thresholds(scores, labels, n_classes=2, alpha=0.5)
    assert q[0] == pytest.approx(0.1)
    assert q[1] == pytest.approx(0.9)


def test_mondrian_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        mondrian_thresholds(np.zeros(5), np.zeros(4), n_classes=2, alpha=0.1)
