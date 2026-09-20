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


def test_a_score_exactly_on_the_threshold_is_inside_the_set():
    """Membership is ``score <= threshold``, inclusive.

    With continuous probabilities exact ties never occur, so a ``<`` would pass
    every statistical coverage test in this suite while quietly dropping the
    boundary point. Discrete scores -- a coarse model, or a tree voting in
    fractions of K -- make ties routine, and there the strict comparison loses
    the guarantee. Asserted on an exact tie rather than left to chance.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    from tabpfn_conformal import ConformalClassifier

    # Ten calibration rows of each class, scores on a 0.1 grid.
    grid = np.round(np.arange(1, 11) * 0.1, 2)          # 0.1 ... 1.0

    class Grid(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p1 = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1.0 - p1, p1])

    # Row i of class 1 gets p(class 1) = 1 - grid[i], so its score is grid[i].
    X_cal = np.concatenate([1.0 - grid, grid]).reshape(-1, 1)
    y_cal = np.array([1] * 10 + [0] * 10)

    cc = ConformalClassifier(Grid(), method="mondrian", prefit=True).fit(X_cal, y_cal)

    # n=10, alpha=0.2 -> k = ceil(11 * 0.8) = 9 -> the 9th smallest score = 0.9.
    t = cc.quantiles(0.2)[1]
    assert t == pytest.approx(0.9)

    # A test row whose score is exactly 0.9 must keep the fraud label.
    on_the_line = np.array([[1.0 - 0.9]])
    assert cc.predict_set(on_the_line, 0.2)[0, 1], "score == threshold must be covered"

    # And one just above it must not, so the test pins both sides.
    assert not cc.predict_set(np.array([[1.0 - 0.95]]), 0.2)[0, 1]
