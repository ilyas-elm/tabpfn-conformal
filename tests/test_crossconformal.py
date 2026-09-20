"""Cross-conformal: every label calibrates, every label is context."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, coverage_by_class, out_of_fold_proba
from tabpfn_conformal.crossconformal import aligned_proba
from conftest import make_imbalanced

ALPHA = 0.1


def test_every_row_scored_out_of_fold():
    X, y = make_imbalanced(n_samples=2000, seed=0)
    proba = out_of_fold_proba(
        LogisticRegression(max_iter=1000), X, y, np.array([0, 1]), n_folds=5, random_state=0
    )
    assert proba.shape == (len(y), 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_cross_uses_all_labels_for_calibration():
    """Split spends labels; cross does not. This is the headline, as an assertion."""
    X, y = make_imbalanced(n_samples=2000, seed=0)

    split = ConformalClassifier(
        LogisticRegression(max_iter=1000), strategy="split", cal_size=0.5, random_state=0
    ).fit(X, y)
    cross = ConformalClassifier(
        LogisticRegression(max_iter=1000), strategy="cross", n_folds=5, random_state=0
    ).fit(X, y)

    minority = cross.classes_[1]
    assert cross.n_calibration_[minority] > split.n_calibration_[minority]
    assert sum(cross.n_calibration_.values()) == len(y)


def test_cross_conformal_covers_both_classes():
    per_class = []
    for seed in range(8):
        X, y = make_imbalanced(n_samples=6000, minority_rate=0.02, seed=seed)
        X_fit, X_test, y_fit, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=seed
        )
        cc = ConformalClassifier(
            LogisticRegression(max_iter=1000),
            method="mondrian",
            strategy="cross",
            n_folds=5,
            random_state=seed,
        ).fit(X_fit, y_fit)
        per_class.append(coverage_by_class(cc.predict_set(X_test, ALPHA), y_test, cc.classes_))

    for k in (0, 1):
        assert np.mean([c[k] for c in per_class]) >= 1 - ALPHA - 0.05, f"class {k}"


def test_cross_rejects_more_folds_than_minority_labels():
    X, y = make_imbalanced(n_samples=400, minority_rate=0.01, seed=0)
    cc = ConformalClassifier(
        LogisticRegression(max_iter=1000), strategy="cross", n_folds=20, random_state=0
    )
    with pytest.raises(ValueError, match="exceeds the smallest class count"):
        cc.fit(X, y)


def test_works_with_a_second_estimator_family():
    X, y = make_imbalanced(n_samples=1500, seed=1)
    cc = ConformalClassifier(
        RandomForestClassifier(n_estimators=25, random_state=0),
        strategy="cross",
        n_folds=3,
        random_state=0,
    ).fit(X, y)
    assert cc.predict_set(X, ALPHA).shape == (len(y), 2)


class _Memoriser(BaseEstimator, ClassifierMixin):
    """Returns probability 1 on the true class of any row it was fitted on.

    A leak is otherwise almost invisible: on easy synthetic data a fold model
    that trained on its own test rows scores them barely better than one that
    did not, so a coverage assertion will not notice. This estimator makes the
    leak total, so out-of-fold-ness becomes an exact assertion rather than a
    statistical one.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.seen_ = {row.tobytes(): label for row, label in zip(X, y)}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
        for i, row in enumerate(X):
            label = self.seen_.get(row.tobytes())
            if label is not None:
                out[i] = 0.0
                out[i, int(np.searchsorted(self.classes_, label))] = 1.0
        return out


def test_no_row_is_scored_by_a_model_that_saw_it():
    """The defining property of cross-conformal, asserted rather than assumed.

    Guards against the fold indices being swapped: training on `test_idx` and
    predicting the same rows leaves every other test in the suite passing.
    """
    X, y = make_imbalanced(n_samples=600, minority_rate=0.1, seed=0)
    assert len({row.tobytes() for row in X}) == len(X), "fixture rows must be unique"

    proba = out_of_fold_proba(
        _Memoriser(), X, y, np.array([0, 1]), n_folds=5, random_state=0
    )
    # Every row was held out, so no row can carry a memorised 0/1 answer.
    np.testing.assert_allclose(proba, 0.5)


def test_aligned_proba_reorders_columns_to_the_global_class_order():
    """`classes_` in a different order must not be trusted positionally."""

    class Reversed(BaseEstimator, ClassifierMixin):
        classes_ = np.array([1, 0])  # deliberately not sorted

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))  # p(class 1)=0.2, p(class 0)=0.8

    out = aligned_proba(Reversed(), np.zeros((3, 2)), np.array([0, 1]))
    np.testing.assert_allclose(out, np.tile([0.8, 0.2], (3, 1)))


def test_aligned_proba_zero_fills_a_class_the_fold_never_saw():
    """A fold with no minority rows returns a narrower matrix; the gap is a zero."""

    class MajorityOnly(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0])

        def predict_proba(self, X):
            return np.ones((len(X), 1))

    out = aligned_proba(MajorityOnly(), np.zeros((2, 2)), np.array([0, 1]))
    np.testing.assert_allclose(out, np.tile([1.0, 0.0], (2, 1)))
