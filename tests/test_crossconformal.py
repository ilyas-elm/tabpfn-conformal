"""Cross-conformal: every label calibrates, every label is context."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, coverage_by_class, out_of_fold_proba
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
