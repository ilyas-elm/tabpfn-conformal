"""Tests for the conformal extension.

CPU only: these use scikit-learn estimators, not TabPFN, because the conformal
machinery is model-agnostic and testing it against TabPFN would make the suite
slow without testing anything extra. Honours FAST_TEST_MODE.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_extensions.conformal import (
    ACI,
    ConformalClassifier,
    conformal_quantile,
    coverage_by_class,
    decision_summary,
    marginal_coverage,
    route,
)
from tabpfn_extensions.conformal.calibration import InsufficientCalibrationWarning

FAST = os.environ.get("FAST_TEST_MODE") == "1"
N_SEEDS = 3 if FAST else 8
N_SAMPLES = 2000 if FAST else 6000
ALPHA = 0.1


def imbalanced(n=N_SAMPLES, minority_rate=0.02, seed=0):
    return make_classification(
        n_samples=n, n_features=12, n_informative=6, n_redundant=2,
        n_clusters_per_class=2, weights=[1 - minority_rate, minority_rate],
        flip_y=0.01, class_sep=1.0, random_state=seed,
    )


def run(method, seed, strategy="split", minority_rate=0.02):
    X, y = imbalanced(minority_rate=minority_rate, seed=seed)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    cc = ConformalClassifier(
        LogisticRegression(max_iter=1000), method=method,
        strategy=strategy, random_state=seed,
    ).fit(X_fit, y_fit)
    sets = cc.predict_set(X_test, ALPHA)
    return sets, y_test, cc


def test_quantile_uses_the_finite_sample_correction():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.5)
    assert conformal_quantile(scores, 0.5) == pytest.approx(0.3)
    assert conformal_quantile(scores, 0.1) != pytest.approx(float(np.quantile(scores, 0.9)))


def test_too_few_calibration_points_warns_and_is_conservative():
    with pytest.warns(InsufficientCalibrationWarning):
        assert conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4]), 0.1) == np.inf


def test_marginal_method_achieves_marginal_coverage():
    cov = [marginal_coverage(*run("marginal", s)[:2], run("marginal", s)[2].classes_)
           for s in range(N_SEEDS)]
    assert np.mean(cov) >= 1 - ALPHA - 0.03


def test_mondrian_covers_both_classes():
    per_class = []
    for s in range(N_SEEDS):
        sets, y_test, cc = run("mondrian", s)
        per_class.append(coverage_by_class(sets, y_test, cc.classes_))
    for k in (0, 1):
        assert np.mean([c[k] for c in per_class]) >= 1 - ALPHA - 0.05, f"class {k}"


def test_marginal_under_covers_the_minority_class():
    """The failure mode class-conditional calibration exists to fix."""
    def minority(method):
        out = []
        for s in range(N_SEEDS):
            sets, y_test, cc = run(method, s, minority_rate=0.01)
            out.append(coverage_by_class(sets, y_test, cc.classes_)[1])
        return float(np.mean(out))

    assert minority("marginal") < 1 - ALPHA
    assert minority("mondrian") > minority("marginal")


def test_cross_conformal_spends_no_labels_on_calibration():
    X, y = imbalanced(seed=0)
    split = ConformalClassifier(LogisticRegression(max_iter=1000),
                                strategy="split", random_state=0).fit(X, y)
    cross = ConformalClassifier(LogisticRegression(max_iter=1000),
                                strategy="cross", n_folds=5, random_state=0).fit(X, y)
    assert sum(cross.n_calibration_.values()) == len(y)
    assert cross.n_calibration_[1] > split.n_calibration_[1]


def test_alpha_is_a_prediction_time_argument():
    sets, _, cc = run("mondrian", 0)
    X, y = imbalanced(seed=0)
    assert cc.predict_set(X, 0.01).sum() > cc.predict_set(X, 0.2).sum()


def test_aci_updates_one_step_per_round():
    aci = ACI(alpha_target=0.05, gamma=0.02, n_classes=2)
    assert aci.update_round(1, error_rate=1.0) < 0.05
    assert aci.n_updates_[1] == 1


def test_route_respects_the_review_budget():
    sets = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [0, 0]], dtype=bool)
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.45, 0.55], [0.7, 0.3], [0.5, 0.5]])
    for k in range(6):
        assert (route(sets, proba, budget_k=k) == "review").sum() <= k
    summary = decision_summary(route(sets, proba, budget_k=2), [0, 1, 1, 0, 1], budget_k=2)
    assert summary["n_review"] == 2
