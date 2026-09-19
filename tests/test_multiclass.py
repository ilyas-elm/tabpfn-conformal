"""Multiclass behaviour.

The conformal machinery was written for binary fraud detection and the
benchmarks are binary, but nothing in scores, calibration, crossconformal or
wrapper assumes two classes. These tests hold that to account, so multiclass
cannot quietly break -- and because the class-conditional argument gets *more*
compelling with more classes, not less.

(`decision.route` is genuinely binary-only: approve / block / review has no
sensible reading with five classes. `test_decision.py` covers its guard.)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, coverage_by_class, marginal_coverage

ALPHA = 0.1
SEEDS = range(6)


def skewed(n_classes, seed):
    """A dominant class plus small ones -- the regime Mondrian exists for."""
    return make_classification(
        n_samples=4000, n_features=15, n_informative=10, n_classes=n_classes,
        n_clusters_per_class=1,
        weights=[0.7] + [0.3 / (n_classes - 1)] * (n_classes - 1),
        random_state=seed,
    )


def run(n_classes, method, strategy="split"):
    per_class, marginal = [], []
    for seed in SEEDS:
        X, y = skewed(n_classes, seed)
        X_fit, X_test, y_fit, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=seed
        )
        cc = ConformalClassifier(
            LogisticRegression(max_iter=1000), method=method,
            strategy=strategy, n_folds=5, random_state=seed,
        ).fit(X_fit, y_fit)
        sets = cc.predict_set(X_test, ALPHA)
        per_class.append(coverage_by_class(sets, y_test, cc.classes_))
        marginal.append(marginal_coverage(sets, y_test, cc.classes_))
    worst = min(np.mean([c[k] for c in per_class]) for k in range(n_classes))
    return float(np.mean(marginal)), float(worst)


@pytest.mark.parametrize("n_classes", [3, 5])
def test_marginal_achieves_marginal_coverage(n_classes):
    marginal, _ = run(n_classes, "marginal")
    assert marginal >= 1 - ALPHA - 0.03


@pytest.mark.parametrize("n_classes", [3, 5])
def test_mondrian_covers_every_class(n_classes):
    _, worst = run(n_classes, "mondrian")
    assert worst >= 1 - ALPHA - 0.03


@pytest.mark.parametrize("n_classes", [3, 5])
def test_marginal_abandons_the_smallest_class(n_classes):
    """The failure mode grows with the number of classes, not shrinks."""
    _, marginal_worst = run(n_classes, "marginal")
    _, mondrian_worst = run(n_classes, "mondrian")
    assert marginal_worst < 1 - ALPHA
    assert mondrian_worst > marginal_worst + 0.05


def test_cross_conformal_covers_every_class():
    _, worst = run(3, "mondrian", strategy="cross")
    assert worst >= 1 - ALPHA - 0.05


def test_prediction_set_shape_follows_the_class_count():
    X, y = skewed(4, 0)
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, y)
    assert cc.predict_set(X, ALPHA).shape == (len(X), 4)
    assert len(cc.quantiles(ALPHA)) == 4
    assert set(cc.n_calibration_) == set(cc.classes_)


def test_string_labels_multiclass():
    X, y = skewed(3, 0)
    names = np.array(["low", "mid", "high"])[y]
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, names)
    assert set(cc.classes_) == {"low", "mid", "high"}
    assert cc.predict_set(X, ALPHA).shape == (len(X), 3)
