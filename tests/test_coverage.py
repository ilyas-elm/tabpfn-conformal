"""Coverage behaviour: the guarantee, and the failure mode it fixes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, coverage_by_class, marginal_coverage
from conftest import make_imbalanced

SEEDS = range(10)
ALPHA = 0.1


def _run(method, seed, minority_rate=0.02, n_samples=8000, strategy="split", **kw):
    X, y = make_imbalanced(n_samples=n_samples, minority_rate=minority_rate, seed=seed)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    cc = ConformalClassifier(
        LogisticRegression(max_iter=1000),
        method=method,
        strategy=strategy,
        random_state=seed,
        **kw,
    ).fit(X_fit, y_fit)
    sets = cc.predict_set(X_test, ALPHA)
    return (
        marginal_coverage(sets, y_test, cc.classes_),
        coverage_by_class(sets, y_test, cc.classes_),
    )


def test_marginal_method_achieves_marginal_coverage():
    cov = [_run("marginal", s)[0] for s in SEEDS]
    assert np.mean(cov) >= 1 - ALPHA - 0.02


def test_mondrian_covers_both_classes():
    per_class = [_run("mondrian", s)[1] for s in SEEDS]
    for k in (0, 1):
        assert np.mean([c[k] for c in per_class]) >= 1 - ALPHA - 0.03, f"class {k}"


def test_marginal_under_covers_the_minority_class():
    """The published failure mode, pinned as a regression test.

    Marginal conformal spends its error budget where the mass is. At a 1%
    base rate the minority class can fall far below 1 - alpha while the
    headline marginal number looks healthy. Mondrian is the fix.
    """
    marg = [_run("marginal", s, minority_rate=0.01)[1] for s in SEEDS]
    mond = [_run("mondrian", s, minority_rate=0.01)[1] for s in SEEDS]

    marg_minority = np.mean([c[1] for c in marg])
    mond_minority = np.mean([c[1] for c in mond])

    assert marg_minority < 1 - ALPHA, "marginal unexpectedly covered the minority class"
    assert mond_minority >= 1 - ALPHA - 0.03
    assert mond_minority > marg_minority


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_coverage_tracks_alpha(alpha):
    X, y = make_imbalanced(n_samples=8000, seed=0)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0
    )
    cc = ConformalClassifier(
        LogisticRegression(max_iter=1000), method="marginal", random_state=0
    ).fit(X_fit, y_fit)
    cov = marginal_coverage(cc.predict_set(X_test, alpha), y_test, cc.classes_)
    assert cov >= 1 - alpha - 0.03
