"""Numerical agreement with MAPIE.

Reimplementing conformal prediction is only defensible if the reimplementation
is correct. MAPIE's ``SplitConformalClassifier`` with the ``lac`` conformity
score is the same estimator as ours with ``method="marginal"`` and
``score="one_minus_prob"``: LAC *is* ``1 - p_y``. Given the same fitted model,
the same calibration set and the same alpha, the prediction sets must be
identical -- not merely similar.

MAPIE is a dev dependency only; the test skips if it is absent.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, coverage_by_class
from conftest import make_imbalanced

mapie_classification = pytest.importorskip("mapie.classification")


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_marginal_split_matches_mapie(alpha, seed):
    X, y = make_imbalanced(n_samples=4000, minority_rate=0.05, seed=seed)
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )
    X_cal, X_test, y_cal, _ = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )

    base = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)

    ours = ConformalClassifier(
        base, method="marginal", score="one_minus_prob", prefit=True
    ).fit(X_cal, y_cal)
    our_sets = ours.predict_set(X_test, alpha)

    theirs = mapie_classification.SplitConformalClassifier(
        estimator=base, confidence_level=1 - alpha, conformity_score="lac", prefit=True
    )
    theirs.conformalize(X_cal, y_cal)
    _, their_sets = theirs.predict_set(X_test)
    their_sets = np.asarray(their_sets).reshape(our_sets.shape)

    np.testing.assert_array_equal(our_sets, their_sets)


# --- cross-conformal ---------------------------------------------------------
# The split test above pins an *exact* equivalence: same estimator, same
# calibration set, same score, so the sets must be identical. Cross is different.
# Ours is the cross-conformal predictor of Vovk (2015) -- pool the out-of-fold
# scores, predict with a model fitted on everything. MAPIE's is CV+ (Barber et
# al. 2021), which aggregates the fold models' predictions instead. They are not
# the same construction, so exact agreement is not expected and would be
# suspicious. What matters is that they land in the same place, since our
# headline result is about cross-conformal.


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_cross_conformal_lands_where_mapies_cv_plus_does(alpha):
    ours_cov, mapie_cov, ours_w, mapie_w, agree = [], [], [], [], []
    for seed in range(3):
        X, y = make_imbalanced(n_samples=4000, minority_rate=0.05, seed=seed)
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=seed
        )
        cc = ConformalClassifier(
            LogisticRegression(max_iter=1000), method="marginal",
            strategy="cross", n_folds=5, random_state=seed,
        ).fit(X_pool, y_pool)
        ours_sets = cc.predict_set(X_test, alpha)

        theirs = mapie_classification.CrossConformalClassifier(
            estimator=LogisticRegression(max_iter=1000),
            confidence_level=1 - alpha, conformity_score="lac",
            cv=5, random_state=seed,
        )
        theirs.fit_conformalize(X_pool, y_pool)
        _, their_sets = theirs.predict_set(X_test)
        their_sets = np.asarray(their_sets).reshape(ours_sets.shape)

        from tabpfn_conformal import average_set_size, marginal_coverage
        ours_cov.append(marginal_coverage(ours_sets, y_test, cc.classes_))
        mapie_cov.append(marginal_coverage(their_sets, y_test, cc.classes_))
        ours_w.append(average_set_size(ours_sets))
        mapie_w.append(average_set_size(their_sets))
        agree.append(float((ours_sets == their_sets).all(axis=1).mean()))

    # Both reach the target; neither is systematically looser than the other.
    assert np.mean(ours_cov) >= 1 - alpha - 0.02, np.mean(ours_cov)
    assert np.mean(mapie_cov) >= 1 - alpha - 0.02, np.mean(mapie_cov)
    assert abs(np.mean(ours_cov) - np.mean(mapie_cov)) < 0.01, (
        f"coverage diverges: ours {np.mean(ours_cov):.4f}, MAPIE {np.mean(mapie_cov):.4f}"
    )
    assert abs(np.mean(ours_w) - np.mean(mapie_w)) < 0.01, (
        f"set size diverges: ours {np.mean(ours_w):.4f}, MAPIE {np.mean(mapie_w):.4f}"
    )
    # Measured at 99.7% across three alphas; 0.98 leaves room for a MAPIE
    # release changing its aggregation without this becoming a flaky failure.
    assert np.mean(agree) > 0.98, f"only {np.mean(agree):.1%} of sets agree"


@pytest.mark.parametrize("n_folds", [2, 3, 5])
def test_cv_plus_scores_every_test_row_k_plus_one_times_and_ours_scores_it_once(n_folds):
    """The one place the two constructions differ in a way a user pays for.

    CV+ builds its sets from the K fold models, so it must query each of them
    for every test row. Pooled cross-conformal derives thresholds from the
    out-of-fold scores and then predicts with the single full-data model, so a
    test row is scored once no matter how large K is.

    On a local model this is a footnote. On a metered API billed per row
    predicted, which is what TabPFN is, it is a K+1 multiplier on inference
    cost that lasts for as long as the model is in production. The two agree on
    the sets themselves, which the test above establishes, so this is a cost
    difference rather than an accuracy one.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    calls = {"n": 0}

    class Counting(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self._m = LogisticRegression(max_iter=500)

        def fit(self, X, y):
            self._m.fit(X, y)
            self.classes_ = self._m.classes_
            return self

        def predict_proba(self, X):
            calls["n"] += len(X)
            return self._m.predict_proba(X)

        def predict(self, X):
            return self._m.predict(X)

    X, y = make_imbalanced(n_samples=1500, minority_rate=0.1, seed=0)
    X_pool, X_test, y_pool, _ = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    ours = ConformalClassifier(Counting(), method="mondrian", strategy="cross",
                               n_folds=n_folds, random_state=0).fit(X_pool, y_pool)
    calls["n"] = 0
    ours.predict_set(X_test, alpha=0.1)
    ours_per_row = calls["n"] / len(X_test)

    theirs = mapie_classification.CrossConformalClassifier(
        estimator=Counting(), confidence_level=0.9, conformity_score="lac",
        cv=n_folds, random_state=0)
    theirs.fit_conformalize(X_pool, y_pool)
    calls["n"] = 0
    theirs.predict_set(X_test)
    theirs_per_row = calls["n"] / len(X_test)

    assert ours_per_row == 1.0, f"ours scored {ours_per_row} times per row"
    assert theirs_per_row == n_folds + 1, (
        f"expected CV+ to score each row {n_folds + 1} times, got {theirs_per_row}")


def test_mapie_cross_conformal_has_no_class_conditional_option():
    """The capability gap, which matters more than any timing.

    Every conformity score ``CrossConformalClassifier`` accepts (lac, aps,
    raps, top_k, naive) is marginal, and the class has no Mondrian or
    class-wise parameter. Marginal conformal spends its error budget where the
    probability mass is, so under heavy imbalance it buys its headline coverage
    from the majority class and abandons the minority one. That is the
    published behaviour, not a MAPIE defect: it is what marginal calibration
    does, and it is why this package pairs cross-conformal with class-conditional
    thresholds in one object.

    Asserted rather than described, because it is the load-bearing reason this
    is not a MAPIE wrapper.
    """
    alpha = 0.1
    X, y = make_imbalanced(n_samples=4000, minority_rate=0.04, seed=0)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)

    assert "class_wise" not in inspect.signature(
        mapie_classification.CrossConformalClassifier.__init__).parameters, (
        "MAPIE gained a class-conditional option; revisit this comparison")

    theirs = mapie_classification.CrossConformalClassifier(
        estimator=LogisticRegression(max_iter=500), confidence_level=1 - alpha,
        conformity_score="lac", cv=5, random_state=0)
    theirs.fit_conformalize(X_pool, y_pool)
    _, raw = theirs.predict_set(X_test)
    their_sets = np.asarray(raw).reshape(len(X_test), 2, -1)[:, :, 0]
    their_cov = coverage_by_class(their_sets, y_test, np.array([0, 1]))

    ours = ConformalClassifier(LogisticRegression(max_iter=500),
                               method="mondrian", strategy="cross",
                               n_folds=5, random_state=0).fit(X_pool, y_pool)
    our_cov = coverage_by_class(ours.predict_set(X_test, alpha=alpha),
                                y_test, ours.classes_)

    # Marginal holds overall and abandons the minority class; Mondrian holds it.
    assert their_cov[1] < 0.5, (
        f"expected marginal CV+ to under-cover the minority class, got "
        f"{their_cov[1]:.3f}")
    assert our_cov[1] >= 1 - alpha - 0.02, (
        f"ours should hold the minority class, got {our_cov[1]:.3f}")
