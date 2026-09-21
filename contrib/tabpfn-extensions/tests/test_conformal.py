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

from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn_extensions.conformal import (
    ACI,
    ConformalClassifier,
    average_set_size,
    conformal_quantile,
    coverage_by_class,
    decision_summary,
    empty_set_rate,
    marginal_coverage,
    out_of_fold_proba,
    route,
)
from tabpfn_extensions.conformal.crossconformal import aligned_proba
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


# --- regression guards -----------------------------------------------------
# Each of the following was written after a mutation of the library passed the
# rest of this file. They are the tests that make the module's own docstrings
# true, so they travel with it.

class _Memoriser(BaseEstimator, ClassifierMixin):
    """Answers 1.0 on the true class of any row it was fitted on.

    A fold leak is otherwise nearly invisible: on easy data a model that trained
    on its own test rows scores them barely better than one that did not, so a
    coverage assertion will not notice. This makes the leak total.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
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
    """Cross-conformal's defining property, asserted rather than assumed."""
    X, y = imbalanced(n=600, minority_rate=0.1)
    proba = out_of_fold_proba(_Memoriser(), X, y, np.array([0, 1]),
                              n_folds=5, random_state=0)
    np.testing.assert_allclose(proba, 0.5)


def test_the_final_model_is_fitted_on_every_row():
    """Its second property: no label is spent, so none leaves the context."""
    X, y = imbalanced(n=600, minority_rate=0.1)
    cc = ConformalClassifier(_Memoriser(), strategy="cross", n_folds=5,
                             random_state=0).fit(X, y)
    seen = cc.estimator_.predict_proba(X)
    np.testing.assert_allclose(seen[np.arange(len(y)), y], 1.0)


def test_a_score_exactly_on_the_threshold_is_inside_the_set():
    """Membership is `score <= threshold`, inclusive.

    Continuous probabilities never tie, so a strict `<` passes every statistical
    test here while dropping the boundary point. Discrete scores make ties
    routine, and there it loses the guarantee.
    """
    class Grid(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p1 = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1.0 - p1, p1])

    grid = np.round(np.arange(1, 11) * 0.1, 2)
    X_cal = np.concatenate([1.0 - grid, grid]).reshape(-1, 1)
    y_cal = np.array([1] * 10 + [0] * 10)
    cc = ConformalClassifier(Grid(), method="mondrian", prefit=True).fit(X_cal, y_cal)

    # n=10, alpha=0.2 -> k = ceil(11 * 0.8) = 9 -> the 9th smallest score = 0.9.
    assert cc.quantiles(0.2)[1] == pytest.approx(0.9)
    assert cc.predict_set(np.array([[1.0 - 0.9]]), 0.2)[0, 1]
    assert not cc.predict_set(np.array([[1.0 - 0.95]]), 0.2)[0, 1]


def test_a_folds_missing_class_is_realigned_not_trusted_positionally():
    """`classes_` in another order must not be read by column position."""
    class Reversed(BaseEstimator, ClassifierMixin):
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    out = aligned_proba(Reversed(), np.zeros((3, 2)), np.array([0, 1]))
    np.testing.assert_allclose(out, np.tile([0.8, 0.2], (3, 1)))


def test_metrics_are_what_they_say():
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    y = np.array([0, 0, 1, 1])
    assert average_set_size(sets) == pytest.approx(1.0)
    assert empty_set_rate(sets) == pytest.approx(0.25)
    assert marginal_coverage(sets, y, np.array([0, 1])) == pytest.approx(0.5)


def test_classes_are_read_positionally_not_assumed_sorted():
    """`coverage_by_class` keys by position, so resolution must too."""
    classes = np.array(["legit", "fraud"])      # deliberately unsorted
    y = np.array(["fraud", "legit"])
    sets = np.array([[False, True], [True, False]])
    cov = coverage_by_class(sets, y, classes)
    assert cov["fraud"] == pytest.approx(1.0)
    assert cov["legit"] == pytest.approx(1.0)


def test_a_dataframe_with_a_non_default_index_is_taken_positionally():
    """Any filtered or merged frame has a non-default index."""
    pd = pytest.importorskip("pandas")
    X, y = imbalanced(n=600, minority_rate=0.15)
    cols = [f"f{i}" for i in range(X.shape[1])]
    ref = ConformalClassifier(LogisticRegression(max_iter=1000), strategy="cross",
                              n_folds=3, random_state=0)
    ref.fit(pd.DataFrame(X, columns=cols), y)
    base = ref.predict_set(pd.DataFrame(X, columns=cols), ALPHA)

    scrambled = pd.DataFrame(X, columns=cols,
                             index=np.random.default_rng(0).permutation(len(X)))
    other = ConformalClassifier(LogisticRegression(max_iter=1000), strategy="cross",
                                n_folds=3, random_state=0).fit(scrambled, y)
    np.testing.assert_array_equal(base, other.predict_set(scrambled, ALPHA))


def test_non_finite_probabilities_are_rejected():
    """NaN scores would otherwise yield silently empty sets, which `route`
    treats as the highest priority for human review."""
    class NaNEstimator(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)

    X, y = imbalanced(n=200, minority_rate=0.2)
    with pytest.raises(ValueError, match="non-finite"):
        ConformalClassifier(NaNEstimator(), prefit=True).fit(X, y)


def test_alpha_dict_reports_the_adapted_levels_not_the_target():
    """ACI's handoff into `predict_set`.

    If this returned `alpha_target`, the adaptation would be computed and then
    discarded — indistinguishable from the (real) finding that ACI cannot move
    the threshold at a small calibration set.
    """
    aci = ACI(alpha_target=0.05, gamma=0.1, n_classes=2)
    for _ in range(5):
        aci.update_round(1, 0.9)
    assert aci.alphas_[1] != pytest.approx(0.05)
    assert aci.alpha_dict()[1] == pytest.approx(float(aci.alphas_[1]))
    assert aci.alpha_dict()[1] != pytest.approx(aci.alpha_target)
