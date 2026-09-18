"""Public API behaviour and sklearn compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn_conformal import ConformalClassifier
from conftest import make_imbalanced


@pytest.fixture
def fitted():
    X, y = make_imbalanced(n_samples=2000, seed=0)
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, y)
    return cc, X, y


def test_predict_set_shape_and_dtype(fitted):
    cc, X, _ = fitted
    sets = cc.predict_set(X, 0.1)
    assert sets.shape == (len(X), 2)
    assert sets.dtype == bool


def test_alpha_is_a_predict_time_argument(fitted):
    """Sweeping alpha must not require refitting -- on a metered API that is the cost."""
    cc, X, _ = fitted
    tight = cc.predict_set(X, 0.2).sum()
    loose = cc.predict_set(X, 0.01).sum()
    assert loose > tight


def test_predict_set_from_proba_matches_predict_set(fitted):
    cc, X, _ = fitted
    np.testing.assert_array_equal(
        cc.predict_set(X, 0.1),
        cc.predict_set_from_proba(cc.predict_proba(X), 0.1),
    )


def test_predict_is_argmax_passthrough(fitted):
    cc, X, _ = fitted
    np.testing.assert_array_equal(cc.predict(X), cc.estimator_.predict(X))


def test_n_calibration_reports_per_class_counts(fitted):
    cc, _, _ = fitted
    assert set(cc.n_calibration_) == set(cc.classes_)
    assert all(v > 0 for v in cc.n_calibration_.values())


def test_prefit_uses_all_data_for_calibration():
    X, y = make_imbalanced(n_samples=2000, seed=0)
    base = LogisticRegression(max_iter=1000).fit(X, y)
    cc = ConformalClassifier(base, prefit=True).fit(X, y)
    assert cc.estimator_ is base
    assert sum(cc.n_calibration_.values()) == len(y)


def test_calibrate_replaces_scores_without_refitting():
    X, y = make_imbalanced(n_samples=3000, seed=0)
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, y)
    before = cc.estimator_
    cc.calibrate(X[:500], y[:500])
    assert cc.estimator_ is before
    assert sum(cc.n_calibration_.values()) == 500


def test_calibrate_rejects_unseen_labels():
    X, y = make_imbalanced(n_samples=1000, seed=0)
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="labels absent at fit time"):
        cc.calibrate(X[:50], np.full(50, 7))


def test_accepts_dataframes():
    X, y = make_imbalanced(n_samples=1500, seed=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    for strategy in ("split", "cross"):
        cc = ConformalClassifier(
            LogisticRegression(max_iter=1000), strategy=strategy, n_folds=3, random_state=0
        ).fit(df, y)
        assert cc.predict_set(df, 0.1).shape == (len(df), 2)


def test_accepts_a_pipeline_as_base_estimator():
    X, y = make_imbalanced(n_samples=1500, seed=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cc = ConformalClassifier(pipe, strategy="cross", n_folds=3, random_state=0).fit(X, y)
    assert cc.predict_set(X, 0.1).shape == (len(X), 2)


def test_string_labels_round_trip():
    X, y = make_imbalanced(n_samples=1500, seed=0)
    y_str = np.where(y == 1, "fraud", "legit")
    cc = ConformalClassifier(LogisticRegression(max_iter=1000), random_state=0).fit(X, y_str)
    assert set(cc.classes_) == {"fraud", "legit"}
    assert set(cc.n_calibration_) == {"fraud", "legit"}


def test_sklearn_clone_and_params_round_trip():
    cc = ConformalClassifier(
        LogisticRegression(), method="marginal", strategy="cross", cal_size=0.3, n_folds=7
    )
    twin = clone(cc)
    assert twin.get_params(deep=False)["cal_size"] == 0.3
    assert twin.method == "marginal" and twin.n_folds == 7
    twin.set_params(cal_size=0.8)
    assert twin.cal_size == 0.8


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "bogus"}, "method must be one of"),
        ({"strategy": "bogus"}, "strategy must be one of"),
        ({"cal_size": 0.0}, "cal_size must lie"),
        ({"cal_size": 1.0}, "cal_size must lie"),
        ({"n_folds": 1}, "n_folds must be at least"),
    ],
)
def test_invalid_parameters_are_rejected_at_fit(kwargs, match):
    X, y = make_imbalanced(n_samples=500, seed=0)
    with pytest.raises(ValueError, match=match):
        ConformalClassifier(LogisticRegression(), **kwargs).fit(X, y)


def test_base_estimator_without_predict_proba_is_rejected():
    from sklearn.svm import LinearSVC

    X, y = make_imbalanced(n_samples=500, seed=0)
    with pytest.raises(TypeError, match="no predict_proba"):
        ConformalClassifier(LinearSVC()).fit(X, y)
