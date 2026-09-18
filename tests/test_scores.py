from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tabpfn_conformal import ConformalClassifier, get_score, neg_log_prob, one_minus_prob
from conftest import make_imbalanced


def test_one_minus_prob():
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    np.testing.assert_allclose(one_minus_prob(proba), [[0.8, 0.2], [0.1, 0.9]])


def test_neg_log_prob_is_finite_at_zero():
    assert np.isfinite(neg_log_prob(np.array([[0.0, 1.0]]))).all()


def test_get_score_accepts_callable_and_name():
    assert get_score("one_minus_prob") is one_minus_prob
    fn = lambda p: 1 - p  # noqa: E731
    assert get_score(fn) is fn


def test_get_score_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown score"):
        get_score("nope")


def test_monotone_scores_give_identical_sets():
    """The claim made in the scores module docstring, asserted rather than implied."""
    X, y = make_imbalanced(n_samples=2000, seed=3)
    kwargs = dict(method="mondrian", strategy="split", cal_size=0.5, random_state=0)

    a = ConformalClassifier(LogisticRegression(max_iter=1000), score="one_minus_prob", **kwargs).fit(X, y)
    b = ConformalClassifier(LogisticRegression(max_iter=1000), score="neg_log_prob", **kwargs).fit(X, y)

    np.testing.assert_array_equal(a.predict_set(X, 0.1), b.predict_set(X, 0.1))
