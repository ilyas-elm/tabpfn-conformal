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

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier
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
