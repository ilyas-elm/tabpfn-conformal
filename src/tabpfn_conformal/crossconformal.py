"""K-fold cross-conformal calibration.

Why this module is the point of the package
-------------------------------------------
Split conformal must hold out a calibration set the model never sees. With a
fraud base rate near 1%, a pool of 10,000 labelled rows contains perhaps a
hundred frauds, and split conformal spends half of them on calibration --
starving the model -- or half on the model -- starving the guarantee.

Cross-conformal removes the choice: every row is scored by a model that did not
see it, so every row calibrates, and the model used at test time is fitted on
*all* of them. The reason nobody does this routinely is that it costs K refits.

For TabPFN there is no refit. ``fit`` swaps the in-context set; no gradient step
is taken. On the Prior Labs API, fits are not even token-charged -- only
predictions are, and each row is predicted exactly once here regardless of K.
That is what makes the strongest form of calibration affordable, and it is a
property of the model, not of this code.

Honest caveat
-------------
Pooling out-of-fold scores and then applying them to a model fitted on the full
pool gives *approximate* validity, not the exact finite-sample guarantee of
split conformal (see Vovk 2015 on cross-conformal predictors; the related CV+
of Barber et al. 2021 bounds worst-case coverage at ``1 - 2*alpha``). We report
empirical coverage alongside this caveat rather than claiming the split-conformal
guarantee we do not have.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

__all__ = ["aligned_proba", "out_of_fold_proba"]


def _take(X, idx):
    """Positional row selection that works for both DataFrames and arrays."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def aligned_proba(estimator, X, classes: np.ndarray) -> np.ndarray:
    """``predict_proba`` re-indexed onto a fixed global class order.

    A fold's training split can be missing a class entirely under heavy
    imbalance, in which case the estimator's ``classes_`` is shorter and in a
    different order. Silently trusting column order there would mislabel every
    score. Missing classes get probability zero.
    """
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    fitted = np.asarray(getattr(estimator, "classes_", classes))

    if fitted.shape == classes.shape and np.array_equal(fitted, classes):
        return proba

    out = np.zeros((proba.shape[0], len(classes)), dtype=float)
    lookup = {c: j for j, c in enumerate(classes)}
    for i, c in enumerate(fitted):
        if c in lookup:
            out[:, lookup[c]] = proba[:, i]
    return out


def out_of_fold_proba(
    base_estimator,
    X,
    y_idx: np.ndarray,
    classes: np.ndarray,
    n_folds: int = 5,
    random_state=None,
) -> np.ndarray:
    """Predicted probabilities for every row, from a model that never saw it.

    Returns an ``(n_samples, n_classes)`` array. Folds are stratified, which is
    not optional at a 1% base rate.
    """
    y_idx = np.asarray(y_idx)
    n_samples = len(y_idx)

    counts = np.bincount(y_idx, minlength=len(classes))
    present = counts[counts > 0]
    if n_folds > present.min():
        raise ValueError(
            f"n_folds={n_folds} exceeds the smallest class count ({present.min()}). "
            "Stratified folds would leave a class unrepresented; lower n_folds or "
            "collect more minority labels."
        )

    proba = np.empty((n_samples, len(classes)), dtype=float)
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    for train_idx, test_idx in splitter.split(np.zeros(n_samples), y_idx):
        fold_model = clone(base_estimator)
        fold_model.fit(_take(X, train_idx), y_idx[train_idx])
        proba[test_idx] = aligned_proba(
            fold_model, _take(X, test_idx), np.arange(len(classes))
        )

    return proba
