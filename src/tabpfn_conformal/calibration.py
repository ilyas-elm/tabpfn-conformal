"""Split-conformal calibration: marginal and class-conditional (Mondrian).

The whole package rests on one function, :func:`conformal_quantile`. It uses the
finite-sample correction ``ceil((n + 1)(1 - alpha)) / n`` rather than the plain
empirical quantile. Using ``np.quantile(scores, 1 - alpha)`` instead is the most
common conformal bug in the wild; it loses the guarantee for small ``n``, which
is exactly the regime this package targets -- a fraud desk holding a hundred
labelled positives.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = [
    "conformal_quantile",
    "marginal_thresholds",
    "mondrian_thresholds",
    "InsufficientCalibrationWarning",
]


class InsufficientCalibrationWarning(UserWarning):
    """Too few calibration points in a group to certify the requested alpha.

    Raised rather than silently clipping the quantile. The threshold returned is
    ``+inf``, which yields the trivial (all-labels) prediction set: still valid,
    merely useless, and the caller is told so.
    """


def conformal_quantile(scores: np.ndarray, alpha: float, *, group: object = None) -> float:
    """The ``ceil((n + 1)(1 - alpha))``-th smallest calibration score.

    Returns ``+inf`` when ``n`` is too small for the requested ``alpha`` (the
    prediction set becomes everything) and ``-inf`` when ``alpha >= 1`` (the
    prediction set becomes empty). Both are the conservative answers.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")

    n = scores.size
    if n == 0:
        warnings.warn(
            f"No calibration points for group {group!r}; returning a trivial "
            "(all-labels) threshold.",
            InsufficientCalibrationWarning,
            stacklevel=2,
        )
        return np.inf

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k <= 0:
        return -np.inf
    if k > n:
        smallest_alpha = 1.0 / (n + 1)
        warnings.warn(
            f"Group {group!r} has n={n} calibration points, which cannot certify "
            f"alpha={alpha:g} (the smallest achievable alpha is {smallest_alpha:.4g}). "
            "Returning a trivial (all-labels) threshold.",
            InsufficientCalibrationWarning,
            stacklevel=2,
        )
        return np.inf

    return float(np.partition(scores, k - 1)[k - 1])


def marginal_thresholds(cal_scores: np.ndarray, alpha: float) -> dict[object, float]:
    """One threshold shared by every class.

    Guarantees ``P(Y in C(X)) >= 1 - alpha`` *averaged over classes*. Under heavy
    imbalance that average is dominated by the majority class, so the minority
    class can be badly under-covered while the headline number looks fine. That
    failure is asserted as a regression test, not just documented.
    """
    return {None: conformal_quantile(cal_scores, alpha, group="marginal")}


def mondrian_thresholds(
    cal_scores: np.ndarray,
    y_cal_idx: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict[int, float]:
    """One threshold per class, each from that class's own calibration points.

    Guarantees ``P(Y in C(X) | Y = k) >= 1 - alpha`` for every ``k``, which is
    what a fraud desk actually needs: a promise about the fraud class, not about
    the average of fraud and legitimate traffic.
    """
    cal_scores = np.asarray(cal_scores, dtype=float).ravel()
    y_cal_idx = np.asarray(y_cal_idx).ravel()
    if cal_scores.shape != y_cal_idx.shape:
        raise ValueError(
            f"cal_scores {cal_scores.shape} and y_cal_idx {y_cal_idx.shape} must align."
        )

    return {
        k: conformal_quantile(cal_scores[y_cal_idx == k], alpha, group=k)
        for k in range(n_classes)
    }
