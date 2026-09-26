"""Nonconformity scores.

A score function maps predicted probabilities of shape ``(n_samples, n_classes)``
to nonconformity values of the same shape: entry ``[i, k]`` is how *strange* it
would be for sample ``i`` to carry label ``k``. Larger means stranger.

Note on the choice of score
---------------------------
For a fixed per-class threshold, any strictly increasing transform of
``1 - p`` yields *identical* prediction sets, because conformal prediction only
ever compares a test score against a quantile of calibration scores and a
monotone map preserves that ordering. ``neg_log_prob`` is therefore provided
for familiarity with the literature, not because it changes any result --
``test_scores.py`` asserts the equivalence rather than leaving it implied.

What *does* change the sets is which calibration points the quantile is taken
over. That is the job of :mod:`tabpfn_conformal.calibration` (marginal vs
class-conditional) and of the label-budget allocation this package exists to
study.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["one_minus_prob", "neg_log_prob", "get_score", "SCORES"]

ScoreFn = Callable[[np.ndarray], np.ndarray]


def one_minus_prob(proba: np.ndarray) -> np.ndarray:
    """``1 - p``: the default. Bounded in [0, 1], no special cases at p = 0."""
    return 1.0 - np.asarray(proba, dtype=float)


def neg_log_prob(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``-log p``, clipped at ``eps`` to keep p = 0 finite."""
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0)
    return -np.log(p)


SCORES: dict[str, ScoreFn] = {
    "one_minus_prob": one_minus_prob,
    "neg_log_prob": neg_log_prob,
}


def get_score(score: str | ScoreFn) -> ScoreFn:
    """Resolve a score name or pass a callable straight through."""
    if callable(score):
        return score
    try:
        return SCORES[score]
    except KeyError:
        raise ValueError(
            f"Unknown score {score!r}. Available: {sorted(SCORES)}, "
            "or pass a callable mapping (n, k) probabilities to (n, k) scores."
        ) from None
