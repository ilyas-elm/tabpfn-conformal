"""Conformal prediction for TabPFN classification.

Vendored from https://github.com/ilyas-elm/tabpfn-conformal (Apache 2.0).
Regenerate with `scripts/build_extension_pr.py` in that repository rather than
editing here, so the two cannot drift apart.
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
