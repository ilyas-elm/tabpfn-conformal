"""Evaluation helpers shared by the tests and every experiment script.

Kept in the core package on purpose: if each experiment defined its own notion
of "coverage" the numbers in the README would not be comparable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "marginal_coverage",
    "coverage_by_class",
    "average_set_size",
    "empty_set_rate",
]


def _as_idx(y, classes):
    """Map labels onto column positions of ``pred_sets``.

    Uses a positional lookup rather than ``np.searchsorted``. searchsorted is
    only correct when ``classes`` is sorted, and these functions take ``classes``
    from the caller -- while :func:`coverage_by_class` keys its result by
    ``enumerate(classes)``, i.e. positionally. An unsorted ``classes`` therefore
    made the two disagree, attaching each coverage to the wrong label, or raised
    an out-of-bounds IndexError if it happened to run off the end. Both are
    worse than not caring about order, so this does not care about order.
    """
    classes = np.asarray(classes)
    y = np.asarray(y).ravel()
    lookup = {c: k for k, c in enumerate(classes.tolist())}
    try:
        return np.array([lookup[v] for v in y.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"y_true contains label {exc.args[0]!r}, which is not in classes "
            f"{classes.tolist()}."
        ) from None


def marginal_coverage(pred_sets: np.ndarray, y_true, classes) -> float:
    """Fraction of samples whose true label is in the set, over all classes."""
    idx = _as_idx(y_true, classes)
    return float(pred_sets[np.arange(len(idx)), idx].mean())


def coverage_by_class(pred_sets: np.ndarray, y_true, classes) -> dict:
    """Coverage computed separately within each true class.

    This is the number that matters under imbalance: a marginal coverage of 0.95
    is perfectly compatible with catching almost no fraud.
    """
    idx = _as_idx(y_true, classes)
    hit = pred_sets[np.arange(len(idx)), idx]
    return {
        c: (float(hit[idx == k].mean()) if np.any(idx == k) else float("nan"))
        for k, c in enumerate(classes)
    }


def average_set_size(pred_sets: np.ndarray) -> float:
    """Mean number of labels per prediction set: the price of the guarantee."""
    return float(pred_sets.sum(axis=1).mean())


def empty_set_rate(pred_sets: np.ndarray) -> float:
    """Fraction of empty sets -- samples the calibrated model refuses to place."""
    return float((pred_sets.sum(axis=1) == 0).mean())
