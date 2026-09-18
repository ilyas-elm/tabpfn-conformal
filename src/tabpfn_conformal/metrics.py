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
    return np.searchsorted(np.asarray(classes), np.asarray(y).ravel())


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
