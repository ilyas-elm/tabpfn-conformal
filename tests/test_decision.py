"""The decision layer: actions, budget discipline, and priority."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_conformal import decision_summary, route
from tabpfn_conformal.decision import APPROVE, BLOCK, REVIEW

SETS = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [0, 0]], dtype=bool)
PROBA = np.array([[0.9, 0.1], [0.2, 0.8], [0.45, 0.55], [0.7, 0.3], [0.5, 0.5]])


def test_singletons_become_approve_and_block():
    a = route(SETS, PROBA, budget_k=5)
    assert a[0] == APPROVE and a[1] == BLOCK


def test_empty_set_is_reviewed_first():
    """An empty set means both labels were ruled out -- the most informative case."""
    a = route(SETS, PROBA, budget_k=1)
    assert a[4] == REVIEW
    assert REVIEW not in a[[2, 3]]


def test_budget_is_never_exceeded():
    for k in range(6):
        assert (route(SETS, PROBA, budget_k=k) == REVIEW).sum() <= k


def test_zero_budget_reviews_nothing():
    a = route(SETS, PROBA, budget_k=0)
    assert REVIEW not in a
    assert set(a) <= {APPROVE, BLOCK}


def test_priority_spends_budget_on_the_most_suspicious():
    a = route(SETS, PROBA, budget_k=2)
    assert a[4] == REVIEW           # empty set outranks everything
    assert a[2] == REVIEW           # p_fraud 0.55 beats 0.30
    assert a[3] != REVIEW


def test_uncertainty_priority_picks_the_boundary_case():
    sets = np.array([[1, 1], [1, 1]], dtype=bool)
    proba = np.array([[0.49, 0.51], [0.05, 0.95]])
    a = route(sets, proba, budget_k=1, priority="uncertainty")
    assert a[0] == REVIEW and a[1] != REVIEW
    b = route(sets, proba, budget_k=1, priority="fraud_proba")
    assert b[1] == REVIEW and b[0] != REVIEW


@pytest.mark.parametrize("overflow, expected", [("approve", APPROVE), ("block", BLOCK)])
def test_overflow_policies(overflow, expected):
    a = route(SETS, PROBA, budget_k=0, overflow=overflow)
    assert a[2] == expected and a[3] == expected and a[4] == expected


def test_overflow_model_falls_back_to_argmax():
    a = route(SETS, PROBA, budget_k=0, overflow="model")
    assert a[2] == BLOCK       # p_fraud 0.55
    assert a[3] == APPROVE     # p_fraud 0.30


def test_actions_are_exhaustive_and_disjoint():
    a = route(SETS, PROBA, budget_k=2)
    assert len(a) == len(SETS)
    assert set(a) <= {APPROVE, BLOCK, REVIEW}


def test_summary_counts_reviewed_fraud_as_caught():
    a = route(SETS, PROBA, budget_k=2)
    s = decision_summary(a, [0, 1, 1, 0, 1], budget_k=2)
    assert s["n_fraud"] == 3
    assert s["fraud_caught"] == pytest.approx(1.0)
    assert s["n_review"] == 2 and s["budget_used"] == pytest.approx(1.0)


def test_summary_reports_missed_fraud():
    a = route(SETS, PROBA, budget_k=0)      # nothing reviewed
    s = decision_summary(a, [0, 0, 0, 1, 0])
    assert s["fraud_missed"] == 1           # index 3 was approved
    assert s["fraud_caught"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs, err, match",
    [
        ({"budget_k": -1}, ValueError, "non-negative"),
        ({"budget_k": 1, "priority": "nope"}, ValueError, "priority must be"),
        ({"budget_k": 1, "overflow": "nope"}, ValueError, "overflow must be"),
    ],
)
def test_invalid_arguments(kwargs, err, match):
    with pytest.raises(err, match=match):
        route(SETS, PROBA, **kwargs)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        route(SETS, PROBA[:3], budget_k=1)
    with pytest.raises(ValueError, match="must align"):
        decision_summary(np.array([APPROVE]), [0, 1])


@pytest.mark.parametrize("bad", [2, -1, 99])
def test_route_rejects_an_out_of_range_positive_idx(bad):
    """Every other argument reports its own mistake; this one used to not.

    `negative_idx` is derived as `1 - positive_idx`, so a bad value surfaced as
    "index 2 is out of bounds for axis 1 with size 2" from deep inside numpy.
    """
    sets = np.array([[True, False], [False, True]])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(ValueError, match="positive_idx must be 0 or 1"):
        route(sets, proba, budget_k=1, positive_idx=bad)
