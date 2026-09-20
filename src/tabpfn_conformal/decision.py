"""Turn prediction sets into actions under a fixed analyst budget.

A prediction set is not a decision. A fraud desk has to do something with every
transaction, and it has a finite number of analyst-hours per day. This module
makes the mapping explicit:

- a singleton ``{legitimate}``  -> **approve**
- a singleton ``{fraud}``       -> **block**
- anything ambiguous            -> **review**, if a slot is free

"Ambiguous" covers two cases that people often conflate. ``{legitimate, fraud}``
means the calibrated model cannot rule either out. The **empty set** means it
ruled *both* out -- the point is unlike anything in calibration, which is the
most interesting thing a conformal predictor can tell you, so empty sets take
priority for review.

When ambiguity exceeds the budget, something has to give and the choice should
be visible rather than buried. ``overflow`` decides what happens to the cases
that do not get a slot; it defaults to falling back on the model's own point
prediction, which is what an unaided desk would do anyway.

Prior art note: cost-sensitive conformal abstention with human review is
benchmarked in arXiv:2607.27143 (2026). This module is the applied layer of the
project, not a contribution -- see docs/CAHIER-DES-CHARGES.md section 3.6.
"""

from __future__ import annotations

import numpy as np

__all__ = ["route", "decision_summary", "APPROVE", "BLOCK", "REVIEW"]

APPROVE, BLOCK, REVIEW = "approve", "block", "review"
_PRIORITIES = ("fraud_proba", "uncertainty")
_OVERFLOWS = ("model", "approve", "block")


def route(
    pred_sets: np.ndarray,
    proba: np.ndarray,
    budget_k: int,
    *,
    positive_idx: int = 1,
    priority: str = "fraud_proba",
    overflow: str = "model",
) -> np.ndarray:
    """Map prediction sets to approve / block / review under a review budget.

    Parameters
    ----------
    pred_sets : (n, 2) bool
        From :meth:`ConformalClassifier.predict_set`.
    proba : (n, 2) float
        Predicted probabilities, used to rank and to break overflow ties.
    budget_k : int
        Maximum number of cases that may be sent to a human.
    positive_idx : int
        Column of the fraud class.
    priority : {"fraud_proba", "uncertainty"}
        How to spend the budget. ``fraud_proba`` reviews the most suspicious
        ambiguous cases, which is what a desk optimising for caught fraud wants.
        ``uncertainty`` reviews the cases nearest the decision boundary. Empty
        sets outrank everything under either rule.
    overflow : {"model", "approve", "block"}
        What happens to ambiguous cases with no slot left.

    Returns
    -------
    (n,) array of ``"approve"``, ``"block"`` or ``"review"``.
    """
    pred_sets = np.asarray(pred_sets, dtype=bool)
    proba = np.asarray(proba, dtype=float)
    if pred_sets.shape != proba.shape:
        raise ValueError(
            f"pred_sets {pred_sets.shape} and proba {proba.shape} must have the same shape."
        )
    if pred_sets.ndim != 2 or pred_sets.shape[1] != 2:
        raise ValueError(f"binary only: expected (n, 2), got {pred_sets.shape}.")
    if budget_k < 0:
        raise ValueError(f"budget_k must be non-negative, got {budget_k}.")
    if priority not in _PRIORITIES:
        raise ValueError(f"priority must be one of {_PRIORITIES}, got {priority!r}.")
    if overflow not in _OVERFLOWS:
        raise ValueError(f"overflow must be one of {_OVERFLOWS}, got {overflow!r}.")
    # Every other argument here reports its own mistake; without this one,
    # positive_idx=-1 surfaced as "index 2 is out of bounds", because the
    # negative column is derived as 1 - positive_idx.
    if positive_idx not in (0, 1):
        raise ValueError(f"positive_idx must be 0 or 1 for a binary problem, got {positive_idx}.")

    negative_idx = 1 - positive_idx
    size = pred_sets.sum(axis=1)
    p_fraud = proba[:, positive_idx]

    actions = np.empty(len(pred_sets), dtype=object)
    actions[(size == 1) & pred_sets[:, negative_idx]] = APPROVE
    actions[(size == 1) & pred_sets[:, positive_idx]] = BLOCK

    ambiguous = np.flatnonzero(size != 1)
    if ambiguous.size:
        if priority == "fraud_proba":
            score = p_fraud[ambiguous]
        else:
            score = -np.abs(p_fraud[ambiguous] - 0.5)
        # An empty set means the model ruled out both labels: review it first.
        score = np.where(size[ambiguous] == 0, np.inf, score)

        order = ambiguous[np.argsort(-score, kind="stable")]
        reviewed, spilled = order[:budget_k], order[budget_k:]
        actions[reviewed] = REVIEW
        if spilled.size:
            if overflow == "model":
                actions[spilled] = np.where(
                    proba[spilled, positive_idx] >= proba[spilled, negative_idx],
                    BLOCK,
                    APPROVE,
                )
            else:
                actions[spilled] = overflow

    return actions.astype(str)


def decision_summary(
    actions: np.ndarray, y_true, positive_label=1, budget_k: int | None = None
) -> dict:
    """Operational metrics for a routing decision.

    ``fraud_caught`` counts frauds that were blocked *or* sent to review, on the
    assumption a reviewed fraud is eventually caught -- the optimistic reading,
    and the one to state out loud rather than let a reader assume.
    """
    actions = np.asarray(actions, dtype=str)
    y = np.asarray(y_true).ravel()
    if actions.shape != y.shape:
        raise ValueError(f"actions {actions.shape} and y_true {y.shape} must align.")

    is_fraud = y == positive_label
    n_fraud = int(is_fraud.sum())
    reviewed = actions == REVIEW
    caught = (actions == BLOCK) | reviewed

    out = {
        "n": int(len(y)),
        "n_fraud": n_fraud,
        "n_approve": int((actions == APPROVE).sum()),
        "n_block": int((actions == BLOCK).sum()),
        "n_review": int(reviewed.sum()),
        "review_rate": float(reviewed.mean()),
        "fraud_caught": float((caught & is_fraud).sum() / n_fraud) if n_fraud else float("nan"),
        "fraud_missed": int((~caught & is_fraud).sum()),
        "false_block_rate": float(
            ((actions == BLOCK) & ~is_fraud).sum() / max((~is_fraud).sum(), 1)
        ),
        "review_precision": float(
            (reviewed & is_fraud).sum() / reviewed.sum()
        ) if reviewed.any() else float("nan"),
    }
    if budget_k is not None:
        out["budget_k"] = int(budget_k)
        out["budget_used"] = float(out["n_review"] / budget_k) if budget_k else float("nan")
    return out
