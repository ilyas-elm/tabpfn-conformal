"""Adaptive conformal inference (ACI) -- online recalibration under drift.

Split conformal assumes exchangeability. Fraud does not oblige: in the Bank
Account Fraud suite the positive rate falls to 0.875% by month 2 and climbs to
1.475% by month 7. A threshold frozen in month 2 is calibrated for a world that
no longer exists.

ACI (Gibbs & Candes, 2021) fixes this without refitting anything. It treats the
*level* as the thing to adapt: after each observation, if the true label fell
outside the prediction set, lower the effective alpha so future sets widen; if
it fell inside, raise it so they tighten.

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

The long-run empirical error rate converges to ``alpha_target`` for *any*
sequence -- including adversarial ones -- because this is a regret bound, not a
distributional assumption. The price is that coverage holds on average over
time rather than at every step.

Class-conditional by default, which matters here. One shared alpha would be
driven almost entirely by the 99% of traffic that is legitimate and would barely
register a change in fraud behaviour. :class:`ACI` keeps one level per class and
updates a class's level only when a label of that class arrives.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ACI"]


class ACI:
    """Online per-class conformal level.

    Parameters
    ----------
    alpha_target : float
        The miscoverage rate to converge to, per class.
    gamma : float
        Step size. Larger adapts faster and oscillates more. Gibbs & Candes use
        0.005-0.05; 0.01 is a reasonable default for monthly batches.
    n_classes : int
        Number of classes to track.
    clip : tuple[float, float]
        Bounds on the effective level. The update is unconstrained in the
        original formulation and can leave [0, 1]; outside those bounds the
        prediction set degenerates to everything or nothing, which is valid but
        uninformative. Clipping keeps it interpretable and is standard practice.

    Examples
    --------
    >>> aci = ACI(alpha_target=0.05, gamma=0.02, n_classes=2)
    >>> aci.alpha(1)
    0.05
    >>> round(aci.update(1, covered=False), 4)   # missed a fraud -> widen
    0.031
    >>> round(aci.update(1, covered=True), 4)    # caught one -> tighten back
    0.032
    """

    def __init__(
        self,
        alpha_target: float,
        gamma: float = 0.01,
        n_classes: int = 2,
        clip: tuple[float, float] = (1e-4, 0.5),
    ):
        if not 0.0 < alpha_target < 1.0:
            raise ValueError(f"alpha_target must lie in (0, 1), got {alpha_target}.")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        if not clip[0] < clip[1]:
            raise ValueError(f"clip must be (low, high) with low < high, got {clip}.")

        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.n_classes = int(n_classes)
        self.clip = clip

        self.alphas_ = np.full(self.n_classes, float(alpha_target))
        self.n_updates_ = np.zeros(self.n_classes, dtype=int)
        self.n_errors_ = np.zeros(self.n_classes, dtype=float)
        self.history_: list[dict] = []

    # -------------------------------------------------------------- reading

    def alpha(self, class_idx: int) -> float:
        """Current effective level for one class."""
        return float(self.alphas_[class_idx])

    def alpha_dict(self) -> dict[int, float]:
        """Levels for every class, in the form ``predict_set`` accepts."""
        return {k: float(self.alphas_[k]) for k in range(self.n_classes)}

    def empirical_error(self, class_idx: int) -> float:
        """Realised miscoverage so far for one class, or nan before any update."""
        n = self.n_updates_[class_idx]
        return float(self.n_errors_[class_idx] / n) if n else float("nan")

    # -------------------------------------------------------------- writing

    def update(self, class_idx: int, covered: bool) -> float:
        """Fold in one observation of true class ``class_idx``.

        ``covered`` is whether the true label was in its prediction set.
        Returns the class's new level.
        """
        if not 0 <= class_idx < self.n_classes:
            raise IndexError(f"class_idx {class_idx} outside 0..{self.n_classes - 1}.")

        err = 0.0 if covered else 1.0
        new = self.alphas_[class_idx] + self.gamma * (self.alpha_target - err)
        self.alphas_[class_idx] = float(np.clip(new, *self.clip))

        self.n_updates_[class_idx] += 1
        self.n_errors_[class_idx] += int(not covered)
        return self.alpha(class_idx)

    def update_round(self, class_idx: int, error_rate: float) -> float:
        """One ACI step for a whole batch, using its observed miscoverage rate.

        Use this when predictions arrive in batches -- a month of transactions,
        a daily file -- rather than one point at a time.

        The distinction matters more than it looks. :meth:`update` applies one
        step of size ``gamma`` per observation, which is correct when you
        predict one point, see its label, and predict the next. Feeding a whole
        month through it applies as many steps as the month has rows, so a
        batch of 1,400 frauds adapts 1,400 times before the next prediction and
        the level slams into its clip bounds. Observed in E3 before this method
        existed: the fraud level oscillated 0.05 -> 0.5 -> 0.0001 -> 0.5 across
        five months and coverage swung between 0.517 and 1.000.

        One round, one step, using the rate the batch actually realised.
        """
        if not 0 <= class_idx < self.n_classes:
            raise IndexError(f"class_idx {class_idx} outside 0..{self.n_classes - 1}.")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must lie in [0, 1], got {error_rate}.")

        new = self.alphas_[class_idx] + self.gamma * (self.alpha_target - error_rate)
        self.alphas_[class_idx] = float(np.clip(new, *self.clip))
        self.n_updates_[class_idx] += 1
        self.n_errors_[class_idx] += error_rate
        return self.alpha(class_idx)

    def update_rounds(self, y_idx, covered) -> dict[int, float]:
        """One :meth:`update_round` per class, from a batch of outcomes."""
        y_idx = np.asarray(y_idx).ravel()
        covered = np.asarray(covered).ravel().astype(bool)
        if y_idx.shape != covered.shape:
            raise ValueError(
                f"y_idx {y_idx.shape} and covered {covered.shape} must align."
            )
        for k in range(self.n_classes):
            mask = y_idx == k
            if mask.any():
                self.update_round(k, float(1.0 - covered[mask].mean()))
        return self.alpha_dict()

    def update_batch(self, y_idx, covered) -> dict[int, float]:
        """Fold in a batch one observation at a time, in order.

        This is the genuinely online protocol: one step per point. For
        batched arrivals use :meth:`update_rounds` instead -- see the note
        there for why the difference is not cosmetic.
        """
        y_idx = np.asarray(y_idx).ravel()
        covered = np.asarray(covered).ravel().astype(bool)
        if y_idx.shape != covered.shape:
            raise ValueError(
                f"y_idx {y_idx.shape} and covered {covered.shape} must align."
            )
        for k, c in zip(y_idx, covered):
            self.update(int(k), bool(c))
        return self.alpha_dict()

    def record(self, **fields) -> None:
        """Append a snapshot to ``history_`` -- used by the drift experiment."""
        self.history_.append(
            {
                **fields,
                "alphas": self.alpha_dict(),
                "empirical_error": {
                    k: self.empirical_error(k) for k in range(self.n_classes)
                },
            }
        )

    def __repr__(self) -> str:
        levels = ", ".join(f"{a:.4f}" for a in self.alphas_)
        return (
            f"ACI(alpha_target={self.alpha_target:g}, gamma={self.gamma:g}, "
            f"levels=[{levels}])"
        )
