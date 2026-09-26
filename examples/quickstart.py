#!/usr/bin/env python3
"""Everything this library does, on synthetic data, in about five seconds.

No API key, no GPU, no dataset, no TabPFN. The conformal machinery is
model-agnostic, so this uses a scikit-learn classifier; swap in
``TabPFNClassifier`` and nothing below changes except the numbers. See
``examples/with_tabpfn.py`` for that version.

    python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from tabpfn_conformal import (
    ConformalClassifier,
    average_set_size,
    coverage_by_class,
    route,
)

ALPHA = 0.1          # we want 90% coverage of the positive class
POSITIVE = 1


def fraud_like(n=6000, rate=0.02, seed=0):
    """An imbalanced binary problem, which is where conformal gets interesting."""
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=6, n_redundant=2,
        weights=[1 - rate, rate], flip_y=0.01, class_sep=1.0, random_state=seed,
    )
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


def report(title, sets, y_true, classes):
    cov = coverage_by_class(sets, y_true, classes)
    print(f"  {title:<34} positive-class coverage {cov[POSITIVE]:.3f}   "
          f"negative {cov[0]:.3f}   mean set size {average_set_size(sets):.3f}")


def main() -> int:
    X_pool, X_test, y_pool, y_test = fraud_like()
    print(f"pool {X_pool.shape[0]} rows, {int(y_pool.sum())} positives "
          f"({y_pool.mean():.1%});  test {X_test.shape[0]} rows, "
          f"{int(y_test.sum())} positives")
    print(f"target: {1 - ALPHA:.0%} coverage of the positive class\n")

    def base():
        return HistGradientBoostingClassifier(max_iter=120, random_state=0)

    # 1. Marginal conformal spends its error budget where the mass is, so under
    #    imbalance the minority class can sit far below 1 - alpha while the
    #    overall number looks healthy. This is the failure Mondrian fixes.
    print("1. marginal calibration vs class-conditional (Mondrian)")
    for method in ("marginal", "mondrian"):
        cc = ConformalClassifier(base(), method=method, strategy="split",
                                 cal_size=0.5, random_state=0).fit(X_pool, y_pool)
        report(f"method={method!r}", cc.predict_set(X_test, alpha=ALPHA),
               y_test, cc.classes_)
    print("  marginal hits its overall target and gives the positive class "
          "nothing: a mean\n  set size below 1 means it is returning empty "
          "sets for the cases that matter.")

    # 2. Split holds out half the labelled positives to calibrate. Cross-conformal
    #    scores every row by a model that did not see it, so every positive
    #    calibrates and none is spent. On TabPFN the K folds cost no training.
    print("\n2. how many labelled positives each strategy actually calibrates on")
    fitted = {}
    for strategy in ("split", "cross"):
        cc = ConformalClassifier(base(), method="mondrian", strategy=strategy,
                                 cal_size=0.5, n_folds=5,
                                 random_state=0).fit(X_pool, y_pool)
        fitted[strategy] = cc
        n_cal = cc.n_calibration_[POSITIVE]
        # A calibration set of size n can only certify alpha >= 1/(n+1).
        print(f"  strategy={strategy!r:<8} calibrates on {n_cal:>4} positives, "
              f"so the tightest certifiable alpha is {1 / (n_cal + 1):.4f} "
              f"({1 - 1 / (n_cal + 1):.2%} coverage)")
        report(f"  -> at alpha={ALPHA}", cc.predict_set(X_test, alpha=ALPHA),
               y_test, cc.classes_)

    # 3. Scores are stored at fit time, so alpha is a prediction-time argument.
    #    Sweeping it costs nothing: no refit, and on a metered API no new calls.
    print("\n3. alpha is a predict-time argument, so sweeping it is free")
    cc = fitted["cross"]
    proba = cc.predict_proba(X_test)
    n_cal = cc.n_calibration_[POSITIVE]
    floor = 1 / (n_cal + 1)
    for a in (0.01, 0.05, 0.1, 0.2):
        sets = cc.predict_set_from_proba(proba, alpha=a)
        cov = coverage_by_class(sets, y_test, cc.classes_)
        note = "  <- below the feasibility floor, so every label is returned" \
            if a < floor else ""
        print(f"  alpha={a:<5} positive coverage {cov[POSITIVE]:.3f}   "
              f"mean set size {average_set_size(sets):.3f}{note}")
    print(f"  (the floor here is 1/(n_cal+1) = {floor:.4f} at n_cal={n_cal}; "
          "the library warns rather than silently clipping)")

    # 4. A prediction set is not yet a decision. Singletons decide themselves;
    #    ambiguous ones need a human, and there are only so many humans.
    print("\n4. turning sets into decisions under a review budget")
    sets = cc.predict_set_from_proba(proba, alpha=ALPHA)
    for budget in (0, 50, 200):
        actions = route(sets, proba, budget, positive_idx=POSITIVE)
        caught = np.mean(actions[y_test == POSITIVE] != "approve")
        print(f"  review budget {budget:>4}: "
              f"{np.sum(actions == 'review'):>4} reviewed, "
              f"{np.sum(actions == 'block'):>4} blocked, "
              f"{caught:.1%} of true positives not auto-approved")

    print("\nCoverage is the guarantee and does not move with the budget. "
          "What moves is how much of it you can act on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
