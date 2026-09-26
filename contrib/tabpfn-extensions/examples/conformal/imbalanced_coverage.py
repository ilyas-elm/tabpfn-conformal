"""Conformal prediction sets for TabPFN classification.

Shows the two things this module adds over an uncalibrated probability: a
per-class coverage guarantee, and the cross-conformal strategy that spends none
of the scarce positive labels on calibration.

    python examples/conformal/imbalanced_coverage.py
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.conformal import (
    ConformalClassifier,
    average_set_size,
    coverage_by_class,
)

ALPHA = 0.1

X, y = make_classification(
    n_samples=2000,
    n_features=12,
    n_informative=6,
    weights=[0.98, 0.02],
    flip_y=0.01,
    random_state=0,
)
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=0
)
print(f"pool: {len(y_pool)} rows, {int(y_pool.sum())} positives")

for strategy in ("split", "cross"):
    cc = ConformalClassifier(
        TabPFNClassifier(),
        method="mondrian",
        strategy=strategy,
        n_folds=5,
        random_state=0,
    ).fit(X_pool, y_pool)

    sets = cc.predict_set(X_test, ALPHA)
    cov = coverage_by_class(sets, y_test, cc.classes_)
    print(
        f"{strategy:>6}: calibrated on {cc.n_calibration_[1]:>3} positives | "
        f"minority coverage {cov[1]:.3f} (target {1 - ALPHA:.2f}) | "
        f"mean set size {average_set_size(sets):.3f}"
    )

# Coverage and set size must be read together: a predictor returning every label
# has perfect coverage and no value.
