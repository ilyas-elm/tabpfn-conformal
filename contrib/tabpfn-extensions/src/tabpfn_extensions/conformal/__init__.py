"""Distribution-free coverage guarantees for TabPFN classification.

`cp_missing_data` already provides conformal *regression* intervals specialised
to missing-data patterns. This module covers the classification side: prediction
sets with a finite-sample, distribution-free coverage guarantee, including the
class-conditional (Mondrian) variant that extreme class imbalance needs.

    from tabpfn_extensions.conformal import ConformalClassifier
    from tabpfn_extensions.utils import TabPFNClassifier

    cc = ConformalClassifier(TabPFNClassifier(), method="mondrian", strategy="cross")
    cc.fit(X_pool, y_pool)
    sets = cc.predict_set(X_new, alpha=0.05)     # (n, n_classes) boolean

Nothing here imports TabPFN: it wraps any estimator exposing `predict_proba`,
and depends only on numpy and scikit-learn. `strategy="cross"` is worth knowing
about specifically for TabPFN — K-fold cross-conformal spends none of the
labelled positives on calibration, which normally costs K refits and costs K
forward passes for a model with no training step.
"""

from .adaptive import ACI
from .calibration import (
    InsufficientCalibrationWarning,
    conformal_quantile,
    marginal_thresholds,
    mondrian_thresholds,
)
from .crossconformal import out_of_fold_proba
from .decision import decision_summary, route
from .metrics import (
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_coverage,
)
from .scores import SCORES, get_score, neg_log_prob, one_minus_prob
from .wrapper import ConformalClassifier

__all__ = [
    "ConformalClassifier",
    "ACI",
    "route",
    "decision_summary",
    "conformal_quantile",
    "marginal_thresholds",
    "mondrian_thresholds",
    "InsufficientCalibrationWarning",
    "out_of_fold_proba",
    "one_minus_prob",
    "neg_log_prob",
    "get_score",
    "SCORES",
    "marginal_coverage",
    "coverage_by_class",
    "average_set_size",
    "empty_set_rate",
]
