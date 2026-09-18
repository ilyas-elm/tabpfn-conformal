"""Distribution-free coverage guarantees for tabular classifiers.

Built for TabPFN-3.5, but the core deliberately depends only on numpy, pandas
and scikit-learn -- no torch, no tabpfn, no GPU. TabPFN appears in
``experiments/`` and nowhere else, so this package installs and its tests run in
seconds on a laptop.
"""

from .calibration import (
    InsufficientCalibrationWarning,
    conformal_quantile,
    marginal_thresholds,
    mondrian_thresholds,
)
from .crossconformal import out_of_fold_proba
from .metrics import (
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_coverage,
)
from .scores import SCORES, get_score, neg_log_prob, one_minus_prob
from .wrapper import ConformalClassifier

__version__ = "0.1.0"

__all__ = [
    "ConformalClassifier",
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
    "__version__",
]
