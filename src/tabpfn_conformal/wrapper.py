"""The sklearn-compatible entry point: :class:`ConformalClassifier`."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .calibration import marginal_thresholds, mondrian_thresholds
from .crossconformal import _take, aligned_proba, out_of_fold_proba
from .scores import get_score

__all__ = ["ConformalClassifier"]

_METHODS = ("marginal", "mondrian")
_STRATEGIES = ("split", "cross")


class ConformalClassifier(BaseEstimator, ClassifierMixin):
    """Wrap any ``predict_proba`` classifier in a coverage guarantee.

    Parameters
    ----------
    base_estimator
        Any estimator exposing ``predict_proba``. Nothing here is specific to
        TabPFN -- the core package deliberately imports neither ``tabpfn`` nor
        ``torch`` -- but see ``strategy="cross"`` for why TabPFN changes what is
        affordable.
    method : {"marginal", "mondrian"}
        One shared threshold, or one per class. Use ``"mondrian"`` whenever the
        minority class is the one you care about.
    strategy : {"split", "cross"}
        ``"split"`` holds out ``cal_size`` of the data to calibrate.
        ``"cross"`` calibrates on out-of-fold scores from ``n_folds`` folds and
        predicts with a model fitted on everything, so no label is spent.
    cal_size : float
        Fraction held out for calibration when ``strategy="split"``. This is a
        constructor argument rather than an explicit ``X_cal`` because how a
        scarce label budget should be divided between a foundation model's
        in-context set and its calibration set is the open question this package
        was built to measure -- so the sweep is a loop over the public API.
    n_folds : int
        Folds when ``strategy="cross"``. Stratified.
    score : str or callable
        Nonconformity score. See :mod:`tabpfn_conformal.scores`.
    prefit : bool
        If True, ``base_estimator`` is already fitted and ``fit(X, y)`` treats
        all of ``(X, y)`` as calibration data. This is also the path used when
        calibration data arrives over time.
    random_state : int or None
        Controls the split and the folds.

    Notes
    -----
    ``alpha`` is supplied at prediction time, not calibration time: the
    calibration *scores* are stored, and thresholds are derived on demand. This
    means a whole sweep over ``alpha`` costs no extra calls to the base
    estimator -- which matters when each call is a metered API request.
    """

    def __init__(
        self,
        base_estimator,
        method: str = "mondrian",
        strategy: str = "split",
        cal_size: float = 0.5,
        n_folds: int = 5,
        score="one_minus_prob",
        prefit: bool = False,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.strategy = strategy
        self.cal_size = cal_size
        self.n_folds = n_folds
        self.score = score
        self.prefit = prefit
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def _validate(self):
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}.")
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_STRATEGIES}, got {self.strategy!r}."
            )
        if not 0.0 < self.cal_size < 1.0:
            raise ValueError(f"cal_size must lie in (0, 1), got {self.cal_size}.")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {self.n_folds}.")
        if not hasattr(self.base_estimator, "predict_proba"):
            raise TypeError(
                f"{type(self.base_estimator).__name__} has no predict_proba; "
                "conformal scores need probabilities."
            )

    def fit(self, X, y):
        """Fit the base estimator (unless ``prefit``) and collect calibration scores."""
        self._validate()
        self.score_fn_ = get_score(self.score)

        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)

        if self.prefit:
            self.estimator_ = self.base_estimator
            cal_proba = aligned_proba(self.estimator_, X, self.classes_)
            cal_idx = y_idx

        elif self.strategy == "split":
            train_pos, cal_pos = train_test_split(
                np.arange(len(y_idx)),
                test_size=self.cal_size,
                stratify=y_idx,
                random_state=self.random_state,
            )
            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(_take(X, train_pos), y[train_pos])
            cal_proba = aligned_proba(self.estimator_, _take(X, cal_pos), self.classes_)
            cal_idx = y_idx[cal_pos]

        else:  # "cross"
            cal_proba = out_of_fold_proba(
                self.base_estimator,
                X,
                y_idx,
                self.classes_,
                n_folds=self.n_folds,
                random_state=self.random_state,
            )
            cal_idx = y_idx
            # Every label calibrated above; every label is now also context.
            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(X, y)

        self._store_calibration(cal_proba, cal_idx)
        return self

    def calibrate(self, X_cal, y_cal):
        """Recalibrate an already-fitted wrapper on fresh labelled data.

        Replaces the stored calibration scores without touching the base
        estimator. This is the path for the drift setting, where labels arrive
        month by month and re-encoding the context is the expensive part.
        """
        check_is_fitted(self, "estimator_")
        y_cal = np.asarray(y_cal).ravel()
        unknown = set(np.unique(y_cal)) - set(self.classes_)
        if unknown:
            raise ValueError(f"calibrate() saw labels absent at fit time: {unknown}.")
        cal_proba = aligned_proba(self.estimator_, X_cal, self.classes_)
        self._store_calibration(cal_proba, np.searchsorted(self.classes_, y_cal))
        return self

    def _store_calibration(self, cal_proba, cal_idx):
        scores_all = np.asarray(self.score_fn_(cal_proba), dtype=float)
        if scores_all.shape != cal_proba.shape:
            raise ValueError(
                f"score function returned {scores_all.shape}, expected {cal_proba.shape}."
            )
        self.calibration_scores_ = scores_all[np.arange(len(cal_idx)), cal_idx]
        self.calibration_labels_ = cal_idx
        self.n_calibration_ = {
            self.classes_[k]: int((cal_idx == k).sum()) for k in range(self.n_classes_)
        }

    # -------------------------------------------------------------- predict

    def quantiles(self, alpha: float) -> dict:
        """Per-class thresholds at ``alpha``. Marginal returns ``{None: t}``."""
        check_is_fitted(self, "calibration_scores_")
        if self.method == "marginal":
            return marginal_thresholds(self.calibration_scores_, alpha)
        return mondrian_thresholds(
            self.calibration_scores_, self.calibration_labels_, self.n_classes_, alpha
        )

    def predict_set_from_proba(self, proba, alpha: float) -> np.ndarray:
        """Prediction sets from probabilities you already hold.

        Lets an experiment score a test set once and then sweep ``alpha``
        offline -- no further calls to the base estimator, and on a metered API
        no further tokens.
        """
        check_is_fitted(self, "calibration_scores_")
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or proba.shape[1] != self.n_classes_:
            raise ValueError(
                f"proba must have shape (n, {self.n_classes_}), got {proba.shape}."
            )

        scores = np.asarray(self.score_fn_(proba), dtype=float)
        q = self.quantiles(alpha)
        thresholds = (
            np.full(self.n_classes_, q[None])
            if self.method == "marginal"
            else np.array([q[k] for k in range(self.n_classes_)], dtype=float)
        )
        return scores <= thresholds[None, :]

    def predict_set(self, X, alpha: float = 0.1) -> np.ndarray:
        """Boolean ``(n_samples, n_classes)`` membership matrix."""
        check_is_fitted(self, "estimator_")
        return self.predict_set_from_proba(
            aligned_proba(self.estimator_, X, self.classes_), alpha
        )

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return aligned_proba(self.estimator_, X, self.classes_)

    def predict(self, X) -> np.ndarray:
        """Point prediction, unchanged by calibration. Provided for sklearn."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
