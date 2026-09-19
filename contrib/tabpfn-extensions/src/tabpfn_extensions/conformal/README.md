# conformal

Distribution-free coverage guarantees for TabPFN **classification**.

`cp_missing_data` covers conformal *regression* intervals under missing-data
patterns. This module covers classification: prediction sets carrying a
finite-sample, distribution-free guarantee, with the class-conditional variant
that imbalanced problems need.

```python
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.conformal import ConformalClassifier, coverage_by_class

cc = ConformalClassifier(TabPFNClassifier(), method="mondrian", strategy="cross")
cc.fit(X_pool, y_pool)
sets = cc.predict_set(X_test, alpha=0.05)        # (n, n_classes) boolean
coverage_by_class(sets, y_test, cc.classes_)     # per-class realised coverage
```

## What it provides

| | |
|---|---|
| `ConformalClassifier` | scikit-learn compatible wrapper around any `predict_proba` estimator |
| `method="mondrian"` | class-conditional calibration — one threshold per class |
| `strategy="cross"` | K-fold cross-conformal: no labels spent on calibration |
| `ACI` | adaptive conformal inference for drifting streams |
| `route` | prediction sets to approve / block / review under a review budget |

## Why `strategy="cross"` matters for TabPFN

Split conformal holds out part of the labelled data to calibrate, so under
extreme imbalance a large share of the scarce positives never reach the model.
Cross-conformal avoids that, and normally costs K refits. TabPFN has no training
step, so it costs K forward passes.

Measured on Bank Account Fraud (NeurIPS 2022) with TabPFN-3.5, comparing at an
identical targeted coverage level: cross-conformal reached the same level from
**half the confirmed fraud labels**, with narrower prediction sets in five of six
comparisons.

## Two behaviours worth knowing

**`alpha` is supplied at prediction time**, not calibration time. The calibration
scores are stored and thresholds derived on demand, so sweeping `alpha` costs no
further calls to the estimator — which matters when each call is a metered API
request. `predict_set_from_proba` goes further: score once, sweep offline.

**Too few calibration points raises rather than clips.** A calibration set of
size `n` can only certify `alpha >= 1/(n+1)`. Below that the module returns the
trivial all-labels set and raises `InsufficientCalibrationWarning`, instead of
silently returning a threshold that does not carry the guarantee.

## Attribution

Vendored from [tabpfn-conformal](https://github.com/ilyas-elm/tabpfn-conformal)
(Apache 2.0), where the benchmarks, figures and full method notes live.
