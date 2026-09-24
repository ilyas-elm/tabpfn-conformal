# Changelog

All notable changes to `tabpfn-conformal`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Corrections to *results* are logged separately and chronologically in
[`docs/FINDINGS.md`](docs/FINDINGS.md), which is the more interesting file: it
records what was measured, what was wrong, and how it was caught.

## [Unreleased]

## [0.1.0], 2026-09-21

First release, built for the Prior Labs TabPFN-3.5 Hackathon.

### Added

- `ConformalClassifier`, scikit-learn compatible, wraps any `predict_proba`
  estimator. `alpha` is a *prediction-time* argument: calibration scores are
  stored and thresholds derived on demand, so sweeping `alpha` costs no further
  calls to the base estimator.
- `method="mondrian"`, class-conditional calibration, one threshold per class.
  Marginal conformal spends its error budget where the mass is, so under heavy
  imbalance the minority class can fall far below `1 - alpha` while the headline
  number looks healthy.
- `strategy="cross"`, K-fold cross-conformal. Every label calibrates *and*
  every label stays in the context, so none is spent on calibration.
- `ACI`, adaptive conformal inference for drifting streams, with
  `update_round`/`update_rounds` (one step per batch) kept distinct from
  `update`/`update_batch` (one step per observation).
- `route` and `decision_summary`, prediction sets to approve / block / review
  under a fixed analyst budget, with empty sets taking priority.
- `marginal_coverage`, `coverage_by_class`, `average_set_size`,
  `empty_set_rate`, the shared metric vocabulary every experiment reports in.
- `one_minus_prob` and `neg_log_prob` nonconformity scores, plus support for a
  custom callable.
- Multiclass support throughout except `route`, which is binary by nature.
- `py.typed` (PEP 561): the package is fully annotated, so downstream type
  checkers now resolve it.

### Notes

- The core depends on numpy, pandas and scikit-learn only. It never imports
  TabPFN or torch; TabPFN appears exclusively under `experiments/`.
- Split conformal carries the exact finite-sample guarantee. Cross-conformal is
  *approximately* valid (Vovk 2015; CV+ worst case `1 - 2*alpha`), and the
  module says so where it matters.
- Verified against MAPIE's `SplitConformalClassifier` with the `lac` conformity
  score: identical prediction sets, not merely similar.
