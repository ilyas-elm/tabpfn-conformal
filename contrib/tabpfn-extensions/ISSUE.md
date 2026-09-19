# Draft issue (post before the PR, per CONTRIBUTING.md)

**Title:** Conformal prediction for classification

---

`cp_missing_data` provides conformal **regression** intervals specialised to
missing-data patterns. There is currently no conformal prediction for
**classification** in the repo — no prediction sets, no class-conditional
calibration, no cross-conformal.

I'd like to contribute a `conformal` module covering that. Briefly:

- `ConformalClassifier`, scikit-learn compatible, wrapping any `predict_proba`
  estimator — prediction sets with a finite-sample distribution-free guarantee
- class-conditional (Mondrian) calibration, which is what imbalanced problems
  need: marginal conformal can leave the minority class far below `1 - alpha`
  while the overall number looks fine
- K-fold cross-conformal, which spends no labelled positives on calibration.
  This is the piece that seems worth having in *this* repo specifically —
  cross-conformal normally costs K refits, and TabPFN has no training step.
- adaptive conformal inference for drifting streams, and a decision layer that
  maps prediction sets to approve / block / review under a review budget

No dependencies beyond numpy and scikit-learn; nothing in it imports TabPFN.
Tests are CPU-only, `FAST_TEST_MODE` aware, and run in under a second.

Working implementation, benchmarks on Bank Account Fraud with TabPFN-3.5, and
the method notes are here: https://github.com/ilyas-elm/tabpfn-conformal

Happy to adjust the scope or the API before opening a PR — in particular whether
you'd prefer the decision layer left out, since it is more applied than the rest.
