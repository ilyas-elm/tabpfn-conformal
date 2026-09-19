# Contributing `conformal` to tabpfn-extensions

Everything under `contrib/tabpfn-extensions/` is **generated** by
`scripts/build_extension_pr.py` from `src/tabpfn_conformal/`. Regenerate rather
than editing here, so the contribution cannot drift from the library.

## Before the PR

CONTRIBUTING.md asks for an issue first:

> Before developing a feature / opening a PR, please open a GitHub issue and
> describe the bug or feature request.

Draft issue text is in [`ISSUE.md`](ISSUE.md).

## Opening it

```bash
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions && uv sync
cp -R <this-repo>/contrib/tabpfn-extensions/src/tabpfn_extensions/conformal src/tabpfn_extensions/
cp <this-repo>/contrib/tabpfn-extensions/tests/test_conformal.py tests/
cp -R <this-repo>/contrib/tabpfn-extensions/examples/conformal examples/
FAST_TEST_MODE=1 pytest tests/test_conformal.py -v
```

Verified locally: the module imports as `tabpfn_extensions.conformal` and all
**9 tests pass under `FAST_TEST_MODE=1` in 0.84s**, with no TabPFN import and no
GPU.

## PR description

> ### Conformal prediction for classification
>
> `cp_missing_data` already provides conformal **regression** intervals
> specialised to missing-data patterns. This adds the classification side:
> prediction sets with a finite-sample, distribution-free coverage guarantee.
>
> **What it adds**
>
> - `ConformalClassifier` — scikit-learn compatible, wraps any `predict_proba`
>   estimator
> - `method="mondrian"` — class-conditional calibration, one threshold per class.
>   Marginal conformal spends its error budget where the mass is, so under heavy
>   imbalance the minority class can fall far below `1 - alpha` while the headline
>   number looks healthy.
> - `strategy="cross"` — K-fold cross-conformal, which spends no labelled
>   positives on calibration
> - `ACI` — adaptive conformal inference for drifting streams
> - `route` — prediction sets to approve / block / review under a review budget
>
> **Why `strategy="cross"` is worth having here specifically.** Cross-conformal
> normally costs K refits, which is why split conformal is the default everywhere.
> TabPFN has no training step, so it costs K forward passes. Measured on Bank
> Account Fraud (NeurIPS 2022) with TabPFN-3.5, comparing at an identical targeted
> coverage level, cross-conformal reached the same level from **half the confirmed
> fraud labels**, with narrower prediction sets in five of six comparisons.
>
> **Dependencies:** none beyond numpy and scikit-learn. Nothing in the module
> imports TabPFN, so it adds nothing to the base install.
>
> **Tests:** 9 tests, CPU only, `FAST_TEST_MODE` aware, under a second. They use
> scikit-learn estimators rather than TabPFN, since the machinery is model-agnostic
> and testing against TabPFN would slow the suite without covering anything extra.
>
> **Honest scope.** Binary and multiclass prediction sets; the benchmarks are
> binary only. Cross-conformal is approximately valid rather than exactly valid
> (Vovk 2015; CV+ worst case `1 - 2*alpha`) and the module says so where it matters.
> The methods are standard — this is packaging, not new statistics.
>
> Benchmarks, figures and full method notes:
> https://github.com/ilyas-elm/tabpfn-conformal
