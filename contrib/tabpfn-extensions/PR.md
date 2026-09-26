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
# their check-changelog workflow fails a PR without this; rename it to the PR number
cp <this-repo>/contrib/tabpfn-extensions/changelog/PRNUMBER.added.md changelog/<PR>.added.md
FAST_TEST_MODE=1 pytest tests/test_conformal.py -v
```

Verified locally: the module imports as `tabpfn_extensions.conformal` and all
**all tests pass under `FAST_TEST_MODE=1` in under a second**, with no TabPFN import and no
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
> - `ConformalClassifier`, scikit-learn compatible, wraps any `predict_proba`
>   estimator
> - `method="mondrian"`, class-conditional calibration, one threshold per class.
>   Marginal conformal spends its error budget where the mass is, so under heavy
>   imbalance the minority class can fall far below `1 - alpha` while the headline
>   number looks healthy.
> - `strategy="cross"`, K-fold cross-conformal, which spends no labelled
>   positives on calibration
> - `ACI`, adaptive conformal inference for drifting streams
> - `route`, prediction sets to approve / block / review under a review budget
>
> **Why `strategy="cross"` is worth having here specifically.** Cross-conformal
> normally costs K refits, which is why split conformal is the default everywhere.
> TabPFN has no training step, so it costs K forward passes. Measured on Bank
> Account Fraud (NeurIPS 2022) with TabPFN-3.5, comparing at an identical targeted
> coverage level, cross-conformal reached the same level from **half the confirmed
> fraud labels** at no cost in set width: across four datasets and nine paired
> comparisons it was significantly wider in **zero** and significantly narrower
> in one, at the scarcest label budget.
>
> **On "why not just use MAPIE".** A fair question. For split conformal the
> answer is that you should: the marginal sets here are bit-identical to
> `SplitConformalClassifier` with `lac`, which is how correctness is tested.
> Cross-conformal is where the two stop being interchangeable, for two
> reasons.
>
> *Capability.* Every conformity score `CrossConformalClassifier` accepts
> (lac, aps, raps, top_k, naive) is marginal, and it exposes no
> class-conditional option, so cross-conformal and Mondrian cannot be
> combined there. On a 4% minority at a 90% target that is measurable: MAPIE
> CV+ gives the minority class **0.043** coverage, this gives it **0.957**.
> Not a MAPIE defect, just what marginal calibration does, but the minority
> class is the one that matters in an imbalanced problem.
>
> *Cost per prediction.* MAPIE cross-conformal is CV+ (Barber et al. 2021),
> which forms each set from the K fold models and so must query every one of
> them for every test row. This is Vovk (2015): thresholds from pooled
> out-of-fold scores, prediction from the single full-data model, so a test
> row is scored once whatever K is. Counted by instrumenting the base
> estimator: `K+1` calls per test row against `1`, exactly, at K in
> {2, 3, 5, 10}. On a local model that is a footnote. On a model billed per
> row predicted it is a `K+1` multiplier on inference cost for as long as it
> is deployed, and the two agree on 99.7% of sets, so it buys nothing back.
>
> What CV+ buys instead is theory, a `1 - 2*alpha` worst-case bound. Pooling
> is only approximately valid and the cost of that is measured below rather
> than waved at. Neither the speed of the conformal code nor its memory is an
> argument: both are a rounding error next to the base model. MAPIE is also
> not currently a dependency of this repository, so adopting this adds none.
>
> **Dependencies:** none beyond numpy and scikit-learn, both already required
> here. Nothing in the module imports TabPFN, so it adds nothing to the base
> install.
>
> **Tests:** 18 tests, CPU only, `FAST_TEST_MODE` aware, about a second. They use
> scikit-learn estimators rather than TabPFN, since the machinery is model-agnostic
> and testing against TabPFN would slow the suite without covering anything extra.
>
> Nine are regression guards, each written after a deliberate mutation of the
> library slipped past the rest: a fold leak in cross-conformal, `<` in place of
> `<=` at the threshold, a class remapping ignored inside a fold, an off-by-one
> in `average_set_size`, and the ACI level being discarded on handoff. Every one
> was confirmed to fail on the mutation it guards.
>
> **Scope.** Prediction sets for any number of classes, `tests/test_multiclass.py`
> covers 3 and 5, where the class-conditional argument is if anything stronger
> (marginal leaves the worst class at 0.675 coverage against a 0.90 target;
> Mondrian holds 0.890). Should compose with `ManyClassClassifier`. The
> benchmarks here are binary, because the motivating problem is.
>
> **What it costs, measured.** Cross-conformal is approximately valid rather
> than exactly valid (Vovk 2015; CV+ worst case `1 - 2*alpha`), and that is not
> only theoretical. Comparing each run against the level it actually certifies
> -- `ceil((n+1)(1-alpha))/n`, not the nominal `1-alpha` -- cross sat below its
> own certified level in **3 of 6** dataset-alpha combinations, by 1.5 to 2.4
> points, concentrated at tight alpha; split was below in **0 of 6**. So the
> trade is half the labels against about two points of realized coverage, not a
> free lunch. The module docstring carries this caveat, so it reaches anyone who
> reads `strategy="cross"` rather than only this PR.
> The methods are standard; this is packaging, not new statistics.
>
> Benchmarks, figures and full method notes:
> https://github.com/ilyas-elm/tabpfn-conformal
