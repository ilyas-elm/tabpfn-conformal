# tabpfn-conformal

**Conformal prediction, priced for a model that never trains.**

Split conformal prediction makes you choose. With a fraud base rate near 1%, a
pool of 10,000 labelled rows holds roughly a hundred frauds — and split
conformal spends half of them calibrating a guarantee instead of teaching the
model. Cross-conformal removes the choice: every label calibrates *and* every
label is context. The reason nobody reaches for it by default is that it costs
K refits.

**TabPFN-3.5 has no training step.** `fit` swaps the in-context set; no gradient
descent happens. So K folds are K forward passes, and on the Prior Labs API
fits are not even token-charged — only predictions are, and each row is
predicted exactly once regardless of K. The strongest form of calibration
becomes affordable because of a property of the model, not because of anything
clever in this library.

```python
from tabpfn_conformal import ConformalClassifier

cc = ConformalClassifier(base_model, method="mondrian", strategy="cross", n_folds=5)
cc.fit(X_pool, y_pool)
sets = cc.predict_set(X_new, alpha=0.05)   # (n, 2) boolean: is each label in the set?
```

Changing `strategy="split"` to `strategy="cross"` is the whole diff.

> **Status: work in progress** for the Prior Labs TabPFN-3.5 Hackathon
> (deadline 6 October 2026). The library and its test suite are complete and
> run on CPU. The TabPFN experiments and their figures are not yet in the repo;
> the full plan, including pre-registered predictions and what would falsify
> them, is in [`docs/CAHIER-DES-CHARGES.md`](docs/CAHIER-DES-CHARGES.md).
> No results are reported here until they have been measured.

## Why conformal prediction on top of TabPFN

TabPFN is reported to be the best-calibrated tabular model available — the
lowest ECE and Brier scores in independent comparisons. But average calibration
is not a guarantee, and it is not a guarantee *about the class you care about*.
The same literature finds TabPFN becomes increasingly majority-biased as
imbalance grows, and fraud lives entirely in the minority class.

Conformal prediction converts that strength into something a risk committee can
act on: a finite-sample, distribution-free bound on how often the true label is
in the predicted set. Class-conditional (Mondrian) calibration makes that a
promise about fraud specifically, not about the average of fraud and
legitimate traffic.

## Install

```bash
pip install -e ".[dev]"
pytest
```

The core package depends on **numpy, pandas and scikit-learn only**. No torch,
no `tabpfn`, no GPU: it installs in seconds and its tests run on a laptop. The
TabPFN code lives in `experiments/` and is never imported by `src/`.

## What's in the library

| Module | What it does |
|---|---|
| `scores.py` | Nonconformity scores (`one_minus_prob` by default) |
| `calibration.py` | Split-conformal quantiles: marginal and class-conditional (Mondrian) |
| `crossconformal.py` | K-fold out-of-fold scoring — the cheap-with-TabPFN path |
| `wrapper.py` | `ConformalClassifier`, sklearn-compatible |
| `metrics.py` | Coverage by class, set size, empty-set rate |

Coming before the deadline: adaptive conformal inference (`adaptive.py`) for
monthly drift, and a review-budget decision layer (`decision.py`).

### Two design decisions worth knowing

**`alpha` is a prediction-time argument, not a calibration-time one.** The
calibration *scores* are stored and thresholds are derived on demand, so
sweeping `alpha` costs no extra calls to the base estimator. When each call is a
metered API request, that is the difference between one experiment and thirty.
`predict_set_from_proba` goes further: score a test set once, sweep offline.

**`cal_size` is a constructor argument.** How a scarce label budget should be
divided between a foundation model's in-context set and its calibration set is
an open question — one you can only ask of a model with no training step. Making
it a first-class parameter means the sweep is a loop over the public API.

## Relation to MAPIE and crepes

[MAPIE](https://github.com/scikit-learn-contrib/MAPIE) and
[crepes](https://github.com/henrikbostrom/crepes) are excellent, mature
conformal prediction libraries. To be explicit about what is and is not new
here: **MAPIE 1.5 already ships both `SplitConformalClassifier` and
`CrossConformalClassifier`, and crepes ships Mondrian conformal classifiers.**
Cross-conformal is a standard method (Vovk, 2015). This package did not invent
it and does not claim to.

The claim is about *economics*, not method. Cross-conformal is standard and
rarely used, because K refits is a real cost for a gradient-boosted model. For a
model with no training step that cost collapses — and TabPFN-3.5 is that model.
The contribution is measuring what that changes for a fraud-sized label budget,
and packaging it so the change is a one-word diff.

What this package adds on top of the existing libraries:

- **Label-budget allocation as a first-class parameter** (`cal_size`), because
  the context-versus-calibration split is the open question for a training-free
  model.
- **`alpha` at prediction time and `predict_set_from_proba`**, so an experiment
  scores a test set once and sweeps `alpha` offline. Against a metered API this
  is the difference between one billed pass and thirty.
- **Mondrian combined with cross-conformal** in one object, which is the
  combination the imbalanced case actually needs.
- **Online adaptive conformal inference** for monthly drift (in progress).
- A dependency footprint small enough to vendor into
  [`tabpfn-extensions`](https://github.com/PriorLabs/tabpfn-extensions), which
  today has interpretability, embeddings, unsupervised learning and Bayesian
  optimization but no conformal prediction at all.

Correctness is verified against MAPIE rather than asserted:
`tests/test_agreement_with_mapie.py` checks that our marginal split-conformal
prediction sets are **exactly identical** to MAPIE's `SplitConformalClassifier`
with the `lac` conformity score, across three alphas and three seeds.

## Honest caveats

- **Cross-conformal gives approximate validity**, not the exact finite-sample
  guarantee of split conformal. Pooling out-of-fold scores and applying them to
  a model fitted on the full pool is the cross-conformal predictor of Vovk
  (2015); the related CV+ of Barber et al. (2021) bounds worst-case coverage at
  `1 - 2·alpha`. Empirical coverage is reported alongside this caveat, never
  instead of it.
- **Marginal conformal under-covers the minority class** under imbalance. This
  is a known result ([arXiv:2607.27143](https://arxiv.org/abs/2607.27143), *MAKE*
  8(7):190), not a discovery of ours — it is pinned here as a regression test.
- **Too few calibration points in a class** makes the requested `alpha`
  uncertifiable. The library returns the trivial all-labels set and emits
  `InsufficientCalibrationWarning`. It never silently clips.

## Licence

This repository is licensed under the **Apache License 2.0** (see `LICENSE`).

Note that **TabPFN-3.5's model weights are released by Prior Labs under a
separate, non-commercial licence**. This package does not bundle, depend on, or
redistribute them; the experiment scripts call TabPFN through the public
`tabpfn-client` API or through locally downloaded weights that you obtain and
license yourself.
