# Submission text

Paste into the "description of what your project does" field. The form asks for
enough detail that a third-party developer can comprehend the project, so this is
written for a developer, not as marketing.

---

## tabpfn-conformal, distribution-free coverage guarantees for TabPFN-3.5

**Repository:** https://github.com/ilyas-elm/tabpfn-conformal (Apache 2.0)
**Interactive demo:** https://claude.ai/artifact/RQdPAtjvKefEv1iUT1RB1q

### What it does

TabPFN-3.5 returns a probability. A bank's risk committee needs a *guarantee*:
what fraction of fraud will this catch, and can you prove it. This project is a
small scikit-learn-compatible library that wraps any `predict_proba` classifier
in conformal prediction, turning probabilities into **prediction sets with a
finite-sample, distribution-free coverage guarantee**, plus seven experiments
measuring what that buys on TabPFN-3.5 and the Bank Account Fraud dataset
(Jesus et al., NeurIPS 2022; 1M rows, 1.1% fraud, real temporal drift).

The core depends only on numpy, pandas and scikit-learn. It never imports TabPFN
or torch, installs in seconds, and its 130 tests run on a laptop CPU in
about three. TabPFN appears only in `experiments/`, reached through the managed
Prior Labs API, so **every figure regenerates from committed results with no API
key and no GPU.**

### The finding

Split conformal, the default everywhere, holds out half your labels to calibrate.
At a 1% base rate that means spending half of ~100 confirmed frauds, the
scarcest thing a fraud desk owns. Cross-conformal spends none, but costs K
refits, which is why nobody reaches for it.

**TabPFN has no training step**, so those K refits are K forward passes and 0
gradient-trained fits. Measured at an *identical targeted coverage level*,
split at budget 2F and cross at budget F calibrate on the same number of
positives, so they compare directly. The one asymmetry favours split, which at
2F also gets twice the in-context rows:

**Cross-conformal reaches the same guarantee from half the confirmed frauds**,
at no cost in set width. Across four datasets and nine paired comparisons it is
significantly wider in **zero** and significantly narrower in one, the scarcest
label budget, which is the regime that matters. The raw win count is 6 of 9, but
the seeds are shared, so the paired test is the one we report.

It **replicates on a second domain**: Forest Cover Type (581k cartographic
observations, cover type 4 at 0.473%, no shared column with BAF, ships with
scikit-learn), where cross-conformal is significantly wider in **0 of 3** matched
comparisons.

**And it is not free, which we measured rather than assumed.** Split conformal's
guarantee is exact; cross-conformal's is only approximate (Vovk 2015; CV+ worst
case 1−2α). Comparing every run's realized coverage against *the level that run
actually certifies*, split is below it in **0 of 6** dataset-α combinations and
cross in **3 of 6**, by 1.5–2.4 points, concentrated at tight α and replicating
on both datasets. So the trade is: the same targeted level from half the
confirmed positives, against about two points of realized coverage at tight α. A
desk that needs the exact guarantee should use split and find the labels; a desk
that cannot find them now knows the price.

Three supporting results:

- **TabPFN gives narrower sets than LightGBM** at every matched level (6.9–12.4%),
  and the mechanism is measurable: TabPFN's calibration error is **74–86% lower**
  on identical rows. Conformal is distribution-free, so calibration does not buy
  validity; it buys *efficiency*, meaning fewer cases in a human's queue.
- **Under drift, TabPFN-3.5-Thinking is directionally more robust**, as the fraud
  rate climbs 0.92% → 1.47%. Across three seeds the base model is below its target
  coverage in 9 of 15 seed-months and Thinking in 3 of 15; Thinking was never worse
  on any seed and strictly better on two of three, for ~5% wider sets. Paired by
  seed the gap is 2.0 ± 1.2 months (t ≈ 1.7, n = 3), **directional, not
  statistically established.** The single-seed version of this looked decisive and
  did not replicate; we report the weaker claim. Thinking has no local weights, so
  this is reachable only through the API.
- **The KV cache makes the evaluation pass 6.8× faster at 200k context**, same
  answer to four decimals (the probabilities differ by at most 7e-04, two orders
  of magnitude below the seed spread, not bit-identical, and the README says
  so). Conformal is exactly the workload it assumes: one fixed context, scored
  twice. The cache is not free: it front-loads the attention state, so `fit`
  gets *slower* and only the prediction pass gets faster.

### What of TabPFN-3.5 this uses

Conformal prediction is model-agnostic, so it would be easy to claim TabPFN
without leaning on it. Specifically:

- **No training step**, the premise. K-fold cross-conformal is K forward passes,
  0 gradient fits against LightGBM's 6.
- **`thinking_mode`**, the drift comparison above; Thinking has no local weights,
  so it exists only through the managed API.
- **`time_col`**, hands the temporal structure to the model natively. Rejected
  outside thinking mode, so it is a capability, not a free upgrade.
- **`fit_mode="fit_with_cache"`**, the 6.8× evaluation pass.
- **`balance_probabilities`**, TabPFN's own imbalance tooling, used as the
  baseline to beat, since it produces no coverage guarantee.
- **`estimate_cost`**, prices a run from array dimensions before spending. It is
  how we caught our own K× cost overclaim, for free.
- **Local weights**, to take the network out of the wall-clock comparison. On
  one T4 they settle it against TabPFN, which is the point of running it.

Two incompatibilities we had to design around, both settled by a dimension-only
probe before any experiment depended on them: `time_col`/`group_col` are
Thinking-only, and the KV cache is mutually exclusive with Thinking
(`HTTP 422`), which is why the cache and drift results are separate experiments.

### What we got wrong

Five predictions were registered before the experiments ran. **Four were
falsified**, including two of our own about cost, cross-conformal turned out to
be exactly K× the API tokens, not cheaper, and adaptive conformal inference
cannot help at a 46-positive calibration set because only 46 distinct thresholds
exist. All of it is in the README and in `docs/FINDINGS.md`, with the numbers.
`scripts/verify_claims.py` recomputes **178 claims** from the committed
results and exits non-zero on any drift; it runs in CI, needs no API key, and
`python examples/quickstart.py` runs the whole library on synthetic data in
seconds with no key, no GPU and no dataset.

### Library

`ConformalClassifier(estimator, method="mondrian", strategy="cross")` with
`.fit()` and `.predict_set(X, alpha)`. Marginal and class-conditional (Mondrian)
calibration, K-fold cross-conformal, adaptive conformal inference for drifting
streams, and a decision layer mapping prediction sets to approve / block / review
under a fixed analyst budget. Multiclass is supported and tested.

Correctness is verified against MAPIE rather than asserted, at two levels. For
split conformal the sets are **exactly identical** to `SplitConformalClassifier`
with the `lac` score, same estimator, same calibration set, so they must be. For
cross-conformal, which is the headline, ours is Vovk (2015) and MAPIE's is CV+
(Barber et al. 2021), genuinely different constructions, and they produce
**99.7% identical prediction sets**, with coverage and set size agreeing to
within 0.002 at α ∈ {0.05, 0.1, 0.2}.

### Relation to tabpfn-extensions

The extensions repo ships `cp_missing_data`, a conformal *regression* interval
estimator specialised to missing-data patterns. There is no conformal prediction
for **classification**, no prediction sets, no class-conditional calibration, no
cross-conformal. The obvious question is why not MAPIE, and for split conformal
the answer is that you should: our sets are bit-identical to it. For cross it
has no class-conditional option at all, which on a 4% minority at a 90% target
is **0.043** coverage of the minority class against **0.957** here, and its CV+
construction queries `K+1` models per test row where this queries one, which on
a metered model is a standing multiplier on inference. This fills that gap and
is offered upstream as a PR;
`scripts/build_extension_pr.py` generates the contribution from the library so
the two cannot drift.

### Honest scope

Cross-conformal is approximately valid rather than exactly valid (Vovk 2015;
CV+ worst case 1−2α), and that is measured, not just noted: against the level
each run actually certifies, cross sits below in **3 of 6** dataset-α
combinations by 1.5 to 2.4 points where split sits below in **0 of 6**. The
headline replicates on a second domain, Forest Cover Type, which shares no
column with fraud, but both are tabular binary problems with a rare positive
class and nothing here speaks to anything else.

**TabPFN is the slower model, and we measured that rather than leaving it
confounded.** The API-based timings were not a race, since TabPFN ran remotely
and LightGBM on a laptop CPU, so all 36 of those rows stay tagged
`wallclock_comparable: false`. Putting both on one Tesla T4 with local weights
settles it against us: **TabPFN is 35 to 73 times slower**, and removing the
confound moved the result further against it rather than rescuing it. The run
is public, with its log and hardware, at
https://www.kaggle.com/code/ilyaselmaazouzi/tabpfn-conformal. What no hardware
changes is the count: **0 gradient-trained fits against LightGBM's 6**.

The methods are standard; the contribution is measurement and packaging, not new
statistics.
