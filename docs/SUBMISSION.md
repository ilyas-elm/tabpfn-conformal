# Submission text

Paste into the "description of what your project does" field. The form asks for
enough detail that a third-party developer can comprehend the project, so this is
written for a developer, not as marketing.

---

## tabpfn-conformal — distribution-free coverage guarantees for TabPFN-3.5

**Repository:** https://github.com/ilyas-elm/tabpfn-conformal (Apache 2.0)
**Interactive demo:** https://claude.ai/artifact/RQdPAtjvKefEv1iUT1RB1q

### What it does

TabPFN-3.5 returns a probability. A bank's risk committee needs a *guarantee*:
what fraction of fraud will this catch, and can you prove it. This project is a
small scikit-learn-compatible library that wraps any `predict_proba` classifier
in conformal prediction — turning probabilities into **prediction sets with a
finite-sample, distribution-free coverage guarantee** — plus five experiments
measuring what that buys on TabPFN-3.5 and the Bank Account Fraud dataset
(Jesus et al., NeurIPS 2022; 1M rows, 1.1% fraud, real temporal drift).

The core depends only on numpy, pandas and scikit-learn. It never imports TabPFN
or torch, installs in seconds, and its 119 tests run on a laptop CPU in
about three. TabPFN appears only in `experiments/`, reached through the managed
Prior Labs API — so **every figure regenerates from committed results with no API
key and no GPU.**

### The finding

Split conformal, the default everywhere, holds out half your labels to calibrate.
At a 1% base rate that means spending half of ~100 confirmed frauds — the
scarcest thing a fraud desk owns. Cross-conformal spends none, but costs K
refits, which is why nobody reaches for it.

**TabPFN has no training step**, so those K refits are K forward passes and 0
gradient-trained fits. Measured at an *identical targeted coverage level* —
split at budget 2F and cross at budget F calibrate on the same number of
positives, so they compare directly. The one asymmetry favours split, which at
2F also gets twice the in-context rows:

**Cross-conformal reaches the same guarantee from half the confirmed frauds**,
at no cost in set width. Across four datasets and nine paired comparisons it is
significantly wider in **zero** and significantly narrower in one — the scarcest
label budget, which is the regime that matters. The raw win count is 6 of 9, but
the seeds are shared, so the paired test is the one we report.

Three supporting results:

- **TabPFN gives narrower sets than LightGBM** at every matched level (6.9–12.4%),
  and the mechanism is measurable: TabPFN's calibration error is **74–86% lower**
  on identical rows. Conformal is distribution-free, so calibration does not buy
  validity — it buys *efficiency*, meaning fewer cases in a human's queue.
- **Under drift, TabPFN-3.5-Thinking is directionally more robust**, as the fraud
  rate climbs 0.92% → 1.47%. Across three seeds the base model is below its target
  coverage in 9 of 15 seed-months and Thinking in 3 of 15; Thinking was never worse
  on any seed and strictly better on two of three, for ~5% wider sets. Paired by
  seed the gap is 2.0 ± 1.2 months (t ≈ 1.7, n = 3) — **directional, not
  statistically established.** The single-seed version of this looked decisive and
  did not replicate; we report the weaker claim. Thinking has no local weights, so
  this is reachable only through the API.
- **The KV cache makes the evaluation pass 6.8× faster at 200k context** with
  identical prediction sets. Conformal is exactly the workload it assumes: one
  fixed context, scored twice.

### What we got wrong

Five predictions were registered before the experiments ran. **Four were
falsified**, including two of our own about cost — cross-conformal turned out to
be exactly K× the API tokens, not cheaper, and adaptive conformal inference
cannot help at a 46-positive calibration set because only 46 distinct thresholds
exist. All of it is in the README and in `docs/FINDINGS.md`, with the numbers.
`scripts/verify_claims.py` recomputes every figure in the README from the
committed results and fails if any has drifted.

### Library

`ConformalClassifier(estimator, method="mondrian", strategy="cross")` with
`.fit()` and `.predict_set(X, alpha)`. Marginal and class-conditional (Mondrian)
calibration, K-fold cross-conformal, adaptive conformal inference for drifting
streams, and a decision layer mapping prediction sets to approve / block / review
under a fixed analyst budget. Multiclass is supported and tested. Correctness is
verified against MAPIE rather than asserted: prediction sets are exactly
identical to `SplitConformalClassifier` with the `lac` score.

### Relation to tabpfn-extensions

The extensions repo ships `cp_missing_data`, a conformal *regression* interval
estimator specialised to missing-data patterns. There is no conformal prediction
for **classification** — no prediction sets, no class-conditional calibration, no
cross-conformal. This fills that gap and is offered upstream as a PR;
`scripts/build_extension_pr.py` generates the contribution from the library so
the two cannot drift.

### Honest scope

One dataset family. Cross-conformal is approximately valid rather than exactly
valid (Vovk 2015; CV+ worst case 1−2α), and the README says so wherever the
numbers appear. The wall-clock comparison against LightGBM is confounded —
TabPFN runs remotely, LightGBM locally — and every result row is tagged
`wallclock_comparable: false` rather than presented as a speed claim. The
methods are standard; the contribution is measurement and packaging, not new
statistics.
