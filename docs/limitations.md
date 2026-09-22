# Limitations

Written to be read by someone looking for the weak points, because they exist
and finding them here is better for everyone than finding them in the code.

## What this project does not claim

Conformal prediction is a mature field. **Nothing in the method is new.**

| Component | Prior art |
|---|---|
| Conformal prediction | Vovk, Gammerman & Shafer, ~1998–2005 |
| Class-conditional (Mondrian) CP | Vovk et al.; standard |
| Cross-conformal | Vovk 2015 — and MAPIE 1.5 ships `CrossConformalClassifier` |
| Adaptive conformal inference | Gibbs & Candès 2021 |
| Marginal CP under-covering the minority class, and Mondrian fixing it | [arXiv:2607.27143](https://arxiv.org/abs/2607.27143); *MAKE* 8(7):190 — both 2026 |
| Cost-sensitive abstention with human review | Same 2607.27143 benchmark |
| Conformal prediction over TabPFN | Moudiki's nnetsauce posts — **regression only** |

What is actually contributed: a tested, installable conformal layer for TabPFN
**classification**. (`tabpfn-extensions` does ship one conformal module,
`cp_missing_data`, but it is regression-only and specialised to missing-data
patterns — there is no conformal classification anywhere in the TabPFN
ecosystem that we could find.) a measurement of how a scarce
label budget should be divided between a training-free model's in-context set
and its calibration set, which nobody has published; and the observation that a
model with no training step changes which conformal variant is practical.

## Statistical caveats

**Cross-conformal is approximately valid, not exactly valid.** Pooling
out-of-fold scores and applying them to a model fitted on the full pool is the
cross-conformal predictor of Vovk (2015); the related CV+ of Barber et al.
(2021) bounds worst-case coverage at `1 − 2α`. Split conformal's guarantee is
exact and finite-sample; cross-conformal's is not. Every reported
cross-conformal coverage is empirical and is printed next to this caveat, never
instead of it.

**The feasibility boundary is about what can be *certified*, not about what is
*achieved*.** `α ≥ 1/(n_cal+1)` says which levels are attainable at all. It does
not promise that an attainable level is well estimated — with 20 calibration
points the realised coverage still has a standard deviation of several
percentage points.

**Coverage and set size must be read together.** A predictor that returns every
label has perfect coverage and no value. This is not hypothetical here: at 25
fraud labels, split conformal scores coverage 1.000 with mean set size 1.807
because it cannot certify α=0.05 on 13 calibration positives. The library emits
`InsufficientCalibrationWarning` in exactly that case rather than clipping the
quantile and pretending.

**ACI trades pointwise coverage for long-run coverage.** Adaptive conformal
inference converges to the target error rate over a sequence; it does not
guarantee coverage in any individual month.

**Exchangeability is assumed and is false.** Bank Account Fraud drifts — that is
why ACI is here — but the split-conformal arms still assume it, and their
guarantees are correspondingly approximate on the later months.

## Cost and performance caveats

**Cross-conformal is not cheap in relative terms.** Measured: exactly K× split
conformal in API tokens (2.0× at K=2, 20.0× at K=20), because the API prices a
call by total rows touched, so each fold is a full pass over the pool. An
earlier version of this project claimed the folds were nearly free. That was
wrong and is corrected here. What is true is the absolute figure — five folds on
a 10,000-row pool is roughly 50,000 tokens, about 0.25% of a monthly budget.

**Wall-clock is worse than the token ratio.** In the E1 pilot, cross at 200
fraud labels took 673 seconds against 11 for split, because each fold is a
separate upload and round-trip. Roughly 60×, not 5×.

**The KV-cache saving is invisible below ~100,000 context rows.** Measured
`estimate_cost`: 50k rows sit on the 10,000-token minimum charge for both cached
and uncached calls; the documented 75% saving only appears from about 200k rows
up. At very small contexts the cache is actively slower — 9.7s versus 1.8s for a
repeat prediction on a 200-row context.

**The LightGBM wall-clock comparison is confounded, and we did not settle it.**
P5 predicted LightGBM cross-conformal would cost far more wall-clock than
TabPFN's. It measured the other way — about 6 s against TabPFN's 50 s at the same
budget — but the two are not comparable: **TabPFN runs remotely on Prior Labs'
GPUs, over the network, and LightGBM runs locally on this laptop's CPU.** The
TabPFN figure is dominated by upload and round-trip, not by inference. Every one
of the 36 rows in `results/e4.jsonl` carries `wallclock_comparable: false` for
this reason. The README quotes these numbers in exactly two places — the opening
cost paragraph and the P5 scoreboard row — and both say the comparison is
confounded rather than resting a claim on it.

What *is* comparable, because it is a count rather than a duration, is gradient
fits: **0 for TabPFN against 6 for LightGBM cross-conformal** at K=5, and 1 for
LightGBM split. Settling the timing properly needs both models on one machine
with one accelerator.

That run is now written and tested end to end —
[`experiments/kaggle/wallclock.py`](../experiments/kaggle/wallclock.py), with
[`analyze_kaggle.py`](../experiments/analyze_kaggle.py) to read it — but it has
not been *run*, because it needs one GPU session. **So P5 remains open, and the
numbers above remain the confounded ones.** The script refuses to pretend
otherwise: on CPU it warns on stderr and the analysis says in its own output
that the run settles nothing. It measures local TabPFN rather than the managed
API, deliberately, since removing the network is the entire point.

**The KV cache and Thinking mode are mutually exclusive** on the managed API,
server-enforced: `HTTP 422 — FIT_WITH_CACHE fit mode is not compatible with
thinking mode`. So cache economics and Thinking results cannot appear in the
same experiment.

## Scope

No regression and no fine-tuning of TabPFN. **Multiclass is supported and
tested** (`tests/test_multiclass.py`) even though the benchmarks are binary —
only `decision.route` is binary by nature. Treat multiclass as working software
with no benchmark evidence behind it here, rather than as a validated claim. One dataset (Bank Account Fraud Base); the other five BAF variants and
other imbalanced datasets are untested. `one_minus_prob` is the only score that
changes anything — for a fixed threshold, any strictly increasing transform of
it produces identical prediction sets, which `tests/test_scores.py` asserts.

## Reproducibility

The core library and its 129 tests run on CPU with numpy, pandas and
scikit-learn, in under a second, with no TabPFN and no GPU. The experiments need
a free Prior Labs account. `TabPFN-3.5-Thinking` and `-Plus` have **no local
weights** and are reachable only through the managed API, so those results
cannot be reproduced offline at all. TabPFN's model weights carry a separate
**non-commercial** licence; this repository is Apache 2.0 and neither bundles nor
redistributes them.
