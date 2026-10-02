# Limitations

Written to be read by someone looking for the weak points, because they exist
and finding them here is better for everyone than finding them in the code.

## What this project does not claim

Conformal prediction is a mature field. **Nothing in the method is new.**

| Component | Prior art |
|---|---|
| Conformal prediction | Vovk, Gammerman & Shafer, ~1998–2005 |
| Class-conditional (Mondrian) CP | Vovk et al.; standard |
| Cross-conformal | Vovk 2015, and MAPIE 1.5 ships `CrossConformalClassifier` |
| Adaptive conformal inference | Gibbs & Candès 2021 |
| Marginal CP under-covering the minority class, and Mondrian fixing it | [arXiv:2607.27143](https://arxiv.org/abs/2607.27143); *MAKE* 8(7):190, both 2026 |
| Cost-sensitive abstention with human review | Same 2607.27143 benchmark |
| Conformal prediction over TabPFN | Moudiki's nnetsauce posts, **regression only** |

What is actually contributed: a tested, installable conformal layer for TabPFN
**classification** (`tabpfn-extensions` does ship one conformal module,
`cp_missing_data`, but it is regression-only and specialised to missing-data
patterns, and there is no conformal classification anywhere in the TabPFN
ecosystem that we could find); a measurement of how a scarce label budget
should be divided between a training-free model's in-context set and its
calibration set, which nobody has published; and the observation that a model
with no training step changes which conformal variant is practical.

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
not promise that an attainable level is well estimated, with 20 calibration
points the realised coverage still has a standard deviation of several
percentage points.

**Coverage and set size must be read together.** A predictor that returns every
label has perfect coverage and no value. This is not hypothetical here: at 25
fraud labels, split conformal scores coverage 1.000 with mean set size 1.806
because it cannot certify α=0.05 on 13 calibration positives. The library emits
`InsufficientCalibrationWarning` in exactly that case rather than clipping the
quantile and pretending.

**ACI trades pointwise coverage for long-run coverage.** Adaptive conformal
inference converges to the target error rate over a sequence; it does not
guarantee coverage in any individual month.

**Exchangeability is assumed and is false.** Bank Account Fraud drifts; that is
why ACI is here, but the split-conformal arms still assume it, and their
guarantees are correspondingly approximate on the later months.

**The zero-shot arm in E8 is a model used outside its domain, and its number
is not a verdict on that model.** Laya is a text decision model; E8 hands it a
JSON serialisation of tabular features and asks a `noul` question. Its AUC on
this task is **0.447**, below chance on all three seeds, against TabPFN's 0.886
and LightGBM's 0.843. Nothing here measures Laya on the inputs it was built for,
and the comparison is not evidence about zero-shot decision models in general.
What E8 does establish is narrower and stronger: a distribution-free guarantee
held at its certified level on a predictor with no usable signal, and the entire
cost of that uselessness appeared in set width and in the review rate rather
than in coverage. Read it as a property of conformal prediction, not as a
benchmark of Laya.

**Seed-to-seed spread understates the uncertainty in *fraud* coverage.**
`make_eval` keeps **every** evaluation fraud and samples the legitimate rows, by
design: it buys a tight fraud-coverage estimate cheaply. The consequence is that
all 2,878 positives are identical in all three seeds, while the 3,000 negatives
are redrawn (overlap 41 of 3,000). So a seed in E4 and E8 varies the negative
sample, the context draw and the calibration split, but never the positive
sample. Every fraud-coverage standard error in this repository is therefore
conditional on one fixed set of frauds, and a genuine resampling of the
positives would be wider.

## Cost and performance caveats

**Cross-conformal is not cheap in relative terms.** Measured: exactly K× split
conformal in API tokens (2.0× at K=2, 20.0× at K=20), because the API prices a
call by total rows touched, so each fold is a full pass over the pool. An
earlier version of this project claimed the folds were nearly free. That was
wrong and is corrected here. What is true is the absolute figure, five folds on
a 10,000-row pool is roughly 50,000 tokens, about 0.25% of a monthly budget.

**Wall-clock is worse than the token ratio.** In the E1 pilot, cross at 200
fraud labels took 673 seconds against 11 for split, because each fold is a
separate upload and round-trip. Roughly 60×, not 5×.

**The KV-cache saving is invisible below ~100,000 context rows.** Measured
`estimate_cost`: 50k rows sit on the 10,000-token minimum charge for both cached
and uncached calls; the documented 75% saving only appears from about 200k rows
up. At very small contexts the cache is actively slower, 9.7s versus 1.8s for a
repeat prediction on a 200-row context.

**The LightGBM wall-clock comparison was confounded. It has been settled, and
it went against us.** P5 predicted LightGBM cross-conformal would cost far more
wall-clock than TabPFN's. Through the API it measured the other way, about 6 s
against TabPFN's 50 s, and the two were not comparable: TabPFN ran remotely on
Prior Labs' GPUs while LightGBM ran on this laptop's CPU, so the TabPFN figure
was dominated by round-trip rather than inference. All 36 rows in
`results/e4.jsonl` still carry `wallclock_comparable: false`, and they always
will, because that run genuinely was not comparable.

The rerun put both on one machine with no network inside the measurement:
TabPFN on a Tesla T4 with local weights, LightGBM on that machine's CPU, which
is where `LGBMClassifier` runs unless it is built and told otherwise. That
removes the network and the remote-API confound, which is what made the first
run incomparable; it does not equalise the accelerator, and this report does not
claim it does. **TabPFN is slower in all four configurations,
by 22 s to 221 s, a factor of 35 to 73.** Removing the confound moved the result
further against TabPFN rather than rescuing it: the network was not what made
TabPFN look slow. Twenty-four rows are in `results/kaggle_wallclock.json`, every
one tagged `wallclock_comparable: true`, produced by
[`wallclock.ipynb`](../experiments/kaggle/wallclock.ipynb). The run is
public, so the hardware, the log and the timings can be read without
rerunning anything: <https://www.kaggle.com/code/ilyaselmaazouzi/tabpfn-conformal>.

Two honest qualifications. The measurement uses *local* weights, not the managed
API, so it does not reproduce the API timings and is not meant to. And one real
effect does survive: split to cross costs TabPFN 4.46× against LightGBM's 6.42×,
so cross-conformal is relatively cheaper on a model with no training step, which
is the mechanism this project rests on. At this scale that effect is swamped,
because LightGBM trains on 9,000 rows in under a second.

What was never in doubt, because it is a count rather than a duration, is
gradient fits: **0 for TabPFN against 6 for LightGBM cross-conformal** at K=5,
and 1 for LightGBM split. That is the number the README leads with, and no
choice of hardware changes it.

**The KV cache and Thinking mode are mutually exclusive** on the managed API,
server-enforced: `HTTP 422, FIT_WITH_CACHE fit mode is not compatible with
thinking mode`. So cache economics and Thinking results cannot appear in the
same experiment.

## Scope

No regression and no fine-tuning of TabPFN. **Multiclass is supported and
tested** (`tests/test_multiclass.py`) even though the benchmarks are binary;
only `decision.route` is binary by nature. Treat multiclass as working software
with no benchmark evidence behind it here, rather than as a validated claim. One dataset (Bank Account Fraud Base); the other five BAF variants and
other imbalanced datasets are untested. `one_minus_prob` is the only score that
changes anything, for a fixed threshold, any strictly increasing transform of
it produces identical prediction sets, which `tests/test_scores.py` asserts.

## Reproducibility

The core library and its 132 tests run on CPU with numpy, pandas and
scikit-learn, in under a second, with no TabPFN and no GPU. The experiments need
a free Prior Labs account. `TabPFN-3.5-Thinking` and `-Plus` have **no local
weights** and are reachable only through the managed API, so those results
cannot be reproduced offline at all. TabPFN's model weights carry a separate
**non-commercial** licence; this repository is Apache 2.0 and neither bundles nor
redistributes them.
