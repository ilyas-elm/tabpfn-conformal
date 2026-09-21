# Findings log

Chronological record of what was discovered, what it changed, and what turned
out to be wrong. Kept because the interesting parts of this project are the
corrections, and because four of them exist nowhere else.

Conventions: **⚑** = changed the plan · **✗** = something we claimed that was
wrong · **⚙** = engineering, not science.

---

## 18 September — before any code

**⚑ The 50% criterion is the project's structural weakness.** A deliberately
model-agnostic library is an argument that TabPFN is interchangeable. Fix: keep
the clean core, but make the headline a claim that is *false for other models*.

**⚑ The planned headline was already published.** [arXiv:2607.27143](https://arxiv.org/abs/2607.27143)
(July 2026) benchmarks marginal vs Mondrian CP **plus** cost-sensitive abstention
with human review across 15 imbalanced tabular datasets — the Mondrian claim and
the decision layer both. *MAKE* 8(7):190 does Mondrian at 1:345 imbalance. Both
demoted to cited background; the decision layer became an application demo.

**⚑ TabPFN-3.5-Thinking and -Plus have no local weights.** API, SageMaker or SAP
only. Thinking tops TabArena, BeyondArena, STRABLE and MulTaBench. So a Kaggle
GPU runs the *second-best* model, and the API is not a budget compromise but the
only route to the best one. This settled the compute plan.

**⚑ `tabpfn-extensions` has no conformal prediction for classification.**
*(Originally recorded as "no conformal prediction at all" — corrected 20 Sept,
see below.)* Apache 2.0, takes contributions.

**Correction to the brief:** BAF Base is **1.10%** fraud (~1 in 91), not 1 in 300.

---

## 19 September — building, then measuring

**✗ Cross-conformal is not new, and MAPIE already ships it.** MAPIE 1.5 has
`CrossConformalClassifier` as well as `SplitConformalClassifier`; crepes has
Mondrian classifiers. Found while writing the agreement test. The claim had to
narrow from "we bring cross-conformal to tabular data" to a claim about
*economics*. Said outright in the README before a judge could say it first.

**⚙ The auth variable is `TABPFN_TOKEN`.** An earlier draft invented
`PRIORLABS_API_TOKEN`. Read from the client source. A plausible-looking guess is
worse than an open question because it silently becomes a fact in committed code.

**⚑ `estimate_cost` takes an `operation` argument** — `predict`, `cache_predict`,
`thinking_fit`, `thinking_predict` — and transmits dimensions only. So the entire
cost model can be measured for **zero tokens**.

**Spike S1 — the KV cache and Thinking are mutually exclusive.** Server-enforced:
`HTTP 422 — FIT_WITH_CACHE fit mode is not compatible with thinking mode`. The
documentation contradicted itself; the measurement settled it. Cache economics
attach to E1/E2, Thinking to E3, and they cannot share a figure.

**Cost is not a constraint, and the cache saving is invisible at our scale.**

| context rows | normal | cached | saving |
|---:|---:|---:|---:|
| 50,000 | 10,000 | 10,000 | 0% |
| 100,000 | 19,039 | 10,000 | 47% |
| 200,000 | 68,319 | 17,080 | **75%** |

Everything under ~100k rows sits on a 10,000-token minimum. The documented 75%
is real but only appears above that. At 200 rows the cache is *slower* — 9.7s vs
1.8s for a repeat prediction. **⚑ Consequence: any experiment meant to
demonstrate cache economics must run at ≥100k context rows.**

**✗ P2 falsified — cross-conformal costs exactly K×, not under 2×.** Measured
2.0× at K=2, 5.0× at K=5, 20.0× at K=20. The API prices a call by *total rows
touched*, so each fold is a full pass over the pool and the folds add up
linearly. "Fits are unbilled" is true and irrelevant: the predicts are what cost.
The original "K folds cost about the same as one" framing was wrong and a judge
could have disproved it with one free call.

**Reproduced and committed 20 Sept.** These numbers lived only in a session
transcript for most of the project — the headline correction of an overclaim,
itself unbacked by any artifact. `experiments/api/cost_kfold.py` now re-quotes
them and writes `results/cost_kfold.json`: ratio equals K in 6 of 6 quotes,
across K ∈ {2, 5, 20} and pools of 10k and 100k. `verify_claims` reads the
committed file, so it needs no token.

**⚙ `time_col` is rejected outside thinking mode.** `group_col`,
`time_col`, `group_time_col` — all Thinking-only. So native temporal handling is
not a freebie; it is a capability with no local weights. This *strengthened* the
showcase argument and would have been missed by assuming the parameter worked
everywhere.

**⚙ `majority_downsample` is not a client parameter.** Named in the TabPFN-3.5
changelog, absent from `tabpfn-client`, which exposes `balance_probabilities`
and `inference_config={"SUBSAMPLE_SAMPLES": N}`. E4 uses what exists.

**⚙ A `predict` call blocked for 1h50m on 18 seconds of CPU.** The client sets
no timeout on ordinary calls, so a dropped response stalls forever and every
later configuration queues behind it — and from outside the run looks alive.
Added a SIGALRM watchdog per configuration. It has since caught two real stalls
in E2, costing 10 minutes each instead of the whole grid.

**⚙ Two broken tools on this machine.** Homebrew `git` 2.54.0 crashes on any
network operation (`_curl_global_trace` missing — linked against a newer libcurl
than macOS 14 ships); use `/usr/bin/git`. Homebrew installs nothing needing a
bottle, macOS 14 now being Tier 3.

---

## The E1 results, and two corrections they forced

**✗ Realised coverage was being compared at different levels.** The conformal
index `⌈(n+1)(1−α)⌉` rounds *up*, so a small calibration set silently targets a
higher level: 13 calibration positives at α=0.10 target **100%**, because the
index lands on the maximum score. Split calibrates on half as many positives as
cross at every budget, so split is always the more conservative of the two — its
higher coverage was granularity, not calibration.

**✗ P1 falsified — cross does not have lower coverage variance.** Seed spread
0.107 vs 0.083 at F=25 (cross better) but 0.044 vs 0.081 at F=100 (cross worse).
No consistent direction.

**⚑ The headline, found in data already collected.** Split at budget 2F
calibrates on F positives, exactly as cross at budget F does. Identical
calibration size ⇒ **identical targeted level**, so those pairs compare directly.
Noted 20 Sept: this was written as "no confound", which is not quite right —
split at 2F also holds twice the in-context rows (4,545 vs 2,273 and so on). The
asymmetry favours split, so the margins below are conservative, but it is an
asymmetry and the README now says so:

| targeted level | split needs | set size | cross needs | set size |
|---:|---:|---:|---:|---:|
| 96.0% | 50 frauds | 1.497 | **25** | **1.304** |
| 92.0% | 100 frauds | 1.345 | **50** | **1.263** |
| 91.0% | 200 frauds | 1.300 | **100** | **1.268** |

**Cross-conformal reaches the same guarantee from half the confirmed frauds**,
with narrower sets in 5 of 6 comparisons. This became the README headline.

**⚙ E1 stored only four α values, not the probabilities.** That made the
matched-*realised*-coverage comparison impossible without repaying for the
predictions. E2 onward persist evaluation probabilities (~80 KB per config), so
α can be swept densely offline for free. This later paid for itself twice.

---

## E3 — two bugs of mine, and a structural finding

**✗ ACI was being driven 1,400 times per prediction round.** `update_batch`
applies one step per *observation*; a month of BAF holds ~1,400 frauds. The level
oscillated 0.05 → 0.5 → 0.0001 → 0.5 and coverage swung 0.517 to 1.000. Added
`update_round` / `update_rounds`: one step per batch, from its observed
miscoverage rate. Not an ACI failure — an interface misuse, now pinned as a test.

**✗ A fake win, caught by fixing the replay.** The first γ sweep reported that
γ=0.5 halved the deviation from target. It had recovered thresholds from month 3
because the real months 0–2 calibration set was never persisted. With the exact
set: γ ∈ {0.05, 0.2} are **identical** to frozen and γ ∈ {0.5, 1.0} are *worse*.
Had I not gone back, the README would carry a tuned-hyperparameter result that
does not exist.

**✗ P4 falsified — ACI cannot help at this label budget, for a structural
reason.** With `n` calibration positives only `n` distinct thresholds exist, so α
must move far enough to change which order statistic is selected before anything
changes at all:

| calibration positives | α must move by |
|---:|---:|
| 46 (split, this experiment) | **0.0139** |
| 92 (cross would give this) | 0.0038 |
| 400 | 0.0024 |

ACI at γ=0.05 moves α ~0.0014 per month — ~0.007 across the whole walk. Too
small to move the threshold; raise γ and it jumps a whole order statistic and
overshoots. **⚑ This ties back to the headline: cross-conformal doubles the
calibration set, doubles threshold resolution, and is the precondition for
online adaptation working at all.**

---

**✗ A wrong number nearly shipped in a figure title.** E3 recorded the
*evaluation set's* fraud rate as `fraud_rate`, and the evaluation set keeps every
fraud plus a fixed legitimate sample — so it is ~41% by construction and says
nothing about the month. The drift figure was captioned "as the fraud rate drifts
(41.04% → 41.66%)". Now records `eval_fraud_rate` and `month_fraud_rate`
separately; the real drift is **0.92% → 1.47%** across months 3–7.

**Re-encoding the context monthly made coverage *worse*, not better** (mean
deviation 0.0232 vs 0.0143 frozen). The arm refreshes the model from recent
history but recalibrates against the *same* months 0–2 calibration set, so a
fresher model is scored against a staler calibration. That is a real design
limit, not a bug: tracking drift needs fresh *labels*, and fresh labels are
exactly what a fraud desk does not have. Refreshing the context alone does not
substitute.

## E2 — the original open question, answered

**Largely a negative result.** At 100 frauds there is no detectable optimal
`cal_size`: paired best-vs-worst difference +0.089 ± 0.042 (t=2.1, n=5) at
α=0.05. At 200 frauds there is a real effect (t=3.6, t=6.5) but it is driven by
`cal_size=0.8` being *bad* — starving the model — not by a sharp interior
optimum. **Cross-conformal beats every split setting at both budgets and both
alphas, 4/4.**

**✗ Corrected 21 Sept — two compounding errors, pulling opposite ways.**

*First, the selection.* `best` and `worst` are the extremes of seven `cal_size`
settings, and the paired t compared that pair against a critical value for *one
pre-specified* comparison. Choosing a pair because it is extreme is what inflates
it. `analyze_e2` now permutes the settings within each seed — the seeds are
paired, so they are exchangeable under the null — rebuilding the null
distribution of the max-minus-min statistic actually reported.

*Second, the pairing was broken anyway.* E2 was resumed part-way, so
`results/e2.jsonl` holds two settings whose rows are in a rotated seed order
(`cal_size=0.2` at 100 frauds, `cal_size=0.4` at 200). `analyze_e2` paired
settings by **list position**, so for those two it differenced one seed against
another. The means, and therefore the figure and every width in the README, were
never affected — means do not care about order. The paired statistic did.

Aligning on seed and permuting:

| budget | spread | p (file order, wrong) | p (seed-paired) |
|---:|---:|---:|---:|
| 100 | 0.089 | 0.059 | **0.042** |
| 200 | 0.055 | 0.788 | 0.618 |

So the honest finding moved in *both* directions: the 200-fraud "real optimum"
(t = 3.6) evaporates, and the 100-fraud "no detectable optimum" becomes a real
spread. Stable — p ∈ [0.0416, 0.0458] across 12 permutation seeds at 100, and
[0.611, 0.623] at 200, so this is not Monte-Carlo noise.

What survives is better than what it replaced: at 100 frauds the allocation does
matter, and **it still is not worth choosing**, because the best split setting
(1.505) loses to cross-conformal (1.463). The verdict string no longer says "a
real optimum" either — the test shows the settings differ, not which is best.

The permutation machinery was checked against ground truth before being trusted:
0 of 40 false positives on pure noise, and p = 0.001 on a planted effect of 0.30.

**✗ Nearly shipped a coin flip as a finding.** The first verdict compared the
best-worst gap to pooled seed scatter and called F=100 "a real optimum" on a
margin of 0.089 versus 0.088. The seeds are *paired* across `cal_size` settings,
so the paired test is the correct one, and it reverses the call.

---

## E4 — baselines, the guarantee, and the daily cap

**⚑ TabPFN beats LightGBM at a matched guarantee, in all four comparisons.** Same
strategy and budget ⇒ same calibration size ⇒ identical targeted level, so set
size compares the models cleanly:

| strategy | budget | targeted level | TabPFN | LightGBM | TabPFN narrower by |
|---|---:|---:|---:|---:|---:|
| split | 100 | 98.0% | **1.618** | 1.764 | 8.3% |
| split | 200 | 96.0% | **1.574** | 1.649 | 4.6% |
| cross | 100 | 96.0% | **1.471** | 1.680 | **12.4%** |
| cross | 200 | 95.5% | **1.459** | 1.566 | 6.8% |

This is the cleanest showcase result in the project: conformal prediction
converts model quality into a unit a fraud desk acts on — how many cases land in
a human's queue — and TabPFN wins on it by 5–12%.

**✗ P5 looks falsified too.** LightGBM cross-conformal ran in ~6s against
TabPFN's ~51s. The comparison is confounded (remote vs local CPU) and every row
carries `wallclock_comparable: false`, but the intuition behind P5 — that K
gradient-boosted trainings would be prohibitive — is simply wrong at this scale.
LightGBM trains on 9,000 rows in under a second. The hardware-neutral claim
(0 gradient fits vs 6) stands; the speed claim does not, and should not be made.

**✗ A marginal rate on an enriched evaluation set.** The threshold arms reported
`flag_rate = 0.64`, measured on a set that is ~49% fraud by construction. That is
not a production flag rate. Replaced with **false-positive rate**, a within-class
quantity that the enrichment cannot bias: the tuned-threshold arm flags **41% of
legitimate traffic** to reach 0.92 recall. The same caveat applies to reading set
size as a review load, and is now printed with the table.

**⚙ `balance_probabilities` is a no-op if you tune a threshold.** The two
no-guarantee arms came back identical — 0.925 recall, 0.407 false-positive rate —
in four of six seeds, and within two cases in 2,878 on the other two. Their tuned
thresholds are wildly different (0.29 vs 0.0045), so the flag genuinely rescales
the probabilities; it is just a monotone rescaling, which threshold tuning
absorbs. It only matters against a *fixed* cutoff such as 0.5. Worth stating,
because "we enabled TabPFN's imbalance handling" sounds like a stronger baseline
than it is.

**⚙ The daily cap is 5,000,000 tokens, separate from the 20M monthly.** E4 hit it
at 4.99M and the remaining nine TabPFN configurations failed with HTTP 429. Not a
bug and fully recoverable — the run resumes — but the cahier tracked the monthly
budget and not the daily one. Both need watching.

**⚙ The SIGALRM watchdog does not work on this client.** `tabpfn-client` uses
httpx with its own timeouts: 900s per request and **7200s — two hours — for
uploads and async polling**, which is precisely the 1h50m stall seen earlier. A
signal handler cannot interrupt a socket read happening below Python off the
main thread, and the watchdog silently failed to fire 18 minutes past a 15-minute
limit. The real fix is `TABPFN_CLIENT_TIMEOUT`, `TABPFN_CLIENT_UPLOAD_TIMEOUT`
and `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT`, which are read from the environment and
must be set **before** importing the client. `_common.set_client_timeouts()` does
this and every experiment calls it.

## 20 September — a claim of ours that was wrong

**✗ `tabpfn-extensions` does contain a conformal module.** Four documents said
it had "no conformal prediction at all". That came from a web summary of the
repo's README rather than its source, and it is false: `cp_missing_data` exports
`CPMDATabPFNRegressor`.

Read from the source this time, the accurate statement is narrower and still a
real gap. `cp_missing_data` is a **regression** interval estimator specialised to
**missing-data patterns** — single split-conformal calibration (`val_size=0.3`),
correction terms per missing-data mask, quantile intervals rather than
prediction sets. There is no conformal prediction for **classification**, no
class-conditional calibration, no cross-conformal, and no online variant.

The full module list is also longer than recorded: bayesian_optimization,
benchmarking, cp_missing_data, embedding, image, interpretability, many_class,
misc, pval_crt, scoring, survival, tabebm, unsupervised.

This one mattered. Prior Labs personnel judge this hackathon, they know their own
repository, and "no conformal prediction at all" is the kind of overstatement
that discredits everything stated next to it. Found only because the PR
preparation required cloning the repo rather than reading about it.

It also cuts the other way: `cp_missing_data` existing is **evidence conformal
contributions are in scope**, which is a better argument for the PR than an
empty category would have been.

## The Thinking arm — and a metric that hid its own result

**⚑ TabPFN-3.5-Thinking holds the guarantee under drift; the base model does not.**

| arm | months below the promised level | mean set size |
|---|---:|---:|
| base, frozen thresholds | 4 of 5 | 1.422 |
| base, ACI | 4 of 5 | 1.424 |
| **Thinking, frozen thresholds** | **0 of 5** | 1.536 |
| **Thinking, ACI** | **0 of 5** | 1.542 |

Measured against the level actually *targeted* — 97.83% with 46 calibration
positives at α=0.05, not 95%, because the conformal index rounds up. Base falls
below its own promise from month 4 onward as the fraud rate climbs. Thinking
never does.

This is the clearest TabPFN-3.5-specific result in the project, and it matches
what Prior Labs documents: Thinking is stronger on temporal and grouped data.
Here that shows up in the unit that matters — whether the guarantee survives
drift.

Three caveats that must travel with it. Thinking also passes `time_col`, which
base cannot (it is rejected outside thinking mode), so this compares *recommended
usage* rather than isolating the checkpoint. It costs about 8% wider sets —
1.536 against 1.422 — which is the price of not breaking the promise. And it is
one seed.

**✗ The metric was hiding this.** `analyze_e3` reported mean absolute deviation
from target, which penalises over-coverage exactly as hard as under-coverage. On
that metric Thinking scored **worse** (0.0375 vs base's 0.0143) — precisely
backwards, because Thinking's "error" is over-delivering. For a guarantee the two
directions are not symmetric: under-coverage breaks the promise, over-coverage
only costs set width. The table now reports months-below-target first and width
as the cost, which is the question a risk function actually asks.

## E5 — scale, and the KV cache finally demonstrated

**⚑ The earlier experiments were asking the wrong question about scale.** Every
one shrank the labelled pool to preserve the 1.1% base rate, so 200 confirmed
frauds meant a context of 18,000 rows. A real fraud desk has *millions* of
transactions and a few hundred confirmed frauds — negatives are abundant,
positives are not. E5 holds the frauds fixed at 200 and pours in legitimate data,
down to a context fraud rate of **0.10%** at 200,000 rows. That is both more
faithful to the problem and the first time this project used TabPFN at a scale
the model was built for.

**✗ A documented limit, noted on day one and then designed around badly.** Cached
predicts cap at 10,000 test rows per call. The calibration pass scores 25,100, so
every cached configuration failed with HTTP 422. Not a blocker — batching *is*
the workload the cache assumes: encode the context once, stream batches through
it. Added `BatchedPredictProba` in the experiments (a property of this API, not
of conformal prediction).

**The cache, measured at last:**

| context | predict uncached | predict cached | speedup |
|---:|---:|---:|---:|
| 50,200 | 6.7 s | 2.3 s | 2.9× |
| 100,200 | 13.6 s | 4.4 s | 3.1× |
| 200,200 | **36.9 s** | **5.4 s** | **6.8×** |

Coverage and set width are **identical** in all three pairs (0.958/1.460,
0.983/1.556, 0.954/1.441) — the cache changes cost, not answers. The speedup
grows with context, as documented, and `estimate_cost` puts the token saving at
75% from 200k rows up. Conformal is precisely the workload it was built for: one
fixed context, scored twice.

Note the fit column moves the other way — 15 s → 24 s at 50k, since a cached fit
computes and stores the attention state up front. It pays back on the second
pass and after that it is free.

## 20 September — an overstatement in our own headline

**✗ "Only affordable because TabPFN never trains" was false, and our own data
said so.** It was the README's first line. E4 measured LightGBM cross-conformal
at about **six seconds** — cross-conformal is entirely affordable with a
gradient-boosted model at this scale, which is exactly what falsifying P5
established. A judge could have disproved the headline from our own results
table two screens further down.

Corrected to what is measured: the same guarantee from half the labels, and on
TabPFN **0 gradient-trained fits against LightGBM's 6** — hardware-independent,
and the number that scales.

**✗ The K× cost caveat had been silently deleted.** An earlier edit to the README
overwrote the paragraph stating that cross-conformal costs exactly K×. So the
document was making the affordability claim while no longer carrying the
correction that qualifies it. Restored.

**✗ "Thinking never does" on five months of one seed.** Softened, with the sample
size stated inline rather than three sentences later.

Found by grepping the public docs for absolutes and superlatives. Most hits were
accurate — "never trains", "never a random split", "never clips silently" — which
is why the real ones had survived several readings. Worth repeating before
submission.

## E6 — the replication narrows the headline

**✗ "Narrower in five of six comparisons" was over-reading noise.** Re-running the
matched-level comparison across Variants I–III gave a raw win count of 6 of 9,
which looks like partial failure. It is neither a win nor a failure: the seeds
are *paired* across strategies, so they have to be tested pairwise, and doing
that dissolves almost all of it.

| dataset | calib. positives | paired difference (split − cross) | verdict |
|---|---:|---:|---|
| Base | 25 | +0.1924 ± 0.0402 (n=5) | **cross narrower** |
| Base | 50 | +0.0815 ± 0.0458 (n=5) | tie |
| Base | 100 | +0.0322 ± 0.0274 (n=5) | tie |
| Variant I | 50 | −0.0215 ± 0.0749 (n=3) | tie |
| Variant I | 100 | −0.0005 ± 0.0379 (n=3) | tie |
| Variant II | 50 | +0.0157 ± 0.0161 (n=3) | tie |
| Variant II | 100 | +0.0109 ± 0.0363 (n=3) | tie |
| Variant III | 50 | +0.0152 ± 0.0106 (n=2) | tie |

**Significantly wider in 0 of 8; significantly narrower in 1.** The narrowing is
real only where positives are scarcest — which is the regime the project is
about, but it is not a general property.

**Completed 20 Sept at 36/36.** Variant III finished its second budget, so the
table above is the partial run and the final count is **nine** comparisons, not
eight: still significantly wider in **0**, still significantly narrower in **1**,
same conclusion. The raw win count is 6 of 9 — which is exactly the number the
paired test exists to discount. `analyze_e6` prints both, in that order, so the
weaker framing cannot be quoted without the stronger one next to it. The README
carried "0 of 8" in its summary and intro for a while after the run finished
while its own detail section said 9; `verify_claims` now recomputes the count.

So the headline shrank, and improved: *the same guarantee from half the confirmed
frauds, at no cost in set width*. The halving is what replicates. The narrowing
is a Base-scarce-budget result and is now labelled as one.

This is the third time in this project that a raw count or an unpaired
comparison flattered a result and the paired test took it back (E2's cal_size
optimum, E4's split@200 margin, now this). Worth making a habit: **when the
seeds are shared, never compare means.**

## The Thinking drift result, settled at three seeds

**✗ The single-seed headline was an artifact. What survives is much weaker, and
is now stated that way.**

| model | seed 0 | seed 1 | seed 2 | total |
|---|---:|---:|---:|---:|
| base | 4 of 5 below | 0 of 5 | 5 of 5 | 9 of 15 |
| Thinking | 0 of 5 | 0 of 5 | 3 of 5 | 3 of 15 |

Seed 0 looked decisive. Seed 1 has base holding comfortably. Seed 2 has Thinking
failing three months itself. Paired by seed the gap is 2.0 ± 1.2 months,
t ≈ 1.7 at n = 3 — **directional, not established**.

What can honestly be said: Thinking was never worse than base on any seed and
strictly better on two of three, at about 5% wider sets.

How close this came to shipping is the point. It was found, written up, and
promoted to the README summary table **on one seed**, because it was the
cleanest TabPFN-specific result in the project and it matched what Prior Labs
documents about Thinking on temporal data. Two things saved it: the pre-committed
decision to replicate before submitting, and fixing `analyze_e3` to aggregate
over seeds first — keyed on month alone, it would have silently kept only the
last seed and shown an unchanged table.

**Wanting a result to be true is exactly when replication matters.**

## (superseded) The Thinking drift result does not replicate cleanly

**✗ The strongest TabPFN-specific claim in the project was a single-seed
artifact, at least in part.**

| model | seed 0 | seed 1 |
|---|---|---|
| base | 4 of 5 months below target | **0 of 5** |
| Thinking | 0 of 5 | 0 of 5 |

Seed 0 looked decisive. Seed 1 shows the base model holding comfortably, so all
four failures come from one draw. Thinking has still never fallen below its
promised level — 0 of 10 seed-months — but "base fails and Thinking fixes it" is
not what two seeds support.

Corrected in the README to a directional statement with the seed structure shown,
rather than the 4-of-5 headline. A third seed is running; the claim gets settled
or dropped before submission.

Worth noting how close this came to shipping. The result was found, written up,
and put in the README summary table on one seed, because it was the cleanest
TabPFN-specific finding in the project and it matched what Prior Labs documents
about Thinking on temporal data. Wanting a result to be true is exactly when
replication matters most.

## Fourth audit — 21 September

**The drift figure drew the wrong target line, and it is the figure in the
README.** Both E3 figures put their dashed target at the nominal
`1 - alpha = 0.95`. Every point in both sits above 0.95, so each figure read
*the guarantee always holds* — printed directly above a table stating that base
misses the level in 4 months of 5. The level a month actually has to clear is
`ceil((n+1)(1-alpha))/n = 97.83%` at 46 calibration positives, which is the
project's own central arithmetic, stated in the paragraph beneath the figure and
in `analyze_e3`'s own table. The picture contradicted its caption.

Redrawn at 97.83%. Base now visibly drops below the line from month 4 on, which
is the 4-of-5 the table reports; Thinking's frozen arm clears it in all five
months and its re-encoded arm dips once, which is the 0-of-5 and 1-of-5 the
table reports. `verify_claims` fails if the axhline goes back to `1 - alpha`.

Three smaller things fell out of looking at the figures at all — something none
of the previous five passes had done.

*Five of nine committed figures were never referenced anywhere.* Two deserved to
be: `e5_scale_alpha005.png`, which asks its question in the title and answers
"No" under the axes with the slope against the seed SD, for a section that had
been making that argument in prose only; and `e3_drift_thinking.png`, for a
claim that is entirely about base versus Thinking and was showing only base.
Both added. The remaining three are alpha-variant alternates, which is fine.

*The first caption I wrote for the Thinking figure was wrong.* "Clears the line
in four months of five" describes the re-encoded arm; the frozen arm — the one a
reader takes as the headline — clears all five. Checked against the data before
it shipped, which is the only reason it did not.

*The footer then collided with the legend*, and before that the target label
collided with the data. Both caught by rendering the PNG and looking at it,
not by reading the code that generates it.

## Third audit — 21 September

Prompted by a question with an obvious answer and a non-obvious cause: *why are
there empty folders?*

`experiments/kaggle/` was the tier-2 wall-clock table from §12 of the plan. It
was never built, so it sat as an empty directory in the working copy — and since
git does not track empty directories, it never existed in a clone at all. Two
readers of this repository would have seen different trees. Removed.

Checking it against the plan turned up that **§12 "Repository layout" described a
repository that does not exist**: it listed `src/tabpfn_conformal/budget.py`,
which was folded into the `cal_size` constructor argument back on 19 Sept and
never written; it omitted `metrics.py`, which is shipped; it listed three files
under `docs/` where there are seven; and it had no `demo/` or `contrib/` at all,
which are two of the four deliverables. A judge reading the plan and then the
tree would have found four mismatches. Rewritten as built, with the two
never-built items named as never-built, and `verify_claims` now walks the
diagram and asserts every path exists — plus the reverse, that no shipped module
is missing from it. Both directions confirmed to fail on the exact errors that
were there.

**The P5 confound was not in `limitations.md`.** It is tagged on all 36 rows of
`results/e4.jsonl`, described in `STATUS.md`, and named in the P5 scoreboard row
— but absent from the document titled *Limitations*, which is where a reader
goes looking for it. Added, with the comparable number (0 gradient fits against
6) separated from the incomparable one (6 s against 50 s, remote versus local).

While writing that entry I put "no wall-clock claim is made anywhere in the
README" into it, then checked: the README quotes those seconds twice. The
scoreboard row said "confounded"; the opening cost paragraph only implied it, by
calling the fit count "the hardware-independent number". Now it says so outright.
Writing the caveat is what exposed the gap in the thing being caveated.

**The README never told anyone to download the data.** Its experiment block
installs `[experiments]` and initialises the TabPFN client, and stops — while
every runner needs `data/Base.csv`, which is gitignored because it is a million
rows. The step existed in `docs/method.md` and in the error message
`_common.load_frames` raises, but not on the path a reader actually follows.
Someone reproducing the experiments would have hit a missing file and had to be
rescued by an exception string. Added, along with the `--dry-run` note, since
pricing a run before spending on it is the thing a reader most wants to know.

`verify_claims` now extracts every `python path/to/script.py` the README
instructs a reader to run and asserts it exists, asserts the data download is
among them, and checks that all six experiments really do accept `--dry-run` —
a promise about not spending money should not be taken on trust.

## Second audit — 21 September

Three more, all of the same family: a statistic that was not measuring what its
label said.

**`replay_aci` still used the metric this project had already rejected.** Mean
absolute deviation from the *nominal* 95%, which punishes over-coverage exactly
as hard as under-coverage — the metric that once scored Thinking worse for
holding its guarantee, fixed in `analyze_e3` and never fixed here. It also
compared against 95% rather than the 97.83% that 46 calibration positives
actually certify. Both corrected, and the script now cross-checks against E3:
frozen replays to 4 of 5 months below target at mean set size 1.422, which is
`analyze_e3`'s base row to three decimals.

That correction cost a claim. "At γ ∈ {0.5, 1.0} ACI is worse" was a
mean-|dev| statement; under months-below-target γ=0.5 ties frozen and γ=1.0 is
nominally *better*, 3 of 5 against 4. What is actually true is instability: the
month-to-month coverage swing goes 0.023 → 0.074 → **0.106** as γ rises, and
γ=1.0 buys its one extra month with the widest sets and a 0.894-then-1.000
oscillation. The README now reports the swing column instead of asserting
"worse", and `verify_claims` reads the sweep table cell by cell.

**`e6_variants` carried its own copy of `make_eval`** — the duplication
`_common.py` exists to prevent, and E6 is exactly where it would bite, since its
headline pairs variants against a Base row computed by E1 from the shared
function. Proven identical on four datasets × three seeds, then deleted.

See also the E2 entry above, corrected the same day.

## Pre-submission audit — 20 September

A sweep over everything a judge would actually open, after the experiments were
done. Four real defects, none of them in the library.

**The video script and the submission text still carried the falsified drift
claim.** Both said base fails "four of five months" and Thinking "zero" — the
single-seed version, corrected in the README days earlier but not in the two
documents a judge reads and hears. A video is the worst place to leave a stale
number, because it cannot be edited after the fact and it is checkable against
the README in thirty seconds. Both now state 9 of 15 versus 3 of 15, directional
at n = 3, and the do-not-say list names the old phrasing explicitly.

**`analyze_e3` compared arms averaged over different seed sets.** `frozen` had
three seeds, `aci` and `refit` one. Aggregating each over whatever it happened
to have and plotting them together drew ACI as a line that visibly diverged from
frozen — the opposite of the project's own finding that at 46 calibration
positives ACI cannot move the threshold at all. The figure would have contradicted
the paragraph beneath it. Arms are now restricted to the seeds they all share,
and the caption states which. This is the same standing rule as before, in a new
disguise: *do not compare means across different seed sets.*

**Running `analyze_e3` with no `--model` pooled base and Thinking into one set of
lines and wrote them out as `e3_drift_base`.** Two different models averaged into
a figure labelled as one of them. It now reports each model separately.

**The demo had no committed generator.** `figures/demo_data.json` and
`demo/index.html` were both produced by a script that was never committed, so a
clean clone could not rebuild the demo and nothing checked its numbers against
`results/`. `scripts/build_demo.py` now rebuilds both, and `verify_claims.py`
fails if the committed data differs from what the generator produces. Recovering
the recipe confirmed the data was honest — the calibration scores and the E1
block reproduce bit-for-bit — but "honest and unverifiable" is not the standard
the rest of the repository is held to.

Also corrected: a stale test count in four documents (86 → 99), an "installs in
one second" claim that measured 16 seconds on a clean environment, and a table
row that had escaped its table in the README.

The README's headline three-seed table was recomputed by hand in the session
that produced it. It is now printed by `analyze_e3` and checked by
`verify_claims.py`, which went from 30 verified claims to 49.

## Scoreboard

| | prediction, registered before the experiments ran | outcome |
|---|---|---|
| P1 | cross-conformal has lower seed-variance of fraud coverage | **falsified** |
| P2 | cross-conformal costs under 2× split in tokens | **falsified** — exactly K× |
| P3 | marginal CP under-covers the fraud class; Mondrian fixes it | held (already published) |
| P4 | static thresholds decay under drift; ACI holds coverage | **falsified** — threshold quantization |
| P5 | LightGBM cross-conformal costs far more wall-clock | **falsified** — ~6s vs TabPFN's ~51s. Confounded (remote vs local), but the intuition was wrong: LightGBM trains on 9,000 rows in under a second. |

Four of five resolved predictions failed. What survived is sturdier for it: the
feasibility ceiling and level fidelity are *deterministic* — checkable on paper,
not falsifiable by more data — and the half-the-labels result is measured at
matched level, with the only asymmetry favouring the baseline.
