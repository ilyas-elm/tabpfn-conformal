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
calibration size ⇒ **identical targeted level**, so those pairs compare directly
with no confound:

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
matched level with no confound.
