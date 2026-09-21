# Cahier des charges — `tabpfn-conformal`

**Prior Labs TabPFN-3.5 Hackathon** · drafted 18 Sept 2026 · deadline **6 Oct 2026, 23:59 CEST** (22:59 Morocco) · **18 days**

> **Findings live in [`FINDINGS.md`](FINDINGS.md).** This document is the *plan*;
> the log is what actually happened, in order. Where they disagree, the log wins.
>
> Status: **v2 (19 Sept) — positioning locked, API shape signed off, library and tests done. Blocked on an API token (M0b).** This document is the contract. If a decision is not written here, it has not been made; if it is written here, do not relitigate it mid-build without editing this file first.

---

## 1. Objective

Wrap TabPFN-3.5's probability outputs in a **distribution-free, finite-sample coverage guarantee** for binary fraud detection under extreme class imbalance and temporal drift, and route that guarantee into an approve / block / review decision under a fixed analyst budget.

**The one-line pitch (README first screen, video first 30 s):**

> *Split conformal spends your scarcest resource — confirmed fraud labels — to save your cheapest one. With a model that never trains, that trade is simply wrong.*
>
> **⚠ Rewritten 19 Sept.** The original pitch claimed K folds cost about the same as one. Measured false: cross-conformal costs **exactly K×** split in API tokens. See §7.2 for what survives and why it is stronger.

---

## 2. Why this wins (mapping to the published rubric)

**Read the arithmetic before optimising the wrong axis.** Originality is not a criterion on its own: it shares a 30% bucket with *practical value*. Call it 15% of the total. Meanwhile **70% of the score is "is TabPFN visibly central" (50%) plus "is this built well and reproducible" (20%)** — both execution, not inspiration. This project is strongest exactly there. Effort should follow the weights.

| Criterion | Weight | How we score |
|---|---|---|
| **Showcase of TabPFN-3.5** | **50%** | The headline claim is *false for every other model*. Three TabPFN-3.5-specific properties carry it: (a) no training step → cross-conformal is inference-only; (b) `fit_mode="fit_with_cache"` — conformal calibration is literally the "one fixed context, many points scored" workload the cache exists for; (c) API fits are **not token-charged**, so K folds cost far less than K×. Plus TabPFN-3.5-**Thinking** used where Prior Labs itself claims it is strongest — *temporal and grouped data* — which is our drift experiment. |
| **Creativity / originality / practical value** | **30%** | Do **not** claim methodological novelty; see §3.6. Claim the two things that are true: one genuinely unmeasured number (the context-vs-calibration label split, §7 E2) and clear practical value — a fraud desk gets a per-class guarantee and a fixed review budget instead of an uncalibrated probability. |
| **Technical quality / reproducibility** | **20%** | Tiny-dependency core, full pytest suite on CPU in under a second, **exact** numerical agreement with MAPIE, `estimate_cost()` printed before every API run, and a headline figure reproducible **on a judge's laptop with a free account and no GPU**. |

**Why Prior Labs specifically should care.** They sell into regulated buyers: fraud is named as a use case on their own landing page, their published testimonial is from an insurer, and Plus/Thinking ship through SAP AI Core and SageMaker. Regulated risk functions do not block on accuracy — they block on "prove the error rate." This project is aimed at the thing standing between TabPFN and a production risk deployment. That argument belongs in the submission text, not just in our heads.

**Tie-breaker we are explicitly playing for:** judges are Prior Labs personnel deciding "would we share this?" A live PR filling a verified gap in *their own* repo is the cheapest, loudest possible yes.

---

## 3. Verified technical base

Everything below was checked on 18 Sept 2026. Re-verify anything marked ⚠ before relying on it.

### 3.1 TabPFN-3.5 (released 15 Sept 2026 — hackathon day 1)

| Variant | Max rows | Max cols | Local weights | API | Notes |
|---|---|---|---|---|---|
| TabPFN-3.5 | 1,000,000 | 20,000 | **Yes** | Yes | base multitask checkpoint |
| TabPFN-3.5-Fast (alpha) | 1,000,000 | 20,000 | **Yes** | Yes | up to 6× faster |
| TabPFN-3.5-Plus | 1,000,000 | 20,000 | **No** | Yes | text-rich data |
| **TabPFN-3.5-Thinking** | **200,000** | **2,000** | **No — API/SageMaker/SAP only** | Yes | **#1 on TabArena, BeyondArena, STRABLE, MulTaBench**; "stronger temporal/grouped data handling" |

- **CPU ceiling: 5,000 samples.** GPU strongly recommended; ~8 GB VRAM works.
- `fit()` stores context; there is no gradient training. `fit_mode="fit_with_cache"` precomputes and stores the training-set attention state.
- Class imbalance: `majority_downsample` (preserves all non-majority rows).
- Speed claims: 0.5 s / 1,000 rows base; 0.17 s / 1,000 rows Fast; "840× faster with kv-cache".
- **Weights are under a NON-COMMERCIAL licence.** Our code is Apache 2.0; the dependency is separately licensed. This must be stated explicitly in the README or a careful judge will flag it.
- Python 3.10–3.14 supported.

### 3.2 Prior Labs API — the operational constraints that shape the plan

| Limit | Value |
|---|---|
| Predict | 60/min, 1,500/hr |
| Fit | 60/min, 1,000/hr |
| **Thinking fit** | **10/min, 30/hr** ← hard planning constraint |
| Upload | 120/min, 3,000/hr |
| Token budget | **5M/day, 20M/month** |
| Minimum charge | **10,000 tokens per billable op** |
| **Uploads and standard fits** | **NOT token-charged** |
| Predictions | charged on (train rows × predict rows × cols × n_estimators) |
| KV-cache reuse | **75% lower charge** |
| Promotion | **50% off TabPFN-3.5 rates until 29 Sept 2026** |
| `estimate_cost()` | sends dimensions only — **no upload, no quota consumed** |
| Cache limits | max 10,000 test rows per cached predict call |

✅ **RESOLVED 19 Sept by spike S1 — measured, not inferred.** The KV-cache doc was right. The server rejects the combination outright:

> `HTTP 422 — Value error, FIT_WITH_CACHE fit mode is not compatible with thinking mode`

**Consequence, now binding:** the cache story and the Thinking story cannot share a figure. Cache economics attach to **E1/E2** on base TabPFN-3.5; Thinking attaches to **E3** (temporal drift), where Prior Labs claims it is strongest. This was the pre-planned contingency in §11 and costs us nothing.

### 3.3 The gap in `tabpfn-extensions` is real

**⚠ Corrected 20 Sept from the source, not the README.** The repo has 13 modules: `bayesian_optimization`, `benchmarking`, `cp_missing_data`, `embedding`, `image`, `interpretability`, `many_class`, `misc`, `pval_crt`, `scoring`, `survival`, `tabebm`, `unsupervised`. **`cp_missing_data` is conformal** — but regression-only (`CPMDATabPFNRegressor`), split-conformal, specialised to missing-data patterns. The gap is **conformal prediction for classification**: no class-conditional calibration, no cross-conformal, no online variant. Apache 2.0, contributions welcome, `uv` + `pytest` conventions.

### 3.4 Bank Account Fraud (BAF), Feedzai / NeurIPS 2022

- 1,000,000 rows × 32 columns (30 features + `fraud_bool` + `month`), 6 variants (Base, I–V).
- **Base fraud rate 1.10% (~1 in 91)** — *correction to the brief's "1 in 300"*.
- `month` ∈ 0–7. Recommended protocol: months 0–5 train, 6–7 test.
- **The fraud rate genuinely drifts** — dips around month 2, climbs toward month 7. Good news: the drift experiment has real signal, not a synthetic one.
- Kaggle download only → `scripts/download_data.py` needs a Kaggle API token; document the manual fallback.

### 3.5 Prior art — what is already done, and what is not

| Work | Overlap | Consequence |
|---|---|---|
| arXiv 2607.27143 (Jul 2026) — cost-sensitive CP + human-in-the-loop abstention, 15 imbalanced tabular datasets, 7 models | **Marginal vs Mondrian CP *and* the review-budget decision layer** | Cite it. **Do not lead with Mondrian.** Present marginal-vs-Mondrian as *verified background*. Our decision layer is an application demo, not a contribution. |
| MAKE 8(7):190 — class-conditional CP for anomaly detection at 1:345 imbalance | "Mondrian restores minority coverage" | Same: known result. Citing it is a rigour signal. |
| arXiv 2605.21742 — correcting class imbalance in PFNs | Finds **thresholding** best, *because* PFNs are well-calibrated | **Perfect setup for us**: they tune a threshold, we derive one with a guarantee. Use as a named baseline. |
| arXiv 2509.01840 — E-ICL+FCP, ICL model with CP-aware loss | Full CP via ICL | Different: they train a bespoke model. We use TabPFN off the shelf. Cite as related. |
| Moudiki, nnetsauce — "Conformalized TabPFN" | TabPFN + split conformal | **Regression only.** Classification + imbalance is open. |
| **MAPIE 1.5** | Ships **`SplitConformalClassifier` *and* `CrossConformalClassifier`**; crepes ships Mondrian classifiers | **Cross-conformal is a standard method and we did not invent it.** Say so first, in the README. The claim is economics, not method. Verified by agreement test. |
| CFCP ([2605.24872](https://arxiv.org/abs/2605.24872)), SOCP ([2606.29403](https://arxiv.org/html/2606.29403v1)), SLCP ([2206.13092](https://arxiv.org/abs/2206.13092)) | Localized / embedding-clustered conformal for conditional coverage | Checked 19 Sept as a possible "bigger swing" using TabPFN embeddings. **Already done.** Do not pivot here. |
| [2509.01840](https://arxiv.org/abs/2509.01840) | Full conformal in one forward pass via ICL + attention symmetry | Checked 19 Sept as the other possible big swing. **Already done.** Do not pivot here. |

**Independently reported and useful to us:** TabPFN achieves the lowest ECE/Brier among tabular models, *but* becomes "increasingly majority-biased as data becomes imbalanced." That is the exact hook: TabPFN's best-in-class average calibration is not a per-class guarantee, and fraud lives entirely in the minority class.

**Closest hackathon competitor found:** `IFoA-ADSWP/tabpfn-reserving` (actuarial loss reserving). Strong README, `docs/method.md`, `docs/readiness.md`, honest results, baseline comparison. **No tests, no video.** That is the bar and that is its gap.

### 3.5b Measured on the API, 19 Sept — these override any doc estimate

| Finding | Measurement |
|---|---|
| **Cost is not a constraint at our planned scale.** Every workload up to ~50k context rows quotes at the **10,000-token minimum**. | `predict` at 1k/10k/50k context = 10,000 tokens flat |
| **The 75% cache discount is real but invisible below ~100k rows** — under that, both paths sit on the floor | 100k ctx: 19,039 → 10,000 (47%). 200k: 68,319 → 17,080 (**75%**). 500k: 364,290 → 91,072 (**75%**) |
| **⇒ Design requirement:** to *demonstrate* cache economics, E1/E4 must run at **≥100k context rows**. Below that there is nothing to show. BAF has 1M rows, so this is free — and a bigger demo is a better demo. | |
| **At tiny scale the cache actively hurts**, exactly as the docs warn | 200-row context: repeat predict 1.84 s uncached vs **9.72 s cached** |
| Thinking predict exceeds the floor | 20k ctx / 5k scored = 26,620 tokens |
| `thinking_effort` is valid **only** on `thinking_fit`, not `thinking_predict` | HTTP 422 otherwise |
| Budget observed | 20,000,000/month, **resets 1 Oct** → ~40M across the hackathon. Four S1 probes cost **213,481** (~1%). |

**⇒ The 29 Sept 50%-discount deadline is no longer a scheduling constraint.** We sit on the token floor; halving a floor changes nothing. M3 stays a useful go/no-go date, but not a financial one. *(Earlier estimate of "~40k tokens" for the S1 probes was wrong by 5×; the measured figure is above.)*

---

### 3.6 What we are explicitly NOT claiming

Three separate searches for a novel conformal angle (18–19 Sept) returned prior art every time: on Mondrian-under-imbalance, on cross-conformal, on embedding-localized CP, and on full-conformal-via-ICL. Conformal prediction is a 25-year-old field with an active 2026 literature. **There is no new conformal method available inside this deadline, and pretending otherwise is the fastest way to lose credibility with judges who know the field.**

Not claimed, anywhere in the repo, README, video or submission text:

- that we invented conformal prediction, Mondrian calibration, cross-conformal, or ACI;
- that Mondrian fixing minority-class coverage is our finding (it is published);
- that cross-conformal is new (MAPIE ships it);
- any speed, cost or coverage number that has not been measured and written to `results/`.

Claimed, and defensible:

1. **No conformal prediction exists for TabPFN classification**, in `tabpfn-extensions` or anywhere else we could find. We fill that gap with tested, installable code.
2. **Cross-conformal's cost structure changes qualitatively for a training-free model**, and we measure by how much. Standard method, new economics.
3. **Nobody has measured how to split a scarce label budget between a foundation model's in-context set and its calibration set.** That number is the one genuinely new result in the project (§7 E2).

Point 3 is small. It is one good figure, not a paper. Sized honestly it is an asset; oversold it is a liability.

---

## 4. Scope

### 4.1 In scope — library (`src/tabpfn_conformal/`)

| Module | Contents |
|---|---|
| `scores.py` | Nonconformity scores. Default `one_minus_prob` (1 − p̂_y). Registry + callable support. |
| `calibration.py` | Split conformal: `marginal` and `mondrian` (class-conditional) quantiles, with the finite-sample ⌈(n+1)(1−α)⌉/n correction. |
| `crossconformal.py` | **K-fold cross-conformal** (the headline). Every labelled row contributes to both context and calibration. |
| `adaptive.py` | ACI — online per-class threshold update as labels arrive. |
| `decision.py` | Prediction set + review budget K → approve / block / review. |
| `wrapper.py` | `ConformalClassifier`, sklearn-compatible. |
| ~~`budget.py`~~ | **Dropped 19 Sept.** `cal_size` is already a constructor argument, so the E2 sweep is a plain loop over the public API — which was the point of putting it there. A module wrapping a `for` loop would be ceremony. |
| `metrics.py` | Coverage by class, marginal coverage, set size, empty-set rate — shared so every experiment reports the same definition. |

### 4.2 In scope — experiments (`experiments/`, TabPFN + API)

- **E1 (headline)** Cross-conformal vs split conformal at a fixed labelled-fraud budget.
- **E2** Label-budget allocation sweep: context fraction × α → realised fraud-class coverage.
- **E3** Month-by-month drift: static calibration vs ACI, months 0–7.
- **E4** Baselines: TabPFN `majority_downsample`, tuned-threshold (per 2605.21742), and LightGBM — including the **cost table**.

### 4.3 Explicitly OUT of scope

Fine-tuning TabPFN · regression · ~~multiclass~~ (*the core turned out to be class-count agnostic and is now tested for it; only the decision layer is binary — 20 Sept*) · web UI · beating raw AUC · **any TabPFN/torch/GPU import inside `src/tabpfn_conformal`** · conformal risk control · weighted/covariate-shift conformal (mention as future work).

---

## 5. Non-negotiable constraints

1. Core package depends on **numpy, pandas, scikit-learn only**. It must `pip install` and its tests must pass in under 60 s on an 8 GB laptop with no GPU.
2. Repo is **Apache 2.0, verbatim LICENSE file**. README states the TabPFN weights' separate non-commercial licence.
3. Every API script prints `estimate_cost()` **before** spending tokens.
4. Every experiment is seeded and writes a machine-readable result file (`results/*.json` or `.csv`) that the figure scripts consume. No figure is produced by hand.
5. Repo layout mirrors `tabpfn-extensions` conventions so the upstream PR is a reviewable diff.

---

## 6. Library design

### 6.1 Public API — **SIGNED OFF 19 Sept, implemented**

```python
from tabpfn_conformal import ConformalClassifier

cc = ConformalClassifier(
    base_estimator,              # anything with predict_proba
    method="mondrian",           # "marginal" | "mondrian"
    strategy="split",            # "split" | "cross"
    cal_size=0.5,                # split only — THIS IS THE E2 SWEEP VARIABLE
    n_folds=5,                   # cross only
    score="one_minus_prob",      # name in registry, or a callable
    prefit=False,                # True -> base_estimator is already fitted
    random_state=0,
)

cc.fit(X, y)                     # split: partition internally per cal_size, fit base, calibrate
                                 # cross: K folds over all of X, every row calibrated out-of-fold
cc.calibrate(X_cal, y_cal)       # prefit=True path; also the online/drift path
sets = cc.predict_set(X, alpha=0.1)   # -> bool ndarray (n, 2): [:,0]=legit in set, [:,1]=fraud in set
cc.predict(X); cc.predict_proba(X)    # passthrough, for sklearn compatibility
cc.quantiles(alpha)              # dict {class_index: threshold}; marginal -> {None: t}
cc.n_calibration_                # dict {class_label: count} — honesty about tiny fraud n
cc.predict_set_from_proba(p, alpha)   # sweep alpha on cached probabilities, zero extra API calls
```

Two deviations from the original sketch, both adopted during implementation:

- **`quantiles` is a method, not an attribute.** Calibration stores the *scores*;
  thresholds are derived on demand. So `alpha` is a prediction-time argument and a
  sweep over `alpha` costs nothing extra. Against a metered API that is the
  difference between one billed pass and thirty.
- **`predict_set_from_proba` added** for the same reason: experiments score a test
  set once and then sweep offline.
- **`metrics.py` added** (`coverage_by_class`, `marginal_coverage`, `average_set_size`,
  `empty_set_rate`) so every experiment reports the same definition of coverage.

Two design points worth stating explicitly:

- **`cal_size` *is* the research question.** Making the budget split a first-class constructor argument means E2 is a loop over the public API, not a private hack. Good for the paper-quality story and good for reuse.
- **`strategy="cross"` reuses the same object and the same `predict_set`.** Swapping split → cross must be a one-word change for a user. That is what makes the headline claim land as a *product* feature, not a script.

Adaptive and decision layers compose rather than inherit:

```python
from tabpfn_conformal import ACI, route

cc = ConformalClassifier(base, method="mondrian", adaptive=ACI(gamma=0.01))
cc.update(X_month, y_month)          # per-class alpha_t update; no refit of base

actions = route(sets, scores, budget_k=500)   # -> array of "approve" | "block" | "review"
```

### 6.2 Correctness requirements

- Split conformal quantile uses ⌈(n+1)(1−α)⌉ / n. **Not** `np.quantile(scores, 1-alpha)`. This is the single most common conformal bug and we will have a test that catches it.
- Mondrian: if a class has too few calibration points to achieve α (i.e. ⌈(n_c+1)(1−α)⌉ > n_c), the threshold is **undefined** — return the trivial (full) set and **emit a warning**, never silently clip. With ~100 fraud labels at α=0.05 this is a live case, not a theoretical one.
- Cross-conformal: honest documentation that CV+/cross-conformal gives **approximate** validity (worst-case 2α), not the exact split-conformal guarantee. **We report both empirical coverage and the theoretical caveat.** Overclaiming here is how a submission loses the 20%.
- Ties in scores broken consistently; randomised (smoothed) conformal offered as an option, off by default.

### 6.3 Testing (`tests/`, CPU-only, LogisticRegression / RandomForest on synthetic imbalanced data)

1. **Marginal coverage** ≥ 1−α over many seeds, within Monte-Carlo tolerance.
2. **Mondrian per-class coverage** ≥ 1−α for *both* classes at 1:100 imbalance.
3. **Marginal under-covers the minority class** at 1:100 — the failure mode, asserted as a test. This is the paper's known result turned into a regression test.
4. Quantile formula exact against a hand-computed small example.
5. Degenerate cases: n_cal too small, one class absent from calibration, α=0, α=1, single-row predict.
6. **`test_agreement_with_mapie.py`** — our marginal split quantile equals MAPIE's on a fixed seed. (MAPIE is a *dev* dependency only.)
7. Cross-conformal: every row scored exactly once out-of-fold; coverage holds empirically.
8. sklearn `check_estimator` compatibility to the extent applicable.
9. ACI: threshold converges to target coverage on a synthetic drifting stream.
10. `route`: never exceeds budget K; approve/block are disjoint from review.

### 6.4 README section: "Relation to MAPIE and crepes" — *mandatory*

> crepes and MAPIE are excellent, mature libraries and we verify our marginal split-conformal quantile against MAPIE numerically (`tests/test_agreement_with_mapie.py`). This package exists for the parts they do not cover: allocating a scarce label budget between a foundation model's in-context set and its calibration set, cross-conformal at split-conformal cost for a model that never trains, cache-aware batch scoring, and online ACI under monthly drift — plus a dependency footprint small enough to vendor into `tabpfn-extensions`.

---

## 7. Experiments

All experiments are on BAF. All write `results/*.json`. All are seeded.

### E1 — Headline: cross-conformal vs split conformal at fixed label budget
**Question.** Given a pool of N labelled rows containing F fraud cases, does cross-conformal produce tighter and more stable fraud-class prediction sets than split conformal at the same N?
**Design.** F ∈ {25, 50, 100, 200, 400, 800}; strategy ∈ {split(cal_size=0.5), cross(K=5)}; method=mondrian; α ∈ {0.05, 0.10}; ≥10 seeds. Fixed held-out test set.
**Metrics.** Realised fraud-class coverage; mean set size; **variance of coverage across seeds** (the stability claim); tokens via `estimate_cost()`; wall-clock.
**Figure 1.** Fraud coverage and set width vs F, split vs cross, with seed-variance bands.

### E2 — Label-budget allocation (the original headline, now supporting act)
**Question.** How should a scarce fraud-label budget be divided between TabPFN's in-context set and the calibration set?
**Design.** Sweep `cal_size` ∈ {0.1 … 0.9} × α ∈ {0.01, 0.05, 0.10}, ≥10 seeds, base TabPFN-3.5.
**Deliverable.** A heatmap and an empirical rule of thumb. **This sweep is only cheap because TabPFN does not train** — state that in the caption.

### E3 — Temporal drift: static calibration vs ACI
**Design.** Calibrate on months 0–2, then walk months 3–7. Compare frozen Mondrian thresholds vs ACI. Secondary and genuinely TabPFN-specific: **when is it worth re-encoding the context (invalidating the KV cache) versus just updating the conformal threshold?** That trade-off only exists for a training-free cached model.
**This is where TabPFN-3.5-Thinking earns its place** — Prior Labs claims it is strongest on temporal data. Run E3's headline configuration with and without Thinking.

### E4 — Baselines and the cost table
Compare, at matched fraud-class coverage: TabPFN-3.5 + our conformal · TabPFN-3.5 + `majority_downsample` · TabPFN-3.5 + tuned threshold (arXiv 2605.21742) · LightGBM + conformal.
**Cost table** (the claim that carries the headline): for each method, number of *gradient-trained fits*, wall-clock on identical hardware, and API tokens. TabPFN's "gradient fits" column is **0**.

### 7.1 Pre-registered predictions — and what would falsify them

Write these down *now* so the results are honest either way. A submission that reports a falsified prediction with a clear explanation still scores well on technical quality; one that quietly changes its hypothesis does not.

| # | Prediction | Falsified if |
|---|---|---|
| P1 | Cross-conformal has **lower seed-variance** of fraud coverage than split at F ≤ 200 | variance is equal or higher |
| ~~P2~~ | ~~Cross-conformal costs **< 2× split** in API tokens~~ | **FALSIFIED 19 Sept. Measured: exactly K×** — 2.0× at K=2, 5.0× at K=5, 20.0× at K=20. The API prices a call by total rows touched, so each fold is a full pass over the pool and the folds add up linearly. Fits being unbilled is true and irrelevant: the *predicts* are what cost. See §7.2. |
| P3 | Marginal CP under-covers fraud at α=0.05; Mondrian does not | marginal is fine (would contradict published work) |
| ~~P4~~ | ~~Static Mondrian thresholds lose coverage by month 7; ACI holds it~~ | **FALSIFIED 19 Sept.** Static loses only 2.3pt (0.978 → 0.955) and **ACI does not recover it at any γ**: γ ∈ {0.05, 0.2} are identical to frozen, γ ∈ {0.5, 1.0} are *worse*. The reason is §7.5. |
| P5 | LightGBM cross-conformal costs ≫ TabPFN in wall-clock on identical hardware | comparable — then the headline weakens to "equally cheap", and we re-weight toward E2/E3 |

**If P1 and P2 both fail, the headline changes to E2.** Decide by 28 Sept (milestone M3). **P2 has already failed, so P1 is now load-bearing on its own.**

### 7.2 The headline, corrected after P2 failed

The original argument was *"K folds cost about the same as one."* **That is false, and measured false, on day two.** Cross-conformal costs K× for everybody, TabPFN included. Anyone repeating the original claim in the README or the video would be stating something a judge can disprove with one free `estimate_cost` call.

What survives does not depend on a ratio, which is why it is stronger:

1. **The absolute cost is negligible.** K=5 on a 10,000-row pool is 50,000 tokens — **0.25% of a monthly budget**.
2. **K× of a forward pass is not K× of a training run.** For LightGBM, K-fold conformal means K full fits plus the tuning question that comes with them; for TabPFN, `clone().fit()` swaps a context and takes no gradient step. ⚠ **Not yet measured.** LightGBM trains fast on 100k×30 and may well win on wall-clock. E4 measures it and we report whichever way it falls.
3. **Labels are the scarce resource in fraud, not compute.** A hundred confirmed frauds costs an analyst team weeks; fifty thousand tokens costs nothing. Split conformal trades the expensive resource to save the cheap one — a trade that made sense when refitting meant retraining, and stops making sense the moment it does not.

Point 3 is the durable claim: it is about data efficiency, survives any cost measurement, and still needs a training-free model to be actionable.

### 7.4 Two corrections forced by the E1 data (19 Sept)

**(a) Realised coverage was being compared at different levels.** The conformal
index ⌈(n+1)(1−α)⌉ rounds *up*, so a small calibration set silently targets a
higher level than α asks for:

| calibration positives | level actually targeted at α=0.10 |
|---:|---:|
| 13 | **100.0%** (the threshold *is* the maximum score) |
| 25 | 96.0% |
| 50 | 92.0% |
| 100 | 91.0% |
| 200 | 90.5% |

At any fraud budget split calibrates on half as many positives as cross, so
**split is always the more conservative of the two** and its higher realised
coverage at small budgets is granularity, not calibration. Reporting the two
side by side without this column was comparing a 100% predictor against a 96%
predictor and calling the first one better. `analyze_e1.py` now prints the
targeted level and the gap to it, and the figure draws each method's own target
as a dotted line.

**(b) P1 is NOT supported by the data.** Seed spread of fraud coverage at
α=0.10: F=25 split 0.107 vs cross 0.083 (cross better), but F=100 split 0.044
vs cross 0.081 (cross **worse**). Mixed, with no consistent direction. Per §7.1
the falsification rule applies: P1 does not carry the headline.

**What survives, in order of strength:**

1. **§7.3 feasibility** — deterministic, checkable on paper, unaffected by any of this.
2. **Level fidelity** — cross targets what you asked for; split overshoots at fraud-scale n. Also deterministic.
3. **Set width** — cross is narrower at every budget (1.30 vs 1.50 at F=25). ⚠ *Partly confounded*: cross also targets a lower level, and narrower sets at a lower target is not a clean win. The honest comparison is width at **matched realised coverage**, which needs a dense α sweep.

**Process fix:** E1 stored only four α values, so (3) cannot be settled from the
data already paid for. E2 and E3 now persist the evaluation probabilities
(`results/proba/`, ~80 KB per configuration) so α can be swept densely offline
for free and the frontier computed after the fact. This should have been in E1
from the start.

### 7.5 Why ACI fails here — threshold quantization (19 Sept)

E3's ACI arm does nothing, and the reason is the same scarcity the whole project
is about. With `n` calibration positives the threshold is the `k`-th order
statistic — **only `n` distinct thresholds exist**. α must move far enough to
change `k` before the prediction sets change *at all*:

| calibration positives | α must move by | to shift the threshold one step |
|---:|---:|---|
| 46 (split, this experiment) | **0.0139** | |
| 92 (cross would give this) | 0.0038 | |
| 200 | 0.0048 | |
| 400 | 0.0024 | |

ACI at γ=0.05 with monthly rounds moves α by ~0.0014 per month — about 0.007
across the whole five-month walk, against the 0.0139 needed. So it is *exactly*
equivalent to frozen. Raise γ enough to move and it jumps a whole order
statistic, overshooting: γ=0.5 and γ=1.0 both score worse than doing nothing.

**This is not a defect in ACI.** ACI assumes the threshold responds smoothly to
the level, which holds when calibration data is plentiful and fails when you have
46 positives. It ties straight back to the headline: cross-conformal doubles the
calibration set, so it doubles the threshold resolution and is the precondition
for online adaptation working at all.

Report E3 as: the drift in BAF months 3–7 is mild (2.3pt), and adaptive
calibration cannot help at this label budget for a structural reason worth
stating. That is more useful than a tuned γ that happens to look good.

### 7.3 The feasibility boundary — sharper than P1, and deterministic

Found in the E1 pilot, 19 Sept. Split conformal can only certify

> **α ≥ 1 / (n_cal + 1)**

because the threshold is the ⌈(n_cal+1)(1−α)⌉-th smallest calibration score and that index cannot exceed n_cal. Split spends half the positives on calibration, so with F confirmed frauds it reaches 2/(F+2); cross-conformal calibrates on all F and reaches 1/(F+1). **Cross-conformal exactly halves the tightest guarantee you can ask for.**

| confirmed frauds | split can certify | cross can certify |
|---:|---:|---:|
| 50 | 96.2% | **98.0%** |
| 100 | 98.0% | **99.0%** |
| 200 | 99.0% | **99.5%** |
| 400 | 99.5% | **99.75%** |

Why this beats P1 as the headline:

- **Deterministic, not statistical.** No seed averaging, no error bars, no "in expectation". A reader checks it on paper.
- **Confirmed exactly by the pilot.** At F=50 both methods fail at α=0.01 as the boundary predicts (minima 0.0385 and 0.0196); at F=200 split sits right on the edge at 0.0099 and cross is comfortable at 0.0050.
- **It is the question a regulated buyer actually asks** — "can you promise 99%?" — and for a hundred-fraud desk the honest answer under standard practice is *no, not at any price*.

Not novel: `1/(n+1)` is textbook conformal arithmetic. What is ours is the demonstration at fraud scale on a tabular foundation model, and the observation that the alternative only becomes practical when there is no training run to repeat.

**P1 (variance reduction) is demoted to supporting evidence.** It remains worth measuring — the full grid is running — but it is a √2 effect that needs seed averaging to see, and §7.3 does not.

⚠ **Honest caveat that must appear beside this table:** cross-conformal's guarantee is *approximate* (Vovk 2015; CV+ worst case 1−2α), so the certifiable-α table states what each method can *nominally* certify. E1's measured coverage is the check on whether the nominal promise holds, and it is reported next to it, never instead of it.

---

## 8. Compute plan — answering "I need to use TabPFN's full potential"

**The research settles this: you cannot reach TabPFN-3.5's full potential on Kaggle.** TabPFN-3.5-**Thinking** — the variant that is #1 on TabArena, BeyondArena, STRABLE and MulTaBench — is **not released as local weights**. It exists only via the Prior Labs API (and SageMaker/SAP). A Kaggle GPU runs the *base* checkpoint. So the API is not the budget compromise; it is the only door to the best model. Use three tiers:

| Tier | Hardware | What runs there | Why |
|---|---|---|---|
| **T0** | Your laptop, CPU | Core library + full pytest suite. **No TabPFN at all.** | Fast iteration; proves the zero-dependency claim. |
| **T1 — primary** | **Prior Labs API** (`tabpfn-client`) | E1, E2, E3, E4 quality results. Base TabPFN-3.5 for the wide sweeps; **Thinking for E3 and the E1 headline confirmation.** | Only route to Thinking/Plus. Fits are unbilled → cross-conformal is cheap, which *is* the headline. And **any judge reproduces it with a free account and no GPU** — that is the 20% criterion handed to us. |
| **T2 — supporting** | **Kaggle GPU** (P100 16 GB or T4×2 32 GB, 30 h/week, 12 h sessions) | **E4's wall-clock cost table only**, plus full-scale BAF runs if API quota binds. | A fair TabPFN-vs-LightGBM wall-clock comparison must run both on *identical* hardware. API latency vs local CPU is not a comparison, it is a confound. Local 3.5 weights + LightGBM on one T4 is. |

Practical notes: develop against ≤3,000-row BAF subsamples locally (CPU ceiling is 5,000). ⚠ TabPFN-3.5 is not yet on Kaggle Models (only 2.5) — the Kaggle notebook must `pip install tabpfn` with internet enabled and accept the weights licence. Verify in spike S2.

### 8.1 API token budget

Budget: **5M/day, 20M/month**, 10k minimum per billable op, **50% off until 29 Sept**. Fits and uploads are free; only predictions bill.

~~Rule: run every wide sweep before 29 Sept.~~ **Superseded by measurement (§3.5b): we sit on the token floor, so the discount is irrelevant and there is no cost reason to rush.** Spend the freed-up schedule on running E1 at ≥100k context instead, which is what makes the cache result visible at all.

**Thinking fits are capped at 30/hour.** A 5-fold cross-conformal Thinking run = 5 fits = 6 runs/hour maximum. Plan Thinking runs as a short, scripted, overnight batch — never interactively.

Before any sweep: run a dry-run script that sums `estimate_cost()` over the whole grid and refuses to proceed above a configured ceiling. Request hackathon credits via the platform on **day 1**, not day 15.

---

## 9. Deliverables

| # | Deliverable | Criterion served |
|---|---|---|
| D1 | Public GitHub repo, **Apache 2.0 verbatim** | mandatory |
| D2 | `src/tabpfn_conformal` — installable, zero heavy deps | 20% |
| D3 | `tests/` — green on CPU in <60 s, incl. MAPIE agreement test | 20% |
| D4 | `experiments/` — API scripts, seeded, cost-printing | 20% + 50% |
| D5 | **README** — headline claim, Figure 1, repro in ≤10 min with no GPU, MAPIE/crepes positioning, licence note, prior-art citations | all three |
| D6 | `docs/method.md` + `docs/limitations.md` (incl. the CV+ 2α caveat) | 20% |
| D7 | **Demo video, 2–3 min** — *optional under the official rules ("optional but encouraged"), but it is the main vehicle for the 50% showcase criterion and the nearest competitor has none* | 50% vehicle |
| D8 | **PR to `PriorLabs/tabpfn-extensions`** | 50% |
| D9 | Submission text on the hackathon platform | mandatory |

### 9.1 Demo video skeleton (2–3 min)
1. 0:00–0:25 — the problem: one fraud in ninety, ~100 labels, a probability with no guarantee.
2. 0:25–1:10 — **the headline**: one-word change `strategy="split"` → `"cross"`, Figure 1, and the cost table with TabPFN's `0` gradient fits.
3. 1:10–1:50 — drift: months 0–7, static thresholds decaying, ACI holding, Thinking on temporal data.
4. 1:50–2:30 — the decision layer under a 500-analyst-hour budget.
5. 2:30–end — `pip install`, tests green, the extensions PR on screen.

---

## 10. Milestones

| ID | Date | Gate |
|---|---|---|
| **M0** ✅ | 19 Sept | **Done:** repo at `~/project_hub/tabpfn-conformal`, Apache 2.0, CI workflow, cahier, API shape signed off. |
| **M0b** ✅ | 19 Sept | **Unblocked.** `TABPFN_TOKEN` in `.env` (gitignored), API reachable, spike S1 run and answered, cost curve measured. Credits turned out to be unnecessary. |
| **M1** ✅ | 22 Sept | *Done early, 19 Sept.* Library complete: scores, marginal, Mondrian, cross-conformal, wrapper, metrics. 53 tests green on CPU. **This was the local-only phase — no TabPFN needed.** |
| **M2** | 25 Sept | First real TabPFN numbers: E1 at small scale via API. Figure 1 v0 exists. |
| **M3** | **28 Sept** | **E1 + E2 complete at full scale** (before the discount ends 29 Sept). **Go/no-go on the headline** per §7.1. |
| **M4** | 1 Oct | ACI + decision layer done. E3 and E4 complete, including the Kaggle wall-clock table. |
| **M5** | 3 Oct | README, docs, figures final. **Repo flipped to public** (deliberately local until now). **Extensions PR opened.** |
| **M6** | 5 Oct | Video recorded. Full reproduction from a clean clone, timed. |
| **M7** | **6 Oct, by 18:00 Morocco** | Submitted. **Five hours of slack before the 22:59 cutoff — not five minutes.** |

### 10.1 Minimum podium-viable cut line

If time collapses, ship in this order and drop from the bottom:

1. Library + tests + README + Apache 2.0 *(without this there is no entry)*
2. E1 headline + Figure 1 + cost table *(without this it is a generic conformal library)*
3. Extensions PR *(cheap, high signal)*
4. **E2 budget sweep** *(promoted 19 Sept: per §3.6 this is the only genuinely unmeasured result in the project — it is the whole originality case, and it is cheap because TabPFN does not train)*
5. Video
6. E3 drift + ACI
7. E4 full baseline matrix
8. Decision layer

Items 6–8 are the ones to sacrifice. **Never sacrifice 1–4.**

---

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **50% criterion**: judges read a model-agnostic library as "TabPFN is interchangeable" | fatal | The headline claim is false for other models. Cost table makes it checkable. Extensions PR. Thinking used where it is claimed strongest. |
| **P1/P2 falsified** — cross-conformal is not cheaper or not better | high | Pre-registered fallback to E2 as headline, decided at M3. |
| **Originality** — Mondrian result is published | medium | Cite it, demote it to background, lead with cross-conformal. |
| **"Why not MAPIE?"** | medium | §6.4 section + numerical agreement test. |
| API quota | **downgraded to low** | Measured: 20M/month, everything at the 10k floor below 100k rows, 213k spent so far. Credits not needed. **Thinking's 30 fits/hour remains the real limit** — batch those. |
| ~~KV-cache × Thinking incompatibility~~ ✅ | resolved | **Measured 19 Sept: incompatible, server-enforced.** Cache → E1/E2, Thinking → E3. Contingency was pre-planned; no cost. |
| **Kaggle has no TabPFN-3.5 model page** | low | `pip install tabpfn` with internet on; accept licence. Spike S2. |
| **Licence confusion** (Apache 2.0 repo, non-commercial weights) | medium | Explicit README section. |
| **Scope overrun**, solo, 18 days, 8 GB laptop | high | §10.1 cut line, enforced at M3 and M5. |
| **Local env**: no `uv`, Python 3.14 default | low | Install `uv`; pin Python 3.11 or 3.12 for the venv — do not fight 3.14 wheels during a deadline. |
| Last-minute submission failure | fatal | M7 is 18:00, five hours early. |

---

## 12. Repository layout

As built, 21 Sept. `scripts/verify_claims.py` asserts every path below exists,
so this cannot drift from the repository again.

```
tabpfn-conformal/
├── LICENSE                     # Apache 2.0, verbatim
├── README.md                   # headline, results, repro, positioning
├── pyproject.toml              # core deps: numpy, pandas, scikit-learn ONLY
├── src/tabpfn_conformal/
│   ├── __init__.py
│   ├── scores.py
│   ├── calibration.py
│   ├── crossconformal.py
│   ├── adaptive.py
│   ├── decision.py
│   ├── metrics.py
│   └── wrapper.py
├── tests/                      # CPU-only, sklearn models, a few seconds
├── experiments/                # TabPFN lives HERE and nowhere else
│   ├── api/                    # the six experiments, against the Prior Labs API
│   ├── kaggle/                 # tier T2 — the fair-hardware wall-clock run
│   └── analyze_*.py            # read committed results; no API key needed
├── scripts/
│   ├── download_data.py
│   ├── build_demo.py           # demo/index.html + figures/demo_data.json
│   ├── build_extension_pr.py   # generates the whole contrib/ payload
│   └── verify_claims.py        # recomputes every numeric claim in the README
├── results/                    # committed JSON/NPZ — judges can replot without running
├── figures/
├── demo/                       # _template.html; index.html is generated
├── contrib/tabpfn-extensions/  # the PR payload, generated from src/
└── docs/
    ├── CAHIER-DES-CHARGES.md   # this file — the plan, kept as written
    ├── FINDINGS.md             # chronological log of results and corrections
    ├── STATUS.md               # handoff: what is left, in order
    ├── SUBMISSION.md           # the submission description
    ├── VIDEO.md                # the demo script, read aloud
    ├── method.md
    └── limitations.md
```

**One thing in the original plan was never built.**
`src/tabpfn_conformal/budget.py` was folded into the `cal_size` constructor
argument (Q1, approved 19 Sept) rather than becoming its own module.

`experiments/kaggle/` was an empty directory until 21 Sept — the tier-2
wall-clock table, planned and never run, which is why P5 is still reported as
confounded. It now holds a runnable script rather than nothing: `wallclock.py`
puts both models on one accelerator with local TabPFN weights, and
`experiments/analyze_kaggle.py` reads what it writes. The measurement still
needs one GPU session; the code no longer does. See
[`limitations.md`](limitations.md).

---

## 13. Open items

**Q1 — API shape. APPROVED 19 Sept**: `cal_size` in the constructor. Implemented.

**Q2 — three-tier compute plan. APPROVED 19 Sept**: laptop for library and tests, Prior Labs API for the science, Kaggle for the wall-clock cost comparison.

**New, discovered during implementation:** MAPIE 1.5 ships `CrossConformalClassifier` as well as `SplitConformalClassifier`, and crepes ships Mondrian classifiers. Cross-conformal is a standard method and we must not imply otherwise. The README now says this outright — the claim is about *economics* (K refits collapse to K forward passes on a training-free model), not about inventing a method. Prior Labs judges will know the conformal ecosystem; being first to say it is much stronger than being caught.

### Day-1 spikes

- ~~**S1**~~ ✅ **Done 19 Sept: incompatible, server-enforced.** See §3.2 and §3.5b.
- **S2** — Kaggle notebook: `pip install tabpfn`, download 3.5 weights, one prediction on a T4. Confirms tier T2 exists.
- **S3** — Create the Prior Labs account, request hackathon credits, run `estimate_cost()` on a BAF-shaped grid, and record real numbers in `docs/method.md`.

---

## 14. Definition of done

A stranger with a free Prior Labs account, no GPU, and ten minutes clones the repo, runs one command, and reproduces Figure 1 — and the README's first screen tells them, in one sentence they can verify from the cost table, why this only works because the model underneath is TabPFN.
