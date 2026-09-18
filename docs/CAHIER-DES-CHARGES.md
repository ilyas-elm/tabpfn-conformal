# Cahier des charges — `tabpfn-conformal`

**Prior Labs TabPFN-3.5 Hackathon** · drafted 18 Sept 2026 · deadline **6 Oct 2026, 23:59 CEST** (22:59 Morocco) · **18 days**

> Status: **v1 — locked on positioning, open on two items** (§13). This document is the contract. If a decision is not written here, it has not been made; if it is written here, do not relitigate it mid-build without editing this file first.

---

## 1. Objective

Wrap TabPFN-3.5's probability outputs in a **distribution-free, finite-sample coverage guarantee** for binary fraud detection under extreme class imbalance and temporal drift, and route that guarantee into an approve / block / review decision under a fixed analyst budget.

**The one-line pitch (README first screen, video first 30 s):**

> *Conformal prediction, priced for a model that never trains.*
> Split conformal makes you throw away half of your ~100 labelled fraud cases to calibrate a guarantee. TabPFN-3.5 has no training step — so K-fold **cross-conformal** costs K forward passes instead of K retrainings, and every fraud label counts twice: once in the context, once in the calibration. On the Prior Labs API, fits are not even billed. This is not affordable with LightGBM. It is nearly free with TabPFN.

---

## 2. Why this wins (mapping to the published rubric)

| Criterion | Weight | How we score |
|---|---|---|
| **Showcase of TabPFN-3.5** | **50%** | The headline claim is *false for every other model*. It rests on three TabPFN-3.5-specific properties: (a) no training step → cross-conformal is inference-only; (b) `fit_mode="fit_with_cache"` KV cache — conformal calibration is literally the "one fixed context, thousands of points scored" workload it was built for; (c) API fits are **not token-charged**, so K folds ≈ the cost of one. Plus: TabPFN-3.5-**Thinking** used where Prior Labs claims it is strongest — *temporal and grouped data* — which is exactly our drift experiment. |
| **Creativity / originality / practical value** | **30%** | Novel: the **label-budget allocation** question (context vs calibration) is only askable of a training-free model; cross-conformal-at-split-conformal-price; ACI under real monthly fraud drift. Practical: a fraud desk gets a per-class guarantee and a fixed review budget. |
| **Technical quality / reproducibility** | **20%** | Zero-dependency core (numpy/pandas/scikit-learn), full pytest suite on CPU, numerical agreement test against MAPIE, `estimate_cost()` printed before every API run, and a headline figure reproducible **on a judge's laptop with a free account and no GPU**. Plus an open PR to `PriorLabs/tabpfn-extensions`. |

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

⚠ **Contradiction to resolve on day 1:** the KV-cache doc says caching is *incompatible with Thinking mode on the managed API*; another source says the client *forces* `use_kv_cache=True` when thinking is enabled. Spike S1 resolves this empirically. It changes whether the cache story and the Thinking story can appear in the same experiment.

### 3.3 The gap in `tabpfn-extensions` is real

Confirmed modules: `interpretability`, `many_class`, `unsupervised`, `embedding`, `image`, `tabebm`, `pval_crt`, `bayesian_optimization`. **No conformal prediction, no calibration, no coverage guarantees.** Apache 2.0, contributions welcome, `uv` + `pytest` conventions.

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
| MAPIE, crepes | Mature Mondrian conformal classification | Addressed head-on: §6.4 positioning section + numerical agreement test. |

**Independently reported and useful to us:** TabPFN achieves the lowest ECE/Brier among tabular models, *but* becomes "increasingly majority-biased as data becomes imbalanced." That is the exact hook: TabPFN's best-in-class average calibration is not a per-class guarantee, and fraud lives entirely in the minority class.

**Closest hackathon competitor found:** `IFoA-ADSWP/tabpfn-reserving` (actuarial loss reserving). Strong README, `docs/method.md`, `docs/readiness.md`, honest results, baseline comparison. **No tests, no video.** That is the bar and that is its gap.

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
| `budget.py` | Helpers for the context/calibration allocation sweep. |

### 4.2 In scope — experiments (`experiments/`, TabPFN + API)

- **E1 (headline)** Cross-conformal vs split conformal at a fixed labelled-fraud budget.
- **E2** Label-budget allocation sweep: context fraction × α → realised fraud-class coverage.
- **E3** Month-by-month drift: static calibration vs ACI, months 0–7.
- **E4** Baselines: TabPFN `majority_downsample`, tuned-threshold (per 2605.21742), and LightGBM — including the **cost table**.

### 4.3 Explicitly OUT of scope

Fine-tuning TabPFN · regression · multiclass · web UI · beating raw AUC · **any TabPFN/torch/GPU import inside `src/tabpfn_conformal`** · conformal risk control · weighted/covariate-shift conformal (mention as future work).

---

## 5. Non-negotiable constraints

1. Core package depends on **numpy, pandas, scikit-learn only**. It must `pip install` and its tests must pass in under 60 s on an 8 GB laptop with no GPU.
2. Repo is **Apache 2.0, verbatim LICENSE file**. README states the TabPFN weights' separate non-commercial licence.
3. Every API script prints `estimate_cost()` **before** spending tokens.
4. Every experiment is seeded and writes a machine-readable result file (`results/*.json` or `.csv`) that the figure scripts consume. No figure is produced by hand.
5. Repo layout mirrors `tabpfn-extensions` conventions so the upstream PR is a reviewable diff.

---

## 6. Library design

### 6.1 Public API — **PROPOSED, needs your sign-off before implementation** (§13, Q1)

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
cc.quantiles_                    # dict {class_label: threshold}; marginal -> {None: t}
cc.n_calibration_                # dict {class_label: count} — honesty about tiny fraud n
```

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
| P2 | Cross-conformal costs **< 2× split** in API tokens (because fits are unbilled) | ≥ 2× |
| P3 | Marginal CP under-covers fraud at α=0.05; Mondrian does not | marginal is fine (would contradict published work) |
| P4 | Static Mondrian thresholds lose coverage by month 7; ACI holds it | static holds — then report that BAF drift is too mild and say so |
| P5 | LightGBM cross-conformal costs ≫ TabPFN in wall-clock on identical hardware | comparable — then the headline weakens to "equally cheap", and we re-weight toward E2/E3 |

**If P1 and P2 both fail, the headline changes to E2.** Decide by 28 Sept (milestone M3).

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

**Rule: run every wide sweep (E1, E2) before 29 Sept.** That is 11 days, and it halves the cost of the two biggest experiments.

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
| D7 | **Demo video, 2–3 min** | 30% + the competitor's blind spot |
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
| **M0** | **19 Sept** | Spikes S1–S3 done. Prior Labs account + hackathon credits requested. Repo public, Apache 2.0, CI green. API shape signed off. |
| **M1** | 22 Sept | Library complete: scores, marginal, Mondrian, cross-conformal, wrapper. Full test suite green on CPU. **This is the local-only phase — no TabPFN needed.** |
| **M2** | 25 Sept | First real TabPFN numbers: E1 at small scale via API. Figure 1 v0 exists. |
| **M3** | **28 Sept** | **E1 + E2 complete at full scale** (before the discount ends 29 Sept). **Go/no-go on the headline** per §7.1. |
| **M4** | 1 Oct | ACI + decision layer done. E3 and E4 complete, including the Kaggle wall-clock table. |
| **M5** | 3 Oct | README, docs, figures final. **Extensions PR opened.** |
| **M6** | 5 Oct | Video recorded. Full reproduction from a clean clone, timed. |
| **M7** | **6 Oct, by 18:00 Morocco** | Submitted. **Five hours of slack before the 22:59 cutoff — not five minutes.** |

### 10.1 Minimum podium-viable cut line

If time collapses, ship in this order and drop from the bottom:

1. Library + tests + README + Apache 2.0 *(without this there is no entry)*
2. E1 headline + Figure 1 + cost table *(without this it is a generic conformal library)*
3. Extensions PR *(cheap, high signal)*
4. Video
5. E3 drift + ACI
6. E2 full sweep
7. E4 full baseline matrix
8. Decision layer

Items 5–8 are the ones to sacrifice. **Never sacrifice 1–3.**

---

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **50% criterion**: judges read a model-agnostic library as "TabPFN is interchangeable" | fatal | The headline claim is false for other models. Cost table makes it checkable. Extensions PR. Thinking used where it is claimed strongest. |
| **P1/P2 falsified** — cross-conformal is not cheaper or not better | high | Pre-registered fallback to E2 as headline, decided at M3. |
| **Originality** — Mondrian result is published | medium | Cite it, demote it to background, lead with cross-conformal. |
| **"Why not MAPIE?"** | medium | §6.4 section + numerical agreement test. |
| **API quota / rate limits**, esp. Thinking at 30/hr | high | `estimate_cost()` dry runs; request credits day 1; sweeps before 29 Sept; Thinking batched overnight; Kaggle fallback. |
| **KV-cache × Thinking incompatibility** ⚠ | medium | Spike S1. If incompatible, the cache story attaches to E1/E2 (base model) and Thinking attaches to E3 — they simply do not share a figure. |
| **Kaggle has no TabPFN-3.5 model page** | low | `pip install tabpfn` with internet on; accept licence. Spike S2. |
| **Licence confusion** (Apache 2.0 repo, non-commercial weights) | medium | Explicit README section. |
| **Scope overrun**, solo, 18 days, 8 GB laptop | high | §10.1 cut line, enforced at M3 and M5. |
| **Local env**: no `uv`, Python 3.14 default | low | Install `uv`; pin Python 3.11 or 3.12 for the venv — do not fight 3.14 wheels during a deadline. |
| Last-minute submission failure | fatal | M7 is 18:00, five hours early. |

---

## 12. Repository layout

```
tabpfn-conformal/
├── LICENSE                     # Apache 2.0, verbatim
├── README.md                   # headline, Figure 1, repro, positioning, licence note
├── pyproject.toml              # core deps: numpy, pandas, scikit-learn ONLY
├── src/tabpfn_conformal/
│   ├── __init__.py
│   ├── scores.py
│   ├── calibration.py
│   ├── crossconformal.py
│   ├── adaptive.py
│   ├── decision.py
│   ├── budget.py
│   └── wrapper.py
├── tests/                      # CPU-only, sklearn models, <60 s
├── experiments/                # TabPFN lives HERE and nowhere else
│   ├── api/                    # tier T1 — reproducible, no GPU
│   └── kaggle/                 # tier T2 — wall-clock cost table
├── scripts/download_data.py
├── results/                    # committed JSON/CSV — judges can replot without running
├── figures/
└── docs/
    ├── CAHIER-DES-CHARGES.md   # this file
    ├── method.md
    └── limitations.md
```

---

## 13. Open items

**Q1 — sign off the public API in §6.1 before M1.** Expensive to reverse once experiments depend on it. Specifically: is `cal_size` as a constructor arg (rather than passing an explicit `X_cal`) the right shape, given it is also the E2 sweep variable?

**Q2 — confirm the three-tier compute plan in §8.** In particular, that Kaggle is demoted to *only* the wall-clock cost table, and that the Prior Labs API carries the science because Thinking has no local weights.

### Day-1 spikes

- **S1** — Does `fit_mode="fit_with_cache"` work together with `thinking_effort` on the managed API? Resolves the §3.2 contradiction.
- **S2** — Kaggle notebook: `pip install tabpfn`, download 3.5 weights, one prediction on a T4. Confirms tier T2 exists.
- **S3** — Create the Prior Labs account, request hackathon credits, run `estimate_cost()` on a BAF-shaped grid, and record real numbers in `docs/method.md`.

---

## 14. Definition of done

A stranger with a free Prior Labs account, no GPU, and ten minutes clones the repo, runs one command, and reproduces Figure 1 — and the README's first screen tells them, in one sentence they can verify from the cost table, why this only works because the model underneath is TabPFN.
