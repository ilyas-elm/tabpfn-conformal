# Method

What the library computes, why each choice was made, and the protocol the
experiments follow. For what this project does *not* claim, see
[`limitations.md`](limitations.md).

## 1. The setting

Binary fraud detection at a ~1% base rate. A model produces `p̂(fraud | x)`. The
question a risk function asks is not "how accurate is it" but **"what fraction
of fraud will this catch, and can you prove it."**

Conformal prediction answers that. Given a calibration set drawn exchangeably
with the test data, it produces a **prediction set** `C(x) ⊆ {legitimate, fraud}`
with `P(Y ∈ C(X)) ≥ 1 − α`, distribution-free and finite-sample — no asymptotics,
no assumption about the model or the data-generating process beyond
exchangeability.

## 2. Nonconformity score

For candidate label `k`, the score is how strange it would be for `x` to carry
that label. The default is

```
s(x, k) = 1 − p̂(k | x)
```

Any strictly increasing transform of this gives **identical prediction sets**,
because conformal only ever compares a test score against a quantile of
calibration scores and a monotone map preserves that ordering.
`tests/test_scores.py` asserts this rather than leaving it implied, which is why
`neg_log_prob` exists in the registry and changes nothing.

## 3. The quantile, and why the correction matters

Given calibration scores `s₁ … sₙ` for the true labels, the threshold is the
**⌈(n+1)(1−α)⌉-th smallest**, not the plain empirical `(1−α)` quantile. Using
`np.quantile(scores, 1-alpha)` is the most common conformal bug in the wild and
it silently loses the guarantee at small `n` — exactly the regime a fraud desk
lives in.

Two consequences follow directly from that index, and both drive this project's
results.

**Feasibility.** The index cannot exceed `n`, so a calibration set of size `n`
can only certify

```
α ≥ 1 / (n + 1)
```

Below that no threshold exists. The library returns `+inf` — the trivial
all-labels set, still valid, merely useless — and raises
`InsufficientCalibrationWarning`. It never clips silently.

**Level fidelity.** The index rounds *up*, so the level actually targeted is
`⌈(n+1)(1−α)⌉ / n`, which is above `1−α`. With 13 calibration positives at
α = 0.10 the index is 13 — the maximum score — so the predictor targets **100%**
coverage, not 90%. Any comparison of two methods' realised coverage that ignores
this is comparing them at different levels.

## 4. Marginal versus class-conditional (Mondrian)

**Marginal** takes one threshold over all calibration points. It guarantees
coverage *averaged over classes*, which at a 1% base rate is dominated by the
majority class — the fraud class can be badly under-covered while the headline
number looks healthy.

**Mondrian** partitions the calibration set by class and takes a quantile per
class, guaranteeing `P(Y ∈ C(X) | Y = k) ≥ 1 − α` for every `k`. That is what a
fraud desk needs: a promise about fraud, not about the average of fraud and
legitimate traffic.

This is established (arXiv:2607.27143; *MAKE* 8(7):190, both 2026), not a finding
here. `tests/test_coverage.py` pins the failure and the fix as regression tests.
Mondrian is the default.

## 5. Cross-conformal, and why TabPFN changes the calculus

Split conformal partitions the labelled pool: part trains the model, part
calibrates. Every fraud label spent on calibration is one the model never sees.

Cross-conformal removes the choice. `K` stratified folds; each row is scored by a
model that did not see it; all out-of-fold scores form the calibration set; the
model used at test time is fitted on the whole pool.

The reason this is not routine is cost — `K` refits. **TabPFN has no training
step.** `fit` stores the in-context set and takes no gradient, so `K` folds are
`K` forward passes. The library's `crossconformal.py` is model-agnostic, but the
economics only work for a model that does not train.

Stratified folds are not optional at a 1% base rate, and the number of folds is
validated against the smallest class count rather than failing obscurely later.

**Validity caveat.** Pooling out-of-fold scores and applying them to a model
fitted on the full pool is the cross-conformal predictor of Vovk (2015). Its
validity is *approximate*; the related CV+ of Barber et al. (2021) bounds
worst-case coverage at `1 − 2α`. Split conformal's guarantee is exact. Every
cross-conformal coverage number in this repository is empirical and is reported
next to this caveat.

## 6. Adaptive conformal inference

Exchangeability fails under drift, and Bank Account Fraud drifts: the positive
rate falls to 0.875% by month 2 and rises to 1.475% by month 7.

ACI (Gibbs & Candès, 2021) adapts the *level* rather than refitting:

```
α_{t+1} = α_t + γ (α_target − err_t),    err_t = 1{Y_t ∉ C_{α_t}(X_t)}
```

The long-run empirical error converges to `α_target` for any sequence, including
adversarial ones — a regret bound, not a distributional assumption. The price is
that coverage holds on average over time, not at every step.

`ACI` is **class-conditional by default**. One shared level would be driven
almost entirely by the 99% of traffic that is legitimate and would barely
register a change in fraud behaviour; each class keeps its own level and updates
only when a label of that class arrives.

## 7. Decision layer

A prediction set is not a decision. Under a fixed analyst budget `K`:

- singleton `{legitimate}` → **approve**
- singleton `{fraud}` → **block**
- ambiguous → **review**, if a slot is free

Two cases are ambiguous and they are not the same. `{legitimate, fraud}` means
neither label could be ruled out. The **empty set** means *both* were ruled out —
the point is unlike anything in calibration, the most informative signal a
conformal predictor produces — so empty sets take priority for review. Overflow
behaviour is an explicit parameter, defaulting to the model's own point
prediction, which is what an unaided desk would do.

## 8. Experimental protocol

**Data.** Bank Account Fraud Base (Jesus et al., NeurIPS 2022; Feedzai).
1,000,000 rows, 32 columns, `fraud_bool` at 1.1029%, `month` 0–7. Verified on
download by `scripts/download_data.py`, which checks rather than assumes.

**Split.** Months 0–5 are the labelled pool, months 6–7 the evaluation set.
**Never random** — the fraud rate climbs across the window, so a random split
leaks the future.

**Evaluation set.** Every fraud from months 6–7 (2,878 of them) plus 3,000
sampled legitimate rows. Class-conditional coverage is estimated *within* each
class, so the class mix cannot bias it, and keeping all positives buys a tight
fraud-coverage estimate for a fraction of the tokens a proportionally sampled set
would cost.

**Label budgets.** A pool holding exactly `F` positives at the base rate, so
`F ∈ {25, 50, 100, 200}` corresponds to pools of roughly 2,300 to 18,000 rows.

**Alphas** are swept offline through `predict_set_from_proba`, so additional
levels cost no further API calls. E2 onward also persist the evaluation
probabilities to `results/proba/`, which lets any α be evaluated after the fact
for free.

**Seeds.** Five per configuration for E1 and E2, three for E4. Bands in every
figure span min–max across seeds.

**Pre-registration.** Predictions and their falsification conditions were written
into [`CAHIER-DES-CHARGES.md`](CAHIER-DES-CHARGES.md) §7.1 before the experiments
ran. Two were falsified and are reported as such.

## 9. Compute

TabPFN-3.5 through the managed Prior Labs API (`tabpfn-client`), not local
weights — **TabPFN-3.5-Thinking and -Plus have no local weights at all**, and
Thinking is the variant Prior Labs documents as strongest on temporal data, which
is the E3 setting. `time_col` is rejected outside thinking mode, so native
temporal handling is a Thinking-only capability.

`estimate_cost()` transmits dimensions only and is called before every run.
Every configuration is wrapped in a SIGALRM watchdog after a plain `predict` call
was observed to block for 1h50m on 18 seconds of CPU.

## 10. Reproducing

```bash
pip install -e ".[dev]" && pytest          # 119 tests, CPU, a few seconds
python scripts/download_data.py            # Kaggle credentials needed
pip install -e ".[experiments]"
python -c "import tabpfn_client; tabpfn_client.init()"
python experiments/api/e1_cross_vs_split.py --dry-run   # price it first
python experiments/api/e1_cross_vs_split.py
python experiments/analyze_e1.py --alpha 0.1
```

Every analysis script reads only the committed results files, so all figures and
tables regenerate from a clean clone **without an API key**.
