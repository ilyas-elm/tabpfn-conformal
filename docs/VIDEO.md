# Demo video script

Target 2:30–3:00. The submission form calls the video optional; the nearest
competitor has none, and the 50% criterion is *showcase*, so it is worth making.

**What to have open:** the [interactive demo](https://claude.ai/artifact/RQdPAtjvKefEv1iUT1RB1q)
in one tab, the GitHub repo in another, a terminal in a third. Record the demo
full-screen — the terminal appears once, briefly, near the end.

**Tone:** state results, then state what failed. The falsification table is not
an apology, it is the reason to believe the rest.

---

## 0:00–0:25 · The problem

> *On screen: the demo's opening question.*

"A fraud model gives you a probability. A risk committee needs a guarantee —
*what fraction of fraud will this catch, and can you prove it.*

Conformal prediction converts one into the other. But how tight a guarantee you
can even ask for is fixed by something most teams never check: how many
confirmed frauds you have."

## 0:25–1:00 · The ceiling *(panel 01, drag the slider)*

> *Drag the budget slider down to ~100, then up.*

"A calibration set of size n can only certify alpha at least one over n plus
one. That is arithmetic, not a tendency.

Split conformal — the default everyone uses — spends half your positives on
calibration. So with a hundred confirmed frauds it tops out at 98%. **The 99%
your regulator asked for is not expensive. It is unavailable.**

Cross-conformal spends none of them. Same hundred frauds, 99%."

## 1:00–1:35 · Why this is a TabPFN project *(panel 02)*

"Cross-conformal is not new — MAPIE ships it. It is rarely used because it costs
K refits.

**TabPFN-3.5 has no training step.** `fit` swaps the in-context set and takes no
gradient. So K folds are K forward passes.

Measured on Bank Account Fraud, comparing at an identical targeted level:
cross-conformal reaches the same guarantee from **half the confirmed frauds**,
with narrower sets. A hundred confirmed frauds is weeks of analyst work. Fifty
thousand tokens is a quarter of one percent of a monthly budget."

## 1:35–2:05 · The desk *(panel 03 — the money shot)*

> *Drag the review-budget slider from 0 up to 200. Let the grid recolour.*

"Four hundred real TabPFN predictions. Singleton sets decide themselves —
approve, block. Ambiguous ones need a human, and you only have so many.

Watch the two numbers. Coverage stays at 95% — that is the guarantee, and it
holds. Fraud *caught* moves from 30% to 95% as the budget rises.

That gap is the honest part. Conformal promises the true label is in the set.
Acting on an ambiguous set still needs a free analyst. **An uncalibrated
probability hides that trade completely; this makes it a number you can budget
for.**"

## 2:05–2:35 · What TabPFN brings, and what failed

> *Cut to the README's two tables.*

"Why TabPFN specifically? Two measurements.

Its calibration error is **74 to 86% lower** than LightGBM's on identical rows —
and conformal turns that into narrower sets, because a better-calibrated model
reaches the same guarantee with less ambiguity.

And under drift, as the fraud rate climbs from 0.92 to 1.47 percent, the base
model falls below its own promised coverage in four months out of five.
**TabPFN-3.5-Thinking does it in zero.** Thinking has no local weights — that
result exists only through the Prior Labs API.

We pre-registered five predictions. **Four were falsified** — including two of
our own about cost. They are in the README with the numbers. What survived is
the part that is arithmetic."

## 2:35–end · It is real software

> *Terminal: `pytest` (99 passing), then the extensions PR page.*

"Ninety-nine tests, CPU only, about a second. The core depends on numpy, pandas
and scikit-learn — no TabPFN, no GPU — so it installs in one second and every
figure regenerates from committed results without an API key.

`tabpfn-extensions` has conformal regression for missing data. It has nothing
for classification. This fills that gap, and the PR is open."

---

## Recording notes

- **Drag the sliders slowly.** The recolouring grid is the one moment of motion;
  let it land.
- **Do not narrate the terminal.** One shot of green tests, two seconds.
- Say "we predicted X, measured not-X" at least once. Most submissions overclaim;
  a judge who checks one number and finds it honest trusts the rest.
- Do **not** claim cross-conformal is cheaper than split. It is exactly K× the
  tokens. The claim is that labels cost more than compute.
- Do **not** say the extensions repo has no conformal prediction. It has
  `cp_missing_data`, which is regression-only.
