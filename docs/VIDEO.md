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

Watch the two numbers. Coverage sits at **94.2% against a 95.1% target** and does
not move — the budget cannot change it, because that is the guarantee. Fraud
*caught* moves from **49% to 94%** as the budget rises from zero to 200.

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
model drops below its own promised coverage in **nine of fifteen seed-months.
Thinking, in three.** Thinking was never worse on any seed — but across three
seeds that gap is directional, not statistically established, and we say so in
the README. On the first seed it looked decisive. It did not replicate. Thinking
has no local weights, so that comparison exists only through the Prior Labs API.

We pre-registered five predictions. **Four were falsified** — including two of
our own about cost. They are in the README with the numbers. What survived is
the part that is arithmetic."

## 2:35–end · It is real software

> *Terminal: `pytest` (99 passing), then the extensions PR page — or
> `contrib/tabpfn-extensions/` if the PR is not open yet.*

"A hundred and fourteen tests, CPU only, a few seconds. The core depends on numpy, pandas
and scikit-learn — no TabPFN, no GPU — so it installs in seconds and every
figure regenerates from committed results without an API key.

`tabpfn-extensions` has conformal regression for missing data. It has nothing
for classification. This fills that gap."

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
- Do **not** say Thinking holds the guarantee and base does not, or quote
  "four of five versus zero". That was one seed and it did not replicate. The
  honest line is 9 of 15 versus 3 of 15 seed-months, directional at n = 3.
- Only say "the PR is open" once it actually is. Until then: "the PR is ready
  to open", and show `contrib/tabpfn-extensions/` instead of the PR page.
