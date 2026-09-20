# Demo video script

Runs **about 4:10** read at a normal pace, including the pauses for dragging the
sliders — 508 spoken words. The submission form calls the video optional; the 50%
criterion is *showcase*, so it is worth making. If you need it under three
minutes, cut the 'What TabPFN brings' section and keep the desk: the desk is the
part that cannot be read off the README.

The [demo](https://claude.ai/artifact/RQdPAtjvKefEv1iUT1RB1q) is a console, not a
document — it carries numbers and controls, no prose. **Everything explanatory is
below, to be read aloud.** Nothing in this script needs to appear on screen.

**What to have open:** the demo full-screen in one tab, the GitHub repo in
another, a terminal in a third. The terminal appears once, briefly, near the end.

**Tone:** state a result, then state what failed. The falsification table is not
an apology, it is the reason to believe the rest.

Everything in **bold** is a number checked by `scripts/verify_claims.py`. Re-run
it before recording.

---

## 0:00–0:27 · Cold open

> *On screen: the console, top bar visible, nothing touched yet.*

"A fraud model gives you a probability. A risk committee needs a guarantee —
what fraction of fraud will this catch, and can you prove it?

Conformal prediction converts one into the other. How tight a guarantee you can
even ask for is fixed by something most teams never check: how many confirmed
frauds you have."

## 0:27–1:04 · Panel 01 — the ceiling

> *Drag the label-budget slider down to 100, pause, then back up.*

"A calibration set of size n can only certify alpha of at least one over n plus
one. Arithmetic, not a tendency.

Split conformal — the default in every library — spends half your positives on
calibration. At a hundred confirmed frauds it tops out at **98%**. The 99% your
regulator asked for is not expensive. It is unavailable.

Cross-conformal spends none of them. Same hundred frauds, **99%**."

## 1:04–1:43 · Panel 02 — why this is a TabPFN project

> *Point at the table. Toggle α 0.05 / α 0.10 once.*

"Cross-conformal is not new — MAPIE ships it. It is rarely used because it costs
K refits.

TabPFN-3.5 has no training step. `fit` swaps the in-context set and takes no
gradient, so K folds are K forward passes.

This table compares them at an identical targeted level. Read the two 'needs'
columns: cross reaches the same guarantee from **half the confirmed frauds**.

A hundred confirmed frauds is weeks of analyst work. Fifty thousand tokens is a
quarter of one percent of a monthly budget."

## 1:43–2:58 · Panels 03 and 04 — the desk

> *Drag the review budget from 0 up to 200. Let the grid recolour. Then pull it
> back to 60 and let the queue table settle.*

"Four hundred real held-out transactions, routed live. Each square is one case.
Singleton prediction sets decide themselves — approve, block. Ambiguous ones need
a human, and you only have so many.

Watch the two leading numbers. Coverage sits at **94.2% against a 95.1% target**
and does not move — the budget cannot change it, because that is the guarantee.
Fraud *caught* moves from **49% to 94%**.

That gap is the honest part. Conformal promises the true label is in the set;
acting on an ambiguous set still needs a free analyst. An uncalibrated
probability hides that trade. This makes it a number you can staff against.

And this is the queue an analyst actually opens — real cases, ranked, with the
set that put them there. Push alpha up and empty sets appear: the model ruling
out *both* labels. Those go to the front."

## 2:58–3:47 · What TabPFN brings, and what failed *(cut this first if you need to)*

> *Cut to the README's two tables.*

"Why TabPFN specifically? Two measurements.

Its calibration error is **74 to 86% lower** than LightGBM's on identical rows,
and conformal turns that into narrower sets.

Under drift, the base model drops below its own promised coverage in **nine of
fifteen seed-months. Thinking, in three.** Thinking was never worse on any seed —
but at three seeds that gap is directional, not established, and the README says
so. On the first seed it looked decisive. It did not replicate. Thinking has no
local weights, so that comparison exists only through the Prior Labs API.

We pre-registered five predictions. **Four were falsified** — including two of
our own about cost."

## 3:47–4:11 · It is real software

> *Terminal: `pytest`, then the extensions PR page — or
> `contrib/tabpfn-extensions/` if the PR is not open yet.*

"A hundred and seventeen tests, CPU only, a few seconds. The core is numpy, pandas
and scikit-learn — no TabPFN, no GPU — and every figure regenerates from
committed results without an API key.

`tabpfn-extensions` has conformal regression for missing data. Nothing for
classification. This fills that gap."

---

## Recording notes

- **Drag the sliders slowly.** The recolouring grid is the one moment of motion;
  let it land before speaking over it.
- **The page has no prose on purpose.** Do not read the panel headings aloud —
  they are labels, you are the narration.
- **Do not narrate the terminal.** One shot of green tests, two seconds.
- Say "we predicted X, measured not-X" at least once. Most submissions overclaim;
  a judge who checks one number and finds it honest trusts the rest.

### Do not say

- Do **not** claim cross-conformal is cheaper than split. It is exactly K× the
  tokens, measured. The claim is that labels cost more than compute.
- Do **not** say the extensions repo has no conformal prediction. It has
  `cp_missing_data`, which is regression-only.
- Do **not** say Thinking holds the guarantee and base does not, or quote
  "four of five versus zero". That was one seed and it did not replicate. The
  honest line is 9 of 15 versus 3 of 15 seed-months, directional at n = 3.
- Do **not** say the matched comparison has no confound. Split at 2F also gets
  twice the in-context rows; the asymmetry favours split, which is why the
  margins are conservative — say that instead.
- Do **not** say the KV cache gives identical sets. The probabilities differ in
  the fourth decimal. "Same answer to four decimals" is the line.
- Only say "the PR is open" once it actually is. Until then: "the PR is ready
  to open", and show `contrib/tabpfn-extensions/` instead of the PR page.
