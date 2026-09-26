# Demo video script

Runs **about 4:30** read at a normal pace, including the pauses for dragging the
sliders, 604 spoken words. The submission form calls the video optional; the 50%
criterion is *showcase*, so it is worth making. If you need it shorter, cut
panel 03 and the analyst queue: panel 02 carries the TabPFN argument and panel
06 carries the honesty, and those are the two that earn the marks.

The [demo](https://claude.ai/artifact/RQdPAtjvKefEv1iUT1RB1q) is a console, not a
document; it carries numbers and controls, no prose. **Everything explanatory is
below, to be read aloud.** Nothing in this script needs to appear on screen.

**What to have open:** the demo full-screen, and a terminal. That is all. The
video never leaves the demo until the last twenty seconds, because the page now
carries the model argument and the falsification scoreboard itself, which it did
not when this script was first written.

**Before recording:** the demo link must be public. Open it in a private window,
signed out, and check it loads. It is the only asset in this project that a
viewer cannot reach from the repository.

**Tone:** state a result, then state what failed. Panel 06 is not an apology, it
is the reason to believe the other five.

Everything in **bold** is a number checked by `scripts/verify_claims.py`. Re-run
it before recording.

---

## 0:00–0:25 · Cold open

> *On screen: the console, top bar visible, nothing touched yet.*

"A fraud model gives you a probability. A risk committee needs a guarantee:
what fraction of fraud will this catch, and can you prove it?

Conformal prediction converts one into the other. How tight a guarantee you can
even ask for is fixed by something most teams never check: how many confirmed
frauds you have."

## 0:25–1:05 · Panel 01, the ceiling

> *Drag the label-budget slider down to 100, pause, then back up.*

"A calibration set of size n can only certify alpha of at least one over n plus
one. Arithmetic, not a tendency.

Split conformal, the default in every library, spends half your positives on
calibration. At a hundred confirmed frauds it tops out at **98%**. The 99% your
regulator asked for is not expensive. It is unavailable.

Cross-conformal spends none of them. Same hundred frauds, **99%**."

## 1:05–2:10 · Panel 02, why this is a TabPFN project

> *Let the three tiles sit on screen for a beat each, then drag the fold slider
> from 2 up to 20 and leave it there. Then scroll to the capability table.*

"Cross-conformal is not new. MAPIE ships it. It is rarely used because it costs
K refits, and here is the whole argument for doing this on TabPFN.

TabPFN-3.5 has no training step. `fit` swaps the in-context set and takes no
gradient. Watch the slider: at any number of folds, LightGBM has to train that
many models plus one, and TabPFN trains none. Zero against six at five folds.

That is why the method is affordable. This is why it is also *better*: TabPFN's
calibration error here is a fifth of LightGBM's, and conformal turns
better-calibrated probabilities into narrower sets at the same guarantee.
Twelve percent narrower, same targeted level, same rows.

And this table is the rest of the model I leaned on. The KV cache makes the
evaluation pass six point eight times faster. Thinking mode loses coverage in
three seed-months out of fifteen where the base model loses nine. `estimate_cost`
prices a run before it spends anything, which is how I caught my own cost
overclaim."

## 2:10–2:35 · Panel 03, the same guarantee from half the labels

> *Toggle α 0.05 / α 0.10 once.*

"Measured at an identical targeted level. Read the two 'needs' columns: cross
reaches the same guarantee from **half the confirmed frauds**.

A hundred confirmed frauds is weeks of analyst work. Fifty thousand tokens is a
quarter of one percent of a monthly budget."

## 2:35–3:40 · Panels 04 and 05, the desk

> *Drag the review budget from 0 up to 200. Let the grid recolour. Then pull it
> back to 60 and let the queue table settle.*

"Four hundred real held-out transactions, routed live. Each square is one case.
Singleton prediction sets decide themselves, approve or block. Ambiguous ones
need a human, and you only have so many.

Watch the two leading numbers. Coverage sits at **94.2% against a 95.1% target**
and does not move; the budget cannot change it, because that is the guarantee.
Fraud *caught* moves from **49% to 94%**.

That gap is the honest part. Conformal promises the true label is in the set;
acting on an ambiguous set still needs a free analyst. An uncalibrated
probability hides that trade. This makes it a number you can staff against.

And this is the queue an analyst actually opens: real cases, ranked, with the
set that put them there. Push alpha up and empty sets appear, the model ruling
out *both* labels. Those go to the front."

## 3:40–4:10 · Panel 06, what it cost

> *Stay on the page. Do not cut away.*

"And this is what it cost, on the same page rather than in a footnote.

On one T4 with both models local, TabPFN is fifty-nine times slower than
LightGBM. Cross-conformal's guarantee is approximate where split's is exact,
and it lands below its own certified level in three of six combinations.

Five predictions were registered before any of this ran. Four were falsified,
including two of my own about cost."

## 4:10–4:30 · It is real software

> *Terminal: `pytest`, then `python examples/quickstart.py`.*

"A hundred and thirty tests, CPU only, a few seconds. The core is numpy, pandas
and scikit-learn; no TabPFN, no GPU. Every figure and every number in the README
regenerates from committed results without an API key, and a script checks all
hundred and seventy-eight of them.

`tabpfn-extensions` has conformal regression for missing data. Nothing for
classification. This fills that gap."

---

## Recording notes

- **Drag the sliders slowly.** The recolouring grid is the one moment of motion;
  let it land before speaking over it.
- **The page has no prose on purpose.** Do not read the panel headings aloud;
they are labels; you are the narration.
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
  margins are conservative; say that instead.
- Do **not** say the KV cache gives identical sets. The probabilities differ in
  the fourth decimal. "Same answer to four decimals" is the line.
- Do **not** say cross-conformal is free. It reaches the same *targeted* level
  from half the labels, and it sits about two points below its own certified
  level at tight α, where split holds. Say "half the labels, two points of
  coverage", the measured trade is more convincing than a free lunch.
- Do **not** say this beats MAPIE, or is faster than it. Given the same scores
  and alpha there is one right answer, and the split-conformal sets here are
  bit-identical to MAPIE's; that is how correctness is tested. The two real
  differences are that MAPIE's cross-conformal has no class-conditional option
  at all, which on a 4% minority is 0.043 coverage against 0.957, and that CV+
  queries K+1 models per test row where this queries one. Say those.
- Do **not** say TabPFN is faster than LightGBM. On one T4 it is **35 to 73
  times slower**, which panel 06 shows on screen. The claim is that labels cost
  more than compute, not that the compute is cheap.
- Only say "the PR is open" once it actually is. Until then: "the PR is ready
  to open", and show `contrib/tabpfn-extensions/` instead of the PR page.
