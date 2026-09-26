# Status, 24 September 2026 (updated)

Written so the project can be picked up from the repository alone. Deadline
**6 October 2026, 23:59 CEST** (22:59 Morocco), 12 days.

## Where things stand

Roughly twelve days ahead of the plan in [`CAHIER-DES-CHARGES.md`](CAHIER-DES-CHARGES.md).
Every deliverable exists in draft or better.

| deliverable | state |
|---|---|
| Library (`src/tabpfn_conformal`) | complete, 126 tests, multiclass, verified against MAPIE |
| E1 cross vs split | complete, 40/40 |
| E2 budget allocation | complete, 80/80 |
| E3 drift + ACI + Thinking | complete, 3 seeds, claim weakened, see below |
| E4 baselines | complete, 36/36 |
| E5 scale + KV cache | complete, 24 rows |
| E6 variant replication | complete, 36/36 |
| E7 second domain (Forest Cover Type) | complete, 24/24, cross wider in 0 of 3 |
| P5 fair-hardware wall-clock (Kaggle T4) | complete, 24/24, **P5 falsified and settled** |
| Validity audit (what cross costs) | complete, `experiments/analyze_validity.py` |
| Calibration analysis | complete |
| README | complete, all claims verified |
| `docs/` method, limitations, findings | complete |
| Interactive demo | published |
| Video script | written, not recorded |
| Extensions PR | generated and tested, **not opened**, unblocked now the repo is public |
| Submission text | drafted |

## The one open question, SETTLED 20 Sept

The Thinking drift claim was replicated across three seeds and **weakened**:
base is below target in 9 of 15 seed-months, Thinking in 3 of 15. Thinking is
never worse on any seed and strictly better on two of three, but paired by seed
the difference is 2.0 ± 1.2 months (t ≈ 1.7, n=3), directional, not
established. README and FINDINGS state it at that strength. No further action
needed unless more seeds are wanted.

## The headline changed on 22 September; read this first

The central claim is now **qualified**, and the qualification is the most
important thing in the project. Cross-conformal reaches the same *targeted*
level from half the confirmed positives; that still holds, and it now
replicates on a second domain (E7, Forest Cover Type). But split's guarantee is
exact and cross's is only approximate, and **cross sits below its own certified
level in 3 of 6 dataset-α combinations by 1.5–2.4 points, where split sits below
in 0 of 6.**

So the trade is: the same targeted level from half the labels, against about two
points of realized coverage at tight α. The README, SUBMISSION and FINDINGS all
say this. **Do not revert to "at no cost" anywhere**,
`experiments/analyze_validity.py` reproduces the table, and `verify_claims.py`
recomputes both counts.

## What is left to do, in order

1. ~~**Flip the repo public.**~~ **Done 24 September.**
   Audited first: no credentials in any blob of any commit, `.env` and `data/`
   never committed, no sensitive filename ever added, CI references no secrets,
   and the committed `.npz` files hold probabilities and labels only.
   `python scripts/check_secrets.py --history` re-runs the whole sweep and is a
   CI step, so a future commit cannot quietly leak one. Two cosmetic exposures
   were redacted from HEAD (a truncated, expired server-side upload id in
   `results/spike_s1.json`, and local absolute paths in run logs). **History was
   deliberately not rewritten**; neither item is a credential, and 84 commits of
   visible corrections are worth more than removing them.

   **Going public immediately exposed a defect no local check could see: CI had
   failed 56 times out of 56 and had never once passed.** The badge at the top of
   the README was red from the first run on 19 September, and nothing behind it
   had ever executed, not the tests, not the secret scan, not the claim
   verifier. Three causes, fixed in `0c91584`: `uv pip install --system` refuses
   on ubuntu-latest (externally-managed interpreter) and would have targeted the
   wrong interpreter anyway, so the matrix was never testing 3.10 or 3.13; an
   unanchored `data/` in `.gitignore` also matched `tests/data/`, so
   `metric_goldens.json` was never committed and the three results-integrity
   tests failed on every fresh clone; and `verify_claims.py` imported `tomllib`,
   which is 3.11+, under a bare `except`, silently skipping two checks on 3.10.
   **All 30 steps now pass on all three interpreters.**

2. **Open the extensions PR.** `python scripts/build_extension_pr.py` regenerates
   the payload; [`../contrib/tabpfn-extensions/ISSUE.md`](../contrib/tabpfn-extensions/ISSUE.md)
   is the issue to post first (their CONTRIBUTING asks for an issue before a PR),
   and `PR.md` is the description. **Rename
   `contrib/tabpfn-extensions/changelog/PRNUMBER.added.md` to the real PR number
   before pushing**; their CI fails any PR without a towncrier fragment.
   Verified: imports as
   `tabpfn_extensions.conformal`, 18 tests pass under `FAST_TEST_MODE=1`
   (locally: `FAST_TEST_MODE=1 PYTHONPATH=contrib/tabpfn-extensions/src pytest
   contrib/tabpfn-extensions/tests`).
3. **Share the demo artifact.** Still private, unlike the Kaggle notebook. See the section below: it is private, and the
   README links to it prominently. One menu, thirty seconds, and it is the
   difference between the showcase criterion landing and 404ing.
4. **Record the video**; [`VIDEO.md`](VIDEO.md) has the script and a list of
   seven claims not to make on camera. Its numbers are checked by
   `scripts/verify_claims.py`, so re-run that before recording.
5. **Submit**; [`SUBMISSION.md`](SUBMISSION.md) is the description field.

## P5 is settled, 26 September

The fair-hardware run is done: both models on one Tesla T4, local TabPFN
weights, no network inside the measurement. **TabPFN is slower in all four
configurations, by 22 s to 221 s, a factor of 35 to 73.** Removing the confound
moved the result further against TabPFN rather than rescuing it.

Twenty-four rows are committed in `results/kaggle_wallclock.json`, every one
tagged `wallclock_comparable: true`, and `experiments/analyze_kaggle.py`
recomputes the verdict from them with no GPU and no key. Ten checks in
`verify_claims.py` recompute every P5 number the docs quote.

One effect survives inside the loss, and the README says so: split to cross
costs TabPFN 4.46× against LightGBM's 6.42×, so cross-conformal *is* relatively
cheaper without a training step. It is swamped at this scale because LightGBM
trains on 9,000 rows in under a second.

The notebook [`experiments/kaggle/wallclock.ipynb`](../experiments/kaggle/wallclock.ipynb)
is committed, and the run is **public** at
<https://www.kaggle.com/code/ilyaselmaazouzi/tabpfn-conformal>, linked from the P5 row and
from `docs/limitations.md`. Verified signed out: the cells, outputs and
timings all render, and the credentials cell prints the secret's label
only, never its value.

## Standing rules learned the hard way

- **Never average two arms over different seed sets.** `analyze_e3` aggregated
  `frozen` over three seeds and `aci` over one, then drew them together, which
  made ACI look like it moved coverage when at a shared seed it does not. Arms
  are now restricted to the seeds they all have, and the figure caption says so.
- **A generated artifact needs a committed generator.** The demo's data had no
  build script, so nothing could check it against `results/`. `scripts/build_demo.py`
  now rebuilds it and `verify_claims.py` fails if the committed copy differs.
- **When the seeds are shared, never compare means.** A raw win count flattered
  a result four times in this project; the paired test took it back every time.
- **Re-run `scripts/verify_claims.py` after any experiment re-run.** It recomputes
  every number in the README from `results/` and has caught real drift.
- **Grep the docs for absolutes before submitting.** Most are accurate, which is
  why the false ones survive several readings.
- **A badge you have never clicked is not a green badge.** CI here failed 56
  times out of 56 while six review passes called it green, because every one
  of them simulated the workflow locally instead of reading the runs GitHub
  actually executed. Locally the steps all passed; on the runner the install
  died before the first test. Check the API, not your own re-enactment.
- **Clone the repository before believing it reproduces.** `tests/data/` was
  gitignored by accident, so three tests passed for the author and failed for
  everyone else. A working copy cannot show you what you forgot to commit.
- **Anything generated for an outside audience loses what the generator does
  not carry.** `build_extension_pr.py` replaces module docstrings, which
  deleted the approximate-validity caveat from the payload while the PR text
  claimed the module carried it. Diff the generated artefact against what you
  claim about it, not against the source it came from.
- Check CPU time against elapsed time when a run looks slow; the client can
  block for hours on a dropped response.

## Environment

- Repo: `~/project_hub/tabpfn-conformal`, remote `git@github.com:ilyas-elm/tabpfn-conformal` (**public** since 24 Sept)
- The GitHub repo still has no description, topics or homepage set, the web
  UI is the only way, and it is the first thing a judge sees. Suggested text
  is at the end of this file.
- Homebrew `git` is broken on this machine (libcurl mismatch); use `/usr/bin/git`
- Homebrew cannot install bottles (macOS 14 is Tier 3), so no `gh`; use the web UI
- `uv` at `~/Library/Python/3.14/bin/uv`, not on PATH; venv is Python 3.12
- API token in `.env` (gitignored). Budget: 5M/day, 20M/month, resets 1 Oct
- TabPFN runs on Prior Labs' GPUs via `tabpfn-client`; nothing local needs a GPU

## The demo artifact is PRIVATE, and the README links to it

The publish API reports it as private: *only its owner and the people the owner
has given access can open the link.* The README's "Try the interactive demo"
link is the showcase centrepiece of a submission judged 50% on showcase, so
until this is changed a judge clicking it gets nothing. It has to be set to
"anyone with the link" from the page's own Share menu; no tooling can do it.
Check it the way a judge would: open the link in a private window, signed out.

## The published demo is a second copy, and nothing can check it automatically

`scripts/build_demo.py` rebuilds `figures/demo_data.json` and `demo/index.html`,
and `verify_claims.py` fails if the committed data differs from the generator.
It cannot reach the *published* artifact, which is a separate copy behind an
account. Checked by hand on 24 Sept: the published page's embedded data is
identical to `figures/demo_data.json` and its script is byte-identical to the
locally built page. **If `build_demo.py` output ever changes, re-publish the
artifact**, otherwise the README links to a demo showing older numbers, and
nothing in CI will say so.

## Repository presentation, done 24 Sept

Description and topics are set. **Homepage is still empty**; it should point at
the demo, so it is blocked on the demo being shared (above). Check with
`curl -s https://api.github.com/repos/ilyas-elm/tabpfn-conformal | python3 -c
"import sys,json;d=json.load(sys.stdin);print(d['description'],d['homepage'],d['topics'])"`.
