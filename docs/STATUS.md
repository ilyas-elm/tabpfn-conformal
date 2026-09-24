# Status — 24 September 2026 (updated)

Written so the project can be picked up from the repository alone. Deadline
**6 October 2026, 23:59 CEST** (22:59 Morocco) — 12 days.

## Where things stand

Roughly twelve days ahead of the plan in [`CAHIER-DES-CHARGES.md`](CAHIER-DES-CHARGES.md).
Every deliverable exists in draft or better.

| deliverable | state |
|---|---|
| Library (`src/tabpfn_conformal`) | complete — 126 tests, multiclass, verified against MAPIE |
| E1 cross vs split | complete, 40/40 |
| E2 budget allocation | complete, 80/80 |
| E3 drift + ACI + Thinking | complete, 3 seeds — claim weakened, see below |
| E4 baselines | complete, 36/36 |
| E5 scale + KV cache | complete, 24 rows |
| E6 variant replication | complete, 36/36 |
| E7 second domain (Forest Cover Type) | complete, 24/24 — cross wider in 0 of 3 |
| Validity audit (what cross costs) | complete — `experiments/analyze_validity.py` |
| Calibration analysis | complete |
| README | complete, all claims verified |
| `docs/` method, limitations, findings | complete |
| Interactive demo | published |
| Video script | written, not recorded |
| Extensions PR | generated and tested, **not opened** — unblocked now the repo is public |
| Submission text | drafted |

## The one open question — SETTLED 20 Sept

The Thinking drift claim was replicated across three seeds and **weakened**:
base is below target in 9 of 15 seed-months, Thinking in 3 of 15. Thinking is
never worse on any seed and strictly better on two of three, but paired by seed
the difference is 2.0 ± 1.2 months (t ≈ 1.7, n=3) — directional, not
established. README and FINDINGS state it at that strength. No further action
needed unless more seeds are wanted.

## The headline changed on 22 September — read this first

The central claim is now **qualified**, and the qualification is the most
important thing in the project. Cross-conformal reaches the same *targeted*
level from half the confirmed positives — that still holds, and it now
replicates on a second domain (E7, Forest Cover Type). But split's guarantee is
exact and cross's is only approximate, and **cross sits below its own certified
level in 3 of 6 dataset-α combinations by 1.5–2.4 points, where split sits below
in 0 of 6.**

So the trade is: the same targeted level from half the labels, against about two
points of realized coverage at tight α. The README, SUBMISSION and FINDINGS all
say this. **Do not revert to "at no cost" anywhere** —
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
   deliberately not rewritten** — neither item is a credential, and 84 commits of
   visible corrections are worth more than removing them.

   **Going public immediately exposed a defect no local check could see: CI had
   failed 56 times out of 56 and had never once passed.** The badge at the top of
   the README was red from the first run on 19 September, and nothing behind it
   had ever executed — not the tests, not the secret scan, not the claim
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
   before pushing** — their CI fails any PR without a towncrier fragment.
   Verified: imports as
   `tabpfn_extensions.conformal`, 18 tests pass under `FAST_TEST_MODE=1`
   (locally: `FAST_TEST_MODE=1 PYTHONPATH=contrib/tabpfn-extensions/src pytest
   contrib/tabpfn-extensions/tests`).
3. **Record the video** — [`VIDEO.md`](VIDEO.md) has the script and a list of
   seven claims not to make on camera. Its numbers are checked by
   `scripts/verify_claims.py`, so re-run that before recording.
4. **Submit** — [`SUBMISSION.md`](SUBMISSION.md) is the description field.

**Optional, and only after the four above: P5 on fair hardware.** The wall-clock
comparison against LightGBM is confounded (TabPFN remote, LightGBM local) and all
36 E4 rows are tagged `wallclock_comparable: false`.

The run that settles it is written and tested; what it needs is a GPU, which this
laptop does not have. It is a **Kaggle notebook** — a free Jupyter notebook that
runs on Kaggle's machines — and it needs a free Kaggle account with **phone
verification**, which is what unlocks the GPU. About fifteen minutes.
[`../experiments/kaggle/README.md`](../experiments/kaggle/README.md) is the
step-by-step, written for someone who has never opened Kaggle.

The repository is public as of 24 Sept, so the notebook can clone it. This is
no longer blocked on anything but a Kaggle account.

Until it is actually run, P5 stays open and `docs/limitations.md` says so. The
submission does not depend on it.

## Standing rules learned the hard way

- **Never average two arms over different seed sets.** `analyze_e3` aggregated
  `frozen` over three seeds and `aci` over one, then drew them together — which
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
- Check CPU time against elapsed time when a run looks slow — the client can
  block for hours on a dropped response.

## Environment

- Repo: `~/project_hub/tabpfn-conformal`, remote `git@github.com:ilyas-elm/tabpfn-conformal` (**public** since 24 Sept)
- The GitHub repo still has no description, topics or homepage set — the web
  UI is the only way, and it is the first thing a judge sees. Suggested text
  is at the end of this file.
- Homebrew `git` is broken on this machine (libcurl mismatch); use `/usr/bin/git`
- Homebrew cannot install bottles (macOS 14 is Tier 3), so no `gh` — use the web UI
- `uv` at `~/Library/Python/3.14/bin/uv`, not on PATH; venv is Python 3.12
- API token in `.env` (gitignored). Budget: 5M/day, 20M/month, resets 1 Oct
- TabPFN runs on Prior Labs' GPUs via `tabpfn-client`; nothing local needs a GPU

## Repository presentation — not set, and it is the first thing seen

GitHub's API reports `description: null`, `topics: []`, `homepage: null`. On a
submission judged 50% on showcase, the one-line description is what appears in
search, in the org feed, and in every link preview. Only the web UI can set
these (no `gh` on this machine). On the repo page, "About" → the gear icon:

- **Description:**
  `Distribution-free coverage guarantees for TabPFN-3.5. Cross-conformal reaches the same targeted level from half the confirmed positives — measured, with the cost stated.`
- **Website:** the published demo URL (the one in the README's "Try the
  interactive demo" link)
- **Topics:** `conformal-prediction`, `tabpfn`, `uncertainty-quantification`,
  `tabular`, `fraud-detection`, `prediction-sets`, `imbalanced-classification`,
  `machine-learning`
