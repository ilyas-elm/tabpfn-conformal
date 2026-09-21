# Status — 20 September 2026 (updated)

Written so the project can be picked up from the repository alone. Deadline
**6 October 2026, 23:59 CEST** (22:59 Morocco) — 16 days.

## Where things stand

Roughly twelve days ahead of the plan in [`CAHIER-DES-CHARGES.md`](CAHIER-DES-CHARGES.md).
Every deliverable exists in draft or better.

| deliverable | state |
|---|---|
| Library (`src/tabpfn_conformal`) | complete — 119 tests, multiclass, verified against MAPIE |
| E1 cross vs split | complete, 40/40 |
| E2 budget allocation | complete, 80/80 |
| E3 drift + ACI + Thinking | ⚠ **seed 2 running** — see below |
| E4 baselines | complete, 36/36 |
| E5 scale + KV cache | complete, 24 rows |
| E6 variant replication | complete, 36/36 |
| Calibration analysis | complete |
| README | complete, all claims verified |
| `docs/` method, limitations, findings | complete |
| Interactive demo | published |
| Video script | written, not recorded |
| Extensions PR | generated and tested, **not opened** |
| Submission text | drafted |

## The one open question — SETTLED 20 Sept

The Thinking drift claim was replicated across three seeds and **weakened**:
base is below target in 9 of 15 seed-months, Thinking in 3 of 15. Thinking is
never worse on any seed and strictly better on two of three, but paired by seed
the difference is 2.0 ± 1.2 months (t ≈ 1.7, n=3) — directional, not
established. README and FINDINGS state it at that strength. No further action
needed unless more seeds are wanted.

## What is left to do, in order

1. **Flip the repo public.** It is private now, and this has to come *first*:
   both [`../contrib/tabpfn-extensions/ISSUE.md`](../contrib/tabpfn-extensions/ISSUE.md)
   and `PR.md` link to `github.com/ilyas-elm/tabpfn-conformal`, so posting either
   while the repo is private gives Prior Labs a 404.
2. **Open the extensions PR.** `python scripts/build_extension_pr.py` regenerates
   the payload; [`../contrib/tabpfn-extensions/ISSUE.md`](../contrib/tabpfn-extensions/ISSUE.md)
   is the issue to post first (their CONTRIBUTING asks for an issue before a PR),
   and `PR.md` is the description. Verified: imports as
   `tabpfn_extensions.conformal`, 9 tests pass under `FAST_TEST_MODE=1`
   (locally: `FAST_TEST_MODE=1 PYTHONPATH=contrib/tabpfn-extensions/src pytest
   contrib/tabpfn-extensions/tests`).
3. **Record the video** — [`VIDEO.md`](VIDEO.md) has the script and a list of
   four claims not to make on camera. Its numbers are checked by
   `scripts/verify_claims.py`, so re-run that before recording.
4. **Submit** — [`SUBMISSION.md`](SUBMISSION.md) is the description field.

Optional, if time allows: **P5 on fair hardware.** The wall-clock comparison
against LightGBM is confounded (TabPFN remote, LightGBM local) and every E4 row
is tagged `wallclock_comparable: false`. Settling it needs both models on one
Kaggle GPU. It is the only explicitly unfinished item from the plan.

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
- Check CPU time against elapsed time when a run looks slow — the client can
  block for hours on a dropped response.

## Environment

- Repo: `~/project_hub/tabpfn-conformal`, remote `git@github.com:ilyas-elm/tabpfn-conformal` (**private**)
- Homebrew `git` is broken on this machine (libcurl mismatch); use `/usr/bin/git`
- Homebrew cannot install bottles (macOS 14 is Tier 3), so no `gh` — use the web UI
- `uv` at `~/Library/Python/3.14/bin/uv`, not on PATH; venv is Python 3.12
- API token in `.env` (gitignored). Budget: 5M/day, 20M/month, resets 1 Oct
- TabPFN runs on Prior Labs' GPUs via `tabpfn-client`; nothing local needs a GPU
