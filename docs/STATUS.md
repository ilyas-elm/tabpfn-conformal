# Status — 20 September 2026

Written so the project can be picked up from the repository alone. Deadline
**6 October 2026, 23:59 CEST** (22:59 Morocco) — 16 days.

## Where things stand

Roughly twelve days ahead of the plan in [`CAHIER-DES-CHARGES.md`](CAHIER-DES-CHARGES.md).
Every deliverable exists in draft or better.

| deliverable | state |
|---|---|
| Library (`src/tabpfn_conformal`) | complete — 99 tests, multiclass, verified against MAPIE |
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

## The one open question

**The Thinking drift result does not replicate cleanly on two seeds.** Base
fails 4 of 5 months on seed 0 and 0 of 5 on seed 1; Thinking is 0 of 5 on both.
A third seed is running via `chain_thinking.sh`.

When it lands:

```bash
python experiments/analyze_e3.py --model base
python experiments/analyze_e3.py --model thinking
python scripts/verify_claims.py
```

Then **settle or drop the claim**. If seed 2 has base failing again, it is two
of three and worth stating as directional. If base holds again, the honest
conclusion is that base's failure was a single bad draw and the claim should
come out of the README summary table entirely.

Do not ship it on n=2.

## What is left to do, in order

1. **Settle the Thinking claim** (above).
2. **Open the extensions PR.** `python scripts/build_extension_pr.py` regenerates
   the payload; [`../contrib/tabpfn-extensions/ISSUE.md`](../contrib/tabpfn-extensions/ISSUE.md)
   is the issue to post first (their CONTRIBUTING asks for an issue before a PR),
   and `PR.md` is the description. Verified: imports as
   `tabpfn_extensions.conformal`, 9 tests pass under `FAST_TEST_MODE=1`.
3. **Record the video** — [`VIDEO.md`](VIDEO.md) has the script and a list of two
   claims not to make on camera.
4. **Flip the repo public** before submitting. It is private now.
5. **Submit** — [`SUBMISSION.md`](SUBMISSION.md) is the description field.

Optional, if time allows: **P5 on fair hardware.** The wall-clock comparison
against LightGBM is confounded (TabPFN remote, LightGBM local) and every E4 row
is tagged `wallclock_comparable: false`. Settling it needs both models on one
Kaggle GPU. It is the only explicitly unfinished item from the plan.

## Standing rules learned the hard way

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
