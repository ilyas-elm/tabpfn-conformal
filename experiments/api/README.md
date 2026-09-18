# API experiments (tier T1)

These run against the **Prior Labs API**, not a local GPU. You write and run them
from VS Code on your laptop; TabPFN executes on Prior Labs' servers.

Why the API rather than Kaggle: **TabPFN-3.5-Thinking and -Plus have no local
weights.** Thinking is the variant that tops TabArena, BeyondArena, STRABLE and
MulTaBench, and it is reachable only through the managed API (or SageMaker /
SAP AI Core). A Kaggle GPU runs the base checkpoint. The API is also what makes
these results reproducible by a judge with a free account and no GPU.

## Setup

```bash
pip install -e ".[experiments]"
export PRIORLABS_API_TOKEN=...   # from https://platform.priorlabs.ai
```

## Budget discipline

- `estimate_cost()` sends dimensions only — no upload, no quota consumed. Call it
  before every sweep.
- Budgets are **5M tokens/day, 20M/month**, with a **10,000-token minimum per
  billable operation**.
- **Uploads and standard fits are not charged.** Only predictions are. This is
  why K-fold cross-conformal is cheap here: K fits, but each row predicted once.
- **Thinking fits are rate-limited to 30/hour.** Batch them overnight; never
  iterate on them interactively.
- Rates are **50% off until 29 September 2026** — run the wide sweeps before then.

## Scripts

| Script | Purpose |
|---|---|
| `spike_s1_cache_thinking.py` | Settles whether the KV cache works with Thinking mode, and prints free cost quotes for the planned experiments. Run this first. |
