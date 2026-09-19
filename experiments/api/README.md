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
```

Then authenticate once. The environment variable is **`TABPFN_TOKEN`** (verified
against `tabpfn-client` 0.6.0, not guessed). Three ways, pick one:

```bash
# a) let the client handle it -- logs in and caches the token itself
python -c "import tabpfn_client; tabpfn_client.init()"
```

```bash
# b) write .env without the token ever touching your shell history or the screen
printf 'TABPFN_TOKEN=' > .env && read -rs T && printf '%s\n' "$T" >> .env && chmod 600 .env && unset T && echo written
```

```bash
# c) plain export, this shell only
export TABPFN_TOKEN=...
```

`.env` is already in `.gitignore`. Never commit the token, and never paste it
into a chat.

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
| `spike_s1_cache_thinking.py` | Settles whether the KV cache works with Thinking mode, and prices every planned experiment. Run this first. |

Start with the free half — it spends nothing:

```bash
python experiments/api/spike_s1_cache_thinking.py --quotes-only
```

`estimate_cost` accepts `operation` in `{predict, cache_predict, thinking_fit,
thinking_predict}` and transmits dimensions only, so this prints the token cost
of E1–E3 *and* measures the KV-cache discount without consuming quota. Drop the
flag to run the four live probes (~40k tokens of a 5M daily budget).
