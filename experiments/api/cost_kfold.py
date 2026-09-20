#!/usr/bin/env python3
"""What K-fold cross-conformal actually costs, in API tokens.

The README's headline correction -- cross-conformal is **exactly K×** the API
cost of split conformal, not "about the same" -- was measured once in a session
and never written down, so nothing in `results/` backed it. This reproduces it.

`estimate_cost` transmits array *dimensions* only, never data, and is not
charged, so the whole sweep is free. It still needs a token, because the quote
comes from the server.

    python experiments/api/cost_kfold.py            # writes results/cost_kfold.json
    python experiments/api/cost_kfold.py --offline  # re-print the committed result
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import REPO, load_token, set_client_timeouts  # noqa: E402

OUT = REPO / "results" / "cost_kfold.json"
POOLS = (10_000, 100_000)
FOLDS = (2, 5, 20)
N_FEATURES = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="print the committed result instead of re-quoting")
    args = ap.parse_args()

    if args.offline:
        if not OUT.exists():
            raise SystemExit(f"No {OUT.relative_to(REPO)} yet -- run without --offline.")
        saved = json.loads(OUT.read_text())
        report(saved["rows"])
        return 0

    set_client_timeouts()
    if not load_token():
        raise SystemExit("No TABPFN_TOKEN (env or .env).")
    from tabpfn_client import estimate_cost

    def quote(n_context: int, n_scored: int) -> int:
        return int(estimate_cost(
            np.zeros((max(n_context, 1), N_FEATURES)),
            np.zeros((max(n_scored, 1), N_FEATURES)),
        ).estimated_cost)

    rows = []
    for pool in POOLS:
        # Split: fit on half the pool, score the other half. One billed pass.
        split = quote(pool // 2, pool - pool // 2)
        for k in FOLDS:
            # Cross: K folds, each fitting on (K-1)/K of the pool and scoring 1/K.
            per_fold = quote(pool - pool // k, pool // k)
            cross = per_fold * k
            rows.append({
                "n_pool": pool, "k": k,
                "split_tokens": split,
                "per_fold_tokens": per_fold,
                "cross_tokens": cross,
                "ratio": round(cross / split, 3),
            })
            print(f"pool={pool:>7,} K={k:>2}  split={split:>8,}  "
                  f"cross={cross:>9,}  ratio={cross / split:.2f}×", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_features": N_FEATURES, "rows": rows}, indent=2))
    print(f"\nWritten to {OUT.relative_to(REPO)}")
    report(rows)
    return 0


def report(rows) -> None:
    print("\n| pool | K | split tokens | cross tokens | ratio |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['n_pool']:,} | {r['k']} | {r['split_tokens']:,} | "
              f"{r['cross_tokens']:,} | **{r['ratio']:.1f}×** |")
    exact = [r for r in rows if abs(r["ratio"] - r["k"]) < 0.05]
    print(f"\nRatio equals K in {len(exact)} of {len(rows)} quotes.")


if __name__ == "__main__":
    raise SystemExit(main())
