"""E1 -- does cross-conformal beat split conformal at a scarce fraud-label budget?

This is the headline experiment, and after P2 was falsified on 19 Sept it is the
*only* empirical support left for the headline (see docs/CAHIER-DES-CHARGES.md
sections 7.1 and 7.2). The cost argument is settled and modest: cross-conformal
costs exactly K x split in API tokens. What remains to be shown is the
statistical claim:

    P1 -- at a fixed number of labelled frauds, cross-conformal achieves
          fraud-class coverage with LOWER VARIANCE ACROSS SEEDS and no worse
          set width than split conformal, because it spends none of the
          scarce positives on calibration.

If P1 fails, the cahier says the headline moves to E2. That decision is made
from this script's output, not from hope.

Protocol
--------
Temporal, as the BAF authors recommend: months 0-5 are the labelled pool,
months 6-7 are the evaluation set. Never a random split -- fraud drifts upward
across the eight months and a random split would quietly leak the future.

The evaluation set keeps ALL frauds from months 6-7 plus a sample of legitimate
rows. That is deliberate: class-conditional coverage is estimated *within* each
class, so the class mix of the evaluation set does not bias it, and keeping
every positive buys a tight fraud-coverage estimate for a fraction of the
tokens a proportionally-sampled set would cost.

Cost and safety
---------------
Alphas are swept offline through predict_set_from_proba, so extra alphas are
free. Results append to results/e1.jsonl and completed configurations are
skipped, so the run is resumable after a rate limit or a dropped connection.

    python experiments/api/e1_cross_vs_split.py --dry-run   # price it, spend nothing
    python experiments/api/e1_cross_vs_split.py --pilot     # 1 seed, 2 budgets
    python experiments/api/e1_cross_vs_split.py             # the full grid
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "data" / "Base.csv"
OUT = REPO / "results" / "e1.jsonl"

LABEL, TIME = "fraud_bool", "month"
POOL_MONTHS, EVAL_MONTHS = (0, 1, 2, 3, 4, 5), (6, 7)

FRAUD_BUDGETS = (25, 50, 100, 200, 400)
SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5
ALPHAS = (0.01, 0.05, 0.10, 0.20)
EVAL_LEGIT = 3_000          # legitimate rows kept alongside every eval fraud
BASE_RATE = 0.011           # used only to size the pool around a fraud budget


def load_frames():
    if not DATA.exists():
        sys.exit(f"Missing {DATA.relative_to(REPO)} -- run scripts/download_data.py first.")
    df = pd.read_csv(DATA)
    pool = df[df[TIME].isin(POOL_MONTHS)].reset_index(drop=True)
    ev = df[df[TIME].isin(EVAL_MONTHS)].reset_index(drop=True)
    return pool, ev


def make_eval(ev: pd.DataFrame, seed: int):
    """All evaluation frauds, plus a fixed sample of legitimate rows."""
    rng = np.random.default_rng(seed)
    pos = ev[ev[LABEL] == 1]
    neg = ev[ev[LABEL] == 0]
    take = min(EVAL_LEGIT, len(neg))
    neg = neg.iloc[rng.choice(len(neg), take, replace=False)]
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.drop(columns=[LABEL, TIME]), out[LABEL].to_numpy()


def make_pool(pool: pd.DataFrame, n_frauds: int, seed: int):
    """A labelled pool holding exactly ``n_frauds`` positives at the base rate."""
    rng = np.random.default_rng(1000 + seed)
    pos = pool[pool[LABEL] == 1]
    neg = pool[pool[LABEL] == 0]
    n_neg = int(round(n_frauds / BASE_RATE)) - n_frauds
    if n_frauds > len(pos) or n_neg > len(neg):
        return None, None
    take = pd.concat([
        pos.iloc[rng.choice(len(pos), n_frauds, replace=False)],
        neg.iloc[rng.choice(len(neg), n_neg, replace=False)],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return take.drop(columns=[LABEL, TIME]), take[LABEL].to_numpy()


def grid(budgets, seeds):
    return [
        {"strategy": s, "n_frauds": f, "seed": d}
        for f in budgets for d in seeds for s in ("split", "cross")
    ]


def key(cfg) -> str:
    return f"{cfg['strategy']}|{cfg['n_frauds']}|{cfg['seed']}"


def done_keys() -> set[str]:
    if not OUT.exists():
        return set()
    out = set()
    for line in OUT.read_text().splitlines():
        try:
            out.add(key(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="price the grid, spend nothing")
    ap.add_argument("--pilot", action="store_true", help="1 seed, 2 budgets")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="fraud-label budgets to sweep; default is the full grid",
    )
    args = ap.parse_args()

    budgets = (50, 200) if args.pilot else tuple(args.budgets or FRAUD_BUDGETS)
    seeds = SEEDS[:1] if args.pilot else SEEDS[: args.seeds]
    configs = grid(budgets, seeds)

    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        if line.startswith("TABPFN_TOKEN=") and not os.environ.get("TABPFN_TOKEN"):
            os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()

    from tabpfn_client import TabPFNClassifier, estimate_cost

    from tabpfn_conformal import ConformalClassifier, coverage_by_class, average_set_size

    # --- price it before spending anything -------------------------------
    n_eval = EVAL_LEGIT + 2_000  # approx; refined once the data is loaded
    total = 0
    for cfg in configs:
        n_pool = int(round(cfg["n_frauds"] / BASE_RATE))
        if cfg["strategy"] == "split":
            calls = [(n_pool // 2, n_pool // 2), (n_pool // 2, n_eval)]
        else:
            calls = [(n_pool - n_pool // N_FOLDS, n_pool // N_FOLDS)] * N_FOLDS
            calls.append((n_pool, n_eval))
        for ctx, scored in calls:
            total += estimate_cost(
                np.zeros((max(ctx, 1), 30)), np.zeros((max(scored, 1), 30))
            ).estimated_cost

    print(f"{len(configs)} configurations, {len(ALPHAS)} alphas swept offline (free)")
    print(f"estimated cost: {total:,} tokens  (~{total / 20_000_000:.1%} of the monthly budget)")
    if args.dry_run:
        print("dry run -- nothing spent")
        return 0

    pool_df, eval_df = load_frames()
    print(f"pool months {POOL_MONTHS}: {len(pool_df):,} rows, {int(pool_df[LABEL].sum()):,} frauds")
    print(f"eval months {EVAL_MONTHS}: {len(eval_df):,} rows, {int(eval_df[LABEL].sum()):,} frauds")

    skip = done_keys()
    if skip:
        print(f"resuming: {len(skip)} configuration(s) already done")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        if key(cfg) in skip:
            continue
        X_pool, y_pool = make_pool(pool_df, cfg["n_frauds"], cfg["seed"])
        if X_pool is None:
            print(f"[{i}/{len(configs)}] {key(cfg)}: not enough rows, skipped", flush=True)
            continue
        X_eval, y_eval = make_eval(eval_df, cfg["seed"])

        cc = ConformalClassifier(
            TabPFNClassifier(),
            method="mondrian",
            strategy=cfg["strategy"],
            cal_size=0.5,
            n_folds=N_FOLDS,
            random_state=cfg["seed"],
        )
        t0 = time.perf_counter()
        try:
            cc.fit(X_pool, y_pool)
            proba = cc.predict_proba(X_eval)          # one billed call, reused below
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key(cfg)}: FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        elapsed = round(time.perf_counter() - t0, 1)

        rec = {
            **cfg,
            "n_pool": int(len(y_pool)),
            "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
            "n_eval_fraud": int((y_eval == 1).sum()),
            "seconds": elapsed,
            "alphas": {},
        }
        for a in ALPHAS:                               # free: no further API calls
            sets = cc.predict_set_from_proba(proba, a)
            cov = coverage_by_class(sets, y_eval, cc.classes_)
            rec["alphas"][str(a)] = {
                "coverage_legit": cov[0],
                "coverage_fraud": cov[1],
                "set_size": average_set_size(sets),
            }

        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        c10 = rec["alphas"]["0.1"]
        print(
            f"[{i}/{len(configs)}] {cfg['strategy']:<5} F={cfg['n_frauds']:<4} s={cfg['seed']} "
            f"cal_frauds={rec['n_cal_fraud']:<4} "
            f"fraud_cov@0.1={c10['coverage_fraud']:.3f} width={c10['set_size']:.3f} "
            f"({elapsed}s)",
            flush=True,
        )

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
