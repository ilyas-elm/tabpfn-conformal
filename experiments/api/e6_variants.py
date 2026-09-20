"""E6 -- does the headline replicate across the BAF variants?

Everything so far runs on BAF Base. One dataset is one result, and the honest
question a reader asks next is whether it generalises. The Bank Account Fraud
suite exists precisely for this: six one-million-row datasets at the same 1.103%
fraud rate, differing in the *bias* deliberately injected into them (Jesus et
al., NeurIPS 2022). Same task, different data-generating quirks — a real
replication test rather than a friendly one.

Deliberately narrow. This does not re-run E1; it re-runs the single comparison
the README leads with, which is the matched-level one:

    split conformal at a budget of 2F calibrates on F positives
    cross-conformal at a budget of  F calibrates on F positives
    -> identical calibration size, identical targeted level, comparable widths

Two matched pairs per variant (n_cal = 50 and n_cal = 100), three seeds. If
cross-conformal reaches the same guarantee from half the labels only on Base,
that shows up here.

    python experiments/api/e6_variants.py --dry-run
    python experiments/api/e6_variants.py --variants "Variant I" "Variant III"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from _common import (  # noqa: E402
    EVAL_LEGIT, EVAL_MONTHS, LABEL, POOL_MONTHS, REPO, TIME, load_token,
    make_pool, resume_keys, set_client_timeouts, split_xy, time_limit,
)

OUT = REPO / "results" / "e6.jsonl"

# (strategy, fraud budget) pairs that land on the same calibration size.
MATCHED = [("split", 100), ("cross", 50),      # n_cal = 50
           ("split", 200), ("cross", 100)]     # n_cal = 100
VARIANTS = ["Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]
SEEDS = (0, 1, 2)
ALPHAS = (0.05, 0.10)
N_FOLDS = 5


def load_variant(name: str):
    path = REPO / "data" / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(REPO)} — run scripts/download_data.py")
    df = pd.read_csv(path)
    return (df[df[TIME].isin(POOL_MONTHS)].reset_index(drop=True),
            df[df[TIME].isin(EVAL_MONTHS)].reset_index(drop=True))


def make_eval_local(ev: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    pos, neg = ev[ev[LABEL] == 1], ev[ev[LABEL] == 0]
    neg = neg.iloc[rng.choice(len(neg), min(EVAL_LEGIT, len(neg)), replace=False)]
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_xy(out)


def key(c) -> str:
    return f"{c['variant']}|{c['strategy']}|{c['n_frauds']}|{c['seed']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variants", nargs="+", default=VARIANTS[:3])
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    configs = [{"variant": v, "strategy": s, "n_frauds": f, "seed": d}
               for v in args.variants for d in SEEDS for (s, f) in MATCHED]

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2
    set_client_timeouts()

    from tabpfn_client import TabPFNClassifier, estimate_cost
    from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class

    n_eval, base_rate, total = 5_878, 0.011, 0
    for c in configs:
        n = int(round(c["n_frauds"] / base_rate))
        calls = ([(n - n // N_FOLDS, n // N_FOLDS)] * N_FOLDS
                 if c["strategy"] == "cross" else [(n // 2, n // 2)])
        calls.append((n, n_eval))
        for ctx, sc in calls:
            total += estimate_cost(np.zeros((max(ctx, 1), 30)),
                                   np.zeros((max(sc, 1), 30))).estimated_cost
    print(f"{len(configs)} configurations across {len(args.variants)} variants")
    print(f"estimated cost: {total:,} tokens ({total / 5_000_000:.0%} of the 5M daily cap)")
    if args.dry_run:
        return 0

    skip = resume_keys(OUT, key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames: dict[str, tuple] = {}

    for i, cfg in enumerate(configs, 1):
        if key(cfg) in skip:
            continue
        if cfg["variant"] not in frames:
            frames[cfg["variant"]] = load_variant(cfg["variant"])
            pool, ev = frames[cfg["variant"]]
            print(f"\n{cfg['variant']}: pool {len(pool):,} rows "
                  f"({pool[LABEL].mean():.3%} fraud), eval {len(ev):,} "
                  f"({int(ev[LABEL].sum()):,} frauds)", flush=True)
        pool_df, eval_df = frames[cfg["variant"]]

        X_pool, y_pool = make_pool(pool_df, cfg["n_frauds"], cfg["seed"])
        if X_pool is None:
            continue
        X_eval, y_eval = make_eval_local(eval_df, cfg["seed"])

        cc = ConformalClassifier(
            TabPFNClassifier(), method="mondrian", strategy=cfg["strategy"],
            cal_size=0.5, n_folds=N_FOLDS, random_state=cfg["seed"],
        )
        t0 = time.perf_counter()
        try:
            with time_limit(args.timeout):
                cc.fit(X_pool, y_pool)
                proba = cc.predict_proba(X_eval)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key(cfg)}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            continue

        rec = {**cfg, "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
               "n_pool": int(len(y_pool)), "seconds": round(time.perf_counter() - t0, 1),
               "alphas": {}}
        for a in ALPHAS:
            sets = cc.predict_set_from_proba(proba, a)
            cov = coverage_by_class(sets, y_eval, cc.classes_)
            rec["alphas"][str(a)] = {"coverage_fraud": cov[1], "coverage_legit": cov[0],
                                     "set_size": average_set_size(sets)}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        c10 = rec["alphas"]["0.1"]
        print(f"[{i}/{len(configs)}] {cfg['variant']:<12} {cfg['strategy']:<5} "
              f"F={cfg['n_frauds']:<4} s={cfg['seed']} n_cal={rec['n_cal_fraud']:<4} "
              f"cov@.10={c10['coverage_fraud']:.3f} width={c10['set_size']:.3f} "
              f"({rec['seconds']}s)", flush=True)

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
