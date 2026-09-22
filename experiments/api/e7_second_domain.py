#!/usr/bin/env python3
"""E7 — does the headline hold on a genuinely different dataset?

Every result so far is Bank Account Fraud. E6 replicated across BAF's Variants
I–III, but those share the schema: same 32 columns, resampled under different
bias. "One dataset family" is the honest description and the README says so.

This is the other domain. Forest Cover Type (Blackard & Dean, UCI): 581,012
cartographic observations, 54 numeric features, predicting tree species from
elevation, slope, hillshade and soil type. Binarised to **cover type 4
(Cottonwood/Willow) against the rest**, which occurs at **0.473%** — comparable
to BAF's 1.1% and arrived at naturally rather than by subsampling. No fraud, no
transactions, no temporal drift, no shared column.

It reruns E1's matched comparison unchanged: split at a budget of 2F positives
against cross at F, which calibrate on the same number and so target the same
level. If the halving result is a property of conformal prediction on an
imbalanced problem rather than a property of this one dataset, it survives here.

    python experiments/api/e7_second_domain.py --dry-run   # price it, spend nothing
    python experiments/api/e7_second_domain.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import REPO, load_token, resume_keys, save_proba, set_client_timeouts, time_limit  # noqa: E402

OUT = REPO / "results" / "e7.jsonl"
BUDGETS = (25, 50, 100, 200)
SEEDS = (0, 1, 2)
N_FOLDS = 5
ALPHAS = (0.01, 0.05, 0.1, 0.2)
POSITIVE_CLASS = 4          # Cottonwood/Willow
EVAL_NEG = 3_000
BASE_RATE = 0.00473         # the rate the dataset actually has


def load_covtype():
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype()
    y = (d.target == POSITIVE_CLASS).astype(int)
    return d.data.astype(float), y


def split_pool_eval(X, y, seed: int):
    """Disjoint pool and evaluation halves, stratified. No time column here."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)


def make_pool(X, y, n_frauds: int, seed: int):
    """Exactly ``n_frauds`` positives at the dataset's own base rate."""
    rng = np.random.default_rng(1000 + seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_neg = int(round(n_frauds / BASE_RATE)) - n_frauds
    if n_frauds > len(pos) or n_neg > len(neg):
        return None, None
    take = np.concatenate([rng.choice(pos, n_frauds, replace=False),
                           rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(take)
    return X[take], y[take]


def make_eval(X, y, seed: int):
    """Every positive, plus a fixed sample of negatives — as in E1."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    neg = rng.choice(neg, min(EVAL_NEG, len(neg)), replace=False)
    take = np.concatenate([pos, neg])
    rng.shuffle(take)
    return X[take], y[take]


def key(c) -> str:
    return f"{c['strategy']}|{c['n_frauds']}|{c['seed']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="price the grid, spend nothing")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    configs = [{"strategy": s, "n_frauds": f, "seed": d}
               for f in args.budgets for d in args.seeds for s in ("split", "cross")]

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2
    set_client_timeouts()
    from tabpfn_client import TabPFNClassifier, estimate_cost

    from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class

    n_eval = EVAL_NEG + 1_100          # refined once the data is loaded
    total = 0
    for cfg in configs:
        n_pool = int(round(cfg["n_frauds"] / BASE_RATE))
        if cfg["strategy"] == "split":
            calls = [(n_pool // 2, n_pool // 2), (n_pool // 2, n_eval)]
        else:
            calls = [(n_pool - n_pool // N_FOLDS, n_pool // N_FOLDS)] * N_FOLDS
            calls.append((n_pool, n_eval))
        for ctx, scored in calls:
            total += estimate_cost(np.zeros((max(ctx, 1), 54)),
                                   np.zeros((max(scored, 1), 54))).estimated_cost

    print(f"{len(configs)} configurations, {len(ALPHAS)} alphas swept offline (free)")
    print(f"estimated cost: {total:,} tokens "
          f"(~{total / 5_000_000:.1%} of the daily cap, "
          f"~{total / 20_000_000:.1%} of the monthly budget)")
    if args.dry_run:
        print("dry run -- nothing spent")
        return 0

    X, y = load_covtype()
    print(f"covtype: {len(y):,} rows, {X.shape[1]} features, "
          f"{int(y.sum()):,} positives ({y.mean():.3%})", flush=True)

    skip = resume_keys(OUT, lambda r: f"{r['strategy']}|{r['n_frauds']}|{r['seed']}")
    if skip:
        print(f"resuming: {len(skip)} configuration(s) already done")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        if key(cfg) in skip:
            continue
        Xp_all, Xe_all, yp_all, ye_all = split_pool_eval(X, y, cfg["seed"])
        X_pool, y_pool = make_pool(Xp_all, yp_all, cfg["n_frauds"], cfg["seed"])
        if X_pool is None:
            print(f"[{i}/{len(configs)}] {key(cfg)}: not enough rows, skipped", flush=True)
            continue
        X_eval, y_eval = make_eval(Xe_all, ye_all, cfg["seed"])

        cc = ConformalClassifier(
            TabPFNClassifier(), method="mondrian", strategy=cfg["strategy"],
            cal_size=0.5, n_folds=N_FOLDS, random_state=cfg["seed"],
        )
        t0 = time.perf_counter()
        try:
            with time_limit(args.timeout):
                cc.fit(X_pool, y_pool)
                proba = cc.predict_proba(X_eval)       # one billed call, reused below
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key(cfg)}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            continue
        elapsed = round(time.perf_counter() - t0, 1)

        rec = {
            **cfg, "dataset": "covtype", "n_pool": int(len(y_pool)),
            "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
            "n_eval_pos": int((y_eval == 1).sum()),
            "seconds": elapsed,
            "proba_file": save_proba("e7", key(cfg), proba, y_eval),
            "alphas": {},
        }
        for a in ALPHAS:                                # free: no further API calls
            sets = cc.predict_set_from_proba(proba, a)
            cov = coverage_by_class(sets, y_eval, cc.classes_)
            rec["alphas"][str(a)] = {
                "coverage_legit": cov[0], "coverage_fraud": cov[1],
                "set_size": average_set_size(sets),
            }
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        c10 = rec["alphas"]["0.1"]
        print(f"[{i}/{len(configs)}] {cfg['strategy']:<5} F={cfg['n_frauds']:<4} "
              f"s={cfg['seed']} cal_pos={rec['n_cal_fraud']:<4} "
              f"cov@0.1={c10['coverage_fraud']:.3f} width={c10['set_size']:.3f} "
              f"({elapsed}s)", flush=True)

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
