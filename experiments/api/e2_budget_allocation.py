"""E2 -- where should a scarce fraud-label budget go?

This is the one genuinely unmeasured question in the project (see
docs/CAHIER-DES-CHARGES.md section 3.6). Split conformal forces a choice: every
labelled fraud you move into the calibration set is one the model no longer
sees in context, and vice versa. For a model that trains, nobody sweeps this,
because each point on the curve is a retraining run. TabPFN has no training
step, so the sweep is a loop over ``cal_size`` and costs one forward pass per
point.

Two forces pull against each other, and the interesting part is that they are
not symmetric:

- **Too little calibration** and the guarantee is not merely noisy, it becomes
  *unavailable*: split conformal can only certify alpha >= 1/(n_cal+1), so below
  n_cal = 1/alpha - 1 the predictor must return every label (section 7.3).
- **Too much calibration** and the model itself is starved of positives, so the
  sets it produces are wide even when the threshold is well estimated.

Cross-conformal is run at each budget as a reference line, because the honest
question is not only "where is the best split" but "does the best split beat
not splitting at all".

    python experiments/api/e2_budget_allocation.py --dry-run
    python experiments/api/e2_budget_allocation.py --pilot
    python experiments/api/e2_budget_allocation.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from _common import (  # noqa: E402
    REPO, load_frames, load_token, make_eval, make_pool, resume_keys,
)

OUT = REPO / "results" / "e2.jsonl"

CAL_SIZES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
FRAUD_BUDGETS = (100, 200)
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.01, 0.05, 0.10, 0.20)
N_FOLDS = 5
BASE_RATE = 0.011


def grid(budgets, cal_sizes, seeds):
    rows = [
        {"strategy": "split", "cal_size": c, "n_frauds": f, "seed": s}
        for f in budgets for s in seeds for c in cal_sizes
    ]
    # Cross-conformal once per (budget, seed) as the "don't split at all" line.
    rows += [
        {"strategy": "cross", "cal_size": None, "n_frauds": f, "seed": s}
        for f in budgets for s in seeds
    ]
    return rows


def key(r) -> str:
    return f"{r['strategy']}|{r['cal_size']}|{r['n_frauds']}|{r['seed']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    budgets = (100,) if args.pilot else FRAUD_BUDGETS
    cal_sizes = (0.2, 0.5, 0.8) if args.pilot else CAL_SIZES
    seeds = SEEDS[:1] if args.pilot else SEEDS
    configs = grid(budgets, cal_sizes, seeds)

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2

    from tabpfn_client import TabPFNClassifier, estimate_cost
    from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class

    n_eval = 5_000
    total = 0
    for c in configs:
        n_pool = int(round(c["n_frauds"] / BASE_RATE))
        if c["strategy"] == "split":
            n_cal = int(round(n_pool * c["cal_size"]))
            calls = [(n_pool - n_cal, n_cal), (n_pool - n_cal, n_eval)]
        else:
            calls = [(n_pool - n_pool // N_FOLDS, n_pool // N_FOLDS)] * N_FOLDS
            calls.append((n_pool, n_eval))
        for ctx, scored in calls:
            total += estimate_cost(
                np.zeros((max(ctx, 1), 30)), np.zeros((max(scored, 1), 30))
            ).estimated_cost

    print(f"{len(configs)} configurations, {len(ALPHAS)} alphas swept offline (free)")
    print(f"estimated cost: {total:,} tokens (~{total / 20_000_000:.1%} of the monthly budget)")
    if args.dry_run:
        return 0

    pool_df, eval_df = load_frames()
    skip = resume_keys(OUT, key)
    if skip:
        print(f"resuming: {len(skip)} configuration(s) already done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        if key(cfg) in skip:
            continue
        X_pool, y_pool = make_pool(pool_df, cfg["n_frauds"], cfg["seed"])
        if X_pool is None:
            continue
        X_eval, y_eval = make_eval(eval_df, cfg["seed"])

        cc = ConformalClassifier(
            TabPFNClassifier(),
            method="mondrian",
            strategy=cfg["strategy"],
            cal_size=cfg["cal_size"] if cfg["cal_size"] is not None else 0.5,
            n_folds=N_FOLDS,
            random_state=cfg["seed"],
        )
        t0 = time.perf_counter()
        try:
            cc.fit(X_pool, y_pool)
            proba = cc.predict_proba(X_eval)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key(cfg)}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            continue

        n_cal_fraud = int(cc.n_calibration_.get(1, 0))
        rec = {
            **cfg,
            "n_pool": int(len(y_pool)),
            "n_cal_fraud": n_cal_fraud,
            # The hard floor from section 7.3, recorded per row so the analysis
            # can mark infeasible points rather than plotting them as successes.
            "min_certifiable_alpha": 1.0 / (n_cal_fraud + 1) if n_cal_fraud else None,
            "seconds": round(time.perf_counter() - t0, 1),
            "alphas": {},
        }
        for a in ALPHAS:
            sets = cc.predict_set_from_proba(proba, a)
            cov = coverage_by_class(sets, y_eval, cc.classes_)
            rec["alphas"][str(a)] = {
                "coverage_legit": cov[0],
                "coverage_fraud": cov[1],
                "set_size": average_set_size(sets),
                "feasible": bool(rec["min_certifiable_alpha"] is not None
                                 and a >= rec["min_certifiable_alpha"]),
            }

        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        c5 = rec["alphas"]["0.05"]
        cs = "cross" if cfg["cal_size"] is None else f"{cfg['cal_size']:.1f}"
        print(f"[{i}/{len(configs)}] F={cfg['n_frauds']:<4} cal_size={cs:<5} s={cfg['seed']} "
              f"cal_frauds={n_cal_fraud:<4} cov@.05={c5['coverage_fraud']:.3f} "
              f"width={c5['set_size']:.3f} {'' if c5['feasible'] else '(INFEASIBLE)'} "
              f"({rec['seconds']}s)", flush=True)

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
