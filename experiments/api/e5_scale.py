"""E5 -- fixed confirmed frauds, growing legitimate context.

Every experiment before this one shrank the labelled pool to preserve the 1.1%
base rate, so a budget of 200 confirmed frauds meant a context of 18,000 rows.
That is not the situation a fraud desk is in. A real desk has *millions* of
transactions and a few hundred confirmed frauds: negatives are abundant,
positives are not.

So the faithful question, and one nobody appears to have measured: **hold the
confirmed frauds fixed and pour in legitimate data. Does the guarantee get
cheaper?** For a model that trains, each point on that curve is another training
run over a larger dataset. For TabPFN it is a longer context and one forward
pass, which is why the sweep is affordable at all.

The scale also finally makes the KV cache visible. Below ~100k context rows
every call sits on the API's 10,000-token minimum and the documented 75% saving
cannot be seen; at 200k it is fully exposed. Conformal is the workload the cache
was built for -- one fixed context, scored twice (calibration, then evaluation) --
so `--cache` measures what that is worth.

    python experiments/api/e5_scale.py --dry-run
    python experiments/api/e5_scale.py                 # statistics, 3 seeds
    python experiments/api/e5_scale.py --cache-study   # cache on/off timing
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
    LABEL, POOL_MONTHS, REPO, BatchedPredictProba, load_frames, load_token,
    make_eval, resume_keys, save_proba, set_client_timeouts, split_xy, time_limit,
)

OUT = REPO / "results" / "e5.jsonl"

N_FRAUDS = 200                                    # held fixed throughout
CONTEXTS = (10_000, 25_000, 50_000, 100_000, 200_000)
CROSS_CONTEXTS = (10_000, 25_000)   # see the note in main(): wall-clock, not a result
SEEDS = (0, 1, 2)
ALPHAS = (0.01, 0.05, 0.10, 0.20)
N_FOLDS = 5


def build_pool(pool: pd.DataFrame, n_legit: int, seed: int):
    """Exactly N_FRAUDS confirmed frauds plus `n_legit` legitimate rows."""
    rng = np.random.default_rng(2000 + seed)
    pos, neg = pool[pool[LABEL] == 1], pool[pool[LABEL] == 0]
    if N_FRAUDS > len(pos) or n_legit > len(neg):
        return None, None
    take = pd.concat([
        pos.iloc[rng.choice(len(pos), N_FRAUDS, replace=False)],
        neg.iloc[rng.choice(len(neg), n_legit, replace=False)],
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_xy(take)


def key(c) -> str:
    return f"{c['strategy']}|{c['n_legit']}|{c['seed']}|{int(c['cache'])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-study", action="store_true",
                    help="one seed, cache on and off, for the timing comparison")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    if args.cache_study:
        configs = [{"strategy": "split", "n_legit": n, "seed": 0, "cache": c}
                   for n in (50_000, 100_000, 200_000) for c in (False, True)]
    else:
        # Cross-conformal refits K times, and each fold's server-side fit grows
        # with the context: 40s at a 10k context, ~25 minutes at 25k. At 100k and
        # 200k it would be hours per configuration, so cross is capped at the
        # contexts where it is affordable and split carries the scale curve.
        # A wall-clock limit, not a result -- said plainly in the writeup.
        configs = [{"strategy": "split", "n_legit": n, "seed": d, "cache": False}
                   for n in CONTEXTS for d in SEEDS]
        configs += [{"strategy": "cross", "n_legit": n, "seed": d, "cache": False}
                    for n in CROSS_CONTEXTS for d in SEEDS]

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2
    set_client_timeouts()

    from tabpfn_client import TabPFNClassifier, estimate_cost
    from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class

    n_eval = 5_878
    total = 0
    for c in configs:
        n = c["n_legit"] + N_FRAUDS
        op = "cache_predict" if c["cache"] else "predict"
        calls = ([(n - n // N_FOLDS, n // N_FOLDS)] * N_FOLDS
                 if c["strategy"] == "cross" else [(n // 2, n // 2)])
        calls.append((n, n_eval))
        for ctx, scored in calls:
            total += estimate_cost(np.zeros((ctx, 30)), np.zeros((scored, 30)),
                                   operation=op).estimated_cost
    print(f"{len(configs)} configurations, {N_FRAUDS} confirmed frauds held fixed")
    print(f"estimated cost: {total:,} tokens "
          f"({total / 5_000_000:.0%} of the 5M daily cap)")
    if args.dry_run:
        return 0

    pool_df, eval_df = load_frames()
    print(f"pool months {POOL_MONTHS}: {len(pool_df):,} rows available", flush=True)
    skip = resume_keys(OUT, key)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        if key(cfg) in skip:
            continue
        X_pool, y_pool = build_pool(pool_df, cfg["n_legit"], cfg["seed"])
        if X_pool is None:
            continue
        X_eval, y_eval = make_eval(eval_df, cfg["seed"])

        def model():
            # A cached fit caps predicts at 10,000 rows, so the cached arm must
            # batch. That is the workload the cache exists for, not a workaround.
            if cfg["cache"]:
                return BatchedPredictProba(TabPFNClassifier(fit_mode="fit_with_cache"))
            return TabPFNClassifier()

        cc = ConformalClassifier(
            model(), method="mondrian", strategy=cfg["strategy"],
            cal_size=0.5, n_folds=N_FOLDS, random_state=cfg["seed"],
        )
        t0 = time.perf_counter()
        try:
            with time_limit(args.timeout):
                cc.fit(X_pool, y_pool)
                t_fit = time.perf_counter() - t0
                t1 = time.perf_counter()
                proba = cc.predict_proba(X_eval)
                t_pred = time.perf_counter() - t1
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key(cfg)}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            continue

        rec = {
            **cfg,
            "n_frauds": N_FRAUDS,
            "n_context": int(len(y_pool)),
            "fraud_rate_context": float(y_pool.mean()),
            "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
            "fit_seconds": round(t_fit, 1),
            "predict_seconds": round(t_pred, 1),
            "seconds": round(time.perf_counter() - t0, 1),
            "proba_file": save_proba("e5", key(cfg), proba, y_eval),
            "alphas": {},
        }
        for a in ALPHAS:
            sets = cc.predict_set_from_proba(proba, a)
            cov = coverage_by_class(sets, y_eval, cc.classes_)
            rec["alphas"][str(a)] = {"coverage_fraud": cov[1], "coverage_legit": cov[0],
                                     "set_size": average_set_size(sets)}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        c5 = rec["alphas"]["0.05"]
        print(f"[{i}/{len(configs)}] {cfg['strategy']:<5} ctx={rec['n_context']:>7,} "
              f"({rec['fraud_rate_context']:.2%} fraud) s={cfg['seed']} "
              f"cache={'on ' if cfg['cache'] else 'off'} "
              f"cov@.05={c5['coverage_fraud']:.3f} width={c5['set_size']:.3f} "
              f"fit={rec['fit_seconds']}s pred={rec['predict_seconds']}s", flush=True)

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
