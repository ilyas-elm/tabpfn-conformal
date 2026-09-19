"""E4 -- baselines, and the cost of the guarantee.

Two questions, deliberately kept apart because they are not the same question.

**Is the guarantee worth anything?** TabPFN's own imbalance handling
(``balance_probabilities``) and the tuned-threshold approach that arXiv:2605.21742
found works best for prior-data fitted networks both produce good recall. Neither
produces a *guarantee*: nothing tells you the fraud catch rate will hold on next
month's traffic. Conformal arms are compared against them on recall, and the
difference in what you can promise is the point.

**What does cross-conformal actually cost?** For LightGBM, K-fold conformal means
K gradient-boosted training runs. For TabPFN it means K forward passes and no
training at all. The cost table records, per arm: gradient-trained fits,
wall-clock, and API tokens.

⚠ **The wall-clock column is confounded and is reported as such.** TabPFN runs
against a remote API (network latency, queueing) while LightGBM runs on this
laptop's CPU. That is not a like-for-like comparison and must not be presented as
one. The honest number is the *gradient fits* column, which is hardware-independent.
A fair wall-clock comparison needs both models on identical hardware -- tier T2 in
docs/CAHIER-DES-CHARGES.md section 8, a Kaggle GPU running local TabPFN weights.

    python experiments/api/e4_baselines.py --dry-run
    python experiments/api/e4_baselines.py --pilot
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
    REPO, load_frames, load_token, make_eval, make_pool, resume_keys, save_proba,
    time_limit,
)

OUT = REPO / "results" / "e4.jsonl"

FRAUD_BUDGETS = (100, 200)
SEEDS = (0, 1, 2)
ALPHAS = (0.01, 0.05, 0.10, 0.20)
N_FOLDS = 5
BASE_RATE = 0.011

# (arm, model family, conformal strategy or None)
ARMS = (
    ("tabpfn_cross",   "tabpfn",   "cross"),
    ("tabpfn_split",   "tabpfn",   "split"),
    ("lightgbm_cross", "lightgbm", "cross"),
    ("lightgbm_split", "lightgbm", "split"),
    ("tabpfn_balanced_threshold", "tabpfn_balanced", None),
    ("tabpfn_tuned_threshold",    "tabpfn",          None),
)


def build(family: str):
    """A fresh unfitted estimator for the given family."""
    if family.startswith("tabpfn"):
        from tabpfn_client import TabPFNClassifier
        return TabPFNClassifier(balance_probabilities=family.endswith("balanced"))
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, n_jobs=-1)


def to_numeric(X):
    """LightGBM needs categoricals encoded; TabPFN takes the raw frame.

    That asymmetry is itself a result worth recording: the five BAF string
    columns go into TabPFN untouched and need explicit handling for LightGBM.
    """
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    return X


def tuned_threshold(proba_cal, y_cal, target_recall: float = 0.90) -> float:
    """Lowest threshold reaching ``target_recall`` on held-out data.

    The approach arXiv:2605.21742 found strongest for PFNs. It gives no coverage
    guarantee: the recall is an in-sample estimate on the tuning split.
    """
    pos = proba_cal[y_cal == 1, 1]
    return float(np.quantile(pos, 1 - target_recall)) if len(pos) else 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    budgets = (100,) if args.pilot else FRAUD_BUDGETS
    seeds = SEEDS[:1] if args.pilot else SEEDS
    configs = [
        {"arm": a, "family": f, "strategy": s, "n_frauds": b, "seed": d}
        for b in budgets for d in seeds for (a, f, s) in ARMS
    ]

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2

    from tabpfn_client import estimate_cost
    from tabpfn_conformal import (
        ConformalClassifier, average_set_size, coverage_by_class,
    )

    n_eval = 5_000
    total = 0
    for c in configs:
        if not c["family"].startswith("tabpfn"):
            continue
        n_pool = int(round(c["n_frauds"] / BASE_RATE))
        calls = ([(n_pool - n_pool // N_FOLDS, n_pool // N_FOLDS)] * N_FOLDS
                 if c["strategy"] == "cross" else [(n_pool // 2, n_pool // 2)])
        calls.append((n_pool, n_eval))
        for ctx, scored in calls:
            total += estimate_cost(np.zeros((max(ctx, 1), 30)),
                                   np.zeros((max(scored, 1), 30))).estimated_cost
    print(f"{len(configs)} configurations ({len(ARMS)} arms)")
    print(f"estimated cost: {total:,} tokens (~{total / 20_000_000:.1%} of the monthly budget)")
    if args.dry_run:
        return 0

    pool_df, eval_df = load_frames()
    skip = resume_keys(OUT, lambda r: f"{r['arm']}|{r['n_frauds']}|{r['seed']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        key = f"{cfg['arm']}|{cfg['n_frauds']}|{cfg['seed']}"
        if key in skip:
            continue
        X_pool, y_pool = make_pool(pool_df, cfg["n_frauds"], cfg["seed"])
        if X_pool is None:
            continue
        X_eval, y_eval = make_eval(eval_df, cfg["seed"])
        if cfg["family"] == "lightgbm":
            X_pool, X_eval = to_numeric(X_pool), to_numeric(X_eval)

        # Gradient-trained fits: the hardware-independent cost. TabPFN is 0 by
        # construction -- fit swaps a context and takes no gradient step.
        n_grad = 0
        if cfg["family"] == "lightgbm":
            n_grad = N_FOLDS + 1 if cfg["strategy"] == "cross" else 1

        t0 = time.perf_counter()
        try:
            with time_limit(args.timeout):
                if cfg["strategy"] is None:
                    # No conformal: fit, tune a threshold on held-out data, apply it.
                    from sklearn.model_selection import train_test_split
                    tr, ca = train_test_split(
                        np.arange(len(y_pool)), test_size=0.5,
                        stratify=y_pool, random_state=cfg["seed"],
                    )
                    est = build(cfg["family"])
                    est.fit(X_pool.iloc[tr], y_pool[tr])
                    thr = tuned_threshold(est.predict_proba(X_pool.iloc[ca]), y_pool[ca])
                    proba = est.predict_proba(X_eval)
                    flagged = proba[:, 1] >= thr
                    rec = {
                        **cfg, "n_grad_fits": n_grad,
                        "recall": float((flagged & (y_eval == 1)).sum() / max((y_eval == 1).sum(), 1)),
                        "flag_rate": float(flagged.mean()),
                        "threshold": thr,
                        "has_guarantee": False,
                        "alphas": {},
                    }
                else:
                    cc = ConformalClassifier(
                        build(cfg["family"]), method="mondrian",
                        strategy=cfg["strategy"], cal_size=0.5, n_folds=N_FOLDS,
                        random_state=cfg["seed"],
                    )
                    cc.fit(X_pool, y_pool)
                    proba = cc.predict_proba(X_eval)
                    rec = {
                        **cfg, "n_grad_fits": n_grad,
                        "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
                        "has_guarantee": True,
                        "proba_file": save_proba("e4", key, proba, y_eval),
                        "alphas": {},
                    }
                    for a in ALPHAS:
                        sets = cc.predict_set_from_proba(proba, a)
                        cov = coverage_by_class(sets, y_eval, cc.classes_)
                        rec["alphas"][str(a)] = {
                            "coverage_fraud": cov[1],
                            "coverage_legit": cov[0],
                            "set_size": average_set_size(sets),
                        }
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {key}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            continue

        rec["seconds"] = round(time.perf_counter() - t0, 1)
        # Flagged as confounded: TabPFN is remote, LightGBM is local CPU.
        rec["wallclock_comparable"] = False
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        summary = (f"recall={rec['recall']:.3f}" if not rec["has_guarantee"]
                   else f"cov@.05={rec['alphas']['0.05']['coverage_fraud']:.3f} "
                        f"width={rec['alphas']['0.05']['set_size']:.3f}")
        print(f"[{i}/{len(configs)}] {cfg['arm']:<26} F={cfg['n_frauds']:<4} s={cfg['seed']} "
              f"grad_fits={rec['n_grad_fits']:<2} {summary} ({rec['seconds']}s)", flush=True)

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume", file=sys.stderr)
        sys.exit(130)
