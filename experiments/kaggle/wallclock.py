#!/usr/bin/env python3
"""P5, on hardware where the comparison is fair.

Every wall-clock number in E4 is tagged ``wallclock_comparable: false``, because
TabPFN ran remotely on Prior Labs' GPUs while LightGBM ran on a laptop CPU. The
TabPFN figure there is dominated by upload and round-trip, not by inference, so
the two are not a race — see ``docs/limitations.md``.

This settles it: **both models, one machine, one accelerator.** It runs the E4
protocol unchanged — matched label budgets, the same pool and evaluation
construction from ``experiments/api/_common.py`` — but against *local* TabPFN
weights instead of the API, so there is no network in the measurement.

    # Kaggle: new notebook, Accelerator = GPU T4 x2, add the BAF dataset,
    #         then in one cell:
    !pip install -q tabpfn lightgbm
    !git clone -q https://github.com/ilyas-elm/tabpfn-conformal.git
    %cd tabpfn-conformal
    !pip install -q -e .
    !python experiments/kaggle/wallclock.py --data /kaggle/input/bank-account-fraud-dataset-neurips-2022

Writes ``results/kaggle_wallclock.json``. Download it, drop it into ``results/``
and run ``python experiments/analyze_kaggle.py``.

Honest about what it measures: *local* TabPFN, not the managed API. That is the
point — it removes the network, which is the confound. It does not reproduce the
API numbers and is not meant to.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "api"))
sys.path.insert(0, str(REPO / "src"))

from _common import BASE_RATE, EVAL_MONTHS, LABEL, POOL_MONTHS, TIME, split_xy  # noqa: E402

from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class  # noqa: E402

BUDGETS = (100, 200)
SEEDS = (0, 1, 2)
N_FOLDS = 5
ALPHAS = (0.05, 0.1)
EVAL_LEGIT = 3_000


def device_report() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "device": "cpu", "gpu": None}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu", "gpu": gpu}


def load_frames(data_dir: pathlib.Path):
    path = data_dir / "Base.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. On Kaggle, add the BAF dataset and pass --data.")
    df = pd.read_csv(path)
    return (df[df[TIME].isin(POOL_MONTHS)].reset_index(drop=True),
            df[df[TIME].isin(EVAL_MONTHS)].reset_index(drop=True))


def make_pool(pool: pd.DataFrame, n_frauds: int, seed: int):
    rng = np.random.default_rng(1000 + seed)
    pos, neg = pool[pool[LABEL] == 1], pool[pool[LABEL] == 0]
    n_neg = int(round(n_frauds / BASE_RATE)) - n_frauds
    if n_frauds > len(pos) or n_neg > len(neg):
        return None, None
    take = pd.concat([pos.iloc[rng.choice(len(pos), n_frauds, replace=False)],
                      neg.iloc[rng.choice(len(neg), n_neg, replace=False)]])
    return split_xy(take.sample(frac=1.0, random_state=seed).reset_index(drop=True))


def make_eval(ev: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    pos, neg = ev[ev[LABEL] == 1], ev[ev[LABEL] == 0]
    neg = neg.iloc[rng.choice(len(neg), min(EVAL_LEGIT, len(neg)), replace=False)]
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_xy(out)


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM needs categoricals encoded; TabPFN takes the frame as it is."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X


def build(family: str, device: str):
    if family == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=device)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path,
                    default=pathlib.Path("/kaggle/input/bank-account-fraud-dataset-neurips-2022"))
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = device_report()
    print(f"device: {dev['device']}" + (f" ({dev['gpu']})" if dev["gpu"] else ""))
    if dev["device"] != "cuda":
        print("WARNING: no GPU. The whole point of this script is to put both "
              "models on the same accelerator; on CPU it settles nothing.",
              file=sys.stderr)

    pool_df, eval_df = load_frames(args.data)
    out, rows = REPO / "results" / "kaggle_wallclock.json", []

    for budget in args.budgets:
        for seed in args.seeds:
            X_pool, y_pool = make_pool(pool_df, budget, seed)
            if X_pool is None:
                continue
            X_eval, y_eval = make_eval(eval_df, seed)

            for family in ("tabpfn", "lightgbm"):
                Xp = to_numeric(X_pool) if family == "lightgbm" else X_pool
                Xe = to_numeric(X_eval) if family == "lightgbm" else X_eval
                for strategy in ("split", "cross"):
                    n_grad = 0 if family == "tabpfn" else (
                        N_FOLDS + 1 if strategy == "cross" else 1)
                    cc = ConformalClassifier(
                        build(family, dev["device"]), method="mondrian",
                        strategy=strategy, cal_size=0.5, n_folds=N_FOLDS,
                        random_state=seed,
                    )
                    t0 = time.perf_counter()
                    cc.fit(Xp, y_pool)
                    t_fit = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    proba = cc.predict_proba(Xe)
                    t_pred = time.perf_counter() - t0

                    rec = {
                        "family": family, "strategy": strategy, "n_frauds": budget,
                        "seed": seed, "n_pool": int(len(y_pool)),
                        "n_cal_fraud": int(cc.n_calibration_.get(1, 0)),
                        "n_grad_fits": n_grad,
                        "fit_seconds": round(t_fit, 2),
                        "predict_seconds": round(t_pred, 2),
                        "seconds": round(t_fit + t_pred, 2),
                        # The whole reason this file exists.
                        "wallclock_comparable": True,
                        "device": dev["device"], "gpu": dev["gpu"],
                        "alphas": {},
                    }
                    for a in ALPHAS:
                        sets = cc.predict_set_from_proba(proba, a)
                        cov = coverage_by_class(sets, y_eval, cc.classes_)
                        rec["alphas"][str(a)] = {
                            "coverage_fraud": cov[1], "coverage_legit": cov[0],
                            "set_size": average_set_size(sets),
                        }
                    rows.append(rec)
                    print(f"  {family:9} {strategy:5} F={budget:<4} s={seed} "
                          f"fit={t_fit:6.1f}s pred={t_pred:5.1f}s "
                          f"grad_fits={n_grad}", flush=True)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps({"device": dev, "rows": rows}, indent=2))

    print(f"\nWritten to {out.relative_to(REPO)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
