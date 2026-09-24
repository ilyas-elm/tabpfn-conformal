#!/usr/bin/env python3
"""P5, on hardware where the comparison is fair.

Every wall-clock number in E4 is tagged ``wallclock_comparable: false``, because
TabPFN ran remotely on Prior Labs' GPUs while LightGBM ran on a laptop CPU. The
TabPFN figure there is dominated by upload and round-trip, not by inference, so
the two are not a race; see ``docs/limitations.md``.

This settles it: **both models, one machine, one accelerator.** It runs the E4
protocol unchanged, matched label budgets, the same pool and evaluation
construction from ``experiments/api/_common.py``, but against *local* TabPFN
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
point, it removes the network, which is the confound. It does not reproduce the
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


def find_base(data_dir: pathlib.Path) -> pathlib.Path:
    """Locate ``Base.csv``, looking past whatever Kaggle called the folder.

    A Kaggle input directory is named after the dataset slug, which is not
    something this script can know, and some datasets nest their files a level
    down. Guessing wrong used to fail *after* the warm-up, so the search costs
    nothing and removes a whole class of "add the dataset and try again".
    """
    if (data_dir / "Base.csv").exists():
        return data_dir / "Base.csv"
    for root in (data_dir, pathlib.Path("/kaggle/input")):
        if root.is_dir():
            found = sorted(root.rglob("Base.csv"))
            if found:
                print(f"note: --data did not hold Base.csv; using {found[0].parent}")
                return found[0]
    attached = ([p.name for p in pathlib.Path("/kaggle/input").glob("*")]
                if pathlib.Path("/kaggle/input").is_dir() else [])
    raise SystemExit(
        f"No Base.csv under {data_dir} or /kaggle/input.\n"
        f"Inputs currently attached: {attached or 'none'}\n"
        "On Kaggle: sidebar, + Add Input, search 'Bank Account Fraud Dataset "
        "NeurIPS 2022', Add. Nothing was measured and no GPU time was spent."
    )


def load_frames(data_dir: pathlib.Path):
    path = find_base(data_dir)
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
    """LightGBM needs categoricals encoded; TabPFN takes the frame as it is.

    Tested against what the column *is*, not against ``dtype == object``: under
    pandas 3 a text column comes back as ``str`` rather than ``object``, so the
    old test silently encoded nothing and LightGBM rejected the frame. Which
    pandas a Kaggle image ships is not something this script gets to choose.
    """
    X = X.copy()
    for col in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])):
            X[col] = X[col].astype("category").cat.codes
    return X


def build(family: str, device: str):
    if family == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=device)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)


def warm_up(device: str, families=("tabpfn", "lightgbm")) -> None:
    """Fit both families once on throwaway data, before anything is timed.

    TabPFN does not load its weights until ``fit``, and on a fresh machine that
    first call also *downloads* them. Timing it would have charged TabPFN a few
    hundred megabytes of network for the one measurement whose entire purpose is
    to have no network in it. LightGBM is warmed too, so neither family pays a
    first-call import cost the other does not.

    It fails early and loudly as a side effect: TabPFN needs a one-time licence
    acceptance before it will fetch weights, and a notebook is not an
    interactive terminal, so without ``TABPFN_TOKEN`` this raises here, in the
    first seconds, rather than part-way through a GPU session.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(32, 4)), columns=list("abcd"))
    # BAF has five string columns, which go into TabPFN untouched and are
    # encoded for LightGBM. A purely numeric warm-up would prove nothing about
    # the frame this actually runs on, so carry one text column here too.
    X["text"] = ["alpha", "beta"] * 16
    y = np.array([0, 1] * 16)
    for family in families:
        t0 = time.perf_counter()
        Xw = to_numeric(X) if family == "lightgbm" else X
        try:
            build(family, device).fit(Xw, y)
        except Exception as exc:                                  # noqa: BLE001
            if family != "tabpfn":
                raise
            blurb = f"{type(exc).__name__}: {exc}".lower()
            if "licen" in blurb or "tabpfn_token" in blurb or "api key" in blurb:
                raise SystemExit(
                    f"TabPFN could not load its weights: {exc}\n\n"
                    "Local inference needs a one-time licence acceptance at "
                    "https://ux.priorlabs.ai (Licenses tab), and then the API key "
                    "from https://ux.priorlabs.ai/account in TABPFN_TOKEN. On "
                    "Kaggle put it in Add-ons -> Secrets, never in a cell.\n"
                    "The token only authorises the download; inference stays local, "
                    "which is what keeps this measurement clean."
                ) from exc
            raise SystemExit(
                f"TabPFN failed on the warm-up frame, before any timing: {exc}\n\n"
                "This is not a licence problem. The warm-up frame is four numeric "
                "columns and one text column, which is the shape BAF has, so "
                "whatever this is would have hit the real run too. Nothing was "
                "measured and no GPU time was spent."
            ) from exc
        print(f"warm-up: {family} ready in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path,
                    default=pathlib.Path("/kaggle/input/bank-account-fraud-dataset-neurips-2022"))
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    # If a session dies part-way, the JSON already holds the finished rows and
    # this lets you rerun only what is missing instead of the whole grid.
    ap.add_argument("--families", nargs="+", default=["tabpfn", "lightgbm"],
                    choices=["tabpfn", "lightgbm"])
    args = ap.parse_args()

    dev = device_report()
    print(f"device: {dev['device']}" + (f" ({dev['gpu']})" if dev["gpu"] else ""))
    if dev["device"] != "cuda":
        print("WARNING: no GPU. The whole point of this script is to put both "
              "models on the same accelerator; on CPU it settles nothing.",
              file=sys.stderr)

    warm_up(dev["device"], args.families)

    pool_df, eval_df = load_frames(args.data)
    out, rows = REPO / "results" / "kaggle_wallclock.json", []

    for budget in args.budgets:
        for seed in args.seeds:
            X_pool, y_pool = make_pool(pool_df, budget, seed)
            if X_pool is None:
                continue
            X_eval, y_eval = make_eval(eval_df, seed)

            for family in args.families:
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
                    out.write_text(json.dumps(
                        {"device": dev, "warmed_up": True, "rows": rows}, indent=2))

    print(f"\nWritten to {out.relative_to(REPO)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
