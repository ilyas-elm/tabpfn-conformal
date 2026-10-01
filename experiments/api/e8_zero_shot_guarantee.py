#!/usr/bin/env python3
"""E8, what a guarantee costs on a model that needs no labels to predict.

Every arm so far learns from labelled frauds: TabPFN takes them in context,
LightGBM trains on them. Laya does neither. It is a 421M-parameter
non-autoregressive **System One** model (Apache-2.0, `pip install laya`) that
takes a state and a typed question and returns a probability in one forward
pass, **zero-shot**: no labelled fraud is required to make a prediction at all.

That makes it the sharpest possible test of this project's claim. If confirmed
positives were only needed to *fit* a model, Laya would need none. It still
needs them, because a coverage guarantee is calibrated from labels whatever
produced the probability. **The labels are not for training. They are for the
guarantee.**

So this measures, on identical rows and through the same conformal object:

* whether conformal gives a zero-shot model valid coverage at all;
* what the guarantee costs each model, in prediction-set width and in the
  fraction of cases that need a human;
* where the boundary in `docs/limitations.md` actually sits, the one that says
  a predictor returning every label has perfect coverage and no value. That is
  asserted there and measured here.

**It costs nothing to run and needs no credentials.** The TabPFN and LightGBM
probabilities are already committed under `results/proba/e4/`, and
`make_eval(ev, seed)` reconstructs exactly the rows they were computed on, which
this asserts rather than assumes. Laya runs locally on CPU at about 96 ms a row.

Design. The evaluation set is split in half, stratified: one half calibrates,
the other is scored. Both halves are random halves of one draw, so they are
exchangeable, which is what the finite-sample guarantee requires. Every arm
sees the identical halves and the identical conformal code, so the only thing
that varies is which model produced the probabilities.

A footnote on Laya's calibration, checked rather than repeated: the checkpoint
ships temperatures [1.637, 1.251, 1.983] for choice, score and noul, and warns
at load that one *other* entry, `choice:11+`, is outside the valid range. The
noul head used here is not affected. Its calibration claim is not the thing
being tested; conformal holds whether or not the claim is true, which is the
point.

    python experiments/api/e8_zero_shot_guarantee.py --dry-run
    python experiments/api/e8_zero_shot_guarantee.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import REPO, load_frames, make_eval  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from tabpfn_conformal import (  # noqa: E402
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_thresholds,
    mondrian_thresholds,
    one_minus_prob,
)

SEEDS = (0, 1, 2)
ALPHAS = (0.05, 0.1, 0.2)
# Committed arms to situate the zero-shot model against, by npz stem.
BASELINES = ("tabpfn_cross_200", "tabpfn_split_200", "lightgbm_cross_200")
OUT = REPO / "results" / "e8.jsonl"
CACHE = REPO / "results" / "proba" / "e8"
PROBA = REPO / "results" / "proba" / "e4"

QUESTION = {
    "type": "noul",
    "instructions": "Is this bank account application fraudulent?",
    "criteria": {"true": "The application is fraudulent.",
                 "false": "The application is legitimate."},
}


def halves(n: int, y: np.ndarray, seed: int):
    """Two stratified halves of the evaluation set: calibrate on one, score the other."""
    rng = np.random.RandomState(seed)
    cal = np.zeros(n, dtype=bool)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        cal[rng.permutation(idx)[: len(idx) // 2]] = True
    return cal, ~cal


def laya_probabilities(X, y, seed: int, limit: int | None):
    """Zero-shot, local, one forward pass per row. Cached so a rerun is free."""
    try:
        from laya import Router
    except ImportError:
        sys.exit('laya is not installed.  pip install laya')

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"laya_eval_{seed}.json"
    done = json.loads(path.read_text()) if path.exists() else {}

    rows = json.loads(X.to_json(orient="records"))
    ids = [str(i) for i in X.index]
    todo = [i for i, k in enumerate(ids) if k not in done]
    if limit is not None:
        todo = todo[:limit]
    if todo:
        router = Router()
        t0 = time.time()
        for n, i in enumerate(todo, 1):
            r = router.predict(json.dumps(rows[i]), {"fraud": QUESTION})
            done[ids[i]] = float(r["answers"]["fraud"]["noul"])
            if n % 200 == 0 or n == len(todo):
                path.write_text(json.dumps(done))
                rate = n / max(time.time() - t0, 1e-9)
                print(f"    laya seed {seed}: {n}/{len(todo)}  "
                      f"{rate:.1f} rows/s, {(len(todo)-n)/max(rate,1e-9)/60:.1f} min left",
                      flush=True)
        path.write_text(json.dumps(done))
    have = np.array([k in done for k in ids])
    p1 = np.array([done.get(k, np.nan) for k in ids], dtype=float)
    return np.column_stack([1.0 - p1, p1]), have


def measure(arm, seed, proba, y, cal_mask, te_mask):
    """Split conformal from probabilities alone: marginal and class-conditional."""
    scores = one_minus_prob(proba)
    y_cal, y_te = y[cal_mask], y[te_mask]
    true_cal = scores[cal_mask][np.arange(cal_mask.sum()), y_cal]
    te_scores = scores[te_mask]
    n_fraud = int((y_cal == 1).sum())
    out = []
    for alpha in ALPHAS:
        for method in ("marginal", "mondrian"):
            if method == "marginal":
                q = np.full(2, marginal_thresholds(true_cal, alpha)[None])
            else:
                t = mondrian_thresholds(true_cal, y_cal, 2, alpha)
                q = np.array([t[0], t[1]], dtype=float)
            sets = te_scores <= q[None, :]
            cov = coverage_by_class(sets, y_te, np.array([0, 1]))
            out.append({
                "arm": arm, "seed": int(seed), "alpha": alpha, "method": method,
                "n_cal": int(cal_mask.sum()), "n_cal_fraud": n_fraud,
                "n_test": int(te_mask.sum()),
                "coverage_fraud": float(cov[1]), "coverage_legit": float(cov[0]),
                "set_size": float(average_set_size(sets)),
                "empty_rate": float(empty_set_rate(sets)),
                # the quantity a desk budgets for: both labels still in play
                "review_rate": float(np.mean(sets.sum(axis=1) == 2)),
                "mean_predicted_fraud": float(np.nanmean(proba[:, 1])),
                "observed_fraud_rate": float((y == 1).mean()),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most this many new Laya rows per seed")
    args = ap.parse_args()

    _, ev = load_frames()
    X, y = make_eval(ev, args.seeds[0])
    y = np.asarray(y)
    cal, te = halves(len(y), y, args.seeds[0])
    print(f"evaluation set {len(y)} rows, {int(y.sum())} fraud")
    print(f"  calibrate on {cal.sum()} ({int(y[cal].sum())} fraud), "
          f"score {te.sum()} ({int(y[te].sum())} fraud)")
    floor = 1.0 / (int(y[cal].sum()) + 1)
    print(f"  smallest certifiable alpha on the fraud class: {floor:.5f}")
    print(f"\nlaya: {len(y) * len(args.seeds):,} local forward passes, no API, no key")
    print("baselines: already committed under results/proba/e4/, nothing to spend")
    if args.dry_run:
        print("dry run.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        X, y = make_eval(ev, seed)
        y = np.asarray(y)
        cal, te = halves(len(y), y, seed)
        print(f"\nseed {seed}")

        for stem in BASELINES:
            f = PROBA / f"{stem}_{seed}.npz"
            if not f.exists():
                print(f"  {stem}: no committed probabilities, skipped")
                continue
            d = np.load(f)
            # The committed arms must be on these exact rows, or the comparison
            # is between different test sets wearing the same name.
            assert len(d["y_true"]) == len(y) and (d["y_true"] == y).all(), \
                f"{f.name} does not line up with make_eval(seed={seed})"
            records += measure(stem, seed, d["proba"], y, cal, te)
            print(f"  {stem}: reused")

        proba, have = laya_probabilities(X, y, seed, args.limit)
        if have.all():
            records += measure("laya_zero_shot", seed, proba, y, cal, te)
            print("  laya_zero_shot: scored")
        else:
            print(f"  laya_zero_shot: {have.sum()}/{len(have)} rows scored, "
                  "rerun to finish before it is measured")

    if records:
        with OUT.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(records)} rows to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
