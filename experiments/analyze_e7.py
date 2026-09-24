#!/usr/bin/env python3
"""E7, the matched comparison on a second domain.

Same analysis as E1, on Forest Cover Type instead of Bank Account Fraud. The
question is narrow: does *the halving* survive a dataset with no shared column,
no fraud, and no temporal structure?

Reads `results/e7.jsonl`. No API key needed.
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e7.jsonl"


def load(alpha: str):
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E7 first.")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    # Keyed by seed, never by list position: E2 was resumed once and two of its
    # settings ended up in a rotated order, which silently unpaired the test.
    cells = defaultdict(lambda: defaultdict(dict))
    ncal = {}
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        k = (r["strategy"], r["n_frauds"])
        cells[k]["width"][r["seed"]] = a["set_size"]
        cells[k]["coverage"][r["seed"]] = a["coverage_fraud"]
        ncal[k] = r["n_cal_fraud"]
    return cells, ncal, len(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.1")
    args = ap.parse_args()
    cells, ncal, n = load(args.alpha)
    alpha = float(args.alpha)
    print(f"{n} result rows, Forest Cover Type (cover type 4 vs rest, 0.473% positive)\n")

    print(f"### Matched on calibration size (alpha = {alpha:g})\n")
    print("| calib. positives | targeted level | split needs | its set size | "
          "cross needs | its set size | labels saved |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    by_ncal = defaultdict(dict)
    for (strategy, budget), c in cells.items():
        by_ncal[ncal[(strategy, budget)]][strategy] = (budget, c)

    paired = []
    for k in sorted(by_ncal):
        pair = by_ncal[k]
        if "split" not in pair or "cross" not in pair:
            continue
        (fs, cs), (fc, cc) = pair["split"], pair["cross"]
        seeds = sorted(set(cs["width"]) & set(cc["width"]))
        if not seeds:
            continue
        ws = float(np.mean([cs["width"][s] for s in seeds]))
        wc = float(np.mean([cc["width"][s] for s in seeds]))
        idx = math.ceil((k + 1) * (1 - alpha))
        lvl = "not certifiable" if idx > k else f"{100 * idx / k:.1f}%"
        delta = 100 * (ws - wc) / ws
        note = f"narrower by {delta:.1f}%" if delta > 0 else f"wider by {-delta:.1f}%"
        print(f"| {k} | {lvl} | {fs} | {ws:.3f} | **{fc}** | **{wc:.3f}** ({note}) "
              f"| **{fs - fc} ({100 * (fs - fc) // fs}%)** |")
        paired.append((k, np.array([cs["width"][s] - cc["width"][s] for s in seeds]), seeds))

    print(f"\n### Paired across seeds (same seed, split at 2F vs cross at F)\n")
    print("| calib. positives | paired difference | verdict |")
    print("|---:|---:|---|")
    crit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
    wider = narrower = 0
    for k, d, seeds in paired:
        if len(d) < 2:
            print(f"| {k} | only {len(d)} seed | no test |")
            continue
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = d.mean() / se if se > 0 else np.inf
        c = crit.get(len(d), 2.0)
        if abs(t) < c:
            verdict = "tie (within noise)"
        elif d.mean() > 0:
            verdict, _ = "**cross narrower**", narrower
            narrower += 1
        else:
            verdict = "**cross wider**"
            wider += 1
        print(f"| {k} | {d.mean():+.4f} ± {se:.4f} (n={len(d)}) | {verdict} |")

    print(f"\n**Cross-conformal is significantly wider in {wider} of {len(paired)} "
          f"comparisons on this dataset**, and significantly narrower in {narrower}.")
    print("\nThe halving is structural, it follows from where the calibration set "
          "comes from, not from the data, so it transfers by construction. What "
          "this tests is whether it costs anything in width on a domain the "
          "method was not tuned on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
