#!/usr/bin/env python3
"""Settle P5 from the fair-hardware run.

Reads ``results/kaggle_wallclock.json``, written by
``experiments/kaggle/wallclock.py`` on a machine where both models share one
accelerator. Prints the comparison E4 could not make, and says so plainly if
the run was done on CPU, where it settles nothing.
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "kaggle_wallclock.json"


def main() -> int:
    if not RESULTS.exists():
        raise SystemExit(
            f"No {RESULTS.relative_to(REPO)}.\n"
            "Run experiments/kaggle/wallclock.py on a GPU (see "
            "experiments/kaggle/README.md) and drop the JSON in results/."
        )
    blob = json.loads(RESULTS.read_text())
    dev, rows = blob["device"], blob["rows"]
    where = dev.get("gpu") or dev.get("device", "unknown")
    print(f"{len(rows)} rows, both models on {where}\n")
    if dev.get("device") != "cuda":
        print("This run was on CPU. It does not settle P5: the point is to put "
              "both models on the same accelerator.\n")

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell = agg[(r["family"], r["strategy"], r["n_frauds"])]
        for k in ("fit_seconds", "predict_seconds", "seconds"):
            cell[k].append(r[k])
        cell["n_grad_fits"].append(r["n_grad_fits"])
        cell["seed"].append(r["seed"])

    print("| family | strategy | budget | fit | predict | total | gradient fits |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for key in sorted(agg):
        c = agg[key]
        print(f"| {key[0]} | {key[1]} | {key[2]} | {np.mean(c['fit_seconds']):.1f} s "
              f"| {np.mean(c['predict_seconds']):.1f} s | {np.mean(c['seconds']):.1f} s "
              f"| {int(np.mean(c['n_grad_fits']))} |")

    print("\n### P5: does LightGBM cross-conformal cost more wall-clock than TabPFN's?\n")
    for strategy in ("split", "cross"):
        for budget in sorted({k[2] for k in agg}):
            t = agg.get(("tabpfn", strategy, budget))
            g = agg.get(("lightgbm", strategy, budget))
            if not (t and g):
                continue
            # The seeds are shared, so pair them rather than compare means.
            seeds = sorted(set(t["seed"]) & set(g["seed"]))
            if len(seeds) < 2:
                continue
            tt = {s: v for s, v in zip(t["seed"], t["seconds"])}
            gg = {s: v for s, v in zip(g["seed"], g["seconds"])}
            d = np.array([gg[s] - tt[s] for s in seeds], dtype=float)
            se = float(d.std(ddof=1) / np.sqrt(d.size))
            verdict = ("LightGBM slower" if d.mean() > 2 * se else
                       "TabPFN slower" if -d.mean() > 2 * se else
                       "no separation")
            print(f"- **{strategy}, {budget} frauds**: LightGBM minus TabPFN "
                  f"{d.mean():+.1f} ± {se:.1f} s over {len(seeds)} paired seeds "
                  f"→ **{verdict}**.")

    print("\nGradient fits are the hardware-independent count and do not change: "
          "0 for TabPFN at either strategy, 1 for LightGBM split and 6 for "
          "LightGBM cross at K=5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
