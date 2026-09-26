#!/usr/bin/env python3
"""Settle P5 from the fair-hardware run.

Reads ``results/kaggle_wallclock.json``, written by
``experiments/kaggle/wallclock.py`` on a machine where both models share one
accelerator. Prints the comparison E4 could not make, and says so plainly if
the run was done on CPU, where it settles nothing.
"""
from __future__ import annotations

import json
import math
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

    # The grid happens to contain the matched comparison as well: split at a
    # budget of 2F and cross at F calibrate on the same number of positives, so
    # the headline claim can be re-checked here on local weights, which is a
    # third setting after the API runs and Forest Cover Type.
    def cell(fam, strat, budget):
        return {r["seed"]: r for r in rows
                if r["family"] == fam and r["strategy"] == strat
                and r["n_frauds"] == budget}

    print("\n### Matched on calibration size: does the headline hold on local weights?\n")
    for a in ("0.05", "0.1"):
        sp, cr = cell("tabpfn", "split", 200), cell("tabpfn", "cross", 100)
        seeds = sorted(set(sp) & set(cr))
        if len(seeds) < 2 or sp[seeds[0]]["n_cal_fraud"] != cr[seeds[0]]["n_cal_fraud"]:
            continue
        d = np.array([cr[s]["alphas"][a]["set_size"] - sp[s]["alphas"][a]["set_size"]
                      for s in seeds])
        se = float(d.std(ddof=1) / np.sqrt(d.size))
        verdict = ("cross narrower" if d.mean() < -2 * se else
                   "cross wider" if d.mean() > 2 * se else "tie")
        print(f"- **alpha={a}**, both calibrating on {sp[seeds[0]]['n_cal_fraud']} "
              f"positives, split spending 200 confirmed frauds and cross 100: "
              f"cross minus split {d.mean():+.4f} ± {se:.4f} over {len(seeds)} "
              f"paired seeds → **{verdict}**.")

    print("\n### Realized coverage against the level each run certifies\n")
    print("Certified is `ceil((n_cal+1)(1-alpha))/n_cal`, not the nominal "
          "`1-alpha`, because the index rounds up. A cell below its own "
          "certified level is the price of approximate validity.\n")
    below = defaultdict(int)
    for fam in ("tabpfn", "lightgbm"):
        for strat in ("split", "cross"):
            for budget in sorted({r["n_frauds"] for r in rows}):
                g = cell(fam, strat, budget)
                if not g:
                    continue
                n_cal = next(iter(g.values()))["n_cal_fraud"]
                for a in ("0.05", "0.1"):
                    tgt = min(1.0, math.ceil((n_cal + 1) * (1 - float(a))) / n_cal)
                    got = float(np.mean([g[s]["alphas"][a]["coverage_fraud"]
                                         for s in sorted(g)]))
                    # A hair under target is sampling, not a broken promise;
                    # count only gaps a reader would call a shortfall.
                    if got - tgt < -0.005:
                        below[(fam, strat)] += 1
                    print(f"| {fam} | {strat} | {budget} | {a} | {n_cal} | "
                          f"{tgt:.4f} | {got:.4f} | {got - tgt:+.4f} |"
                          if False else
                          f"  {fam:9} {strat:5} F={budget:<4} a={a:<5} "
                          f"n_cal={n_cal:<4} certifies {tgt:.4f}  "
                          f"realized {got:.4f}  {got - tgt:+.4f}"
                          + ("  BELOW" if got - tgt < -0.005 else ""))
    print()
    for k in sorted(below):
        print(f"  {k[0]} {k[1]}: below its certified level in {below[k]} of 4 cells")
    if not below:
        print("  nothing below its certified level by more than half a point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
