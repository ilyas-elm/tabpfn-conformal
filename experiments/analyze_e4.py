"""Turn results/e4.jsonl into the baseline and cost tables.

Three separate questions, kept separate because conflating them is how baseline
comparisons mislead:

1. **Model quality at a matched guarantee.** TabPFN and LightGBM with the same
   conformal strategy and the same budget calibrate on the same number of
   positives, so they target an identical level. Set size is then a clean
   comparison of the underlying model.
2. **What the guarantee costs.** Gradient-trained fits is the hardware-neutral
   number. Wall-clock is reported but flagged, because TabPFN runs remotely and
   LightGBM runs on local CPU.
3. **Guarantee versus no guarantee.** The threshold arms produce recall. They
   cannot produce a promise that the recall holds next month.

Reads only the committed results file: no API key needed.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e4.jsonl"


def load(alpha: str):
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    conf, thresh = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in rows:
        bucket = conf if r["has_guarantee"] else thresh
        cell = bucket[(r["arm"], r["n_frauds"])]
        cell["seconds"].append(r["seconds"])
        cell["grad"].append(r["n_grad_fits"])
        if r["has_guarantee"]:
            a = r["alphas"].get(alpha)
            if a is None:
                continue
            cell["coverage"].append(a["coverage_fraud"])
            cell["width"].append(a["set_size"])
            cell["n_cal"].append(r["n_cal_fraud"])
        else:
            cell["recall"].append(r["recall"])
            cell["flag_rate"].append(r["flag_rate"])
    return conf, thresh, len(rows)


def quality(conf, alpha: float):
    print(f"\n### 1. Model quality at a matched guarantee (α = {alpha:g})\n")
    print("Same strategy and budget ⇒ same calibration size ⇒ **identical targeted "
          "level**, so set size compares the models directly.\n")
    print("| strategy | budget | calib. positives | targeted level | TabPFN set size | "
          "LightGBM set size | TabPFN advantage |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for strategy in ("split", "cross"):
        for budget in sorted({b for (_, b) in conf}):
            t = conf.get((f"tabpfn_{strategy}", budget))
            g = conf.get((f"lightgbm_{strategy}", budget))
            if not (t and g):
                continue
            n = int(round(np.mean(t["n_cal"])))
            k = math.ceil((n + 1) * (1 - alpha))
            level = "infeasible" if k > n else f"{k / n:.1%}"
            tw, gw = float(np.mean(t["width"])), float(np.mean(g["width"]))
            print(f"| {strategy} | {budget} | {n} | {level} | **{tw:.3f}** | {gw:.3f} | "
                  f"**{(gw - tw) / gw:+.1%}** narrower |")


def cost(conf, thresh):
    print("\n### 2. What the guarantee costs\n")
    print("| arm | gradient-trained fits | wall-clock (s) |")
    print("|---|---:|---:|")
    for bucket in (conf, thresh):
        for (arm, budget), cell in sorted(bucket.items()):
            if budget != max(b for (_, b) in bucket):
                continue
            print(f"| {arm} | **{int(np.mean(cell['grad']))}** | "
                  f"{np.mean(cell['seconds']):.1f} |")
    print("\n> ⚠ **The wall-clock column is not a like-for-like comparison and must not "
          "be presented as one.** TabPFN runs against a remote API (network latency, "
          "queueing); LightGBM runs on local CPU. The hardware-neutral number is "
          "gradient-trained fits — 0 for TabPFN by construction, since `fit` swaps a "
          "context and takes no gradient step. A fair wall-clock comparison needs both "
          "models on identical hardware.")


def guarantee(conf, thresh, alpha: str):
    print("\n### 3. Guarantee versus no guarantee\n")
    print("| arm | guarantee? | fraud coverage / recall | cost in set size or flag rate |")
    print("|---|:--:|---:|---:|")
    for (arm, budget), cell in sorted(conf.items()):
        if budget != max(b for (_, b) in conf):
            continue
        print(f"| {arm} | **yes** | {np.mean(cell['coverage']):.3f} coverage | "
              f"{np.mean(cell['width']):.3f} set size |")
    for (arm, budget), cell in sorted(thresh.items()):
        if budget != max(b for (_, b) in thresh):
            continue
        print(f"| {arm} | **no** | {np.mean(cell['recall']):.3f} recall | "
              f"{np.mean(cell['flag_rate']):.3f} flag rate |")
    print("\nThe threshold arms are the approach arXiv:2605.21742 found strongest for "
          "prior-data fitted networks, and they reach comparable recall. What they "
          "cannot do is tell you that recall will hold on next month's traffic. That is "
          "the entire difference, and it is not visible in a recall column.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.05")
    args = ap.parse_args()
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} — run E4 first.")
    conf, thresh, n = load(args.alpha)
    print(f"{n} result rows")
    quality(conf, float(args.alpha))
    cost(conf, thresh)
    guarantee(conf, thresh, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
