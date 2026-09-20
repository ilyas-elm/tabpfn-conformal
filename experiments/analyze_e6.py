"""E6 analysis: does the headline replicate across BAF variants?

One table, one question. Split at budget 2F and cross at budget F calibrate on
the same number of positives, so they target an identical level and their set
widths compare directly. If cross-conformal reaches the same guarantee from half
the labels only on Base, this is where it shows.

Reads only committed results: no API key needed.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e6.jsonl"
BASE = REPO / "results" / "e1.jsonl"


def collect(path: pathlib.Path, alpha: str, variant_key=None):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        v = r[variant_key] if variant_key else "Base"
        out[(v, r["n_cal_fraud"])][r["strategy"]].append((r["n_frauds"], a["set_size"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.1")
    args = ap.parse_args()
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E6 first.")

    a = float(args.alpha)
    data = collect(RESULTS, args.alpha, "variant")
    if BASE.exists():
        data.update(collect(BASE, args.alpha))

    print(f"\n### Does the headline replicate? (α = {args.alpha})\n")
    print("Split at budget 2F and cross at budget F calibrate on the same number of")
    print("positives, so they target an identical level and compare directly.\n")
    print("| dataset | calib. positives | targeted level | split needs | width | "
          "cross needs | width | cross wins? |")
    print("|---|---:|---:|---:|---:|---:|---:|:--:|")

    wins = total = 0
    for (variant, ncal) in sorted(data, key=lambda k: (k[0], k[1])):
        pair = data[(variant, ncal)]
        if {"split", "cross"} - set(pair):
            continue
        sb, sw = pair["split"][0][0], float(np.mean([w for _, w in pair["split"]]))
        cb, cw = pair["cross"][0][0], float(np.mean([w for _, w in pair["cross"]]))
        k = math.ceil((ncal + 1) * (1 - a))
        lvl = "infeasible" if k > ncal else f"{k / ncal:.1%}"
        win = cw < sw
        wins += win; total += 1
        print(f"| {variant} | {ncal} | {lvl} | {sb} | {sw:.3f} | **{cb}** | "
              f"**{cw:.3f}** | {'yes' if win else 'no'} |")

    if total:
        print(f"\n**Cross-conformal is narrower in {wins} of {total} matched comparisons"
              f"**, each at half the labels.")
        if wins < total:
            print("Where it loses, say so plainly -- a replication that only reports "
                  "the agreements is not a replication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
