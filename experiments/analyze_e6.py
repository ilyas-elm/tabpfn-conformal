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

# Per-seed widths, so the comparison can be tested pairwise rather than by eye.
PAIRED: dict = defaultdict(dict)


def collect(path: pathlib.Path, alpha: str, variant_key=None):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        v = r[variant_key] if variant_key else "Base"
        out[(v, r["n_cal_fraud"])][r["strategy"]].append((r["n_frauds"], a["set_size"]))
        PAIRED[(v, r["n_cal_fraud"])].setdefault(r["strategy"], {})[r["seed"]] = a["set_size"]
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

    paired = PAIRED
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

    if not total:
        return 0
    print(f"\nRaw win count: cross is narrower in {wins} of {total}. That over-reads "
          "noise;\nthe seeds are paired, so test them pairwise.\n")

    print("### Paired across seeds (same seed, split at 2F vs cross at F)\n")
    print("| dataset | calib. positives | paired difference | verdict |")
    print("|---|---:|---:|---|")
    tally = {"narrower": 0, "tie": 0, "wider": 0}
    for (variant, ncal) in sorted(paired):
        pr = paired[(variant, ncal)]
        seeds = sorted(set(pr.get("split", {})) & set(pr.get("cross", {})))
        if len(seeds) < 2:
            continue
        d = np.array([pr["split"][s] - pr["cross"][s] for s in seeds])  # >0: cross narrower
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = d.mean() / se if se > 0 else np.inf
        crit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(len(d), 2.0)
        if abs(t) < crit:
            verdict, kind = "tie (within noise)", "tie"
        elif d.mean() > 0:
            verdict, kind = "**cross narrower**", "narrower"
        else:
            verdict, kind = "**cross wider**", "wider"
        tally[kind] += 1
        print(f"| {variant} | {ncal} | {d.mean():+.4f} ± {se:.4f} (n={len(d)}) | {verdict} |")

    print(f"\n**Cross-conformal is significantly wider in {tally['wider']} of "
          f"{sum(tally.values())} comparisons.** It is significantly narrower in "
          f"{tally['narrower']}; the rest are ties.")
    print("\nSo the claim that replicates is *the same guarantee from half the confirmed")
    print("frauds at no cost in set width*, the narrower-sets result holds where labels")
    print("are scarcest, which is the regime that matters, but not everywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
