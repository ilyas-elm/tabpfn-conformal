"""E5 analysis: does abundant legitimate data help when positives are scarce?

Two questions, kept apart:

1. **Statistics.** The confirmed frauds are fixed at 200 while the legitimate
   context grows from 10k to 200k rows, driving the context fraud rate from 1.96%
   down to 0.10%. Does the guarantee get cheaper -- narrower sets at the same
   targeted level -- or does the model simply drown in negatives?
2. **Cost.** Above ~100k context rows the KV cache stops being hidden by the
   API's 10,000-token minimum charge. The cache study runs the same
   configuration with `fit_mode="fit_with_cache"` on and off.

Reads only committed results: no API key needed.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e5.jsonl"
FIGS = REPO / "figures"

SPLIT_C, CROSS_C = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"


def load(alpha: str):
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E5 first.")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    stats, cache = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        if r["cache"]:
            cache[r["n_context"]]["on"] = r
        else:
            cache.setdefault(r["n_context"], {}).setdefault("off", r)
            cell = stats[(r["strategy"], r["n_context"])]
            cell["cov"].append(a["coverage_fraud"])
            cell["width"].append(a["set_size"])
            cell["n_cal"].append(r["n_cal_fraud"])
            cell["rate"].append(r["fraud_rate_context"])
            cell["fit"].append(r["fit_seconds"])
    return stats, cache, len(rows)


def figure(stats, alpha: float, path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.grid(False, which="minor"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d1")
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xscale("log")

    for strategy, colour, label in (("split", SPLIT_C, "split conformal"),
                                    ("cross", CROSS_C, "cross-conformal")):
        pts = sorted((n, c) for (s, n), c in stats.items() if s == strategy)
        if not pts:
            continue
        x = np.array([n for n, _ in pts], dtype=float)
        mean = np.array([np.mean(c["width"]) for _, c in pts])
        lo = np.array([np.min(c["width"]) for _, c in pts])
        hi = np.array([np.max(c["width"]) for _, c in pts])
        ax.fill_between(x, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(x, mean, color=colour, linewidth=2.0, zorder=3, label=label,
                solid_capstyle="round")
        ax.plot(x, mean, "o", color=colour, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=4)
        ax.annotate(label, xy=(x[-1], mean[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)

    ticks = sorted({n for _, n in stats})
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("context rows  (200 confirmed frauds throughout)", fontsize=10, color=INK)
    ax.set_ylabel("mean set size  (lower is better)", fontsize=10, color=INK)
    ax.set_title(f"Does abundant legitimate data buy a cheaper guarantee?  (α = {alpha:g})",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="best", labelcolor=INK_2)
    fig.text(0.012, 0.012,
             "The fraud count is held at 200 while the context grows, so the context fraud rate "
             "falls from 1.96% to 0.10%.\nBands span min–max across 3 seeds. "
             "Bank Account Fraud; TabPFN-3.5 via the Prior Labs API.",
             fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom")
    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.24)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO)} and .svg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.05")
    args = ap.parse_args()
    stats, cache, n = load(args.alpha)
    a = float(args.alpha)
    print(f"{n} result rows")

    if stats:
        print(f"\n### Fixed 200 confirmed frauds, growing legitimate context "
              f"(α = {args.alpha})\n")
        print("| context | fraud rate | strategy | calib. positives | targeted level | "
              "coverage | set size |")
        print("|---:|---:|---|---:|---:|---:|---:|")
        for (s, ctx), c in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ncal = int(np.mean(c["n_cal"]))
            k = math.ceil((ncal + 1) * (1 - a))
            lvl = "infeasible" if k > ncal else f"{k / ncal:.1%}"
            print(f"| {ctx:,} | {np.mean(c['rate']):.2%} | {s} | {ncal} | {lvl} | "
                  f"{np.mean(c['cov']):.3f} | {np.mean(c['width']):.3f} |")
        figure(stats, a, FIGS / f"e5_scale_alpha{args.alpha.replace('.', '')}")

    pairs = {n_: v for n_, v in cache.items() if {"on", "off"} <= set(v)}
    if pairs:
        print("\n### What the KV cache is worth\n")
        print("| context | predict uncached | predict cached | speedup | "
              "fit uncached | fit cached | same answer? |")
        print("|---:|---:|---:|---:|---:|---:|:--:|")
        for ctx in sorted(pairs):
            on, off = pairs[ctx]["on"], pairs[ctx]["off"]
            same = (abs(on["alphas"][args.alpha]["set_size"]
                        - off["alphas"][args.alpha]["set_size"]) < 5e-3)
            print(f"| {ctx:,} | {off['predict_seconds']:.1f} s | "
                  f"**{on['predict_seconds']:.1f} s** | "
                  f"**{off['predict_seconds'] / max(on['predict_seconds'], 1e-9):.1f}×** | "
                  f"{off['fit_seconds']:.1f} s | {on['fit_seconds']:.1f} s | "
                  f"{'yes' if same else 'NO'} |")
        print("\nA cached fit front-loads the attention state, so `fit` gets slower and")
        print("every later pass gets faster. Conformal scores the same context twice —")
        print("once to calibrate, once to evaluate — so it pays back immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
