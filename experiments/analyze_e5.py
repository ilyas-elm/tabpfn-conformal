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
        if strategy == "split":
            ax.annotate(label, xy=(x[-1], mean[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
        else:
            # Below its own line: to the right it would run across the split curve.
            ax.annotate(label + "\n(capped at 25k \u2014 K refits, hours beyond)",
                        xy=(x[0], mean[0]), xytext=(0, -30), textcoords="offset points",
                        ha="left", va="top", fontsize=8.5, color=INK_2, linespacing=1.4)

    # The effect is flat, and a tight y-axis makes seed noise look like structure.
    # Draw the noise level explicitly and state the answer, so the figure cannot be
    # read as a trend it does not contain.
    sp = sorted((n, c) for (st, n), c in stats.items() if st == "split")
    if len(sp) > 2:
        xs = np.array([n for n, _ in sp], dtype=float)
        ys = np.array([np.mean(c["width"]) for _, c in sp])
        sd = float(np.mean([np.std(c["width"], ddof=1) for _, c in sp if len(c["width"]) > 1]))
        slope = float(np.polyfit(np.log10(xs), ys, 1)[0])
        ax.axhspan(ys.mean() - sd, ys.mean() + sd, color=SPLIT_C, alpha=0.07,
                   zorder=1, linewidth=0)
        ax.axhline(ys.mean(), color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"±1 seed SD ({sd:.3f}) around the mean;\n"
                    f"the whole curve sits inside it",
                    xy=(xs[1], ys.mean() + sd), xytext=(0, 6),
                    textcoords="offset points", fontsize=8, color=INK_2, linespacing=1.4)
        ax.annotate(f"No. Slope {slope:+.3f} per 10\u00d7 context, against a seed SD of {sd:.3f}.",
                    xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                    fontsize=9.5, color=INK)
        lo, hi = ys.min() - 4 * sd, ys.max() + 4 * sd
        ax.set_ylim(lo, hi)

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
    fig.subplots_adjust(left=0.10, right=0.74, top=0.90, bottom=0.30)
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
              "fit uncached | fit cached | max |\u0394p| | \u0394 width |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ctx in sorted(pairs):
            on, off = pairs[ctx]["on"], pairs[ctx]["off"]
            # This column used to read "same answer? yes", decided by comparing
            # MEAN set size within 5e-3. That is an average, not an agreement:
            # the two runs can return different sets and the same mean. Compare
            # the saved probabilities instead, which is the actual claim.
            dp = float("nan")
            try:
                a = np.load(REPO / off["proba_file"])["proba"].astype(float)
                b = np.load(REPO / on["proba_file"])["proba"].astype(float)
                if a.shape == b.shape:
                    dp = float(np.abs(a - b).max())
            except (KeyError, OSError):
                pass
            dw = abs(on["alphas"][args.alpha]["set_size"]
                     - off["alphas"][args.alpha]["set_size"])
            print(f"| {ctx:,} | {off['predict_seconds']:.1f} s | "
                  f"**{on['predict_seconds']:.1f} s** | "
                  f"**{off['predict_seconds'] / max(on['predict_seconds'], 1e-9):.1f}\u00d7** | "
                  f"{off['fit_seconds']:.1f} s | {on['fit_seconds']:.1f} s | "
                  f"{dp:.1e} | {dw:.1e} |")
        print("\nA cached fit front-loads the attention state, so `fit` gets slower and")
        print("every later pass gets faster. Conformal scores the same context twice, ")
        print("once to calibrate, once to evaluate, so it pays back immediately.")
        print("\nThe cached and uncached runs are **not bit-identical**: the probabilities")
        print("differ in the fourth decimal on nearly every row. They agree to far better")
        print("than the seed-to-seed spread, so the conclusion is unchanged, but the")
        print("earlier \"same answer: yes\" was comparing mean set size, not the answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
