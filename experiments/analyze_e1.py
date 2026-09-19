"""Turn results/e1.jsonl into Figure 1 and the README tables.

No API calls, no network: this reads the committed results file, so a judge can
regenerate every figure from the repo without a Prior Labs account.

    python experiments/analyze_e1.py [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e1.jsonl"
FIGS = REPO / "figures"

# Categorical slots 1 and 2 of the validated default palette. Validated for this
# pair: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1 on #fcfcfb.
SPLIT_C, CROSS_C = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"


def load():
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E1 first.")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("results/e1.jsonl is empty.")
    return rows


def aggregate(rows, alpha: str):
    """{strategy: {n_frauds: {metric: [values across seeds]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        cell = out[r["strategy"]][r["n_frauds"]]
        cell["coverage"].append(a["coverage_fraud"])
        cell["width"].append(a["set_size"])
        cell["n_cal"].append(r["n_cal_fraud"])
        cell["seconds"].append(r["seconds"])
    return out


def _series(agg, strategy, metric):
    budgets = sorted(agg[strategy])
    mean = np.array([np.mean(agg[strategy][b][metric]) for b in budgets])
    lo = np.array([np.min(agg[strategy][b][metric]) for b in budgets])
    hi = np.array([np.max(agg[strategy][b][metric]) for b in budgets])
    n = [len(agg[strategy][b][metric]) for b in budgets]
    return np.array(budgets, dtype=float), mean, lo, hi, n


def figure(agg, alpha: float, path: pathlib.Path):
    fig, (ax_c, ax_w) = plt.subplots(
        2, 1, figsize=(7.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax_c, ax_w):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.grid(False, which="minor")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d1")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.set_xscale("log")

    target = 1 - alpha
    budgets_all = sorted({b for st in agg for b in agg[st]})

    # Split calibrates on F/2 positives, so it can only certify alpha >= 2/(F+2).
    # Below this budget its "coverage" is 1.0 only because it returns every label.
    split_floor = 2.0 / alpha - 2.0
    if split_floor > min(budgets_all):
        for ax in (ax_c, ax_w):
            ax.axvspan(min(budgets_all) * 0.82, split_floor, color="#efeeea",
                       zorder=0, linewidth=0)
        ax_c.annotate(
            f"split cannot certify \u03b1={alpha:g}\nwith fewer than {int(np.ceil(split_floor))} frauds",
            xy=(min(budgets_all) * 0.86, 0.04), xycoords=("data", "axes fraction"),
            fontsize=8, color=INK_2, ha="left", va="bottom", linespacing=1.4,
        )

    ax_c.axhline(target, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax_c.annotate(
        f"target {target:.0%}", xy=(0.5, target), xycoords=("axes fraction", "data"),
        ha="center", va="bottom", fontsize=8.5, color=INK_2,
    )

    for strategy, colour, label in (
        ("split", SPLIT_C, "split conformal"),
        ("cross", CROSS_C, "cross-conformal"),
    ):
        if strategy not in agg:
            continue
        for ax, metric in ((ax_c, "coverage"), (ax_w, "width")):
            x, mean, lo, hi, _ = _series(agg, strategy, metric)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.14, linewidth=0, zorder=2)
            ax.plot(x, mean, color=colour, linewidth=2.0, label=label, zorder=3,
                    solid_capstyle="round")
            ax.plot(x, mean, "o", color=colour, markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
        # Direct label at the right end, in ink -- identity is never colour alone.
        x, mean, *_ = _series(agg, strategy, "coverage")
        ax_c.annotate(
            label, xy=(x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK_2,
        )

    ax_c.set_ylabel("fraud-class coverage", fontsize=10, color=INK)
    ax_w.set_ylabel("mean set size", fontsize=10, color=INK)
    ax_w.set_xlabel("confirmed fraud labels available", fontsize=10, color=INK)

    budgets = sorted({b for s in agg for b in agg[s]})
    ax_w.set_xticks(budgets)
    ax_w.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_w.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax_c.set_title(
        f"Fraud-class coverage at α = {alpha:g}, and what it costs in set size",
        fontsize=11.5, color=INK, loc="left", pad=10,
    )
    ax_c.legend(
        frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2,
        bbox_to_anchor=(1.0, -0.02), ncol=2, handlelength=1.6, columnspacing=1.4,
    )
    fig.text(
        0.012, 0.012,
        "Bands span min\u2013max across seeds. In the shaded region split returns every label,\n"
        "so its coverage of 1.0 is vacuous \u2014 read it against set size below.\n"
        "Bank Account Fraud; months 0\u20135 pool, 6\u20137 evaluation; TabPFN-3.5 via the Prior Labs API.",
        fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.80, top=0.92, bottom=0.17)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO)} and .svg")


def tables(agg, alpha: str):
    print(f"\n### Measured at alpha = {alpha}\n")
    print("| frauds | strategy | calib. frauds | coverage (mean) | spread | set size | seeds |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for b in sorted({x for s in agg for x in agg[s]}):
        for s in ("split", "cross"):
            if b not in agg[s]:
                continue
            c = agg[s][b]
            cov = np.array(c["coverage"])
            print(
                f"| {b} | {s} | {int(np.mean(c['n_cal'])):,} | {cov.mean():.3f} | "
                f"{cov.max() - cov.min():.3f} | {np.mean(c['width']):.3f} | {len(cov)} |"
            )

    print("\n### Feasibility boundary (deterministic)\n")
    print("| confirmed frauds | split can certify | cross can certify |")
    print("|---:|---:|---:|")
    for f in (50, 100, 200, 400):
        print(f"| {f} | {1 - 2 / (f + 2):.2%} | **{1 - 1 / (f + 1):.2%}** |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.05")
    args = ap.parse_args()

    rows = load()
    agg = aggregate(rows, args.alpha)
    if not agg:
        raise SystemExit(f"No results at alpha={args.alpha}.")
    print(f"{len(rows)} result rows; strategies {sorted(agg)}")
    figure(agg, float(args.alpha), FIGS / f"e1_coverage_alpha{args.alpha.replace('.', '')}")
    tables(agg, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
