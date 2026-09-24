"""Turn results/e1.jsonl into Figure 1 and the README tables.

No API calls, no network: this reads the committed results file, so a judge can
regenerate every figure from the repo without a Prior Labs account.

    python experiments/analyze_e1.py [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import math
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


def effective_nominal(n_cal: int, alpha: float) -> float:
    """The level a calibration set of size ``n_cal`` actually targets.

    Conformal uses the ceil((n+1)(1-alpha))-th smallest score, and that index
    rounds up. With 13 calibration positives at alpha=0.10 the index is 13 --
    the maximum -- so the predictor targets 100% coverage, not 90%. Comparing
    two methods' realised coverage without this is comparing them at different
    levels: at any fraud budget split calibrates on half as many positives as
    cross, so split is always the more conservative of the two.
    """
    if n_cal <= 0:
        return float("nan")
    k = math.ceil((n_cal + 1) * (1 - alpha))
    return float("nan") if k > n_cal else k / n_cal


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
        cell["effective"].append(effective_nominal(r["n_cal_fraud"], float(alpha)))
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

    # The nominal level is a reference, not the target: what a point has to
    # clear is ceil((n_cal+1)(1-alpha))/n_cal, which differs per strategy and
    # per budget and is drawn below as the dotted curves. Labelling this flat
    # line "target" put the emphasis on the wrong one.
    ax_c.axhline(target, color="#cfcec9", linewidth=1.0, linestyle=(0, (2, 4)),
                 zorder=1)
    ax_c.annotate(
        f"nominal {target:.0%}", xy=(0.02, target), xycoords=("axes fraction", "data"),
        ha="left", va="bottom", fontsize=8, color="#a9a8a3",
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
        xe, eff, *_ = _series(agg, strategy, "effective")
        ax_c.plot(xe, eff, color=colour, linewidth=1.4, linestyle=(0, (4, 3)),
                  zorder=2, alpha=0.8)

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
        "Dashed curves: the level each strategy actually certifies at that budget, "
        "ceil((n_cal+1)(1-\u03b1))/n_cal \u2014 what a point has to clear.\n"
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
    print("| frauds | strategy | calib. frauds | level actually targeted | coverage (mean) | "
          "vs its own target | spread | set size | seeds |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in sorted({x for s in agg for x in agg[s]}):
        for s in ("split", "cross"):
            if b not in agg[s]:
                continue
            c = agg[s][b]
            cov = np.array(c["coverage"])
            eff = float(np.mean(c["effective"]))
            eff_s = "infeasible" if np.isnan(eff) else f"{eff:.1%}"
            gap = "n/a" if np.isnan(eff) else f"{cov.mean() - eff:+.3f}"
            print(
                f"| {b} | {s} | {int(np.mean(c['n_cal'])):,} | {eff_s} | {cov.mean():.3f} | "
                f"{gap} | {cov.max() - cov.min():.3f} | {np.mean(c['width']):.3f} | {len(cov)} |"
            )

    print("\n### Feasibility boundary (deterministic)\n")
    print("| confirmed frauds | split can certify | cross can certify |")
    print("|---:|---:|---:|")
    for f in (50, 100, 200, 400):
        print(f"| {f} | {1 - 2 / (f + 2):.2%} | **{1 - 1 / (f + 1):.2%}** |")


def matched_figure(agg, alpha: float, path: pathlib.Path):
    """The headline figure: set size against calibration size, not budget.

    Plotting against calibration positives rather than the label budget is what
    makes the claim legible. Both methods land on the same x positions, so they
    are compared at an identical targeted level -- and the annotation says how
    many confirmed frauds each one needed to get there. Cross needs half.
    """
    by_ncal = {}
    for strategy in ("split", "cross"):
        for budget, cell in agg.get(strategy, {}).items():
            n = int(round(np.mean(cell["n_cal"])))
            by_ncal.setdefault(n, {})[strategy] = (
                budget, float(np.mean(cell["width"])),
                float(np.min(cell["width"])), float(np.max(cell["width"])),
            )
    pairs = sorted(n for n, v in by_ncal.items() if {"split", "cross"} <= set(v))
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
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

    for strategy, colour, label in (
        ("split", SPLIT_C, "split conformal"),
        ("cross", CROSS_C, "cross-conformal"),
    ):
        x = np.array(pairs, dtype=float)
        mean = np.array([by_ncal[n][strategy][1] for n in pairs])
        lo = np.array([by_ncal[n][strategy][2] for n in pairs])
        hi = np.array([by_ncal[n][strategy][3] for n in pairs])
        ax.fill_between(x, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(x, mean, color=colour, linewidth=2.0, zorder=3, label=label,
                solid_capstyle="round")
        ax.plot(x, mean, "o", color=colour, markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
        for n, m in zip(pairs, mean):
            ax.annotate(f"{by_ncal[n][strategy][0]} frauds", xy=(n, m),
                        xytext=(0, 11 if strategy == "split" else -17),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK_2)
        ax.annotate(label, xy=(pairs[-1], mean[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)

    ax.set_xticks(pairs)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    labels = []
    for n in pairs:
        k = math.ceil((n + 1) * (1 - alpha))
        labels.append(f"{n}\n({k / n:.1%})" if k <= n else f"{n}\n(infeasible)")
    ax.set_xticklabels(labels)
    ax.set_xlabel("calibration positives  (and the coverage level that implies)",
                  fontsize=10, color=INK)
    ax.set_ylabel("mean set size  (lower is better)", fontsize=10, color=INK)
    ax.set_title("The same guarantee from half the confirmed frauds",
                 fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_2)
    fig.text(
        0.012, 0.012,
        "Both methods are compared at an identical targeted level, because identical calibration size\n"
        "implies an identical level. The annotation is how many confirmed frauds each one needed.\n"
        f"\u03b1 = {alpha:g}; bands span min\u2013max across 5 seeds; Bank Account Fraud; "
        "TabPFN-3.5 via the Prior Labs API.",
        fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom",
    )
    ax.margins(x=0.10)
    fig.subplots_adjust(left=0.13, right=0.79, top=0.90, bottom=0.30)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO)} and .svg")


def matched_level_table(agg, alpha: str):
    """The comparison that is not confounded.

    Split at budget 2F calibrates on F positives, exactly as cross at budget F
    does. Identical calibration size means an identical targeted level, so these
    pairs can be compared directly on set width -- no interpolation, no matching
    on realised coverage, no confound. The question it answers is the practical
    one: how many confirmed frauds does each method need for the same guarantee?
    """
    a = float(alpha)
    by_ncal = {}
    for strategy in ("split", "cross"):
        for budget, cell in agg.get(strategy, {}).items():
            n = int(round(np.mean(cell["n_cal"])))
            by_ncal.setdefault(n, {})[strategy] = (budget, float(np.mean(cell["width"])))

    pairs = {n: v for n, v in by_ncal.items() if {"split", "cross"} <= set(v)}
    if not pairs:
        return

    print(f"\n### Matched on calibration size, identical targeted level (alpha = {alpha})\n")
    print("| calib. positives | targeted level | split needs | its set size | "
          "cross needs | its set size | labels saved |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in sorted(pairs):
        k = math.ceil((n + 1) * (1 - a))
        target = "infeasible" if k > n else f"{k / n:.1%}"
        (sb, sw), (cb, cw) = pairs[n]["split"], pairs[n]["cross"]
        verdict = "narrower" if cw < sw else "wider"
        print(f"| {n} | {target} | {sb} frauds | {sw:.3f} | **{cb} frauds** | "
              f"**{cw:.3f}** ({verdict} by {abs(cw - sw) / sw:.1%}) | "
              f"**{sb - cb} ({(sb - cb) / sb:.0%})** |")


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
    matched_figure(agg, float(args.alpha),
                   FIGS / f"e1_matched_alpha{args.alpha.replace('.', '')}")
    matched_level_table(agg, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
