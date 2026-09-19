"""Turn results/e2.jsonl into the budget-allocation figure.

The question E2 answers: given F confirmed frauds, what fraction should go to
calibration rather than into TabPFN's context? Two asymmetric forces --
too little calibration makes the guarantee *unavailable*, too much starves the
model -- so the useful plot is set size (efficiency) against cal_size, with the
infeasible region marked and cross-conformal drawn as the "don't split" line.

Reads only the committed results file: no API key needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e2.jsonl"
FIGS = REPO / "figures"

# Validated categorical slots 1-2; cross-conformal uses ink, not a third hue,
# because it is a reference line rather than a peer series.
C1, C2 = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"


def load(alpha: str):
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E2 first.")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    split = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cross = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = r["alphas"].get(alpha)
        if a is None:
            continue
        if r["strategy"] == "cross":
            cross[r["n_frauds"]]["coverage"].append(a["coverage_fraud"])
            cross[r["n_frauds"]]["width"].append(a["set_size"])
        else:
            cell = split[r["n_frauds"]][r["cal_size"]]
            cell["coverage"].append(a["coverage_fraud"])
            cell["width"].append(a["set_size"])
            cell["feasible"].append(a["feasible"])
    return split, cross, len(rows)


def figure(split, cross, alpha: float, path: pathlib.Path):
    fig, (ax_c, ax_w) = plt.subplots(
        2, 1, figsize=(7.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_c, ax_w):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d1")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    ax_c.axhline(1 - alpha, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax_c.annotate(f"target {1 - alpha:.0%}", xy=(0.5, 1 - alpha),
                  xycoords=("axes fraction", "data"), ha="center", va="bottom",
                  fontsize=8.5, color=INK_2)

    for colour, budget in zip((C1, C2), sorted(split)):
        cal_sizes = sorted(split[budget])
        x = np.array(cal_sizes)
        feas = np.array([all(split[budget][c]["feasible"]) for c in cal_sizes])

        for ax, metric in ((ax_c, "coverage"), (ax_w, "width")):
            mean = np.array([np.mean(split[budget][c][metric]) for c in cal_sizes])
            lo = np.array([np.min(split[budget][c][metric]) for c in cal_sizes])
            hi = np.array([np.max(split[budget][c][metric]) for c in cal_sizes])
            ax.fill_between(x, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(x, mean, color=colour, linewidth=2.0, zorder=3,
                    label=f"split, {budget} frauds", solid_capstyle="round")
            # Hollow markers where the level cannot be certified at all.
            ax.plot(x[feas], mean[feas], "o", color=colour, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
            ax.plot(x[~feas], mean[~feas], "o", color=SURFACE, markersize=6,
                    markeredgecolor=colour, markeredgewidth=1.8, zorder=4)

        if budget in cross:
            ax_w.axhline(np.mean(cross[budget]["width"]), color=colour, linewidth=1.3,
                         linestyle=(0, (2, 2)), zorder=2)
            ax_w.annotate(f"cross-conformal, {budget}", xy=(0.012, np.mean(cross[budget]["width"])),
                          xycoords=("axes fraction", "data"), fontsize=8, color=INK_2,
                          va="bottom")

    ax_c.set_ylabel("fraud-class coverage", fontsize=10, color=INK)
    ax_w.set_ylabel("mean set size  (lower is better)", fontsize=10, color=INK)
    ax_w.set_xlabel("fraction of the fraud-label budget spent on calibration", fontsize=10, color=INK)
    ax_c.set_title(
        f"Where should a scarce fraud-label budget go?  (α = {alpha:g})",
        fontsize=11.5, color=INK, loc="left", pad=10,
    )
    ax_c.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2, ncol=2,
                handlelength=1.6, columnspacing=1.4)
    fig.text(
        0.012, 0.012,
        "Hollow markers: the level cannot be certified at that split "
        "(n_cal too small, so every label is returned).\n"
        "Dashed horizontal lines: cross-conformal, which spends no labels on calibration at all.\n"
        "Bands span min–max across seeds. Bank Account Fraud; TabPFN-3.5 via the Prior Labs API.",
        fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.18)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO)} and .svg")


def table(split, cross, alpha: str):
    print(f"\n### Budget allocation at alpha = {alpha}\n")
    print("| frauds | cal_size | calib. positives | feasible | coverage | set size |")
    print("|---:|---:|---:|:--:|---:|---:|")
    for b in sorted(split):
        for c in sorted(split[b]):
            cell = split[b][c]
            print(f"| {b} | {c:.1f} | ~{int(round(b * c))} | "
                  f"{'yes' if all(cell['feasible']) else '**no**'} | "
                  f"{np.mean(cell['coverage']):.3f} | {np.mean(cell['width']):.3f} |")
        if b in cross:
            print(f"| {b} | *cross* | {b} | yes | "
                  f"{np.mean(cross[b]['coverage']):.3f} | {np.mean(cross[b]['width']):.3f} |")

    print("\n**Best feasible split per budget (lowest set size):**\n")
    for b in sorted(split):
        feasible = {c: np.mean(v["width"]) for c, v in split[b].items() if all(v["feasible"])}
        if not feasible:
            print(f"- {b} frauds: no split-conformal setting can certify this level")
            continue
        best = min(feasible, key=feasible.get)
        line = f"- {b} frauds: cal_size **{best:.1f}**, set size {feasible[best]:.3f}"
        if b in cross:
            cw = np.mean(cross[b]["width"])
            line += f" — cross-conformal {cw:.3f} ({'better' if cw < feasible[best] else 'worse'})"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.05")
    args = ap.parse_args()
    split, cross, n = load(args.alpha)
    if not split:
        raise SystemExit(f"No split results at alpha={args.alpha}.")
    print(f"{n} result rows")
    figure(split, cross, float(args.alpha), FIGS / f"e2_budget_alpha{args.alpha.replace('.', '')}")
    table(split, cross, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
