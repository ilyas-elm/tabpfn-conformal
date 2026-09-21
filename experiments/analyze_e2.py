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
            # Keyed by seed as well as appended. E2 was resumed part-way, so the
            # rows for two settings are in a rotated seed order, and anything
            # that pairs settings by list position pairs different seeds.
            cell["seeds"].append(r["seed"])
            cell["n_cal"].append(a.get("n_cal_fraud", r.get("n_cal_fraud")))
    return split, cross, len(rows)


def figure(split, cross, alpha: float, path: pathlib.Path):
    any_hollow = False
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

    # One flat line at 1-alpha is wrong here, and wrong by a different amount at
    # every x. The level actually certified is ceil((n_cal+1)(1-alpha))/n_cal,
    # and n_cal IS the x-axis: at 100 frauds it runs from 100% at cal_size 0.2
    # down to 96.25% at 0.8. Drawn flat at 0.95, a point could sit well above
    # the line while being below its own promise -- which is exactly what
    # cal_size 0.2 does. So the target is drawn per budget, as a curve.
    ax_c.axhline(1 - alpha, color="#cfcec9", linewidth=1.0, linestyle=(0, (2, 4)),
                 zorder=1)
    ax_c.annotate(f"nominal {1 - alpha:.0%}", xy=(0.02, 1 - alpha),
                  xycoords=("axes fraction", "data"), ha="left", va="bottom",
                  fontsize=8, color="#a9a8a3")

    for colour, budget in zip((C1, C2), sorted(split)):
        cal_sizes = sorted(split[budget])
        x = np.array(cal_sizes)
        feas = np.array([all(split[budget][c]["feasible"]) for c in cal_sizes])
        any_hollow = any_hollow or bool((~feas).any())

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

        # The level this budget actually certifies at each cal_size.
        n_cal = np.array([int(np.median(split[budget][c]["n_cal"])) for c in cal_sizes])
        target = np.minimum(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        ax_c.plot(x, target, color=colour, linewidth=1.3, linestyle=(0, (4, 3)),
                  zorder=2, alpha=0.75)
        ax_c.annotate("certified", xy=(x[-1], target[-1]), xytext=(5, 0),
                      textcoords="offset points", fontsize=8, color=colour,
                      va="center")

        if budget in cross:
            ax_w.axhline(np.mean(cross[budget]["width"]), color=colour, linewidth=1.3,
                         linestyle=(0, (2, 2)), zorder=2)
            ax_w.annotate(f"cross-conformal, {budget} frauds",
                          xy=(0.012, np.mean(cross[budget]["width"])),
                          xycoords=("axes fraction", "data"), fontsize=8, color=INK_2,
                          va="bottom" if budget == min(split) else "top",
                          xytext=(0, 3 if budget == min(split) else -3),
                          textcoords="offset points")

    ax_c.set_ylabel("fraud-class coverage", fontsize=10, color=INK)
    ax_w.set_ylabel("mean set size  (lower is better)", fontsize=10, color=INK)
    ax_w.set_xlabel("fraction of the fraud-label budget spent on calibration", fontsize=10, color=INK)
    ax_c.set_title(
        f"Where should a scarce fraud-label budget go?  (α = {alpha:g})",
        fontsize=11.5, color=INK, loc="left", pad=10,
    )
    ax_c.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2, ncol=2,
                handlelength=1.6, columnspacing=1.4)
    # Only explain hollow markers when some were drawn. Every setting here is
    # certifiable, so the note used to describe something not on the chart and
    # sent a reader hunting for it among the white-ringed solid markers.
    notes = []
    if any_hollow:
        notes.append("Hollow markers: the level cannot be certified at that split "
                     "(n_cal too small, so every label is returned).")
    notes += [
        "Dashed curves: the level each budget actually certifies, "
        "ceil((n_cal+1)(1-\u03b1))/n_cal, which moves with n_cal and is what a "
        "point has to clear.",
        "Dotted horizontal lines (lower panel): cross-conformal, which spends no "
        "labels on calibration at all.",
        "Bands span min\u2013max across seeds. Bank Account Fraud; TabPFN-3.5 via "
        "the Prior Labs API.",
    ]
    fig.text(0.012, 0.012, "\n".join(notes),
             fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.20)
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

    print("\n**Is the best cal_size a real optimum, or seed noise?**\n")
    print("A best-looking setting means nothing if the gap to the worst setting is no")
    print("bigger than the scatter across seeds within a single setting.\n")
    for b in sorted(split):
        feasible = {c: v for c, v in split[b].items() if all(v["feasible"])}
        if not feasible:
            print(f"- {b} frauds: no split-conformal setting can certify this level")
            continue
        means = {c: float(np.mean(v["width"])) for c, v in feasible.items()}
        best, worst = min(means, key=means.get), max(means, key=means.get)

        # `best` and `worst` are the extremes of several cal_size settings, so a
        # plain paired t between them is testing a pair chosen *because* it is
        # extreme -- that inflates significance, and an earlier version of this
        # called 200 frauds "a real optimum" on t=3.6 against a critical value
        # for one pre-specified comparison.
        #
        # Permute instead. The seeds are paired (the same pools at every
        # cal_size), so under the null that cal_size does not matter the widths
        # within a seed are exchangeable across settings. Shuffling within seed
        # and re-taking max-minus-min builds the null distribution of the
        # statistic actually being reported, selection included. Exact-ish,
        # assumption-light, and needs nothing beyond numpy.
        sizes = sorted(feasible)
        common = (set.intersection(*(set(feasible[c]["seeds"]) for c in sizes))
                  if sizes else set())
        if len(sizes) > 1 and len(common) > 1:
            # Align on seed, never on position: see the note in load().
            seeds = sorted(common)
            W = np.array([[dict(zip(feasible[c]["seeds"], feasible[c]["width"]))[s]
                           for s in seeds] for c in sizes])       # (settings, seeds)
            observed = float(W.mean(axis=1).max() - W.mean(axis=1).min())
            rng = np.random.default_rng(0)
            null = np.empty(20_000)
            for i in range(null.size):
                # An independent permutation of the settings within each seed.
                order = np.argsort(rng.random(W.shape), axis=0)
                m = np.take_along_axis(W, order, axis=0).mean(axis=1)
                null[i] = m.max() - m.min()
            pval = float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))
            # "the settings differ", not "there is an optimum": the test says the
            # spread is larger than chance, and says nothing about *which*
            # setting is best -- picking that out of seven is the selection
            # problem this test exists to avoid.
            verdict = (f"spread {observed:.3f} across {len(sizes)} settings, "
                       f"permutation p={pval:.3f} (n={W.shape[1]} paired seeds) → "
                       + ("**the settings differ** (but which is best is not resolved)"
                          if pval < 0.05 else "**not distinguishable from noise**"))
        else:
            verdict = "unequal seed counts; no paired test"
        print(f"- **{b} frauds**: best cal_size {best:.1f} ({means[best]:.3f}), "
              f"worst {worst:.1f} ({means[worst]:.3f}). {verdict}")
        if b in cross:
            cw = float(np.mean(cross[b]["width"]))
            verdict2 = ("**beats every split setting**" if cw < means[best]
                        else "does not beat the best split")
            print(f"  Cross-conformal: {cw:.3f} — {verdict2}.")


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
