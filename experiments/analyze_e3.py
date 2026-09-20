"""Turn results/e3.jsonl into the drift figure.

Top panel: fraud-class coverage month by month for each arm, against the target.
Bottom panel: the effective level ACI is running, so the adaptation is visible
rather than inferred from the coverage line.

Reads only the committed results file: no API key needed.
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
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "e3.jsonl"
FIGS = REPO / "figures"

SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
# Validated slots 1-3. Slot 3 (aqua) is below 3:1 on this surface, so the
# relief rule applies: every line carries a visible direct label and the
# script prints a table view.
ARM_STYLE = {
    "frozen": ("#2a78d6", "frozen thresholds"),
    "aci":    ("#eb6834", "ACI"),
    "refit":  ("#1baf7a", "context re-encoded monthly"),
}


def load(model_tag: str | None):
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS.relative_to(REPO)} -- run E3 first.")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if model_tag:
        rows = [r for r in rows if r.get("model") == model_tag]
    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["month"]] = r
    return by_arm, rows


def figure(by_arm, rows, path: pathlib.Path):
    alpha = rows[0]["alpha_target"]
    months_all = sorted({m for a in by_arm for m in by_arm[a]})

    def cov(arm):
        return [by_arm[arm][m]["coverage_fraud"] for m in months_all if m in by_arm[arm]]

    # ACI turned out to be numerically IDENTICAL to frozen here (threshold
    # quantization -- see docs/FINDINGS.md). Drawing two lines where one hides
    # under the other would conceal the finding, so merge them and say so.
    merged, drawn = {}, dict(ARM_STYLE)
    if "frozen" in by_arm and "aci" in by_arm and np.allclose(cov("frozen"), cov("aci")):
        merged["frozen"] = "frozen thresholds and ACI (identical)"
        drawn.pop("aci")

    fig, (ax_c, ax_a) = plt.subplots(
        2, 1, figsize=(8.4, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_c, ax_a):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d1")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    ax_c.axhline(1 - alpha, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax_c.annotate(f"target {1 - alpha:.0%}", xy=(months_all[0], 1 - alpha),
                  xytext=(2, 4), textcoords="offset points",
                  fontsize=8.5, color=INK_2, va="bottom")

    labels = []
    for arm, (colour, label) in drawn.items():
        if arm not in by_arm:
            continue
        label = merged.get(arm, label)
        months = sorted(by_arm[arm])
        c = [by_arm[arm][m]["coverage_fraud"] for m in months]
        ax_c.plot(months, c, color=colour, linewidth=2.2, zorder=3, solid_capstyle="round")
        ax_c.plot(months, c, "o", color=colour, markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4, label=label)
        labels.append((c[-1], label, colour))

        if arm == "aci" or (arm == "frozen" and "frozen" in merged):
            src = "aci" if "aci" in by_arm else arm
            lv = [by_arm[src][m]["levels"].get("1", by_arm[src][m]["levels"].get(1))
                  for m in sorted(by_arm[src])]
            ax_a.plot(sorted(by_arm[src]), lv, color=ARM_STYLE["aci"][0], linewidth=2.2, zorder=3)
            ax_a.plot(sorted(by_arm[src]), lv, "o", color=ARM_STYLE["aci"][0], markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)

    # Stagger the right-hand labels so they cannot collide.
    for i, (y, label, _) in enumerate(sorted(labels, reverse=True)):
        ax_c.annotate(label, xy=(months_all[-1], y), xytext=(8, 6 - 16 * i),
                      textcoords="offset points", fontsize=8.5, color=INK_2, va="center")

    ax_a.axhline(alpha, color=INK_MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
    ax_a.annotate(f"start \u03b1 = {alpha:g}", xy=(months_all[0], alpha), xytext=(2, 4),
                  textcoords="offset points", fontsize=8.5, color=INK_2, va="bottom")
    ax_a.annotate("ACI moves the level by 0.003 across five months \u2014 far short of the\n"
                  "0.0139 needed to shift the threshold by a single order statistic,\n"
                  "so the prediction sets never change.",
                  xy=(0.98, 0.06), xycoords="axes fraction", fontsize=8, color=INK_2,
                  linespacing=1.5, ha="right", va="bottom")

    rates = [next(by_arm[a][m]["month_fraud_rate"] for a in by_arm if m in by_arm[a])
             for m in months_all]
    ax_c.set_title(f"Fraud-class coverage as the monthly fraud rate drifts "
                   f"{rates[0]:.2%} \u2192 {rates[-1]:.2%}",
                   fontsize=11.5, color=INK, loc="left", pad=10)
    ax_c.set_ylabel("fraud-class coverage", fontsize=10, color=INK)
    ax_a.set_ylabel("effective \u03b1 (ACI)", fontsize=10, color=INK)
    ax_a.set_xlabel("month", fontsize=10, color=INK)
    ax_a.set_xticks(months_all)
    # Legend below the axes: direct labels already sit at the line ends, so an
    # in-plot legend would only duplicate them and collide with the target line.
    handles, lab = ax_c.get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=9, labelcolor=INK_2,
               loc="lower center", bbox_to_anchor=(0.42, 0.075), ncol=2,
               handlelength=1.4, columnspacing=2.0)
    fig.text(0.012, 0.012,
             f"Bank Account Fraud, {rows[0]['model']} TabPFN-3.5 via the Prior Labs API. "
             "Thresholds calibrated on months 0\u20132, then months revealed one at a time;\n"
             "ACI sees each month's labels only after predicting it.",
             fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom")
    fig.subplots_adjust(left=0.09, right=0.72, top=0.92, bottom=0.21)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO)} and .svg")


def table(by_arm):
    months = sorted({m for a in by_arm for m in by_arm[a]})
    print("\n### Coverage by month (table view — required by the contrast relief rule)\n")
    header = "| month | fraud rate | " + " | ".join(ARM_STYLE[a][1] for a in by_arm) + " |"
    print(header)
    print("|---:|---:|" + "---:|" * len(by_arm))
    for m in months:
        rate = next(by_arm[a][m]["month_fraud_rate"] for a in by_arm if m in by_arm[a])
        cells = " | ".join(
            f"{by_arm[a][m]['coverage_fraud']:.3f}" if m in by_arm[a] else "—" for a in by_arm
        )
        print(f"| {m} | {rate:.2%} | {cells} |")

    # Deviation from target treats over- and under-coverage alike, which is wrong
    # for a guarantee: under-coverage breaks the promise, over-coverage only costs
    # width. Report the promise first, then what it cost.
    n_cal = 46  # calibration positives available in this setup
    print("\n**Did the guarantee hold?**\n")
    print("The level actually targeted is ceil((n+1)(1-alpha))/n, not 1-alpha. With")
    print(f"{n_cal} calibration positives at alpha=0.05 that is "
          f"{math.ceil((n_cal + 1) * 0.95) / n_cal:.3%}.\n")
    print("| arm | months below the promised level | mean set size |")
    print("|---|---:|---:|")
    for a in by_arm:
        alpha = next(iter(by_arm[a].values()))["alpha_target"]
        target = math.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        rs = list(by_arm[a].values())
        below = sum(1 for r in rs if r["coverage_fraud"] < target)
        flag = "**" if below == 0 else ""
        print(f"| {ARM_STYLE[a][1]} | {flag}{below} of {len(rs)}{flag} | "
              f"{np.mean([r['set_size'] for r in rs]):.3f} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=["base", "thinking"])
    args = ap.parse_args()
    by_arm, rows = load(args.model)
    if not rows:
        raise SystemExit("No matching results.")
    tag = args.model or rows[0]["model"]
    print(f"{len(rows)} result rows, arms {sorted(by_arm)}")
    figure(by_arm, rows, FIGS / f"e3_drift_{tag}")
    table(by_arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
