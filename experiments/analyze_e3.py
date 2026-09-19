"""Turn results/e3.jsonl into the drift figure.

Top panel: fraud-class coverage month by month for each arm, against the target.
Bottom panel: the effective level ACI is running, so the adaptation is visible
rather than inferred from the coverage line.

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
    fig, (ax_c, ax_a) = plt.subplots(
        2, 1, figsize=(7.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_c, ax_a):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d1")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    ax_c.axhline(1 - alpha, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax_c.annotate(f"target {1 - alpha:.0%}", xy=(0.02, 1 - alpha),
                  xycoords=("axes fraction", "data"), fontsize=8.5, color=INK_2, va="bottom")

    for arm, (colour, label) in ARM_STYLE.items():
        if arm not in by_arm:
            continue
        months = sorted(by_arm[arm])
        cov = [by_arm[arm][m]["coverage_fraud"] for m in months]
        ax_c.plot(months, cov, color=colour, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax_c.plot(months, cov, "o", color=colour, markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4, label=label)
        # Relief rule: identity never by colour alone.
        ax_c.annotate(label, xy=(months[-1], cov[-1]), xytext=(6, 0),
                      textcoords="offset points", fontsize=8.5, color=INK_2, va="center")

        if arm == "aci":
            lv = [by_arm[arm][m]["levels"]["1"] if "1" in by_arm[arm][m]["levels"]
                  else by_arm[arm][m]["levels"][1] for m in months]
            ax_a.plot(months, lv, color=colour, linewidth=2.0, zorder=3)
            ax_a.plot(months, lv, "o", color=colour, markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
            ax_a.annotate("ACI level for the fraud class", xy=(months[-1], lv[-1]),
                          xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                          color=INK_2, va="center")

    ax_a.axhline(alpha, color=INK_MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
    ax_a.annotate(f"starting level α = {alpha:g}", xy=(0.02, alpha),
                  xycoords=("axes fraction", "data"), fontsize=8.5, color=INK_2, va="bottom")

    # The drift itself, as context for why any of this is needed.
    months_all = sorted({m for a in by_arm for m in by_arm[a]})
    rates = [next(by_arm[a][m]["month_fraud_rate"] for a in by_arm if m in by_arm[a])
             for m in months_all]
    ax_c.set_title(
        "Fraud-class coverage as the monthly fraud rate drifts  "
        f"({rates[0]:.2%} → {rates[-1]:.2%} across these months)",
        fontsize=11.5, color=INK, loc="left", pad=10,
    )
    ax_c.set_ylabel("fraud-class coverage", fontsize=10, color=INK)
    ax_a.set_ylabel("effective α", fontsize=10, color=INK)
    ax_a.set_xlabel("month", fontsize=10, color=INK)
    ax_a.set_xticks(months_all)
    ax_c.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_2)
    fig.text(
        0.012, 0.012,
        f"Bank Account Fraud, {rows[0]['model']} TabPFN-3.5 via the Prior Labs API. "
        "Thresholds calibrated on months 0–2, then months revealed one at a time.\n"
        "ACI sees each month's labels only after predicting it.",
        fontsize=7.5, color=INK_MUTED, linespacing=1.5, va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.74, top=0.92, bottom=0.15)
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

    print("\n**Mean absolute deviation from target across the walk:**\n")
    for a in by_arm:
        target = 1 - next(iter(by_arm[a].values()))["alpha_target"]
        dev = np.mean([abs(r["coverage_fraud"] - target) for r in by_arm[a].values()])
        print(f"- {ARM_STYLE[a][1]}: {dev:.4f}")


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
