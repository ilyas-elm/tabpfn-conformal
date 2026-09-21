"""Replay the ACI walk offline at any gamma, using saved probabilities.

E3 persists each month's evaluation probabilities. The context is frozen across
the ACI arm, so those probabilities do not depend on gamma at all -- only the
threshold does. That means the entire adaptive walk can be re-run for any gamma,
any target, and any update rule **without a single further API call**.

This is the payoff from saving probabilities rather than just summary metrics:
gamma is a hyperparameter that needs tuning, and tuning it against a metered API
would have cost a multiple of the original experiment.

    python experiments/replay_aci.py
    python experiments/replay_aci.py --gammas 0.05 0.2 0.5 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabpfn_conformal import ACI  # noqa: E402
from tabpfn_conformal.calibration import conformal_quantile  # noqa: E402

RESULTS = REPO / "results" / "e3.jsonl"


def load_months(model: str = "base", arm: str = "aci"):
    """Per-month (month, proba, y_true) for one arm, from the saved files."""
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("model") == model and r["arm"] == arm
            and r.get("proba_file")]
    out = []
    for r in sorted(rows, key=lambda r: r["month"]):
        d = np.load(REPO / r["proba_file"])
        out.append((r["month"], d["proba"], d["y_true"]))
    return out


def calibration_scores(model: str = "base", arm: str = "aci", months=None):
    """The real months 0-2 calibration scores, if E3 persisted them.

    Falls back to recovering them from the first evaluation month for runs made
    before that was added; the fallback is approximate and says so.
    """
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    cal = next((r["calibration_file"] for r in rows
                if r.get("model") == model and r["arm"] == arm
                and r.get("calibration_file")), None)
    if cal:
        d = np.load(REPO / cal)
        y = d["y_true"]
        return 1.0 - d["proba"][np.arange(len(y)), y], y, True
    _, proba, y = months[0]
    return 1.0 - proba[np.arange(len(y)), y], y, False


def walk(months, cal_scores, cal_y, alpha: float, gamma: float | None):
    """One pass over the months. ``gamma=None`` freezes the level."""
    aci = ACI(alpha_target=alpha, gamma=gamma or 0.01, n_classes=2)
    trace = []
    for month, proba, y in months:
        level = aci.alpha_dict() if gamma else {0: alpha, 1: alpha}
        scores = 1.0 - proba
        thr = np.array([
            conformal_quantile(cal_scores[cal_y == k], level[k], group=k)
            for k in (0, 1)
        ])
        sets = scores <= thr[None, :]
        covered = sets[np.arange(len(y)), y]
        fraud = y == 1
        trace.append({
            "month": int(month),
            "coverage_fraud": float(covered[fraud].mean()),
            "set_size": float(sets.sum(axis=1).mean()),
            "alpha_fraud": float(level[1]),
        })
        if gamma:
            aci.update_rounds(y, covered)
    return trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.2, 0.5, 1.0])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--model", default="base")
    args = ap.parse_args()

    months = load_months(args.model)
    if not months:
        raise SystemExit("No saved ACI probabilities in results/e3.jsonl.")
    cal_scores, cal_y, exact = calibration_scores(args.model, months=months)
    print(f"{len(months)} months replayed from saved probabilities; zero API calls")
    print("calibration: " + ("exact (months 0-2, persisted)" if exact
          else "APPROXIMATE (recovered from month 3; rerun E3 to persist it)") + "\n")

    # The level actually targeted is ceil((n+1)(1-alpha))/n, not 1-alpha: the
    # index rounds up. And months BELOW it are the failure -- mean absolute
    # deviation punishes over-coverage just as hard, which is the metric that
    # once scored a model worse for holding its guarantee (see docs/FINDINGS.md,
    # "the drift metric hid its own result"). analyze_e3 was fixed for both;
    # this script had kept the old reporting.
    n_cal_fraud = int(np.count_nonzero(cal_y == 1))
    target = min(1.0, math.ceil((n_cal_fraud + 1) * (1 - args.alpha)) / n_cal_fraud)

    rows = [("frozen", walk(months, cal_scores, cal_y, args.alpha, None))]
    rows += [(f"gamma={g:g}", walk(months, cal_scores, cal_y, args.alpha, g))
             for g in args.gammas]

    hdr = ("| arm | " + " | ".join(f"m{t['month']}" for t in rows[0][1])
           + " | months below target | swing | mean set size |")
    print(hdr)
    print("|---|" + "---:|" * (len(rows[0][1]) + 3))
    for name, trace in rows:
        cov = [t["coverage_fraud"] for t in trace]
        below = sum(1 for c in cov if c < target)
        # Month-to-month range. High gamma does not track the drift, it
        # oscillates the level; that shows up here and not in a mean.
        swing = max(cov) - min(cov)
        width = float(np.mean([t["set_size"] for t in trace]))
        flag = "**" if below == 0 else ""
        print(f"| {name} | " + " | ".join(f"{c:.3f}" for c in cov)
              + f" | {flag}{below} of {len(cov)}{flag} | {swing:.3f} | {width:.3f} |")

    print(f"\nTarget {target:.3%} — the level {n_cal_fraud} calibration positives "
          f"actually certify at alpha={args.alpha:g}, not the nominal "
          f"{1 - args.alpha:.0%}. Fewer months below is better; over-coverage is "
          "not a failure, it is width paid for nothing.")
    if not exact:
        print("Note: months 0-2 calibration was not persisted for this run, so "
              "thresholds are recovered from month 3.\nThe comparison across gammas "
              "is valid; absolute coverage levels are approximate.")

    out = REPO / "results" / "aci_gamma_sweep.json"
    out.write_text(json.dumps({name: trace for name, trace in rows}, indent=2))
    print(f"\nWritten to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
