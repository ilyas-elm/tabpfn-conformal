"""E3 -- monthly drift: frozen thresholds vs ACI vs re-encoding the context.

Conformal prediction assumes exchangeability. Bank Account Fraud does not
oblige: the positive rate falls to 0.875% in month 2 and climbs to 1.475% by
month 7. A threshold calibrated in month 2 is calibrated for a world that stops
existing.

Three arms, which between them answer two separate questions:

  frozen   context from the early months, thresholds calibrated once and never
           touched                                    -- the baseline that breaks
  aci      same frozen context, thresholds updated each month by adaptive
           conformal inference                        -- is adapting the LEVEL enough?
  refit    context re-encoded each month from all history so far, thresholds
           recalibrated                               -- is re-encoding worth it?

The second question only exists for a model with no training step. For a
gradient-boosted baseline "refit" means a monthly retraining job and the answer
is decided by engineering cost before anyone measures accuracy. For TabPFN it is
a context swap, so the comparison is actually about statistics.

This is also where TabPFN-3.5 earns its keep beyond being a good classifier:

  * ``time_col="month"`` hands the temporal structure to the model natively
    rather than letting it treat a month index as an ordinary integer feature.
  * ``--thinking`` runs the arms through TabPFN-3.5-Thinking, which Prior Labs
    documents as strongest on grouped and temporal data. Thinking has no local
    weights, so this path only exists through the API.

Note settled by spike S1: Thinking is incompatible with fit_mode="fit_with_cache"
on the managed API (HTTP 422), so the cache story belongs to E1/E4 and the
Thinking story belongs here. They cannot share a figure.

    python experiments/api/e3_drift_aci.py --dry-run
    python experiments/api/e3_drift_aci.py --pilot
    python experiments/api/e3_drift_aci.py --thinking
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from _common import DATA, LABEL, REPO, TIME, load_token, save_proba, split_xy, time_limit  # noqa: E402

OUT = REPO / "results" / "e3.jsonl"

CALIBRATION_MONTHS = (0, 1, 2)     # what the desk knows before the walk starts
WALK_MONTHS = (3, 4, 5, 6, 7)      # revealed one at a time
N_CONTEXT = 20_000                 # a plausible labelled history for a fraud desk
N_CAL = 5_000                      # held out from the same early months
EVAL_LEGIT = 2_000                 # per month, alongside every fraud that month
ALPHA = 0.05
GAMMA = 0.05
ARMS = ("frozen", "aci", "refit")


def month_slice(df, months, n, seed, all_frauds=False):
    """Sample ``n`` rows from the given months; optionally keep every fraud."""
    rng = np.random.default_rng(seed)
    sub = df[df[TIME].isin(months)]
    if all_frauds:
        pos = sub[sub[LABEL] == 1]
        neg = sub[sub[LABEL] == 0]
        take = min(EVAL_LEGIT, len(neg))
        out = pd.concat([pos, neg.iloc[rng.choice(len(neg), take, replace=False)]])
    else:
        take = min(n, len(sub))
        out = sub.iloc[rng.choice(len(sub), take, replace=False)]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="one arm, two months")
    ap.add_argument("--thinking", action="store_true", help="use TabPFN-3.5-Thinking")
    ap.add_argument("--arms", nargs="+", default=None, choices=ARMS,
                    help="subset of arms to run; default is all three")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="seconds before a stalled month is abandoned")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    args = ap.parse_args()

    arms = ("aci",) if args.pilot else tuple(args.arms or ARMS)
    walk = WALK_MONTHS[:2] if args.pilot else WALK_MONTHS

    if not load_token():
        print("No TABPFN_TOKEN -- see experiments/api/README.md", file=sys.stderr)
        return 2

    from tabpfn_client import TabPFNClassifier, estimate_cost
    from tabpfn_conformal import ACI, ConformalClassifier, coverage_by_class, average_set_size

    def model():
        # Verified 19 Sept: time_col / group_col / group_time_col are rejected
        # outside thinking mode ("only supported in thinking mode"). So native
        # temporal handling is not a freebie -- it is a Thinking capability, and
        # Thinking has no local weights. That makes --thinking the arm that
        # actually exercises what TabPFN-3.5 can do that nothing local can.
        if args.thinking:
            return TabPFNClassifier(
                thinking_mode=True, thinking_effort="medium", time_col=TIME
            )
        return TabPFNClassifier()

    # --- price it -------------------------------------------------------
    n_eval = EVAL_LEGIT + 1_400
    total = 0
    for arm in arms:
        total += estimate_cost(np.zeros((N_CONTEXT, 30 + int(args.thinking))), np.zeros((N_CAL, 30 + int(args.thinking)))).estimated_cost
        for i, _ in enumerate(walk):
            ctx = N_CONTEXT + (i * 15_000 if arm == "refit" else 0)
            op = "thinking_predict" if args.thinking else "predict"
            total += estimate_cost(
                np.zeros((ctx, 30 + int(args.thinking))), np.zeros((n_eval, 30 + int(args.thinking))), operation=op
            ).estimated_cost
            if args.thinking:
                total += estimate_cost(
                    np.zeros((ctx, 30 + int(args.thinking))), operation="thinking_fit",
                    thinking_effort="medium",
                ).estimated_cost
    print(f"{len(arms)} arm(s) x {len(walk)} months"
          f"{' with Thinking' if args.thinking else ''}")
    print(f"estimated cost: {total:,} tokens (~{total / 20_000_000:.1%} of the monthly budget)")
    if args.thinking:
        n_fits = len(arms) * (len(walk) + 1)
        print(f"Thinking fits: {n_fits} (rate limit is 30/hour -- "
              f"{'fine' if n_fits <= 30 else 'WILL THROTTLE'})")
    if args.dry_run:
        return 0

    if not DATA.exists():
        raise SystemExit(f"Missing {DATA.relative_to(REPO)} -- run scripts/download_data.py")
    df = pd.read_csv(DATA)

    early = month_slice(df, CALIBRATION_MONTHS, N_CONTEXT + N_CAL, args.seed)
    ctx0, cal0 = early.iloc[:N_CONTEXT], early.iloc[N_CONTEXT:]
    print(f"context months {CALIBRATION_MONTHS}: {len(ctx0):,} rows, "
          f"{int(ctx0[LABEL].sum())} frauds | calibration: {len(cal0):,} rows, "
          f"{int(cal0[LABEL].sum())} frauds", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tag = "thinking" if args.thinking else "base"

    for arm in arms:
        X_ctx, y_ctx = split_xy(ctx0, keep_time=args.thinking)
        # Fit the base on the context, then calibrate on the held-out early rows.
        base = model()
        base.fit(X_ctx, y_ctx)
        cc = ConformalClassifier(base, method="mondrian", prefit=True,
                                 random_state=args.seed)
        X_cal, y_cal = split_xy(cal0, keep_time=args.thinking)
        cc.fit(X_cal, y_cal)

        # Persist the calibration scores too, so experiments/replay_aci.py can
        # re-run the whole walk at any gamma EXACTLY rather than recovering the
        # thresholds from the first evaluation month.
        cal_file = save_proba(
            "e3", f"{tag}_{arm}_CALIBRATION_s{args.seed}",
            cc.predict_proba(X_cal), y_cal,
        )

        aci = ACI(alpha_target=ALPHA, gamma=args.gamma, n_classes=2)
        history = list(CALIBRATION_MONTHS)

        for m in walk:
            ev = month_slice(df, [m], 0, args.seed, all_frauds=True)
            # Only hand the month column to the model when it can interpret it
            # as time. Otherwise drop it: the walk months are index values the
            # context never saw, and a bare integer feature would just be noise
            # the model has no basis to extrapolate from.
            X_ev, y_ev = split_xy(ev, keep_time=args.thinking)

            if arm == "refit":
                # Re-encode the context from everything seen so far. Free of
                # training; this is the arm that only makes sense for a PFN.
                hist = month_slice(df, history, N_CONTEXT, args.seed + 7)
                Xh, yh = split_xy(hist, keep_time=args.thinking)
                base = model()
                base.fit(Xh, yh)
                cc = ConformalClassifier(base, method="mondrian", prefit=True,
                                         random_state=args.seed)
                cc.fit(X_cal, y_cal)

            level = aci.alpha_dict() if arm == "aci" else ALPHA
            t0 = time.perf_counter()
            try:
                with time_limit(args.timeout):
                    proba = cc.predict_proba(X_ev)
            except Exception as exc:  # noqa: BLE001
                print(f"{arm} month {m}: FAILED {type(exc).__name__}: {str(exc)[:110]}",
                      flush=True)
                break
            sets = cc.predict_set_from_proba(proba, level)
            cov = coverage_by_class(sets, y_ev, cc.classes_)

            rec = {
                "arm": arm, "model": tag, "month": int(m), "seed": args.seed,
                "proba_file": save_proba("e3", f"{tag}_{arm}_m{m}_s{args.seed}", proba, y_ev),
                "calibration_file": cal_file,
                "gamma": args.gamma, "alpha_target": ALPHA,
                "n_eval": int(len(y_ev)), "n_eval_fraud": int((y_ev == 1).sum()),
                "fraud_rate": float((y_ev == 1).mean()),
                "coverage_fraud": cov[1], "coverage_legit": cov[0],
                "set_size": average_set_size(sets),
                "levels": aci.alpha_dict() if arm == "aci" else {0: ALPHA, 1: ALPHA},
                "seconds": round(time.perf_counter() - t0, 1),
            }
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{arm:<7} {tag:<8} month {m}  fraud_cov={cov[1]:.3f} "
                  f"width={rec['set_size']:.3f} alpha_fraud={rec['levels'][1]:.4f} "
                  f"({rec['seconds']}s)", flush=True)

            if arm == "aci":
                # Labels for month m arrive after the fact: that is the ACI
                # protocol. One step per MONTH, not one per row -- update_batch
                # would apply ~1,400 steps here and drive the level into its
                # clip bounds. Observed doing exactly that before this fix:
                # 0.05 -> 0.5 -> 0.0001 -> 0.5 across five months.
                covered = sets[np.arange(len(y_ev)), np.searchsorted(cc.classes_, y_ev)]
                aci.update_rounds(np.searchsorted(cc.classes_, y_ev), covered)
            history.append(int(m))

    print(f"\nWritten to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
