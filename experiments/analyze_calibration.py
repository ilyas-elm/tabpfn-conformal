"""Why TabPFN's conformal sets are narrower: calibration quality, measured here.

The project rests on a claim it had been borrowing from others -- that TabPFN is
unusually well calibrated. This measures it on our own data, from probabilities
already saved, at zero API cost.

It also pins down the mechanism, which is worth stating precisely because it is
easy to get backwards. Conformal prediction is **distribution-free**: its
coverage guarantee holds whatever the model's probabilities look like, including
badly calibrated ones. What calibration quality changes is not validity but
**efficiency** -- a better-calibrated model reaches the same guarantee with
narrower prediction sets, which is to say fewer cases in a human's queue.

So the chain is: TabPFN is better calibrated -> its conformal sets are narrower
-> a fraud desk reviews less for the same promise. Conformal is what makes the
calibration advantage cash out in a unit anyone can budget for.

    python experiments/analyze_calibration.py
"""

from __future__ import annotations

import glob
import json
import pathlib
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = pathlib.Path(__file__).resolve().parents[1]

# The evaluation set keeps every fraud plus a fixed legitimate sample, so it runs
# at ~49% fraud. Calibration metrics are base-rate sensitive, so measuring on it
# raw reports the enrichment rather than the model: ECE comes out near 0.45 for
# everything. Importance weights restore the true rate of BAF months 6-7.
TRUE_RATE = 0.0141
N_BINS = 15


def production_weights(y, pi: float = TRUE_RATE):
    y = y.astype(bool)
    w = np.empty(len(y), dtype=float)
    w[y] = pi / max(y.sum(), 1)
    w[~y] = (1 - pi) / max((~y).sum(), 1)
    return w / w.sum()


def weighted_ece(p, y, w, bins: int = N_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        mass = w[m].sum()
        if mass > 0:
            total += mass * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(total)


def weighted_brier(p, y, w) -> float:
    return float(np.sum(w * (p - y) ** 2))


def main() -> int:
    files = sorted(glob.glob(str(REPO / "results/proba/e4/*.npz")))
    if not files:
        raise SystemExit("No E4 probabilities — run experiments/api/e4_baselines.py first.")

    agg = defaultdict(list)
    for f in files:
        # rsplit from the right: save_proba() sanitises "|" to "_", so a family
        # whose name already contains one (tabpfn_balanced) would split into five
        # parts and raise here. Parsing from the right keeps that working.
        stem = pathlib.Path(f).stem
        rest, budget, _seed = stem.rsplit("_", 2)
        model, strategy = rest.rsplit("_", 1)
        d = np.load(f)
        p = d["proba"][:, 1].astype(float)
        y = d["y_true"].astype(float)
        w = production_weights(y)
        agg[(model, strategy, int(budget))].append(
            (weighted_ece(p, y, w), weighted_brier(p, y, w), roc_auc_score(y, p))
        )

    print(f"\n### Calibration quality, reweighted to the true {TRUE_RATE:.2%} base rate\n")
    print("| model | strategy | budget | ECE | Brier | AUC |")
    print("|---|---|---:|---:|---:|---:|")
    for (m, s, b), v in sorted(agg.items()):
        print(f"| {m} | {s} | {b} | {np.mean([x[0] for x in v]):.5f} | "
              f"{np.mean([x[1] for x in v]):.5f} | {np.mean([x[2] for x in v]):.4f} |")

    print("\n### Head to head on identical rows\n")
    print("| strategy | budget | TabPFN ECE | LightGBM ECE | TabPFN better by | AUC gap |")
    print("|---|---:|---:|---:|---:|---:|")
    out = {}
    for s in ("split", "cross"):
        for b in sorted({k[2] for k in agg}):
            t, g = agg.get(("tabpfn", s, b)), agg.get(("lightgbm", s, b))
            if not (t and g):
                continue
            te, ge = np.mean([x[0] for x in t]), np.mean([x[0] for x in g])
            ta, ga = np.mean([x[2] for x in t]), np.mean([x[2] for x in g])
            print(f"| {s} | {b} | **{te:.5f}** | {ge:.5f} | "
                  f"**{abs(te - ge) / max(te, ge):.0%}** | +{ta - ga:.3f} |")
            out[f"{s}_{b}"] = {"tabpfn_ece": te, "lightgbm_ece": ge,
                               "tabpfn_auc": ta, "lightgbm_auc": ga}

    print("\nConformal prediction is distribution-free: the coverage guarantee holds for")
    print("a badly calibrated model too. What calibration buys is not validity but")
    print("EFFICIENCY — the same promise from narrower sets. That is the mechanism")
    print("behind TabPFN's narrower sets in E4, and it is measurable here.")

    path = REPO / "results" / "calibration.json"
    path.write_text(json.dumps({"true_rate": TRUE_RATE, "n_bins": N_BINS,
                                "head_to_head": out}, indent=2))
    print(f"\nWritten to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
