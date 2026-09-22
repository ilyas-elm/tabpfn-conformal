#!/usr/bin/env python3
"""Record per-file golden metric values for tests/test_results_integrity.py.

Run this after an experiment adds probability files to `results/proba/`. It
records what the *current* library computes, so only run it when you are sure
the library is right — the point of the goldens is to catch it changing.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from tabpfn_conformal import (average_set_size, coverage_by_class,
                              empty_set_rate, marginal_coverage)

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBA = REPO / "results" / "proba"
OUT = REPO / "tests" / "data" / "metric_goldens.json"
THRESHOLDS = (0.01, 0.05, 0.2, 0.5, 0.9)
CLASSES = np.array([0, 1])


def main() -> int:
    out = {}
    for f in sorted(PROBA.rglob("*.npz")):
        z = np.load(f)
        proba, y = z["proba"].astype(float), z["y_true"]
        rec = {}
        for thr in THRESHOLDS:
            sets = np.column_stack([1 - proba[:, 1] <= thr, proba[:, 1] <= thr])
            bc = coverage_by_class(sets, y, CLASSES)
            rec[str(thr)] = [round(marginal_coverage(sets, y, CLASSES), 12),
                             round(bc[0], 12), round(bc[1], 12),
                             round(average_set_size(sets), 12),
                             round(empty_set_rate(sets), 12)]
        out[f.relative_to(PROBA).as_posix()] = rec
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=0, sort_keys=True))
    print(f"recorded {len(out)} files to {OUT.relative_to(REPO)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
