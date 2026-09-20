#!/usr/bin/env python3
"""Build the interactive demo from committed results.

Writes ``figures/demo_data.json`` and renders ``demo/index.html`` from
``demo/_template.html``. Needs no API key and no TabPFN install: everything
comes from files already in ``results/``.

    python scripts/build_demo.py

Every number the demo shows is derived here, so the page can be regenerated
and checked the same way ``scripts/verify_claims.py`` checks the README.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# One TabPFN file, split into disjoint calibration and display halves. The
# basename is asserted because save_proba() sanitises "|" to "_", and an
# earlier glob silently matched the LightGBM file with a TabPFN label.
SOURCE = Path("results/proba/e4/tabpfn_cross_200_0.npz")
SPLIT_SEED = 0      # calibration / display halves
CASE_SEED = 1       # which display rows are drawn on the page
N_CASE_FRAUD, N_CASE_LEGIT = 120, 280

NOTE = ("TabPFN-3.5 via the Prior Labs API on Bank Account Fraud months 6-7. "
        "Calibration and displayed cases are disjoint halves of one held-out "
        "set. Regenerate with scripts/build_demo.py.")


def scores_and_cases() -> dict:
    """Split one held-out TabPFN file into calibration scores and display cases."""
    assert SOURCE.name.startswith("tabpfn"), f"{SOURCE.name} is not a TabPFN file"
    z = np.load(REPO / SOURCE)
    proba, y = z["proba"].astype(np.float32), z["y_true"]
    n = len(y)

    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    cal, disp = perm[: n // 2], perm[n // 2 :]

    # Nonconformity score for the true label: 1 - p(true class).
    score = 1.0 - proba[np.arange(n), y]
    cal_fraud = np.sort(score[cal[y[cal] == 1]])
    cal_legit = np.sort(score[cal[y[cal] == 0]])

    rng = np.random.default_rng(CASE_SEED)
    pick_f = rng.choice(disp[y[disp] == 1], N_CASE_FRAUD, replace=False)
    pick_l = rng.choice(disp[y[disp] == 0], N_CASE_LEGIT, replace=False)
    pick = rng.permutation(np.concatenate([pick_f, pick_l]))
    cases = [{"p": round(float(proba[i, 1]), 5), "y": int(y[i])} for i in pick]

    return {
        "source": SOURCE.as_posix(),
        "note": NOTE,
        # Stored in full: the page computes its thresholds from these arrays,
        # so a truncated array would silently mean a smaller calibration set.
        "n_cal_fraud": int(cal_fraud.size),
        "n_cal_legit": int(cal_legit.size),
        "cal_fraud": [round(float(v), 5) for v in cal_fraud],
        "cal_legit": [round(float(v), 5) for v in cal_legit],
        "cases": cases,
        "n_fraud": int(sum(c["y"] for c in cases)),
    }


def e1_block() -> list[dict]:
    """Mean set size per (strategy, label budget, alpha), averaged over seeds."""
    widths: dict[tuple, list[float]] = defaultdict(list)
    ncal: dict[tuple, set] = defaultdict(set)
    with open(REPO / "results/e1.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            key0 = (r["strategy"], r["n_frauds"])
            ncal[key0].add(r["n_cal_fraud"])
            for a, m in r["alphas"].items():
                widths[key0 + (float(a),)].append(m["set_size"])

    out = []
    for (strategy, frauds, alpha), vals in sorted(widths.items()):
        sizes = ncal[(strategy, frauds)]
        assert len(sizes) == 1, f"{strategy}/{frauds} has mixed n_cal_fraud: {sizes}"
        out.append({"s": strategy, "f": frauds, "a": alpha,
                    "w": round(float(np.mean(vals)), 4), "ncal": next(iter(sizes))})
    return out


def main() -> None:
    data = scores_and_cases()
    data["e1"] = e1_block()

    out_json = REPO / "figures/demo_data.json"
    out_json.write_text(json.dumps(data, separators=(",", ":")))

    template = (REPO / "demo/_template.html").read_text()
    assert "__DEMO_DATA__" in template, "template lost its __DEMO_DATA__ placeholder"
    page = template.replace("__DEMO_DATA__", json.dumps(data, separators=(",", ":")))
    out_html = REPO / "demo/index.html"
    out_html.write_text(page)

    print(f"wrote {out_json.relative_to(REPO)} "
          f"({data['n_cal_fraud']} fraud + {data['n_cal_legit']} legit calibration "
          f"scores, {len(data['cases'])} display cases, {len(data['e1'])} E1 rows)")
    print(f"wrote {out_html.relative_to(REPO)} ({out_html.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
