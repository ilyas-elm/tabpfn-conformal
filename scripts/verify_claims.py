"""Recompute every numeric claim in the README from the committed results.

Written because a README drifts the moment an experiment is rerun, and a wrong
number in a headline table costs more credibility than the result was worth.
This recomputes each claim from `results/` and diffs it against what the README
says, so staleness is caught mechanically instead of by rereading.

    python scripts/verify_claims.py        # exits non-zero on any mismatch
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
TOL = 0.0005


def load(name):
    p = REPO / "results" / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def agg(rows, alpha, strategy_key="strategy"):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = r.get("alphas", {}).get(alpha)
        if a is None:
            continue
        cell = out[(r[strategy_key], r["n_frauds"])]
        cell["width"].append(a["set_size"])
        cell["cov"].append(a["coverage_fraud"])
        cell["n_cal"].append(r.get("n_cal_fraud", 0))
    return out


checks: list[tuple[str, bool, str]] = []


def check(label: str, claimed, actual, tol=TOL):
    ok = claimed is not None and abs(claimed - actual) <= tol
    checks.append((label, ok, f"README {claimed} vs computed {actual:.4f}"))


def in_readme(pattern: str):
    """First capture group of `pattern` as a float, or None if absent."""
    m = re.search(pattern, README)
    return float(m.group(1)) if m else None


# ---- 1. The feasibility table is pure arithmetic -------------------------
for frauds, split_c, cross_c in ((50, 96.2, 98.0), (100, 98.0, 99.0),
                                 (200, 99.0, 99.5), (400, 99.5, 99.75)):
    check(f"feasibility split @{frauds}",
          in_readme(rf"\| {frauds} \| ([\d.]+)% \|"), 100 * (1 - 2 / (frauds + 2)), 0.06)
    check(f"feasibility cross @{frauds}",
          in_readme(rf"\| {frauds} \| [\d.]+% \| \*\*([\d.]+)%\*\* \|"),
          100 * (1 - 1 / (frauds + 1)), 0.06)

# ---- 2. Level fidelity is pure arithmetic --------------------------------
for n, level in ((13, 100.0), (25, 96.0), (50, 92.0), (100, 91.0), (200, 90.5)):
    k = math.ceil((n + 1) * 0.9)
    check(f"level fidelity n={n}", level, 100 * k / n, 0.06)

# ---- 3. The matched-level table, recomputed from E1 ----------------------
e1 = agg(load("e1.jsonl"), "0.1")
by_ncal = {}
for (strategy, budget), cell in e1.items():
    by_ncal.setdefault(int(round(np.mean(cell["n_cal"]))), {})[strategy] = (
        budget, float(np.mean(cell["width"]))
    )
for n, pair in sorted(by_ncal.items()):
    if {"split", "cross"} - set(pair):
        continue
    (sb, sw), (cb, cw) = pair["split"], pair["cross"]
    k = math.ceil((n + 1) * 0.9)
    lvl = f"{100 * k / n:.1f}"
    check(f"E1 matched {lvl}% split width",
          in_readme(rf"\| {lvl}% \| {sb} frauds \| ([\d.]+) \|"), sw)
    check(f"E1 matched {lvl}% cross width",
          in_readme(rf"\| {lvl}% \| {sb} frauds \| [\d.]+ \| \*\*{cb} frauds\*\* \| \*\*([\d.]+)\*\*"), cw)

# ---- 4. The E4 baseline table --------------------------------------------
e4 = agg(load("e4.jsonl"), "0.05", strategy_key="arm")
for strategy in ("split", "cross"):
    for budget in (100, 200):
        t = e4.get((f"tabpfn_{strategy}", budget))
        g = e4.get((f"lightgbm_{strategy}", budget))
        if not (t and g):
            checks.append((f"E4 {strategy}@{budget}", None, "not yet run"))
            continue
        n = int(round(np.mean(t["n_cal"])))
        k = math.ceil((n + 1) * 0.95)
        lvl = f"{100 * k / n:.1f}"
        check(f"E4 {strategy}@{budget} TabPFN width",
              in_readme(rf"\| {strategy} \| {budget} \| {lvl}% \| \*\*([\d.]+)\*\*"),
              float(np.mean(t["width"])))
        check(f"E4 {strategy}@{budget} LightGBM width",
              in_readme(rf"\| {strategy} \| {budget} \| {lvl}% \| \*\*[\d.]+\*\* \| ([\d.]+) \|"),
              float(np.mean(g["width"])))

# ---- 5. The drift range ---------------------------------------------------
e3 = load("e3.jsonl")
if e3:
    months = sorted({r["month"] for r in e3})
    rates = {r["month"]: r["month_fraud_rate"] for r in e3}
    check("E3 drift start", in_readme(r"climbs ([\d.]+)% → [\d.]+%"),
          100 * rates[months[0]], 0.01)
    check("E3 drift end", in_readme(r"climbs [\d.]+% → ([\d.]+)%"),
          100 * rates[months[-1]], 0.01)

# ---- 6. Test count --------------------------------------------------------
import subprocess
out = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "pytest", "-q",
                      "--collect-only"], cwd=REPO, capture_output=True, text=True)
m = re.search(r"(\d+) tests? collected", out.stdout)
if m:
    check("test count", in_readme(r"(\d+) tests, CPU"), float(m.group(1)), 0.5)

# ---- report ---------------------------------------------------------------
bad = [c for c in checks if c[1] is False]
skipped = [c for c in checks if c[1] is None]
for label, ok, detail in checks:
    mark = "ok  " if ok else ("skip" if ok is None else "FAIL")
    if not ok:
        print(f"{mark}  {label:<38} {detail}")
print(f"\n{len(checks) - len(bad) - len(skipped)} verified, "
      f"{len(bad)} mismatched, {len(skipped)} not yet runnable")
sys.exit(1 if bad else 0)
