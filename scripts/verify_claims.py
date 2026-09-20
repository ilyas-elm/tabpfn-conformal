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

    # The three-seed replication is the most-corrected claim in the README --
    # it looked decisive on one seed and did not hold -- so recompute it here.
    frozen = [r for r in e3 if r["arm"] == "frozen"]
    if frozen:
        n_cal, alpha = 46, frozen[0]["alpha_target"]
        target = math.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        seeds = sorted({r["seed"] for r in frozen})
        label = {"base": "base TabPFN-3.5", "thinking": "TabPFN-3.5-Thinking"}
        below = {}
        for model, name in label.items():
            per = []
            for sd in seeds:
                rs = [r for r in frozen if r["model"] == model and r["seed"] == sd]
                per.append(sum(1 for r in rs if r["coverage_fraud"] < target))
            below[model] = per
            sizes = [r["set_size"] for r in frozen if r["model"] == model]
            cells = r" \| ".join(r"(\d+) of \d+" for _ in seeds)
            row = (r"\| " + re.escape(name) + r" \| " + cells
                   + r" \| \*\*(\d+) of (\d+)\*\* \| ([\d.]+) \|")
            m = re.search(row, README)
            for i, sd in enumerate(seeds):
                check(f"E3 {model} seed {sd} below",
                      float(m.group(i + 1)) if m else None, float(per[i]), 0.5)
            check(f"E3 {model} total below",
                  float(m.group(len(seeds) + 1)) if m else None, float(sum(per)), 0.5)
            check(f"E3 {model} seed-months",
                  float(m.group(len(seeds) + 2)) if m else None, float(len(seeds) * 5), 0.5)
            check(f"E3 {model} mean set size",
                  float(m.group(len(seeds) + 3)) if m else None, float(np.mean(sizes)))
        d = np.array(below["base"], float) - np.array(below["thinking"], float)
        se = d.std(ddof=1) / np.sqrt(d.size)
        check("E3 paired mean", in_readme(r"([\d.]+) ± [\d.]+ months"), float(d.mean()), 0.05)
        check("E3 paired stderr", in_readme(r"[\d.]+ ± ([\d.]+) months"), float(se), 0.05)
        check("E3 paired t", in_readme(r"t ≈ ([\d.]+), n = 3"), float(d.mean() / se), 0.05)

# ---- 6. Test count --------------------------------------------------------
import subprocess
out = subprocess.run([sys.executable, "-m", "pytest", "-q",
                      "--collect-only"], cwd=REPO, capture_output=True, text=True)
m = re.search(r"(\d+) tests? collected", out.stdout)
if m:
    check("test count", in_readme(r"(\d+) tests, CPU"), float(m.group(1)), 0.5)

# ---- 7. The demo is reproducible, and the video quotes it correctly -------
# The demo page is a headline deliverable, so its data must regenerate from
# results/ and the script must quote what the page actually shows.
sys.path.insert(0, str(REPO / "scripts"))
try:
    import build_demo
except Exception as exc:                                    # pragma: no cover
    checks.append(("demo regenerates", None, f"build_demo import failed: {exc}"))
else:
    committed = json.loads((REPO / "figures/demo_data.json").read_text())
    rebuilt = build_demo.scores_and_cases()
    rebuilt["e1"] = build_demo.e1_block()
    checks.append(("demo data regenerates from results", committed == rebuilt,
                   "figures/demo_data.json differs from scripts/build_demo.py output"))

    def desk(D, alpha, K):
        """Replicates panel 03 of demo/_template.html."""
        def q(scores):
            s_ = sorted(scores)
            k_ = math.ceil((len(s_) + 1) * (1 - alpha))
            return math.inf if k_ > len(s_) else s_[k_ - 1]
        qF, qL = q(D["cal_fraud"]), q(D["cal_legit"])
        rows = []
        for c in D["cases"]:
            inF, inL = (1 - c["p"]) <= qF, c["p"] <= qL
            size = inF + inL
            rows.append({"p": c["p"], "y": c["y"], "size": size,
                         "act": (("block" if inF else "approve") if size == 1 else None)})
        amb = sorted((r for r in rows if r["act"] is None),
                     key=lambda r: (-(r["size"] == 0), -r["p"]))
        for j, r in enumerate(amb):
            r["act"] = "review" if j < K else ("block" if r["p"] >= 0.5 else "approve")
        fraud = [r for r in rows if r["y"] == 1]
        return (100 * sum((1 - r["p"]) <= qF for r in fraud) / len(fraud),
                100 * sum(r["act"] != "approve" for r in fraud) / len(fraud))

    VIDEO = (REPO / "docs/VIDEO.md").read_text()

    def in_video(pattern):
        m = re.search(pattern, VIDEO)
        return float(m.group(1)) if m else None

    cov, caught_lo = desk(committed, 0.05, 0)
    _, caught_hi = desk(committed, 0.05, 200)
    check("video demo coverage", in_video(r"Coverage sits at \*\*([\d.]+)% against"), cov, 0.05)
    check("video demo caught (K=0)",
          in_video(r"moves from \*\*(\d+)% to \d+%\*\*"), round(caught_lo), 0.5)
    check("video demo caught (K=200)",
          in_video(r"moves from \*\*\d+% to (\d+)%\*\*"), round(caught_hi), 0.5)

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
