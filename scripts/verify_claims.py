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
    """First capture group of `pattern` as a float, or None if absent.

    The prose uses a typographic minus (U+2212), which float() rejects, so the
    captured text is normalised before conversion.
    """
    m = re.search(pattern, README)
    return float(m.group(1).replace("\u2212", "-")) if m else None


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

# ---- 5b. The calibration (ECE) table, recomputed from the saved probabilities
# The "74-86% lower calibration error" line is quoted in the README, the
# submission and the video, and was the only headline claim with no check.
def _weights(y, true_rate=0.0141):
    y = np.asarray(y, dtype=float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    w = np.where(y == 1, true_rate / max(n_pos, 1), (1 - true_rate) / max(n_neg, 1))
    return w / w.sum()


def _ece(p, y, w, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        if wb <= 0:
            continue
        total += wb * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(total)


proba_dir = REPO / "results/proba/e4"
if proba_dir.exists():
    from collections import defaultdict as _dd
    ece = _dd(list)
    for f in sorted(proba_dir.glob("*.npz")):
        rest, budget, _seed = f.stem.rsplit("_", 2)
        model, strategy = rest.rsplit("_", 1)
        d = np.load(f)
        pr, yy = d["proba"][:, 1].astype(float), d["y_true"].astype(float)
        ece[(model, strategy, int(budget))].append(_ece(pr, yy, _weights(yy)))
    for strategy in ("split", "cross"):
        for budget in (100, 200):
            t = ece.get(("tabpfn", strategy, budget))
            g = ece.get(("lightgbm", strategy, budget))
            if not (t and g):
                continue
            tm, gm = float(np.mean(t)), float(np.mean(g))
            row = (rf"\| {strategy} \| {budget} \| \*\*([\d.]+)\*\* \| ([\d.]+) \| "
                   rf"\*\*(\d+)%\*\* \|")
            m = re.search(row, README)
            check(f"ECE {strategy}@{budget} TabPFN",
                  float(m.group(1)) if m else None, tm, 0.0001)
            check(f"ECE {strategy}@{budget} LightGBM",
                  float(m.group(2)) if m else None, gm, 0.0001)
            check(f"ECE {strategy}@{budget} reduction %",
                  float(m.group(3)) if m else None, round(100 * (1 - tm / gm)), 0.5)

# ---- 5c. E4 "narrower by", and the E5 scale + cache numbers ---------------
if e4:
    pct = []
    for strategy in ("split", "cross"):
        for budget in (100, 200):
            t = e4.get((f"tabpfn_{strategy}", budget))
            g = e4.get((f"lightgbm_{strategy}", budget))
            if not (t and g):
                continue
            tw, gw = float(np.mean(t["width"])), float(np.mean(g["width"]))
            pct.append(100 * (1 - tw / gw))
    if pct:
        check("E4 narrower-by low", in_readme(r"([\d.]+)–[\d.]+% narrower"), min(pct), 0.05)
        check("E4 narrower-by high", in_readme(r"[\d.]+–([\d.]+)% narrower"), max(pct), 0.05)
        check("E4 comparisons won",
              in_readme(r"narrower, (\d+) of \d+ comparisons"),
              float(sum(x > 0 for x in pct)), 0.5)

e5 = load("e5.jsonl")
if e5:
    # Slope of set size against log10 context, split only -- the arms are not
    # replicated equally (cross stops at 25k), so mixing them would repeat the
    # analyze_e3 mistake.
    from collections import defaultdict as _dd2
    by_ctx = _dd2(list)
    for r in e5:
        a = r.get("alphas", {}).get("0.05")
        if a and r["strategy"] == "split" and not r["cache"]:
            by_ctx[r["n_context"]].append(a["set_size"])
    if len(by_ctx) > 2:
        xs = np.array(sorted(by_ctx), dtype=float)
        ys = np.array([np.mean(by_ctx[x]) for x in sorted(by_ctx)])
        sd = float(np.mean([np.std(by_ctx[x], ddof=1) for x in sorted(by_ctx)
                            if len(by_ctx[x]) > 1]))
        slope = float(np.polyfit(np.log10(xs), ys, 1)[0])
        check("E5 slope",
              in_readme(r"Slope (\u2212?[\d.]+) set size per 10\u00d7 context"),
              slope, 0.0006)
        check("E5 seed SD", in_readme(r"seed standard deviation\nof ([\d.]+)"), sd, 0.0006)

    # The README's cache table: predict and fit, cached and not, per context.
    _cached = {r["n_context"]: r for r in e5 if r["cache"]}
    _plain = {r["n_context"]: r for r in e5 if not r["cache"] and r["seed"] == 0
              and r["strategy"] == "split"}
    for _ctx in sorted(_cached):
        if _ctx not in _plain:
            continue
        c, u = _cached[_ctx], _plain[_ctx]
        B = r"\*{0,2}"   # the table bolds whichever cell is the headline
        row = (rf"\| {_ctx // 1000},{_ctx % 1000:03d} \| {B}([\d.]+) s{B} \| {B}([\d.]+) s{B} "
               rf"\| {B}([\d.]+)×{B} \| {B}([\d.]+) s{B} \| {B}([\d.]+) s{B} \|")
        m = re.search(row, README)
        check(f"E5 cache {_ctx} predict uncached",
              float(m.group(1)) if m else None, u["predict_seconds"], 0.05)
        check(f"E5 cache {_ctx} predict cached",
              float(m.group(2)) if m else None, c["predict_seconds"], 0.05)
        check(f"E5 cache {_ctx} speedup",
              float(m.group(3)) if m else None,
              u["predict_seconds"] / c["predict_seconds"], 0.05)
        check(f"E5 cache {_ctx} fit uncached",
              float(m.group(4)) if m else None, u["fit_seconds"], 0.05)
        check(f"E5 cache {_ctx} fit cached",
              float(m.group(5)) if m else None, c["fit_seconds"], 0.05)
        # The "not bit-identical" claim, recomputed from the saved probabilities.
        try:
            _a = np.load(REPO / u["proba_file"])["proba"].astype(float)
            _b = np.load(REPO / c["proba_file"])["proba"].astype(float)
        except (KeyError, OSError):
            _a = _b = None
        if _a is not None and _a.shape == _b.shape:
            _dp = float(np.abs(_a - _b).max())
            _m2 = re.search(rf"\| {_ctx // 1000},{_ctx % 1000:03d} \|(?:[^|]*\|){{5}} "
                            rf"([\d.]+e-\d+) \|", README)
            check(f"E5 cache {_ctx} max dp",
                  float(_m2.group(1)) if _m2 else None, _dp, 5e-6)

    cached = {r["n_context"]: r for r in e5 if r["cache"]}
    plain = {r["n_context"]: r for r in e5 if not r["cache"] and r["seed"] == 0
             and r["strategy"] == "split"}
    big = max(cached) if cached else None
    if big and big in plain:
        speed = plain[big]["predict_seconds"] / cached[big]["predict_seconds"]
        check("E5 cache speedup", in_readme(r"\*\*([\d.]+)× faster\*\* at 200k"), speed, 0.05)
        check("E5 cache uncached s", in_readme(r"([\d.]+) s → [\d.]+ s"),
              plain[big]["predict_seconds"], 0.05)
        check("E5 cache cached s", in_readme(r"[\d.]+ s → ([\d.]+) s"),
              cached[big]["predict_seconds"], 0.05)

# ---- 5d. E6: the paired replication count ---------------------------------
# The README carried "0 of 8" in two places while its own detail table said 9.
# Mirrors analyze_e6: the Base dataset comes from E1, the variants from E6, at
# alpha = 0.1, with a two-sided t against the small-n critical value.
_paired: dict = defaultdict(dict)
for _path, _vkey in (("e1.jsonl", None), ("e6.jsonl", "variant")):
    for r in load(_path):
        a = r.get("alphas", {}).get("0.1")
        if a is None:
            continue
        v = r[_vkey] if _vkey else "Base"
        _paired[(v, r["n_cal_fraud"])].setdefault(r["strategy"], {})[r["seed"]] = a["set_size"]

_CRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
_tally = {"narrower": 0, "tie": 0, "wider": 0}
for pr in _paired.values():
    seeds = sorted(set(pr.get("split", {})) & set(pr.get("cross", {})))
    if len(seeds) < 2:
        continue
    d = np.array([pr["split"][s] - pr["cross"][s] for s in seeds], dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = d.mean() / se if se > 0 else np.inf
    if abs(t) < _CRIT.get(len(d), 2.0):
        _tally["tie"] += 1
    else:
        _tally["narrower" if d.mean() > 0 else "wider"] += 1

_total = sum(_tally.values())
if _total:
    check("E6 paired comparisons", in_readme(r"0 of (\d+) paired tests"), float(_total), 0.5)
    check("E6 detail-table count", in_readme(r"wider in 0 of (\d+)\."), float(_total), 0.5)
    checks.append(("E6 significantly wider is zero", _tally["wider"] == 0,
                   f'computed {_tally["wider"]} significantly wider, README says 0'))
    check("E6 significantly narrower",
          in_readme(r"Significantly narrower in (\d+)\*\*"), float(_tally["narrower"]), 0.5)

# ---- 5e. The multiclass figures the README attributes to the test suite ---
# The README says tests/test_multiclass.py "pins this"; make that literally so.
_mc = REPO / "tests/test_multiclass.py"
if _mc.exists():
    # Match the value inside a pytest.approx(...) assertion, not anywhere in the
    # file: a bare substring search passed on 0.700 because the fixture weights
    # line contains "0.7".
    _asserted = {float(m) for m in re.findall(r"pytest\.approx\(([\d.]+)", _mc.read_text())}
    for _label, _pat in (
        ("multiclass marginal worst",
         r"marginal conformal leaves the worst class at \*\*([\d.]+)\*\*"),
        ("multiclass mondrian worst", r"Mondrian holds \*\*([\d.]+)\*\*"),
    ):
        _claimed = in_readme(_pat)
        checks.append((
            _label,
            _claimed is not None and any(abs(_claimed - a) < 1e-9 for a in _asserted),
            f"README says {_claimed}; tests/test_multiclass.py asserts "
            f"{sorted(_asserted)} — the README says this file pins it",
        ))

# ---- 5f. The extensions payload is the code we actually tested --------------
# contrib/ is generated from src/ and ships to another repository. The payload
# carries 18 tests; the full suite runs against src/, so the two must be the
# same code or the PR is backed by a suite that never saw it.
def _logic(path, rename=False):
    import ast
    src = path.read_text()
    if rename:
        src = src.replace("tabpfn_conformal", "tabpfn_extensions.conformal")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
               and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(tree)


_payload = REPO / "contrib/tabpfn-extensions/src/tabpfn_extensions/conformal"
if _payload.exists():
    _drifted = []
    for _mod in sorted(REPO.glob("src/tabpfn_conformal/*.py")):
        if _mod.name == "__init__.py":
            continue
        _ship = _payload / _mod.name
        if not _ship.exists():
            _drifted.append(f"{_mod.name} missing from payload")
        elif _logic(_mod, rename=True) != _logic(_ship):
            _drifted.append(f"{_mod.name} logic differs")
    checks.append(("extensions payload matches src", not _drifted,
                   "; ".join(_drifted) + " — run scripts/build_extension_pr.py"))

# ---- 5g. The K-times cost correction, from committed quotes ---------------
# This is the README's own correction of an earlier overclaim, and it had no
# artifact behind it until experiments/api/cost_kfold.py was written.
_cost = REPO / "results/cost_kfold.json"
if _cost.exists():
    _rows = json.loads(_cost.read_text())["rows"]
    _exact = [r for r in _rows if abs(r["ratio"] - r["k"]) < 0.05]
    checks.append(("cost ratio equals K in every quote", len(_exact) == len(_rows),
                   f"{len(_exact)} of {len(_rows)} quotes have ratio == K"))
    for _k in (2, 20):
        _got = [r["ratio"] for r in _rows if r["k"] == _k]
        if _got:
            check(f"README cost ratio at K={_k}",
                  in_readme(rf"([\d.]+)× at K={_k}"), float(_got[0]), 0.05)

# ---- 5h. E2: the permutation test behind "don't split it" -----------------
# Mirrors analyze_e2 exactly: feasible settings only, aligned on SEED (E2 was
# resumed, so two settings sit in a rotated order and pairing by list position
# differences the wrong seeds), 20,000 permutations from default_rng(0).
e2 = load("e2.jsonl")
if e2:
    from collections import defaultdict as _dd4
    _cells = _dd4(lambda: _dd4(dict))
    _feas = _dd4(lambda: _dd4(list))
    _cross = _dd4(list)
    for r in e2:
        a = r.get("alphas", {}).get("0.05")
        if not a:
            continue
        if r["strategy"] == "cross":
            _cross[r["n_frauds"]].append(a["set_size"])
        elif r.get("cal_size") is not None:
            _cells[r["n_frauds"]][r["cal_size"]][r["seed"]] = a["set_size"]
            _feas[r["n_frauds"]][r["cal_size"]].append(a["feasible"])

    for _budget in (100, 200):
        sizes = sorted(c for c in _cells[_budget] if all(_feas[_budget][c]))
        if len(sizes) < 2:
            continue
        seeds = sorted(set.intersection(*(set(_cells[_budget][c]) for c in sizes)))
        W = np.array([[_cells[_budget][c][s] for s in seeds] for c in sizes])
        means = W.mean(axis=1)
        observed = float(means.max() - means.min())
        rng = np.random.default_rng(0)
        null = np.empty(20_000)
        for j in range(null.size):
            order = np.argsort(rng.random(W.shape), axis=0)
            m = np.take_along_axis(W, order, axis=0).mean(axis=1)
            null[j] = m.max() - m.min()
        pval = float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))

        row = rf"\| {_budget} frauds \| ([\d.]+) \| \*{{0,2}}([\d.]+)\*{{0,2}} \|"
        m2 = re.search(row, README)
        check(f"E2 spread @{_budget}", float(m2.group(1)) if m2 else None, observed, 0.001)
        check(f"E2 permutation p @{_budget}", float(m2.group(2)) if m2 else None, pval, 0.002)
        # The claim that actually matters: not splitting beats every split ratio.
        if _cross.get(_budget):
            cw = float(np.mean(_cross[_budget]))
            checks.append((f"E2 cross beats every split @{_budget}",
                           bool(cw < means.min()),
                           f"cross {cw:.4f} vs best split {means.min():.4f}"))

# ---- 5i. The ACI gamma sweep table ----------------------------------------
# Written from results/aci_gamma_sweep.json, which replay_aci.py regenerates
# from saved probabilities with no API calls.
_sweep = REPO / "results/aci_gamma_sweep.json"
if _sweep.exists():
    _arms = json.loads(_sweep.read_text())
    _ncal = 46
    _target = min(1.0, math.ceil((_ncal + 1) * 0.95) / _ncal)
    for _name, _label in (("frozen", "frozen"), ("gamma=0.05", r"0\.05"),
                          ("gamma=0.2", r"0\.2"), ("gamma=0.5", r"0\.5"),
                          ("gamma=1", r"1\.0")):
        _tr = _arms.get(_name)
        if not _tr:
            continue
        _cov = [t["coverage_fraud"] for t in _tr]
        _row = rf"\| {_label} \| (\d+) of \d+ \| ([\d.]+) \| ([\d.]+) \|"
        _m = re.search(_row, README)
        check(f"ACI {_name} months below",
              float(_m.group(1)) if _m else None,
              float(sum(1 for c in _cov if c < _target)), 0.5)
        check(f"ACI {_name} swing",
              float(_m.group(2)) if _m else None, float(max(_cov) - min(_cov)), 0.001)
        check(f"ACI {_name} set size",
              float(_m.group(3)) if _m else None,
              float(np.mean([t["set_size"] for t in _tr])), 0.001)

# ---- 5j. The documented repository layout matches the repository -----------
# Section 12 of the cahier des charges listed budget.py (never built), omitted
# metrics.py, and showed an experiments/kaggle/ that was an empty directory.
_plan = REPO / "docs/CAHIER-DES-CHARGES.md"
if _plan.exists():
    _txt = _plan.read_text()
    _m = re.search(r"## 12\. Repository layout.*?```\n(.*?)```", _txt, re.S)
    if _m:
        missing, stack = [], []
        for line in _m.group(1).splitlines():
            if not line.strip() or line.startswith("tabpfn-conformal/"):
                continue
            depth = (len(line) - len(line.lstrip("│ "))) // 4
            name = re.sub(r"^[│ ]*[├└]──\s*", "", line).split("#")[0].strip()
            if not name:
                continue
            name = name.rstrip("/")
            stack = stack[:depth] + [name]
            path = REPO / "/".join(stack)
            # A wildcard entry stands for a family of files.
            if "*" in name:
                if not list(path.parent.glob(name)):
                    missing.append("/".join(stack))
            elif not path.exists():
                missing.append("/".join(stack))
        checks.append(("documented layout exists", not missing,
                       "listed but absent: " + ", ".join(missing[:6])))

        # And the other direction, for the package itself: every shipped module
        # must appear in the diagram.
        listed = set(re.findall(r"([a-z_]+\.py)", _m.group(1)))
        shipped = {f.name for f in (REPO / "src/tabpfn_conformal").glob("*.py")}
        undocumented = sorted(shipped - listed)
        checks.append(("every src module is documented", not undocumented,
                       "in src/ but not in the layout: " + ", ".join(undocumented)))

# ---- 5k. The wall-clock confound is flagged on every row it applies to ----
if e4:
    _rows = load("e4.jsonl")
    _flagged = sum(1 for r in _rows if r.get("wallclock_comparable") is False)
    checks.append(("E4 wall-clock confound flagged on every row",
                   _flagged == len(_rows),
                   f"{_flagged} of {len(_rows)} rows carry wallclock_comparable: false"))
    # The count that IS hardware-independent, quoted in the README's opening.
    _fits = {r["arm"]: r.get("n_grad_fits") for r in _rows}
    check("README gradient fits, LightGBM cross",
          in_readme(r"0 gradient-trained fits against\nLightGBM's (\d+)"),
          float(_fits.get("lightgbm_cross", -1)), 0.5)
    checks.append(("TabPFN arms do zero gradient fits",
                   all(v == 0 for k, v in _fits.items() if k.startswith("tabpfn")),
                   f"{ {k: v for k, v in _fits.items() if k.startswith('tabpfn')} }"))

# ---- 5l. Package metadata agrees with pyproject ---------------------------
try:
    try:
        import tomllib                       # 3.11+
    except ModuleNotFoundError:              # 3.10, which requires-python allows
        import tomli as tomllib
    _pj = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    sys.path.insert(0, str(REPO / "src"))
    import tabpfn_conformal as _pkg
    checks.append(("package version matches pyproject",
                   _pkg.__version__ == _pj["version"],
                   f'__version__ {_pkg.__version__} vs pyproject {_pj["version"]}'))
    checks.append(("no TODO left in package metadata",
                   "TODO" not in json.dumps(_pj),
                   "pyproject [project] still contains TODO"))
except Exception as exc:                                      # pragma: no cover
    checks.append(("package metadata", None, f"not checkable: {exc}"))

# ---- 5m. Every command the README tells a reader to run must exist ---------
# The README documented the experiment path without the data download step for
# most of the project; the script it never named is the one that fetches a
# dataset the repository cannot ship.
_cmds = set(re.findall(r"python (scripts/[\w/]+\.py|experiments/[\w/]+\.py)", README))
_absent = sorted(c for c in _cmds if not (REPO / c).exists())
checks.append(("every script the README runs exists", not _absent,
               "named but absent: " + ", ".join(_absent)))
checks.append(("README documents the data download",
               "scripts/download_data.py" in _cmds,
               "the experiments need data/ and the README never fetches it"))

# The README says every experiment takes --dry-run; that is a promise about
# spending money, so it is checked rather than trusted.
_runners = sorted((REPO / "experiments/api").glob("e[0-9]_*.py"))
_no_dry = [f.name for f in _runners if '"--dry-run"' not in f.read_text()]
checks.append(("every experiment supports --dry-run", not _no_dry,
               "missing --dry-run: " + ", ".join(_no_dry)))

# ---- 5n. The drift figure draws the certified level, not the nominal one ---
# Both E3 figures drew their target at 1-alpha = 0.95. Every point in both sat
# above it, so the figure read "the guarantee always holds" directly above a
# table saying base misses it in 4 months of 5.
_e3src = (REPO / "experiments/analyze_e3.py").read_text()
checks.append(("E3 figure targets the certified level",
               "ax_c.axhline(effective" in _e3src
               and "ax_c.axhline(1 - alpha" not in _e3src,
               "analyze_e3 draws the coverage target at the nominal 1-alpha"))

# E1 and E2 draw the certified level as a curve, because it moves with n_cal;
# the nominal line stays only as a faint reference and must not be labelled
# "target". Both figures once were, and both read as passing when they were not.
for _name, _src in (("E1", "analyze_e1.py"), ("E2", "analyze_e2.py")):
    _txt = (REPO / "experiments" / _src).read_text()
    checks.append((f"{_name} figure does not label the nominal as the target",
                   'f"target {' not in _txt and '"target ' not in _txt,
                   f"{_src} still labels a flat line 'target'"))

# And every figure the README shows must exist.
_figs = set(re.findall(r"\(figures/([\w.]+\.png)\)", README))
_gone = sorted(f for f in _figs if not (REPO / "figures" / f).exists())
checks.append(("every figure the README shows exists", not _gone,
               "referenced but absent: " + ", ".join(_gone)))

# ---- 5o. The extensions payload is PR-ready ------------------------------
# Their check-changelog workflow fails any PR without a towncrier fragment, and
# their CI runs `pytest tests/`, so the payload has to carry its own tests.
_pay = REPO / "contrib/tabpfn-extensions"
if _pay.exists():
    _frag = sorted((_pay / "changelog").glob("*.added.md")) if (_pay / "changelog").exists() else []
    checks.append(("extensions payload has a changelog fragment", bool(_frag),
                   "PriorLabs/tabpfn-extensions fails any PR without changelog/<PR>.<type>.md"))
    _tf = _pay / "tests/test_conformal.py"
    if _tf.exists():
        _n = len(re.findall(r"^def test_", _tf.read_text(), re.M))
        check("PR.md states the payload test count",
              float(re.search(r"\*\*Tests:\*\* (\d+) tests",
                              (_pay / "PR.md").read_text()).group(1))
              if re.search(r"\*\*Tests:\*\* (\d+) tests", (_pay / "PR.md").read_text())
              else None, float(_n), 0.5)

# ---- 5p. The TabPFN-3.5 capability table names things the code really uses --
# This is the section that answers the "showcase" criterion, so it must not
# claim a capability the experiments never touch.
_api = " ".join(f.read_text() for f in sorted((REPO / "experiments").rglob("*.py")))
_CAPABILITIES = {
    "thinking_mode": "thinking_mode",
    "time_col": 'time_col',
    "fit_with_cache": 'fit_mode="fit_with_cache"',
    "balance_probabilities": "balance_probabilities",
    "estimate_cost": "estimate_cost",
    "local weights": "from tabpfn import TabPFNClassifier",
}
_unused = sorted(k for k, needle in _CAPABILITIES.items() if needle not in _api)
checks.append(("README capability table matches the code", not _unused,
               "claimed in the README, absent from experiments/: "
               + ", ".join(_unused)))

# The 422 quoted in that section has to be the one the server actually returned.
_spike = REPO / "results/spike_s1.json"
if _spike.exists():
    _txt = _spike.read_text()
    checks.append(("the quoted HTTP 422 is the recorded one",
                   "FIT_WITH_CACHE fit mode is not compatible with thinking mode" in _txt
                   and "FIT_WITH_CACHE fit mode is not compatible with thinking mode" in README,
                   "README quotes a server error not present in results/spike_s1.json"))

# ---- 5q. PEP 561 marker actually ships ------------------------------------
# The package is fully annotated; without py.typed in the wheel, type checkers
# treat it as untyped and downstream users get nothing from the hints.
checks.append(("py.typed exists", (REPO / "src/tabpfn_conformal/py.typed").exists(),
               "src/tabpfn_conformal/py.typed is missing"))
checks.append(("py.typed is declared to the build backend",
               "py.typed" in (REPO / "pyproject.toml").read_text(),
               "pyproject does not list py.typed, so the wheel may drop it"))

# ---- 5r. method.md agrees with the results it describes -------------------
# It is a judged deliverable and had drifted: it said two predictions were
# falsified (four were) and listed seeds for three of the six experiments.
_method = (REPO / "docs/method.md").read_text()
for _name, _pat, _want in (
    ("method.md falsification count", r"\*\*Four of the five were falsified\*\*", True),
    ("method.md records the ACI verdict", r"cannot help at this label budget", True),
    ("method.md notes the matched-comparison asymmetry", r"twice the in-context rows", True),
):
    checks.append((_name, bool(re.search(_pat, _method)) == _want,
                   f"docs/method.md is missing: {_pat}"))
# Seeds per experiment, straight from the results files.
for _e in (1, 2, 3, 4, 5, 6):
    _rows = load(f"e{_e}.jsonl")
    if _rows:
        _n = len({r.get("seed") for r in _rows})
        _claim = 5 if _e in (1, 2) else 3
        checks.append((f"method.md seed count for E{_e}", _n == _claim,
                       f"E{_e} has {_n} seeds; method.md says {_claim}"))

# ---- 5s. SUBMISSION.md agrees with the README ------------------------------
# It is the text that actually gets submitted, and it drifted twice: "five
# experiments" when there are six, and "identical prediction sets" for the KV
# cache after the README had been corrected to "same answer to four decimals".
_sub = (REPO / "docs/SUBMISSION.md").read_text()
_n_experiments = len([f for f in (REPO / "results").glob("e[0-9].jsonl")])
_m = re.search(r"plus (\w+) experiments", _sub)
check("SUBMISSION experiment count",
      float({"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}.get(_m.group(1), -1)) if _m else None,
      float(_n_experiments), 0.5)

# Headline figures, compared as literal strings so a different sentence shape in
# either document cannot make the check pass or fail for the wrong reason.
for _label, _literal in (
    ("narrower-by range", "6.9\u201312.4%"),
    ("ECE reduction", "74\u201386%"),
    ("drift, base", "9 of 15"),
    ("drift, Thinking", "3 of 15"),
    ("paired drift gap", "2.0 \u00b1 1.2 months"),
):
    checks.append((f"SUBMISSION and README agree: {_label}",
                   _literal in README and _literal in _sub,
                   f"{_literal!r} in README={_literal in README}, "
                   f"in SUBMISSION={_literal in _sub}"))

# Specific to the KV cache: the phrase "identical prediction sets" is legitimate
# elsewhere (the MAPIE split comparison really is exact), so match the sentence
# that would be wrong rather than the words.
_cache_claim = re.search(r"KV cache[\s\S]{0,160}?identical (?:prediction )?sets",
                         _sub)
checks.append(("SUBMISSION does not claim identical cache sets",
               _cache_claim is None,
               "SUBMISSION.md says the KV cache gives identical sets; it does not"))

# ---- 5t. The handoff doc counts what the video script actually lists -------
_status = (REPO / "docs/STATUS.md").read_text()
_video = (REPO / "docs/VIDEO.md").read_text()
_n_donts = len(re.findall(r"^- (?:Do \*\*not\*\*|Only say)", _video, re.M))
_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
checks.append(("STATUS counts the do-not-say list correctly",
               f"{_words.get(_n_donts, _n_donts)} claims not to make" in _status,
               f"VIDEO.md lists {_n_donts}; STATUS.md says otherwise"))
checks.append(("STATUS names the changelog rename step",
               "PRNUMBER.added.md" in _status,
               "their CI fails a PR without the towncrier fragment, and STATUS "
               "does not say to rename it"))

# ---- 5u. E7 and the measured cost of approximate validity ------------------
# The headline was qualified on the strength of these, so they are recomputed
# here rather than trusted. Seed is the unit: within a seed the split and cross
# arms share an evaluation set, so the runs are not independent.
def _gaps(rows, strategy, alpha):
    from collections import defaultdict as _dd
    per = _dd(list)
    for r in rows:
        if r["strategy"] != strategy:
            continue
        a, n = r["alphas"].get(alpha), r.get("n_cal_fraud")
        if not a or not n:
            continue
        cert = min(1.0, math.ceil((n + 1) * (1 - float(alpha))) / n)
        per[r["seed"]].append(a["coverage_fraud"] - cert)
    return np.array([np.mean(per[s]) for s in sorted(per)])


_CRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
_below = {"split": 0, "cross": 0}
_cells = 0
for _f, _ in (("e1.jsonl", "BAF"), ("e7.jsonl", "covtype")):
    _rows = load(_f)
    if not _rows:
        continue
    for _a in ("0.05", "0.1", "0.2"):
        for _st in ("split", "cross"):
            _g = _gaps(_rows, _st, _a)
            if len(_g) < 2:
                continue
            _se = float(_g.std(ddof=1) / np.sqrt(len(_g)))
            _t = _g.mean() / _se if _se else 0.0
            if _t < -_CRIT.get(len(_g), 2.0):
                _below[_st] += 1
            if _st == "cross":
                _cells += 1

if _cells:
    check("README: cross below its certified level, count",
          in_readme(r"Cross is\nbelow in (\d+) of 6"), float(_below["cross"]), 0.5)
    check("README: split below its certified level, count",
          in_readme(r"below its certified level in (\d+) of 6"),
          float(_below["split"]), 0.5)

# E7 exists, is the second domain, and its paired comparison is what we claim.
_e7 = load("e7.jsonl")
if _e7:
    checks.append(("E7 ran the full grid", len(_e7) == 24,
                   f"{len(_e7)} rows, expected 24"))
    checks.append(("E7 is a different dataset from BAF",
                   all(r.get("dataset") == "covtype" for r in _e7),
                   "E7 rows are not tagged covtype"))
    # cross significantly wider in zero matched comparisons
    from collections import defaultdict as _dd7
    _pair = _dd7(dict)
    for r in _e7:
        a = r["alphas"].get("0.1")
        if a:
            _pair[r["n_cal_fraud"]].setdefault(r["strategy"], {})[r["seed"]] = a["set_size"]
    _wider = 0
    for _k, _pr in _pair.items():
        _s = sorted(set(_pr.get("split", {})) & set(_pr.get("cross", {})))
        if len(_s) < 2:
            continue
        _d = np.array([_pr["split"][x] - _pr["cross"][x] for x in _s], dtype=float)
        _se = float(_d.std(ddof=1) / np.sqrt(len(_d)))
        if _se and _d.mean() / _se < -_CRIT.get(len(_d), 2.0):
            _wider += 1
    checks.append(("E7: cross significantly wider in zero comparisons", _wider == 0,
                   f"cross is significantly wider in {_wider} matched comparisons"))

# ---- 6. Test count --------------------------------------------------------
import subprocess
out = subprocess.run([sys.executable, "-m", "pytest", "-q",
                      "--collect-only"], cwd=REPO, capture_output=True, text=True)
m = re.search(r"(\d+) tests? collected", out.stdout)
if m:
    check("test count", in_readme(r"(\d+) tests, CPU"), float(m.group(1)), 0.5)

# ---- 6a. No committed figure predates the script that draws it ------------
# Two figures shipped for days showing a flat dashed line labelled "target 90%"
# after their scripts had been fixed to draw the level each run actually
# certifies -- which at alpha=0.1 moves with n_cal, so a flat line was wrong at
# every point by a different amount, and every marker sat above it. The scripts
# were right; nobody re-ran them. Byte-comparing figures in CI is hopeless
# across fonts and matplotlib versions, but matplotlib stamps <dc:date> into
# every SVG, and git knows when each script last changed. That comparison is
# exact and platform-independent.
import datetime
_fig_dir = REPO / "figures"
if _fig_dir.exists():
    _stale, _unknown = [], []
    for _svg in sorted(_fig_dir.glob("*.svg")):
        _m = re.match(r"(e\d+)_", _svg.name)
        if not _m:
            continue
        _script = REPO / "experiments" / f"analyze_{_m.group(1)}.py"
        if not _script.exists():
            continue
        _d = re.search(r"<dc:date>([0-9T:.\-]+)</dc:date>", _svg.read_text(errors="ignore"))
        if not _d:
            continue
        _drawn = datetime.datetime.fromisoformat(_d.group(1))
        _out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(_script.relative_to(REPO))],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        if not _out:
            # A shallow clone has no history to ask; say so rather than pass.
            _unknown.append(_svg.name)
            continue
        _changed = datetime.datetime.fromisoformat(_out).astimezone().replace(tzinfo=None)
        if _drawn < _changed:
            _stale.append(f"{_svg.name} drawn {_drawn:%Y-%m-%d %H:%M} but "
                          f"{_script.name} changed {_changed:%Y-%m-%d %H:%M}")
    if _unknown:
        checks.append(("no figure predates its generator", None,
                       f"no git history for {len(_unknown)} figures (shallow clone?)"))
    else:
        checks.append(("no figure predates its generator", not _stale,
                       "; ".join(_stale) or "all figures postdate their script"))

# ---- 6b. The vendored payload still carries its caveat --------------------
# build_extension_pr.py replaces every module docstring with a vendoring header,
# which silently deleted the approximate-validity caveat from the module Prior
# Labs would merge -- while PR.md told them the module "says so where it
# matters". Anyone reading strategy="cross" upstream would have seen no warning.
_vend = REPO / "contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/crossconformal.py"
if _vend.exists():
    _v = _vend.read_text()
    checks.append(("vendored cross-conformal keeps the validity caveat",
                   all(t in _v for t in ("approximate", "Vovk 2015", "3 of 6", "0 of 6")),
                   "the caveat did not survive vendoring; PR.md claims it does"))
    _pr = (REPO / "contrib/tabpfn-extensions/PR.md").read_text()
    checks.append(("the PR description states the measured cost, not just the theory",
                   "3 of 6" in _pr and "0 of 6" in _pr,
                   "PR.md describes approximate validity without the measured number"))

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
    # The page compares against the level actually targeted, not the nominal one.
    n_f = len(committed["cal_fraud"])
    eff = 100 * min(1.0, math.ceil((n_f + 1) * 0.95) / n_f)
    check("video demo coverage", in_video(r"Coverage sits at \*\*([\d.]+)% against"), cov, 0.05)
    check("video demo target", in_video(r"against a ([\d.]+)% target"), eff, 0.05)
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
