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


# The plan, the handoff, the video script and the submission text are working
# material and live in private/, which is gitignored: they carry local paths,
# API budget, milestone dates and instructions to the author. Their checks run
# when the files are there and are skipped entirely when they are not, so a
# clone of the public repository verifies everything it can actually see.
PRIVATE = REPO / "private"


def private_doc(name):
    p = PRIVATE / name
    return p.read_text() if p.exists() else None


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
    # bool() is not decoration. `actual` is often a numpy scalar, which makes
    # the comparison a numpy bool, and `np.False_ is False` is False, so a
    # failing check printed FAIL and was then not counted as one. The script
    # exited 0 with a visible failure on screen.
    ok = bool(claimed is not None and abs(claimed - actual) <= tol)
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

        # ...and every cell of the table above that sentence. The slope was
        # checked from the start; the rows were not, so a cell could read
        # anything at all. It also pins the row count: the table once showed
        # three of the five contexts that were run, and the three it showed
        # were the monotone ones.
        _sc = re.search(
            r"\| context \| context fraud rate \| set size \|\n\|[-: |]+\|\n((?:\|.*\|\n)+)",
            README)
        if _sc:
            _srows = [[c.strip() for c in ln.strip().strip("|").split("|")]
                      for ln in _sc.group(1).strip().split("\n")]
            checks.append(("E5 scale table lists every context run",
                           len(_srows) == len(by_ctx),
                           f"table has {len(_srows)} rows, results have "
                           f"{len(by_ctx)} contexts"))
            for _r in _srows:
                _ctxn = int(_r[0].replace(",", ""))
                _vals = by_ctx.get(_ctxn)
                if not _vals:
                    checks.append((f"E5 scale row {_r[0]}", None, "no such context in results"))
                    continue
                check(f"E5 scale set size @{_r[0]}", float(_r[2]), float(np.mean(_vals)), 0.0005)
                _fr = [x["fraud_rate_context"] for x in e5
                       if x["n_context"] == _ctxn and x["strategy"] == "split"
                       and not x["cache"]]
                check(f"E5 scale fraud rate @{_r[0]}", float(_r[1].rstrip("%")),
                      100 * float(np.mean(_fr)), 0.006)

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

    # Every cell of the detail table, not just the counts drawn from it. Those
    # eighteen numbers were the largest block in the README that nothing
    # recomputed, and the table does not carry its own alpha, so reproducing it
    # by hand means guessing the sign convention and the level. Pinned here.
    _e6_rows = re.findall(
        r"^\| (Base|Variant [IV]+) \| (\d+) \| ([+\u2212-][\d.]+) ± ([\d.]+) \(n=(\d+)\)",
        README, re.M)
    checks.append(("the E6 table has all nine rows", len(_e6_rows) == 9,
                   f"parsed {len(_e6_rows)} rows from the README table"))
    for _v, _nc, _m, _se, _n in _e6_rows:
        pr = _paired.get((_v, int(_nc)), {})
        seeds = sorted(set(pr.get("split", {})) & set(pr.get("cross", {})))
        if len(seeds) < 2:
            checks.append((f"E6 table {_v}@{_nc}", None, "no paired seeds for this cell"))
            continue
        d = np.array([pr["split"][s] - pr["cross"][s] for s in seeds], dtype=float)
        check(f"E6 table {_v}@{_nc} mean", float(_m.replace("\u2212", "-")),
              float(d.mean()), 0.001)
        check(f"E6 table {_v}@{_nc} stderr", float(_se),
              float(d.std(ddof=1) / np.sqrt(len(d))), 0.001)
        check(f"E6 table {_v}@{_nc} n", float(_n), float(len(d)), 0.5)

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
            f"{sorted(_asserted)}; the README says this file pins it",
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
                   "; ".join(_drifted) + "; run scripts/build_extension_pr.py"))

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
_plan_txt = private_doc("CAHIER-DES-CHARGES.md")
if _plan_txt is not None:
    _txt = _plan_txt
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
    _fits_all = [(r["arm"], r.get("n_grad_fits")) for r in _rows]
    check("README gradient fits, LightGBM cross",
          in_readme(r"0 gradient-trained fits against\nLightGBM's (\d+)"),
          float(_fits.get("lightgbm_cross", -1)), 0.5)
    _bad_fits = [(k, v) for k, v in _fits_all if k.startswith("tabpfn") and v != 0]
    checks.append(("TabPFN arms do zero gradient fits",
                   not _bad_fits,
                   f"{len(_bad_fits)} of {sum(1 for k, _ in _fits_all if k.startswith('tabpfn'))} "
                   f"tabpfn rows report a gradient fit: {_bad_fits[:4]}"))

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

    # The README claims the payload passes their pre-commit unmodified. Their
    # mypy hook runs isolated (mypy plus types-requests, no numpy), and their
    # config sets ignore_missing_imports with warn_return_any, so numpy is Any
    # and `return np.inf` from a function annotated -> float is an error. Their
    # own 42 modules have none of these; three of ours did, and nothing here
    # would have noticed, because this repository's CI runs no type checker.
    import ast as _ast
    _anyret = []
    for _mpf in sorted((_pay / "src").rglob("*.py")):
        for _fn in _ast.walk(_ast.parse(_mpf.read_text())):
            if not isinstance(_fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if not (isinstance(_fn.returns, _ast.Name) and _fn.returns.id == "float"):
                continue
            for _r in _ast.walk(_fn):
                if not isinstance(_r, _ast.Return) or _r.value is None:
                    continue
                _val = _r.value.operand if isinstance(_r.value, _ast.UnaryOp) else _r.value
                if (isinstance(_val, _ast.Attribute)
                        and isinstance(_val.value, _ast.Name)
                        and _val.value.id == "np"):
                    _anyret.append(f"{_mpf.name}:{_r.lineno} in {_fn.name}()")
    checks.append(("payload declares float but returns a bare numpy scalar",
                   not _anyret,
                   "their mypy warn_return_any rejects: " + ", ".join(_anyret)))
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
_sub = private_doc("SUBMISSION.md")
if _sub is not None:
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
_status = private_doc("STATUS.md")
_video = private_doc("VIDEO.md")
if _status is not None and _video is not None:
    _n_donts = len(re.findall(r"^- (?:Do \*\*not\*\*|Only say)", _video, re.M))
    # Spelled out to twenty. A short map here has silently broken this check
    # twice, once missing "seven" and once missing "nine", each time reading as
    # a mismatch in the document rather than a gap in the map.
    _words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
              7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
              12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
              16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
              20: "twenty"}
    assert _n_donts in _words, f"extend _words: the list now has {_n_donts} items"
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
else:
    # Say so rather than disappearing. A check that drops itself when its tool
    # is missing takes the total down with it, and the count self-check below
    # then reports a puzzling off-by-one instead of the actual cause.
    checks.append(("test count", None,
                   f"pytest collected nothing (exit {out.returncode}); "
                   "the package is probably not installed in this interpreter"))

# ---- 5n. P5, settled on fair hardware -------------------------------------
# Every number the README and limitations.md quote about the T4 run is
# recomputed here from the 24 committed rows, so none of them is typed.
_kag = REPO / "results" / "kaggle_wallclock.json"
if _kag.exists():
    _blob = json.loads(_kag.read_text())
    _krows = _blob["rows"]
    checks.append(("the fair-hardware run is 24 rows on a GPU",
                   len(_krows) == 24 and _blob["device"]["device"] == "cuda"
                   and all(r["wallclock_comparable"] for r in _krows),
                   f"{len(_krows)} rows on {_blob['device'].get('gpu')}"))

    def _cell(fam, strat, budget):
        return {r["seed"]: r for r in _krows if r["family"] == fam
                and r["strategy"] == strat and r["n_frauds"] == budget}

    _diffs, _ratios = [], []
    for _strat in ("split", "cross"):
        for _b in sorted({r["n_frauds"] for r in _krows}):
            _t, _g = _cell("tabpfn", _strat, _b), _cell("lightgbm", _strat, _b)
            _seeds = sorted(set(_t) & set(_g))
            _diffs += [_t[s]["seconds"] - _g[s]["seconds"] for s in _seeds]
            _ratios.append(np.mean([_t[s]["seconds"] for s in _seeds])
                           / np.mean([_g[s]["seconds"] for s in _seeds]))
    checks.append(("TabPFN is slower in every fair-hardware configuration",
                   all(d > 0 for d in _diffs),
                   f"{sum(d > 0 for d in _diffs)} of {len(_diffs)} runs slower"))

    # "by 22 s to 221 s, a factor of 35 to 73"
    _gaps = []
    for _strat in ("split", "cross"):
        for _b in sorted({r["n_frauds"] for r in _krows}):
            _t, _g = _cell("tabpfn", _strat, _b), _cell("lightgbm", _strat, _b)
            _seeds = sorted(set(_t) & set(_g))
            _gaps.append(np.mean([_t[s]["seconds"] - _g[s]["seconds"] for s in _seeds]))
    check("P5 smallest gap in seconds",
          in_readme(r"by (\d+) s to \d+ s, a factor"), min(_gaps), 0.5)
    check("P5 largest gap in seconds",
          in_readme(r"by \d+ s to (\d+) s, a factor"), max(_gaps), 0.5)
    check("P5 smallest speed factor",
          in_readme(r"a factor of (\d+) to \d+"), min(_ratios), 0.5)
    check("P5 largest speed factor",
          in_readme(r"a factor of \d+ to (\d+)"), max(_ratios), 0.5)

    # The split-to-cross multiplier, the one effect that survives.
    for _fam, _pat in (("tabpfn", r"costs TabPFN\n\*\*([\d.]+)×\*\*"),
                       ("lightgbm", r"and LightGBM \*\*([\d.]+)×\*\*")):
        _m = np.mean([np.mean([_cell(_fam, "cross", _b)[s]["seconds"] for s in (0, 1, 2)])
                      / np.mean([_cell(_fam, "split", _b)[s]["seconds"] for s in (0, 1, 2)])
                      for _b in sorted({r["n_frauds"] for r in _krows})])
        check(f"{_fam} split-to-cross multiplier", in_readme(_pat), _m, 0.01)

    _cr = _cell("tabpfn", "cross", 200); _lg = _cell("lightgbm", "cross", 200)
    check("TabPFN cross at 200 frauds, seconds",
          in_readme(r"([\d.]+) s for cross-conformal at 200 confirmed frauds"),
          float(np.mean([_cr[s]["seconds"] for s in sorted(_cr)])), 0.5)
    check("LightGBM cross at 200 frauds, seconds",
          in_readme(r"\*\*: ([\d.]+) s against"),
          float(np.mean([_lg[s]["seconds"] for s in sorted(_lg)])), 0.05)
else:
    checks.append(("P5 fair-hardware run", None, "results/kaggle_wallclock.json absent"))

# ---- 5m. The falsification counts in prose match the scoreboard -----------
# One sentence said "three of four" for three days after a fifth prediction was
# added and a fourth was falsified, two screens below a table saying otherwise.
# The table is the source of truth; every prose count is checked against it.
_pred_rows = re.findall(r"^\| \*\*P\d+\*\* \|.*$", README, re.M)
if _pred_rows:
    _total = len(_pred_rows)
    _false = sum("falsified" in r for r in _pred_rows)
    _words = {3: "three", 4: "four", 5: "five", 6: "six"}
    _claims = re.findall(r"(\w+) of (\w+) pre-registered predictions were falsified",
                         README, re.I)
    _wrong = [f"'{a} of {b}'" for a, b in _claims
              if a.lower() != _words.get(_false, "?") or b.lower() != _words.get(_total, "?")]
    checks.append((f"prose falsification counts match the P-table ({_false} of {_total})",
                   bool(_claims) and not _wrong,
                   "; ".join(_wrong) or "no such sentence found in the README"))

# ---- 6a. No committed figure predates the script that draws it ------------
# Two figures shipped for days showing a flat dashed line labelled "target 90%"
# after their scripts had been fixed to draw the level each run actually
# certifies, which at alpha=0.1 moves with n_cal, so a flat line was wrong at
# every point by a different amount and every marker sat above it. The scripts
# were right; nobody re-ran them. Byte-comparing figures in CI is hopeless
# across fonts and matplotlib versions, but git dates both files off one clock:
# if the script was committed after the figure, the figure was not regenerated.
# Committing a script together with its regenerated figures gives them the same
# timestamp, so that, the normal case, is not flagged.
def _last_commit(rel: str):
    out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", rel],
                         cwd=REPO, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


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
        _fig_at = _last_commit(str(_svg.relative_to(REPO)))
        _src_at = _last_commit(str(_script.relative_to(REPO)))
        if _fig_at is None or _src_at is None:
            # A shallow clone has no history to ask; say so rather than pass.
            _unknown.append(_svg.name)
        elif _src_at > _fig_at:
            _stale.append(f"{_svg.name} last committed before {_script.name}")
    if _unknown:
        checks.append(("no figure predates its generator", None,
                       f"no git history for {len(_unknown)} figures (shallow clone?)"))
    else:
        checks.append(("no figure predates its generator", not _stale,
                       "; ".join(_stale) or "every figure was committed with or after its script"))

# ---- 6c. The module README that ships upstream ----------------------------
# It is the first thing a tabpfn-extensions user reads, and nothing checked it.
# It carried "narrower prediction sets in five of six comparisons", a raw win
# count, which is exactly the framing this project retracted two screens away
# in its own README.
_mod_readme = REPO / "contrib/tabpfn-extensions/src/tabpfn_extensions/conformal/README.md"
if _mod_readme.exists():
    _mr = _mod_readme.read_text()
    _num = r"(?:\\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    _raw = re.findall(rf"narrower[^.]{{0,60}}in {_num} of {_num}", _mr, re.I)
    checks.append(("the upstream module README reports the paired test, not a win count",
                   not _raw, f"raw win count phrasing: {_raw}"))
    checks.append(("the upstream module README states the measured validity cost",
                   "3 of 6" in _mr and "0 of 6" in _mr,
                   "it describes cross-conformal without what approximate validity costs"))

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
    try:
        rebuilt = build_demo.build_data()
    except Exception as exc:                                # pragma: no cover
        rebuilt = None
        checks.append(("the demo payload can be rebuilt at all", False,
                       f"build_demo.build_data() raised {type(exc).__name__}: {exc}"))
if "rebuilt" in dir() and rebuilt is not None:
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

    VIDEO = private_doc("VIDEO.md")
    # Matched against a whitespace-flattened copy. The script is prose and gets
    # rewrapped every time it is edited, and a phrase that straddles a line
    # break stops matching a pattern written with a literal space in it. That
    # broke four of these checks at once the first time the script was
    # reflowed, which is a property of the document, not of the claim.
    VIDEO_FLAT = re.sub(r"\s+", " ", VIDEO) if VIDEO is not None else None

    def in_video(pattern):
        if VIDEO_FLAT is None:
            return None
        m = re.search(pattern, VIDEO_FLAT)
        return float(m.group(1)) if m else None

    cov, caught_lo = desk(committed, 0.05, 0)
    _, caught_hi = desk(committed, 0.05, 200)
    # The page compares against the level actually targeted, not the nominal one.
    n_f = len(committed["cal_fraud"])
    eff = 100 * min(1.0, math.ceil((n_f + 1) * 0.95) / n_f)
    if VIDEO is not None:
        check("video demo coverage", in_video(r"Coverage sits at \*\*([\d.]+)% against"), cov, 0.05)
        check("video demo target", in_video(r"against a ([\d.]+)% target"), eff, 0.05)
        check("video demo caught (K=0)",
              in_video(r"moves from \*\*(\d+)% to \d+%\*\*"), round(caught_lo), 0.5)
        check("video demo caught (K=200)",
              in_video(r"moves from \*\*\d+% to (\d+)%\*\*"), round(caught_hi), 0.5)

        # The script is read aloud, so its counts are spelled out, and the
        # digit-hunting checks above slid straight past them: it still said a
        # hundred and thirty tests when there were 132, and a hundred and
        # seventy-eight claims when there were 360. Those would have been
        # spoken on camera.
        _UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                  "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                  "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                  "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
                  "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                  "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

        def spoken(phrase):
            """Read "three hundred and sixty" or "a hundred and thirty-two"."""
            total = current = 0
            for word in re.split(r"[\s-]+", phrase.lower().strip()):
                if word in ("and", ""):
                    continue
                if word == "a":
                    current = max(current, 1)
                elif word == "hundred":
                    current = max(current, 1) * 100
                    total += current
                    current = 0
                elif word in _UNITS:
                    current += _UNITS[word]
                else:
                    return None
            return total + current

        # The submission text quotes the same count in digits. It said 130
        # while the suite had 132, and nothing was reading it.
        _SUB = private_doc("SUBMISSION.md")
        if _SUB is not None and m:
            _st = re.search(r"its (\d+) tests", _SUB)
            if _st:
                check("submission states the test count", float(_st.group(1)),
                      float(m.group(1)), 0.5)
            else:
                checks.append(("submission states the test count", False,
                               "the test count is no longer findable in "
                               "SUBMISSION.md; reword the pattern or the check "
                               "stops running"))

        _vt = re.search(r"\"([A-Za-z \-]+?) tests, CPU only", VIDEO_FLAT)
        if _vt and m:          # `m` is the pytest collection match from section 6
            checks.append(("video states the test count",
                           spoken(_vt.group(1)) == int(m.group(1)),
                           f"script says {_vt.group(1)!r} = {spoken(_vt.group(1))}, "
                           f"pytest collects {m.group(1)}"))
        # Matched loosely on purpose: the sentence around this number gets
        # rewritten, and a tight phrase match means the check disappears with
        # it. That happened once already, the first time this closing section
        # was reworded, so a miss is now recorded as a failure rather than
        # being silently skipped.
        _vc = re.search(r"([A-Za-z][A-Za-z \-]*?hundred[A-Za-z \-]*?) of them", VIDEO_FLAT)
        _rc = re.search(r">\s*(\d+) of them from `results/`", README)
        if _vc and _rc:
            checks.append(("video and README quote the same claim count",
                           spoken(_vc.group(1)) == int(_rc.group(1)),
                           f"script says {_vc.group(1)!r} = {spoken(_vc.group(1))}, "
                           f"README says {_rc.group(1)}"))
        else:
            checks.append(("video and README quote the same claim count", False,
                           "the spoken claim count is no longer findable in the "
                           "script; reword the pattern or the check stops running"))

# ---- 7b. The number of experiments the README claims ----------------------
# It said six for the three days after E7 landed, because the sentence was
# written when there were six and nothing counted the directory.
_words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
# The sentence says "All N experiments are complete", so count the ones that
# are complete: a runner with committed results beside it. Counting runners
# alone made an experiment that exists but has not produced anything read as
# finished, which is the opposite of what the sentence promises.
_runners = sorted((REPO / "experiments" / "api").glob("e[0-9]*.py"))
_n_exp = sum(1 for _r in _runners
             if (REPO / "results" / f"{_r.stem.split('_')[0]}.jsonl").exists())
_incomplete = [_r.stem for _r in _runners
               if not (REPO / "results" / f"{_r.stem.split('_')[0]}.jsonl").exists()]
_m_exp = re.search(r"All (\w+) experiments are\s*>?\s*complete", README)
checks.append(("the README states how many experiments there are",
               _m_exp is not None and _m_exp.group(1).lower() == _words.get(_n_exp),
               f"README says {_m_exp and _m_exp.group(1)!r}, "
               f"experiments/api has {_n_exp}"))

# ---- 8. The verifier's own advertised size ---------------------------------
# The README said 135 while this script ran 176, because nothing compared them.
_ANCHOR_MARKS: list = []


def anchored(label: str, pattern: str, text: str = None):
    """Find a block anchor, and say so loudly when it is not there.

    Every regex-gated block in this file used to be written ``if m:``. Reword
    the sentence the regex points at and the whole block stops running, taking
    its checks out of the total with nothing printed. That has now happened
    twice. An anchor that misses is recorded as not-runnable, so it shows up in
    the report and in the count instead of evaporating.
    """
    m = re.search(pattern, README if text is None else text)
    if m is not None:
        _ANCHOR_MARKS.append((label, len(checks)))
    if m is None:
        # A failure, not a skip. Skips are for work that genuinely cannot run
        # here (an experiment not yet executed, a private document absent from
        # a clone) and they do not change the exit code. A moved anchor is
        # different: the document still makes the claim, and the check that
        # used to test it has quietly stopped running. That has to be loud.
        checks.append((f"anchor: {label}", False,
                       "no longer matches the document; reword the pattern or "
                       "the checks behind it silently stop running"))
    return m


# ---- 7. Tables that were published without a check behind them -----------
# Found by mutating one cell of every table in the README and asking whether
# this file noticed. Nine tables did not, roughly sixty numbers, among them the
# approximate-validity table that carries the project's main caveat and the
# head-to-head against MAPIE. Each was correct when recomputed by hand; none was
# protected. These recompute them from results/ so they stay that way.

# 7a. The level a given number of calibration positives actually certifies.
# Pure arithmetic, which is exactly why nothing was checking it.
_lvl = anchored("certified-level table",
    r"\| calibration positives \| level actually targeted at \u03b1 = 0\.10 \|\n"
    r"\|[-: |]+\|\n((?:\|.*\|\n)+)", README)
if _lvl:
    for _line in _lvl.group(1).strip().split("\n"):
        _c = [x.strip() for x in _line.strip().strip("|").split("|")]
        _n = int(_c[0])
        _m = re.search(r"([\d.]+)%", _c[1])
        if _m:
            _k = math.ceil((_n + 1) * 0.90)
            check(f"certified level at {_n} positives", float(_m.group(1)),
                  100 * min(1.0, _k / _n), 0.05)

# 7a-bis. The ratio between the tightest alpha each strategy can certify. The
# README used to call this an exact halving and invite the reader to check it
# on paper, which is where it fails: the +1 in the finite-sample index leaves
# it at 1.96 for 50 confirmed frauds, approaching two but never reaching it.
_rat = anchored("alpha ratio between strategies",
                r"a factor of \*\*([\d.]+)\u00d7\*\* at (\d+) confirmed frauds rising to\s*\n?"
                r"\*\*([\d.]+)\u00d7\*\* at (\d+)")
if _rat:
    for _g_ratio, _g_budget in ((1, 2), (3, 4)):
        _F = int(_rat.group(_g_budget))
        # split calibrates on F/2, cross on F, so the smallest certifiable
        # alphas are 2/(F+2) and 1/(F+1).
        check(f"alpha ratio at {_F} frauds", float(_rat.group(_g_ratio)),
              (2.0 / (_F + 2)) / (1.0 / (_F + 1)), 0.005)

# 7b. Realized coverage minus the level actually certified: the cost of
# cross-conformal's approximate validity, and the reason the headline is
# hedged. experiments/analyze_validity.py prints it; nothing compared it with
# what the README then published.
def _gaps(_rows, _strategy, _alpha):
    _per = {}
    for _r in _rows:
        if _r["strategy"] != _strategy:
            continue
        _a, _n = _r["alphas"].get(_alpha), _r.get("n_cal_fraud")
        if not _a or not _n:
            continue
        # Capped at 1.0: where ceil((n+1)(1-a)) exceeds n the threshold is
        # +inf, the set is every label and coverage is exactly 1.0, so the
        # honest gap is zero rather than negative.
        _cert = min(1.0, math.ceil((_n + 1) * (1 - float(_alpha))) / _n)
        _per.setdefault(_r["seed"], []).append(_a["coverage_fraud"] - _cert)
    return [float(np.mean(_per[_s])) for _s in sorted(_per)]

_vrows = {"Bank Account Fraud": load("e1.jsonl"), "Forest Cover Type": load("e7.jsonl")}
_vtab = anchored("validity table",
    r"\| dataset \| \u03b1 \| split \(exact\) \| cross \(approximate\) \|\n"
    r"\|[-: |]+\|\n((?:\|.*\|\n)+)", README)
_CRITV = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
_below = {"split": 0, "cross": 0}
_seen = {"split": 0, "cross": 0}
if _vtab:
    for _line in _vtab.group(1).strip().split("\n"):
        _c = [x.strip() for x in _line.strip().strip("|").split("|")]
        _ds, _al = _c[0], _c[1]
        _rows = _vrows.get(_ds)
        if not _rows:
            continue
        for _col, _strategy in ((_c[2], "split"), (_c[3], "cross")):
            _m = re.search(r"([+-][\d.]+) \u00b1 ([\d.]+)", _col)
            if not _m:
                continue
            _g = _gaps(_rows, _strategy, _al)
            if len(_g) < 2:
                continue
            _mean = float(np.mean(_g))
            _se = float(np.std(_g, ddof=1) / np.sqrt(len(_g)))
            check(f"validity {_ds[:3]} a={_al} {_strategy} mean", float(_m.group(1)), _mean, 6e-5)
            check(f"validity {_ds[:3]} a={_al} {_strategy} se", float(_m.group(2)), _se, 6e-5)
            # Bold marks "significantly below"; the mark has to match the test.
            _t = _mean / _se if _se else 0.0
            _is_below = _t < -_CRITV.get(len(_g), 2.0)
            checks.append((f"validity {_ds[:3]} a={_al} {_strategy} bold matches the test",
                           _col.startswith("**") == _is_below,
                           f"bold={_col.startswith('**')} but t={_t:.2f} says below={_is_below}"))
            _seen[_strategy] += 1
            _below[_strategy] += _is_below
    for _strategy in ("split", "cross"):
        _m = re.search(
            r"Split is below its certified level in (\d+) of (\d+) dataset-\u03b1 combinations\. Cross is\s*\n?below in (\d+) of (\d+)",
            README)
        if _m:
            _claim = int(_m.group(1)) if _strategy == "split" else int(_m.group(3))
            checks.append((f"validity: {_strategy} below in N of 6",
                           _claim == _below[_strategy] and _seen[_strategy] == 6,
                           f"README says {_claim}, recomputed {_below[_strategy]} "
                           f"of {_seen[_strategy]}"))

# 7c. The second-domain tables: matched set sizes, and the paired differences.
_e7 = load("e7.jsonl")
if _e7:
    _by = {}
    for _r in _e7:
        _a = _r["alphas"].get("0.1")
        _n = _r.get("n_cal_fraud")
        if not _a or not _n:
            continue
        _by.setdefault(_n, {}).setdefault(_r["strategy"], {})[_r["seed"]] = (
            _r["n_frauds"], _a["set_size"])
    _m7 = anchored(
        "E7 matched table",
        r"\| calib\. positives \| targeted level \| split needs \| its set size \| "
        r"cross needs \| its set size \| labels saved \|\n\|[-: |]+\|\n((?:\|.*\|\n)+)",
        README[README.find("Forest Cover"):] if "Forest Cover" in README else README)
    if _m7:
        for _line in _m7.group(1).strip().split("\n"):
            _c = [x.strip() for x in _line.strip().strip("|").split("|")]
            _n = int(_c[0])
            _p = _by.get(_n, {})
            if "split" not in _p or "cross" not in _p:
                continue
            _sd = sorted(set(_p["split"]) & set(_p["cross"]))
            _ss = float(np.mean([_p["split"][x][1] for x in _sd]))
            _cs = float(np.mean([_p["cross"][x][1] for x in _sd]))
            check(f"E7 matched level @{_n}", float(re.search(r"([\d.]+)%", _c[1]).group(1)),
                  100 * math.ceil((_n + 1) * 0.9) / _n, 0.05)
            check(f"E7 matched split size @{_n}", float(_c[3]), _ss, 5e-4)
            check(f"E7 matched cross size @{_n}",
                  float(re.search(r"\*\*([\d.]+)\*\*", _c[5]).group(1)), _cs, 5e-4)
            _rel = re.search(r"\((narrower|wider) by ([\d.]+)%\)", _c[5])
            if _rel:
                _delta = 100 * (_ss - _cs) / _ss
                _want = _delta if _rel.group(1) == "narrower" else -_delta
                check(f"E7 matched {_rel.group(1)}-by @{_n}", float(_rel.group(2)), _want, 0.05)
            check(f"E7 matched split budget @{_n}", float(_c[2]),
                  float(_p["split"][_sd[0]][0]), 0.5)

    _p7 = anchored("E7 paired table",
        r"\| calib\. positives \| paired difference \| verdict \|\n\|[-: |]+\|\n"
        r"((?:\|.*\|\n)+)", README)
    if _p7:
        _n_tie = 0
        for _line in _p7.group(1).strip().split("\n"):
            _c = [x.strip() for x in _line.strip().strip("|").split("|")]
            _n = int(_c[0])
            _m = re.search(r"([+-][\d.]+) \u00b1 ([\d.]+) \(n=(\d+)\)", _c[1])
            _p = _by.get(_n, {})
            if not _m or "split" not in _p or "cross" not in _p:
                continue
            _sd = sorted(set(_p["split"]) & set(_p["cross"]))
            _d = np.array([_p["split"][x][1] - _p["cross"][x][1] for x in _sd])
            _se = float(_d.std(ddof=1) / np.sqrt(len(_d)))
            check(f"E7 paired diff @{_n}", float(_m.group(1)), float(_d.mean()), 6e-5)
            check(f"E7 paired se @{_n}", float(_m.group(2)), _se, 6e-5)
            check(f"E7 paired n @{_n}", float(_m.group(3)), float(len(_d)), 0.5)
            _n_tie += abs(_d.mean() / _se) < _CRITV.get(len(_d), 2.0) if _se else 0
        _m = re.search(r"wider in \*\*(\d+) of (\d+) comparisons here\*\*", README)
        if _m:
            checks.append(("E7: cross wider in N of 3",
                           int(_m.group(1)) == 0 and int(_m.group(2)) == _n_tie,
                           f"README {_m.group(1)} of {_m.group(2)}; recomputed "
                           f"0 wider with {_n_tie} ties"))

# 7d. P1 in the pre-registration scoreboard: the seed spread of fraud coverage.
_e1v = load("e1.jsonl")
_p1 = anchored("P1 seed-spread row",
               r"max minus min at \u03b1 = 0\.10: ([\d.]+) vs ([\d.]+) at F=(\d+) "
               r"\(cross better\) but ([\d.]+) vs ([\d.]+) at F=(\d+)")
if _e1v and _p1:
    def _spread(_strategy, _budget):
        """Max minus min across seeds, which is what the row reports.

        Not the standard deviation: the prediction was worded "seed-variance"
        and the numbers beside it are the spread, so the README now says so.
        """
        _v = [r["alphas"]["0.1"]["coverage_fraud"] for r in _e1v
              if r["strategy"] == _strategy and r["n_frauds"] == _budget
              and "0.1" in r["alphas"]]
        return float(max(_v) - min(_v)) if len(_v) > 1 else float("nan")
    for _i, (_strategy, _b, _g) in enumerate(
            (("split", int(_p1.group(3)), 1), ("cross", int(_p1.group(3)), 2),
             ("split", int(_p1.group(6)), 4), ("cross", int(_p1.group(6)), 5))):
        check(f"P1 {_strategy} seed SD at F={_b}", float(_p1.group(_g)), _spread(_strategy, _b), 6e-4)

# 7e. The scoreboard's ECE range, and the capability table's cache speedup:
# both restate a number checked elsewhere, in a different string, so both were
# free to drift away from the table they summarise.
_cal = REPO / "results" / "calibration.json"
if _cal.exists():
    _h2h = json.loads(_cal.read_text()).get("head_to_head", {})
    if _h2h:
        _t = [v["tabpfn_ece"] for v in _h2h.values()]
        _l = [v["lightgbm_ece"] for v in _h2h.values()]
        _m = anchored("scoreboard ECE range",
                      r"ECE ([\d.]+)\u2013([\d.]+) vs ([\d.]+)\u2013([\d.]+)")
        if _m:
            check("scoreboard ECE tabpfn low", float(_m.group(1)), min(_t), 5e-5)
            check("scoreboard ECE tabpfn high", float(_m.group(2)), max(_t), 5e-5)
            check("scoreboard ECE lightgbm low", float(_m.group(3)), min(_l), 5e-5)
            check("scoreboard ECE lightgbm high", float(_m.group(4)), max(_l), 5e-5)

        # The AUC-gap column of the calibration table, the one column of it
        # that nothing was reading.
        _ct = anchored("calibration table",
                       r"\| strategy \| budget \| TabPFN ECE \| LightGBM ECE \| "
                       r"TabPFN better by \| AUC gap \|\n\|[-: |]+\|\n((?:\|.*\|\n)+)")
        if _ct:
            for _line in _ct.group(1).strip().split("\n"):
                _c = [x.strip() for x in _line.strip().strip("|").split("|")]
                _key = f"{_c[0]}_{_c[1]}"
                _v = _h2h.get(_key)
                if not _v:
                    checks.append((f"calibration row {_key}", None, "no such arm in calibration.json"))
                    continue
                check(f"calibration AUC gap {_key}", float(_c[5]),
                      _v["tabpfn_auc"] - _v["lightgbm_auc"], 5e-4)

# 7f. The "narrower by" percentage columns. The set sizes either side of them
# were checked; the percentage derived from the pair was not.
_mn = anchored("E1 matched table",
               r"\| targeted level \| split needs \| its set size \| cross needs \| "
               r"its set size \| labels saved \|\n\|[-: |]+\|\n((?:\|.*\|\n)+)")
if _mn:
    for _line in _mn.group(1).strip().split("\n"):
        _c = [x.strip() for x in _line.strip().strip("|").split("|")]
        _sz = re.search(r"([\d.]+)", _c[2])
        _cz = re.search(r"\*\*([\d.]+)\*\*", _c[4])
        _pc = re.search(r"\(([\d.]+)% narrower\)", _c[4])
        if _sz and _cz and _pc:
            _a, _b = float(_sz.group(1)), float(_cz.group(1))
            check(f"E1 matched narrower-by at {_c[1]}", float(_pc.group(1)),
                  100 * (_a - _b) / _a, 0.05)

_e4t = anchored("E4 baselines table",
                r"\| strategy \| budget \| targeted level \| TabPFN \| LightGBM \| "
                r"TabPFN narrower by \|\n\|[-: |]+\|\n((?:\|.*\|\n)+)")
if _e4t:
    for _line in _e4t.group(1).strip().split("\n"):
        _c = [x.strip() for x in _line.strip().strip("|").split("|")]
        _t4 = re.search(r"([\d.]+)", _c[3])
        _l4 = re.search(r"([\d.]+)", _c[4])
        _p4 = re.search(r"([\d.]+)", _c[5])
        if _t4 and _l4 and _p4:
            _a, _b = float(_l4.group(1)), float(_t4.group(1))
            check(f"E4 narrower-by {_c[0]}@{_c[1]}", float(_p4.group(1)),
                  100 * (_a - _b) / _a, 0.05)

# 7g. The scoreboard restates the E5 slope and SD in a second, shorter form.
if e5 and len(by_ctx) > 2:
    _sb = anchored("scoreboard E5 summary",
                   r"slope \u2212([\d.]+) vs seed SD ([\d.]+)")
    if _sb:
        check("scoreboard E5 slope", -float(_sb.group(1)), slope, 0.0006)
        check("scoreboard E5 seed SD", float(_sb.group(2)), sd, 0.0006)

# 7h. The second domain's base rate, against the constant the experiment ran
# with. This is a consistency check between prose and code, not a measurement:
# confirming it against covtype itself would mean downloading the dataset,
# which this script must never do.
_e7src = REPO / "experiments/api/e7_second_domain.py"
if _e7src.exists():
    _br = re.search(r"BASE_RATE = ([\d.]+)", _e7src.read_text())
    _rb = anchored("scoreboard covtype base rate",
                   r"forest cover type, ([\d.]+)% positive")
    if _br and _rb:
        check("covtype base rate matches the experiment constant",
              float(_rb.group(1)), 100 * float(_br.group(1)), 5e-4)

# 7i. The paired t behind "directional, not established" for Thinking.
_e3v = load("e3.jsonl")
if _e3v:
    _tm = anchored("Thinking paired t", r"directional, t \u2248 ([\d.]+) at n=(\d+)\*\*")
    if _tm:
        _lvl3 = 45 / 46
        _d = []
        for _sd3 in sorted({r["seed"] for r in _e3v}):
            _cnt = {}
            for _mo in ("base", "thinking"):
                _cnt[_mo] = sum(
                    r["coverage_fraud"] < _lvl3 for r in _e3v
                    if r["arm"] == "frozen" and r["model"] == _mo and r["seed"] == _sd3)
            _d.append(_cnt["base"] - _cnt["thinking"])
        _d = np.array(_d, dtype=float)
        _se3 = float(_d.std(ddof=1) / np.sqrt(len(_d)))
        check("Thinking paired t", float(_tm.group(1)),
              float(_d.mean()) / _se3 if _se3 else 0.0, 0.05)
        check("Thinking paired n", float(_tm.group(2)), float(len(_d)), 0.5)

if e5:
    _cap = re.search(r"`fit_mode=\"fit_with_cache\"` \| E5 \| \*\*([\d.]+)\u00d7 faster", README)
    _c2 = {r["n_context"]: r for r in e5 if r["cache"]}
    _p2 = {r["n_context"]: r for r in e5 if not r["cache"] and r["seed"] == 0
           and r["strategy"] == "split"}
    if _cap and 200200 in _c2 and 200200 in _p2:
        check("capability table cache speedup", float(_cap.group(1)),
              _p2[200200]["predict_seconds"] / _c2[200200]["predict_seconds"], 0.05)

# 7j. The MAPIE head-to-head. Its numbers lived only inside tests that assert
# loose bounds on purpose, so a MAPIE upgrade would not turn into a flaky
# failure; that left the exact figures in the README unchecked. They now come
# from results/mapie_comparison.json, written by experiments/analyze_mapie.py.
_mapf = REPO / "results" / "mapie_comparison.json"
if _mapf.exists():
    _mp = json.loads(_mapf.read_text())
    _cx = _mp["cross_vs_cv_plus"]
    _worst_same = min(v["sets_identical"] for v in _cx.values())
    _worst_cov = max(v["abs_coverage_gap"] for v in _cx.values())
    _worst_sz = max(v["abs_set_size_gap"] for v in _cx.values())
    _mrow = anchored(
        "MAPIE cross agreement row",
        r"\*\*([\d.]+)% of prediction sets identical\*\* at the worst of "
        r"\u03b1 \u2208 \{0\.05, 0\.1, 0\.2\}; coverage within ([\d.]+) and "
        r"set size within ([\d.]+)")
    if _mrow:
        check("MAPIE worst set agreement", float(_mrow.group(1)), 100 * _worst_same, 0.05)
        # A stated bound has to actually hold, so this is an inequality, not a
        # near-equality: it said 0.002 while the measurement was 0.0027.
        checks.append(("MAPIE coverage bound holds",
                       _worst_cov <= float(_mrow.group(2)),
                       f"README claims within {_mrow.group(2)}, worst measured {_worst_cov:.5f}"))
        checks.append(("MAPIE set size bound holds",
                       _worst_sz <= float(_mrow.group(3)),
                       f"README claims within {_mrow.group(3)}, worst measured {_worst_sz:.5f}"))
        # ...and not so loose as to be meaningless.
        checks.append(("MAPIE bounds are tight to within 10x",
                       _worst_cov * 10 >= float(_mrow.group(2))
                       and _worst_sz * 10 >= float(_mrow.group(3)),
                       "the quoted bound is more than ten times the measurement"))

    _sbm = anchored("scoreboard MAPIE minority coverage",
                    r"minority coverage\s*\n?\*\*([\d.]+)\*\* against ours at \*\*([\d.]+)\*\*")
    if _sbm:
        check("scoreboard MAPIE minority coverage", float(_sbm.group(1)),
              _mp["class_conditional"]["mapie_cv_plus_lac"]["minority_coverage"], 5e-4)
        check("scoreboard ours minority coverage", float(_sbm.group(2)),
              _mp["class_conditional"]["ours_mondrian_cross"]["minority_coverage"], 5e-4)

    _cc = _mp["class_conditional"]
    _ctab = anchored("MAPIE class-conditional table",
                     r"\| \| minority coverage \| majority \| mean set size \|\n"
                     r"\|[-: |]+\|\n((?:\|.*\|\n)+)")
    if _ctab:
        _want = {"MAPIE CV+": _cc["mapie_cv_plus_lac"], "ours,": _cc["ours_mondrian_cross"]}
        for _line in _ctab.group(1).strip().split("\n"):
            _c = [x.strip() for x in _line.strip().strip("|").split("|")]
            _key = next((k for k in _want if _c[0].startswith(k)), None)
            if _key is None:
                checks.append((f"MAPIE table row {_c[0][:20]}", None, "unrecognised row"))
                continue
            _v = _want[_key]
            _lab = "mapie" if _key.startswith("MAPIE") else "ours"
            for _i, _field in ((1, "minority_coverage"), (2, "majority_coverage"),
                               (3, "mean_set_size")):
                _num = re.search(r"([\d.]+)", _c[_i])
                if _num:
                    check(f"MAPIE table {_lab} {_field}", float(_num.group(1)),
                          _v[_field], 5e-4)

# ---- 8. Claims in prose, not in a table ----------------------------------
# Same sweep as section 7, run over the sentences instead of the tables: mutate
# a number, see whether anything notices. Most of what it flagged was a
# parameter or a citation rather than a measurement; these are the
# measurements. One of them, the order-statistic step, was wrong in its last
# digit, which is exactly the size of error this kind of check exists to find.

# 8a. The worked example that carries the argument in prose.
_wex = anchored(
    "worked example",
    r"targets 96%, and delivers \*\*coverage ([\d.]+) with mean set size ([\d.]+)\*\*;"
    r"\s*\ncross calibrates on all 50, targets 92%, and delivers \*\*([\d.]+) with set size\s*\n([\d.]+)\*\*")
if _wex and _e1v:
    for _gi, (_st, _fld) in enumerate(
            ((("split", "coverage_fraud")), ("split", "set_size"),
             ("cross", "coverage_fraud"), ("cross", "set_size")), start=1):
        _rs = [r for r in _e1v if r["strategy"] == _st and r["n_frauds"] == 50
               and "0.1" in r["alphas"]]
        if _rs:
            check(f"worked example {_st} {_fld}", float(_wex.group(_gi)),
                  float(np.mean([r["alphas"]["0.1"][_fld] for r in _rs])), 5e-4)

# 8b. E2: the best allocation still loses to not splitting. At alpha 0.05,
# which is the level the section's table is computed at.
_e2v = load("e2.jsonl")
_e2p = anchored("E2 best-split sentence",
                r"splitting at all: ([\d.]+) against cross-conformal's \*\*([\d.]+)\*\*")
if _e2v and _e2p:
    _sp, _cr = {}, []
    for _r in _e2v:
        _a = _r["alphas"].get("0.05")
        if not _a or _r["n_frauds"] != 100:
            continue
        if _r["strategy"] == "split":
            _sp.setdefault(_r.get("cal_size"), []).append(_a["set_size"])
        else:
            _cr.append(_a["set_size"])
    if _sp and _cr:
        check("E2 best split set size", float(_e2p.group(1)),
              min(float(np.mean(v)) for v in _sp.values()), 5e-4)
        check("E2 cross set size", float(_e2p.group(2)), float(np.mean(_cr)), 5e-4)

# 8c. The ACI sweep, and why adaptive calibration cannot move at this budget.
_acif = REPO / "results" / "aci_gamma_sweep.json"
if _acif.exists():
    _aci = json.loads(_acif.read_text())
    _sw = anchored("ACI swing sentence",
                   r"the month-to-month swing goes from ([\d.]+) to \*\*([\d.]+)\*\*")
    if _sw:
        for _gi, _key in ((1, "frozen"), (2, "gamma=1")):
            _c = [x["coverage_fraud"] for x in _aci.get(_key, [])]
            if _c:
                check(f"ACI swing {_key}", float(_sw.group(_gi)), max(_c) - min(_c), 5e-4)
    _ex = anchored("ACI extremes sentence",
                   r"widest sets and the wildest swing, ([\d.]+) one month and ([\d.]+)")
    if _ex:
        _c = [x["coverage_fraud"] for x in _aci.get("gamma=1", [])]
        if _c:
            check("ACI gamma=1 lowest month", float(_ex.group(1)), min(_c), 5e-4)
            check("ACI gamma=1 highest month", float(_ex.group(2)), max(_c), 5e-4)
    _mv = anchored("ACI movement sentence", r"ACI moves \u03b1 by\s*\n?([\d.]+) across the whole walk")
    if _mv:
        _a = [x["alpha_fraud"] for x in _aci.get("gamma=0.05", []) if "alpha_fraud" in x]
        if _a:
            check("ACI alpha movement", float(_mv.group(1)), max(_a) - min(_a), 5e-4)

# 8d. Arithmetic stated in prose. The order-statistic step is how far alpha has
# to rise before a different calibration score is selected; at 46 positives and
# alpha 0.05 that is 1 - 44/47 - 0.05, and the README used to round it up.
_stp = anchored("order-statistic step",
                r"that step is \*\*([\d.]+)\*\*; ACI moves")
if _stp:
    _nn, _aa = 46, 0.05
    _kk = math.ceil((_nn + 1) * (1 - _aa))
    check("order-statistic step at 46 positives", float(_stp.group(1)),
          (1 - (_kk - 1) / (_nn + 1)) - _aa, 5e-5)
_sa = anchored("feasibility floor", r"fires for every \u03b1 below (\d+\.\d+)")
if _sa:
    check("smallest certifiable alpha at 13 positives", float(_sa.group(1)), 1 / 14, 5e-5)
for _lbl, _pat in (("certified level, results section",
                    r"positives actually certify \(([\d.]+)%, not 95%\)"),
                   ("certified level, drift section",
                    r"actually targeted \(([\d.]+)% with 46 calibration positives")):
    _m = anchored(_lbl, _pat)
    if _m:
        check(_lbl, float(_m.group(1)), 100 * math.ceil(47 * 0.95) / 46, 0.005)

# 8e. The base rate the calibration study reweights to.
if _cal.exists():
    _tr = anchored("reweighted base rate", r"reweighted to the true ([\d.]+)% base rate")
    if _tr:
        check("reweighted base rate", float(_tr.group(1)),
              100 * json.loads(_cal.read_text())["true_rate"], 5e-4)

# 8f. The cached round trip, and the K-fold cost multiplier at K=20.
if e5:
    _rt = anchored("round trip sentence",
                   r"round trip is worse overall, ([\d.]+) s cached against ([\d.]+) s uncached")
    _pp = {(r["n_context"], bool(r["cache"])): r for r in e5
           if r["strategy"] == "split" and r["seed"] == 0}
    if _rt and (50200, True) in _pp and (50200, False) in _pp:
        for _gi, _flag in ((1, True), (2, False)):
            _r = _pp[(50200, _flag)]
            check(f"round trip {'cached' if _flag else 'uncached'}", float(_rt.group(_gi)),
                  _r["fit_seconds"] + _r["predict_seconds"], 0.05)

_ck = REPO / "results" / "cost_kfold.json"
if _ck.exists():
    _km = anchored("K multiplier sentence",
                   r"([\d.]+)\u00d7 at K=2 and ([\d.]+)\u00d7 at K=20")
    if _km:
        _rws = json.loads(_ck.read_text())["rows"]
        for _gi, _kk2 in ((1, 2), (2, 20)):
            _hit = [r for r in _rws if r["k"] == _kk2]
            if _hit:
                check(f"cost multiplier at K={_kk2}", float(_km.group(_gi)),
                      float(np.mean([r["ratio"] for r in _hit])), 0.05)

# 8g. The size of the approximate-validity shortfall, quoted twice in prose.
if _vtab:
    _gapsp = []
    for _dsn, _rws in _vrows.items():
        for _al2 in ("0.05", "0.1", "0.2"):
            _g2 = _gaps(_rws, "cross", _al2)
            if len(_g2) < 2:
                continue
            _m2 = float(np.mean(_g2))
            _se2 = float(np.std(_g2, ddof=1) / np.sqrt(len(_g2)))
            if _se2 and _m2 / _se2 < -_CRITV.get(len(_g2), 2.0):
                _gapsp.append(abs(_m2) * 100)
    if _gapsp:
        for _lbl, _pat in (("validity shortfall range, section",
                            r"by about ([\d.]+) to ([\d.]+) points"),
                           ("validity shortfall range, closing",
                            r"combinations by ([\d.]+) to ([\d.]+) points")):
            _m = anchored(_lbl, _pat)
            if _m:
                check(f"{_lbl} low", float(_m.group(1)), min(_gapsp), 0.05)
                check(f"{_lbl} high", float(_m.group(2)), max(_gapsp), 0.05)

# 8h. TabPFN's own imbalance tooling, the baseline with no guarantee at all.
_e4v = load("e4.jsonl")
if _e4v:
    _tt = anchored("tuned-threshold sentence",
                   r"reaches ([\d.]+) recall while\s*\nflagging \*\*([\d.]+)% of legitimate traffic\*\*")
    if _tt:
        _sub = [r for r in _e4v if r["arm"] == "tabpfn_tuned_threshold"
                and r["n_frauds"] == 200]
        if _sub:
            check("tuned threshold recall", float(_tt.group(1)),
                  float(np.mean([r["recall"] for r in _sub])), 5e-4)
            check("tuned threshold flag rate", float(_tt.group(2)),
                  100 * float(np.mean([r["false_positive_rate"] for r in _sub])), 0.05)

    # The pair of thresholds is one run, not an average: same budget, same seed,
    # which is the only way the "identical recall" claim beside it means anything.
    _th = anchored("balance_probabilities threshold shift",
                   r"threshold shifts from ([\d.]+) to ([\d.]+)\)")
    if _th:
        _one = {r["arm"]: r for r in _e4v if r["n_frauds"] == 200 and r["seed"] == 0
                and "threshold" in r["arm"]}
        if len(_one) == 2:
            check("tuned threshold value", float(_th.group(1)),
                  _one["tabpfn_tuned_threshold"]["threshold"], 5e-5)
            check("balanced threshold value", float(_th.group(2)),
                  _one["tabpfn_balanced_threshold"]["threshold"], 5e-3)

    _id = anchored("balance_probabilities identical-seeds claim",
                   r"identical in (\w+) of (\w+) seeds")
    if _id:
        _pairs = {}
        for _r in _e4v:
            if "threshold" in _r["arm"]:
                _pairs.setdefault((_r["n_frauds"], _r["seed"]), {})[_r["arm"]] = _r
        _same = sum(
            1 for _v in _pairs.values()
            if len(_v) == 2
            and _v["tabpfn_tuned_threshold"]["recall"] == _v["tabpfn_balanced_threshold"]["recall"]
            and _v["tabpfn_tuned_threshold"]["false_positive_rate"]
            == _v["tabpfn_balanced_threshold"]["false_positive_rate"])
        _w2n = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        checks.append(("balance_probabilities identical in N of six seeds",
                       _w2n.get(_id.group(1)) == _same
                       and _w2n.get(_id.group(2)) == len(_pairs),
                       f"README {_id.group(1)} of {_id.group(2)}; recomputed "
                       f"{_same} of {len(_pairs)}"))

# 8i. Numbers the README states twice: once where they are measured, once in a
# summary sentence. Only the first copy was ever checked.
_r2 = anchored("E1 narrowest restatement", r"scarcest, ([\d.]+)% at 25 calibration positives")
if _r2 and _e1v:
    _p25 = {}
    for _r in _e1v:
        _a = _r["alphas"].get("0.1")
        if _a and _r.get("n_cal_fraud") == 25:
            _p25.setdefault(_r["strategy"], {})[_r["seed"]] = _a["set_size"]
    if "split" in _p25 and "cross" in _p25:
        _sd2 = sorted(set(_p25["split"]) & set(_p25["cross"]))
        _a2 = float(np.mean([_p25["split"][x] for x in _sd2]))
        _b2 = float(np.mean([_p25["cross"][x] for x in _sd2]))
        check("E1 narrowest restatement", float(_r2.group(1)), 100 * (_a2 - _b2) / _a2, 0.05)

_r3 = anchored("E6 Variant III restatement",
               r"closest thing to a loss\*\*: \u2212([\d.]+) \u00b1 ([\d.]+), t \u2248 ([\d.]+)")
if _r3:
    _e6v = load("e6.jsonl")
    # Keyed on calibration positives, which is what the row says. Keyed on the
    # budget instead, split and cross never line up and the block appends
    # nothing at all, which is how this one first went missing.
    _pv = {}
    for _r in _e6v:
        _a = _r["alphas"].get("0.1")
        if _a and _r.get("variant") == "Variant III" and _r.get("n_cal_fraud") == 100:
            _pv.setdefault(_r["strategy"], {})[_r["seed"]] = _a["set_size"]
    if "split" in _pv and "cross" in _pv:
        _sd3 = sorted(set(_pv["split"]) & set(_pv["cross"]))
        _d3 = np.array([_pv["split"][x] - _pv["cross"][x] for x in _sd3])
        _se3b = float(_d3.std(ddof=1) / np.sqrt(len(_d3)))
        check("E6 Variant III paired diff", -float(_r3.group(1)), float(_d3.mean()), 6e-5)
        check("E6 Variant III se", float(_r3.group(2)), _se3b, 6e-5)
        check("E6 Variant III t", float(_r3.group(3)),
              abs(float(_d3.mean()) / _se3b) if _se3b else 0.0, 0.05)

_r4 = anchored("MAPIE agreement restatement", r"the two produce ([\d.]+)% identical sets")
if _r4 and _mapf.exists():
    check("MAPIE agreement restatement", float(_r4.group(1)),
          100 * min(v["sets_identical"] for v in
                    json.loads(_mapf.read_text())["cross_vs_cv_plus"].values()), 0.05)

_r5 = anchored("E5 context rate restatement",
               r"driving the context fraud rate from ([\d.]+)% down to")
if _r5 and e5:
    _f10 = [r["fraud_rate_context"] for r in e5
            if r["n_context"] == 10200 and r["strategy"] == "split" and not r["cache"]]
    if _f10:
        check("E5 context rate restatement", float(_r5.group(1)),
              100 * float(np.mean(_f10)), 0.006)

_r6 = anchored("covtype rate in the E7 section",
               r"Cottonwood/Willow, which occurs at \*\*([\d.]+)%\*\*")
if _r6 and _e7src.exists():
    _br2 = re.search(r"BASE_RATE = ([\d.]+)", _e7src.read_text())
    if _br2:
        check("covtype rate in the E7 section", float(_r6.group(1)),
              100 * float(_br2.group(1)), 5e-4)

# 8j. The share of a monthly budget one calibration pass costs. Stated twice,
# from the documented free-tier ceiling, so both copies have to agree with it.
_bud = anchored("monthly budget share",
                r"on a 10,000-row pool is roughly ([\d,]+) tokens, about ([\d.]+)% of a monthly budget")
if _bud:
    _tok = float(_bud.group(1).replace(",", ""))
    _apidoc = REPO / "experiments" / "api" / "README.md"
    _cap = None
    if _apidoc.exists():
        _cm = re.search(r"(\d+)M/month", _apidoc.read_text())
        if _cm:
            _cap = float(_cm.group(1)) * 1e6
    if _cap:
        check("monthly budget share", float(_bud.group(2)), 100 * _tok / _cap, 0.005)
    else:
        checks.append(("monthly budget share", None,
                       "the monthly ceiling is not stated in experiments/api/README.md"))

# 8k. The opening claim, taken from the example CI actually runs. The first
# number a reader meets should not be the one nothing re-derives.
_qs = REPO / "examples" / "quickstart.py"
_qm = anchored("opening marginal-coverage claim",
               r"minority class \*\*([\d.]+)\*\* coverage while its overall number looks healthy")
if _qm and _qs.exists():
    _out = subprocess.run([sys.executable, str(_qs)], cwd=REPO,
                          capture_output=True, text=True)
    _mm = re.search(r"method='marginal'\s+positive-class coverage ([\d.]+)", _out.stdout)
    if _mm:
        check("opening marginal-coverage claim", float(_qm.group(1)), float(_mm.group(1)), 5e-4)
    else:
        checks.append(("opening marginal-coverage claim", None,
                       f"quickstart printed nothing parseable (exit {_out.returncode})"))

# ---- 10. The other two published documents ------------------------------
# limitations.md was not read by this script at all, so every number in it was
# free. method.md was read only for three phrases and a seed count. Both
# restate results, and both had a digit wrong: the order-statistic step, the
# same one the README had, and the set size at 13 calibration positives.
_LIM = (REPO / "docs/limitations.md").read_text()
_MET = (REPO / "docs/method.md").read_text()


def anchored_in(label: str, pattern: str, text: str):
    """anchored(), against a document other than the README."""
    return anchored(label, pattern, text)


# 10a. The worked illustration of why coverage and set size must be read together.
_li = anchored_in("limitations: vacuous coverage at 13 positives",
                  r"split conformal scores coverage ([\d.]+) with mean set size ([\d.]+)",
                  _LIM)
if _li and _e1v:
    _s13 = [r for r in _e1v if r["strategy"] == "split"
            and r.get("n_cal_fraud") == 13 and "0.05" in r["alphas"]]
    if _s13:
        check("limitations: coverage at 13 positives", float(_li.group(1)),
              float(np.mean([r["alphas"]["0.05"]["coverage_fraud"] for r in _s13])), 5e-4)
        check("limitations: set size at 13 positives", float(_li.group(2)),
              float(np.mean([r["alphas"]["0.05"]["set_size"] for r in _s13])), 5e-4)

# 10b. The K-fold cost multipliers, restated from the README.
if _ck.exists():
    _lk = anchored_in("limitations: K multipliers",
                      r"\(([\d.]+)\u00d7 at K=2, ([\d.]+)\u00d7 at K=20\)", _LIM)
    if _lk:
        _rws2 = json.loads(_ck.read_text())["rows"]
        for _gi, _kk3 in ((1, 2), (2, 20)):
            _hit2 = [r for r in _rws2 if r["k"] == _kk3]
            if _hit2:
                check(f"limitations: multiplier at K={_kk3}", float(_lk.group(_gi)),
                      float(np.mean([r["ratio"] for r in _hit2])), 0.05)

# 10c. The split-to-cross wall-clock multipliers on fair hardware.
if _kag.exists():
    _lm = anchored_in("limitations: split-to-cross multipliers",
                      r"costs TabPFN ([\d.]+)\u00d7 against LightGBM's ([\d.]+)\u00d7", _LIM)
    if _lm:
        _kr = json.loads(_kag.read_text())["rows"]

        def _cell10(fam, strat, budget):
            return {r["seed"]: r for r in _kr if r["family"] == fam
                    and r["strategy"] == strat and r["n_frauds"] == budget}

        for _gi, _fam10 in ((1, "tabpfn"), (2, "lightgbm")):
            _mult = float(np.mean([
                np.mean([_cell10(_fam10, "cross", _b)[s]["seconds"] for s in (0, 1, 2)])
                / np.mean([_cell10(_fam10, "split", _b)[s]["seconds"] for s in (0, 1, 2)])
                for _b in sorted({r["n_frauds"] for r in _kr})]))
            check(f"limitations: {_fam10} split-to-cross", float(_lm.group(_gi)), _mult, 0.01)

# 10d. method.md restates the ACI arithmetic and the drift swing.
_ms = anchored_in("method: order-statistic step",
                  r"At 46 positives that is (\d+\.\d+)\.", _MET)
if _ms:
    _k10 = math.ceil(47 * 0.95)
    check("method: order-statistic step", float(_ms.group(1)),
          (1 - (_k10 - 1) / 47) - 0.05, 5e-5)
if _acif.exists():
    _aci10 = json.loads(_acif.read_text())
    _mm10 = anchored_in("method: ACI movement",
                        r"ACI moves the level by \*\*([\d.]+)\*\*", _MET)
    if _mm10:
        _a10 = [x["alpha_fraud"] for x in _aci10.get("gamma=0.05", []) if "alpha_fraud" in x]
        if _a10:
            check("method: ACI movement", float(_mm10.group(1)), max(_a10) - min(_a10), 5e-4)
    _sw10 = anchored_in("method: coverage swing",
                        r"coverage swing goes from ([\d.]+) to \*\*([\d.]+)\*\*", _MET)
    if _sw10:
        for _gi, _key10 in ((1, "frozen"), (2, "gamma=1")):
            _c10 = [x["coverage_fraud"] for x in _aci10.get(_key10, [])]
            if _c10:
                check(f"method: swing {_key10}", float(_sw10.group(_gi)),
                      max(_c10) - min(_c10), 5e-4)

# 10e. The index arithmetic method.md uses to explain the feasibility ceiling.
_mi = anchored_in("method: index at 13 positives",
                  r"\u03b1 = 0\.10 the index is (\d+), the maximum score", _MET)
if _mi:
    check("method: index at 13 positives", float(_mi.group(1)),
          float(math.ceil(14 * 0.90)), 0.5)

# ---- 9. Did every anchored block actually produce checks? ----------------
# An anchor that matches and then finds no data behind it appends nothing, and
# the block disappears exactly as quietly as a missed anchor does. That is the
# same bug in a different place, and it has now bitten three times: a reworded
# sentence, a wrong dict key into calibration.json, and an E6 lookup keyed on
# the budget rather than on calibration positives. Each anchor is required to
# leave at least one check behind it.
for _i, (_lbl, _at) in enumerate(_ANCHOR_MARKS):
    _end = _ANCHOR_MARKS[_i + 1][1] if _i + 1 < len(_ANCHOR_MARKS) else len(checks)
    if _end <= _at:
        checks.append((f"anchor {_lbl!r} produced no checks", False,
                       "the pattern matched but the data behind it was not "
                       "found, so the block silently added nothing"))

# Counted last, and counts itself, so the figure in the README is the number of
# checks this file actually performs.
# Counted against the published configuration, the one CI and a reader run.
# With private/ present there are more checks, and the number in the README
# describes what someone cloning the repository will see.
if not PRIVATE.exists():
    _claimed_n = in_readme(r"recomputes\s*>?\s*(\d+) of them")
    checks.append(("the README states how many claims this script checks",
                   _claimed_n is not None and int(_claimed_n) == len(checks) + 1,
                   f"README says {_claimed_n and int(_claimed_n)}, this run has {len(checks) + 1}"))

# ---- report ---------------------------------------------------------------
# Identity tests against False are what let a numpy bool slip through; ask for
# truthiness instead, and treat only an explicit None as "not runnable".
skipped = [c for c in checks if c[1] is None]
bad = [c for c in checks if c[1] is not None and not c[1]]
for label, ok, detail in checks:
    mark = "ok  " if ok else ("skip" if ok is None else "FAIL")
    if not ok:
        print(f"{mark}  {label:<38} {detail}")
_odd = [c[0] for c in checks if c[1] is not None and not isinstance(c[1], bool)]
if _odd:
    print(f"warning: {len(_odd)} checks have a non-bool verdict: {_odd[:3]}")
print(f"\n{len(checks) - len(bad) - len(skipped)} verified, "
      f"{len(bad)} mismatched, {len(skipped)} not yet runnable")
sys.exit(1 if bad else 0)
