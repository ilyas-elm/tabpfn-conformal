"""The committed results must stay reproducible by the current library.

`results/*.jsonl` was produced by the library as it stood when the API
experiments ran. Every change to `src/` since then has to leave those numbers
alone, or the README is quoting figures its own code no longer produces.

These are golden-value tests against the real saved probabilities in
`results/proba/`, not synthetic data: if a metric function changes what it
returns for the inputs the experiments actually used, this fails.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tabpfn_conformal import (
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_coverage,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBA = REPO / "results" / "proba"
CLASSES = np.array([0, 1])
THRESHOLDS = (0.01, 0.05, 0.2, 0.5, 0.9)
GOLDEN_PATH = pathlib.Path(__file__).parent / "data" / "metric_goldens.json"

pytestmark = pytest.mark.skipif(
    not PROBA.exists(), reason="committed probabilities not present"
)


def _sets(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Mondrian-shaped sets at a fixed score threshold, as the experiments build them."""
    return np.column_stack([1 - proba[:, 1] <= threshold, proba[:, 1] <= threshold])


def _files():
    return sorted(PROBA.rglob("*.npz"))


def _digest(path: pathlib.Path) -> dict:
    """One row of golden values for a single probability file."""
    z = np.load(path)
    proba, y = z["proba"].astype(float), z["y_true"]
    out = {}
    for thr in THRESHOLDS:
        sets = _sets(proba, thr)
        by_class = coverage_by_class(sets, y, CLASSES)
        out[str(thr)] = [
            round(marginal_coverage(sets, y, CLASSES), 12),
            round(by_class[0], 12),
            round(by_class[1], 12),
            round(average_set_size(sets), 12),
            round(empty_set_rate(sets), 12),
        ]
    return out


def test_the_committed_probabilities_are_still_there():
    recorded = json.loads(GOLDEN_PATH.read_text())
    present = {f.relative_to(PROBA).as_posix() for f in _files()}
    missing = sorted(set(recorded) - present)
    assert not missing, f"probability files recorded but now absent: {missing[:5]}"


def test_metrics_are_stable_on_the_real_saved_probabilities():
    """Golden values per file, so a later experiment adding results cannot move them.

    Recorded from the library as it stood when the results were produced
    (commit 1418523) and verified identical afterwards across 3,760
    comparisons. A change that moves any of them moves the README.
    """
    recorded = json.loads(GOLDEN_PATH.read_text())
    drifted = []
    for rel, want in recorded.items():
        got = _digest(PROBA / rel)
        for thr, values in want.items():
            if any(abs(a - b) > 1e-9 for a, b in zip(got[thr], values)):
                drifted.append((rel, thr, values, got[thr]))
    assert not drifted, (
        f"{len(drifted)} file/threshold pairs no longer reproduce. First: "
        f"{drifted[0]}. The library no longer reproduces the committed results."
    )


def test_every_probability_file_is_covered_by_a_golden():
    """A new experiment must be recorded, not silently left unprotected."""
    recorded = set(json.loads(GOLDEN_PATH.read_text()))
    present = {f.relative_to(PROBA).as_posix() for f in _files()}
    uncovered = sorted(present - recorded)
    assert not uncovered, (
        f"{len(uncovered)} probability file(s) have no golden value: "
        f"{uncovered[:5]}. Run `python scripts/record_metric_goldens.py`."
    )


def test_stored_set_sizes_match_what_the_library_computes_now():
    """E3 saves its calibration, so its sets can be rebuilt end to end."""
    rows = [json.loads(l) for l in (REPO / "results/e3.jsonl").read_text().splitlines() if l.strip()]
    checked = 0
    for r in rows:
        pf, cf = REPO / r["proba_file"], REPO / r.get("calibration_file", "")
        if not (pf.exists() and cf.exists()):
            continue
        cal = np.load(cf)
        cal_scores = 1.0 - cal["proba"][np.arange(len(cal["y_true"])), cal["y_true"]]
        cal_y = cal["y_true"]
        ev = np.load(pf)
        proba, y = ev["proba"].astype(float), ev["y_true"]

        from tabpfn_conformal import conformal_quantile
        alpha = r["alpha_target"]
        q = {k: conformal_quantile(cal_scores[cal_y == k], alpha, group=k) for k in (0, 1)}
        sets = np.column_stack([1 - proba[:, 0] <= q[0], 1 - proba[:, 1] <= q[1]])
        if r["arm"] == "frozen":                       # ACI/refit move the level
            assert average_set_size(sets) == pytest.approx(r["set_size"], abs=5e-3), (
                f"{r['model']}/{r['arm']} month {r['month']} seed {r['seed']}: "
                f"recomputed {average_set_size(sets):.4f} vs stored {r['set_size']:.4f}"
            )
            checked += 1
    assert checked >= 10, f"only {checked} rows were re-derivable; expected the frozen arm"


# Recorded from the library at commit 1418523, the state that produced the
# results, and re-verified against the current library file by file.
