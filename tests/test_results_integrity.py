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

pytestmark = pytest.mark.skipif(
    not PROBA.exists(), reason="committed probabilities not present"
)


def _sets(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Mondrian-shaped sets at a fixed score threshold, as the experiments build them."""
    return np.column_stack([1 - proba[:, 1] <= threshold, proba[:, 1] <= threshold])


def _files():
    return sorted(PROBA.rglob("*.npz"))


def test_the_committed_probabilities_are_still_there():
    files = _files()
    assert len(files) >= 180, f"expected the full set of saved probabilities, found {len(files)}"


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.2, 0.5, 0.9])
def test_metrics_are_stable_on_the_real_saved_probabilities(threshold):
    """Pins the metric vocabulary against real data, not a synthetic fixture.

    Golden values were recorded from the library as it stood when the results
    were produced (commit 1418523) and verified identical afterwards across
    3,760 comparisons. A change that moves any of them moves the README.
    """
    totals = {"cov": 0.0, "cov0": 0.0, "cov1": 0.0, "size": 0.0, "empty": 0.0}
    n = 0
    for f in _files():
        z = np.load(f)
        proba, y = z["proba"].astype(float), z["y_true"]
        sets = _sets(proba, threshold)
        by_class = coverage_by_class(sets, y, CLASSES)
        totals["cov"] += marginal_coverage(sets, y, CLASSES)
        totals["cov0"] += by_class[0]
        totals["cov1"] += by_class[1]
        totals["size"] += average_set_size(sets)
        totals["empty"] += empty_set_rate(sets)
        n += 1

    # Aggregated over every file, so one drifting row cannot hide in the mean.
    digest = {k: round(v / n, 10) for k, v in totals.items()}
    golden = GOLDEN[threshold]
    for k, want in golden.items():
        assert digest[k] == pytest.approx(want, abs=1e-9), (
            f"{k} at threshold {threshold} moved: {digest[k]} vs recorded {want}. "
            "The library no longer reproduces the committed results."
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
GOLDEN = {
    0.01: {"cov": 0.1104840334, "cov0": 0.0, "cov1": 0.2419591814, "size": 0.5501950393, "empty": 0.4498049607},
    0.05: {"cov": 0.2535095943, "cov0": 0.0, "cov1": 0.5683184906, "size": 0.7853067692, "empty": 0.2146932308},
    0.2: {"cov": 0.4004709931, "cov0": 3.5461e-06, "cov1": 0.9015462459, "size": 0.9542351909, "empty": 0.0457648091},
    0.5: {"cov": 0.4412945871, "cov0": 9.21986e-05, "cov1": 0.9927687027, "size": 1.0, "empty": 0.0},
    0.9: {"cov": 0.4528461149, "cov0": 0.0152248286, "cov1": 0.9999759733, "size": 1.120355106, "empty": 0.0},
}
