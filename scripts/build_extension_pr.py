"""Generate the `tabpfn-extensions` contribution from this package.

Vendoring by hand rots. This regenerates the entire PR payload from
`src/tabpfn_conformal/`, so the contribution cannot drift from the library it
came from. Run it again before opening or updating the PR.

Layout follows CONTRIBUTING.md in tabpfn-extensions:

    src/tabpfn_extensions/conformal/   the module
    tests/test_conformal.py            tests (FAST_TEST_MODE aware)
    examples/conformal/                a runnable example

    python scripts/build_extension_pr.py
"""

from __future__ import annotations

import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tabpfn_conformal"
OUT = REPO / "contrib" / "tabpfn-extensions"

MODULES = [
    "scores.py", "calibration.py", "crossconformal.py",
    "wrapper.py", "metrics.py", "adaptive.py", "decision.py",
]

HEADER = '''"""Conformal prediction for TabPFN classification.

Vendored from https://github.com/ilyas-elm/tabpfn-conformal (Apache 2.0).
Regenerate with `scripts/build_extension_pr.py` in that repository rather than
editing here, so the two cannot drift apart.
"""

'''

INIT = '''"""Distribution-free coverage guarantees for TabPFN classification.

`cp_missing_data` already provides conformal *regression* intervals specialised
to missing-data patterns. This module covers the classification side: prediction
sets with a finite-sample, distribution-free coverage guarantee, including the
class-conditional (Mondrian) variant that extreme class imbalance needs.

    from tabpfn_extensions.conformal import ConformalClassifier
    from tabpfn_extensions.utils import TabPFNClassifier

    cc = ConformalClassifier(TabPFNClassifier(), method="mondrian", strategy="cross")
    cc.fit(X_pool, y_pool)
    sets = cc.predict_set(X_new, alpha=0.05)     # (n, n_classes) boolean

Nothing here imports TabPFN: it wraps any estimator exposing `predict_proba`,
and depends only on numpy and scikit-learn. `strategy="cross"` is worth knowing
about specifically for TabPFN — K-fold cross-conformal spends none of the
labelled positives on calibration, which normally costs K refits and costs K
forward passes for a model with no training step.
"""

from .adaptive import ACI
from .calibration import (
    InsufficientCalibrationWarning,
    conformal_quantile,
    marginal_thresholds,
    mondrian_thresholds,
)
from .crossconformal import out_of_fold_proba
from .decision import decision_summary, route
from .metrics import (
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_coverage,
)
from .scores import SCORES, get_score, neg_log_prob, one_minus_prob
from .wrapper import ConformalClassifier

__all__ = [
    "ConformalClassifier",
    "ACI",
    "route",
    "decision_summary",
    "conformal_quantile",
    "marginal_thresholds",
    "mondrian_thresholds",
    "InsufficientCalibrationWarning",
    "out_of_fold_proba",
    "one_minus_prob",
    "neg_log_prob",
    "get_score",
    "SCORES",
    "marginal_coverage",
    "coverage_by_class",
    "average_set_size",
    "empty_set_rate",
]
'''

MODULE_README = """# conformal

Distribution-free coverage guarantees for TabPFN **classification**.

`cp_missing_data` covers conformal *regression* intervals under missing-data
patterns. This module covers classification: prediction sets carrying a
finite-sample, distribution-free guarantee, with the class-conditional variant
that imbalanced problems need.

```python
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.conformal import ConformalClassifier, coverage_by_class

cc = ConformalClassifier(TabPFNClassifier(), method="mondrian", strategy="cross")
cc.fit(X_pool, y_pool)
sets = cc.predict_set(X_test, alpha=0.05)        # (n, n_classes) boolean
coverage_by_class(sets, y_test, cc.classes_)     # per-class realised coverage
```

## What it provides

| | |
|---|---|
| `ConformalClassifier` | scikit-learn compatible wrapper around any `predict_proba` estimator |
| `method="mondrian"` | class-conditional calibration — one threshold per class |
| `strategy="cross"` | K-fold cross-conformal: no labels spent on calibration |
| `ACI` | adaptive conformal inference for drifting streams |
| `route` | prediction sets to approve / block / review under a review budget |

## Why `strategy="cross"` matters for TabPFN

Split conformal holds out part of the labelled data to calibrate, so under
extreme imbalance a large share of the scarce positives never reach the model.
Cross-conformal avoids that, and normally costs K refits. TabPFN has no training
step, so it costs K forward passes.

Measured on Bank Account Fraud (NeurIPS 2022) with TabPFN-3.5, comparing at an
identical targeted coverage level: cross-conformal reached the same level from
**half the confirmed fraud labels**, with narrower prediction sets in five of six
comparisons.

## Two behaviours worth knowing

**`alpha` is supplied at prediction time**, not calibration time. The calibration
scores are stored and thresholds derived on demand, so sweeping `alpha` costs no
further calls to the estimator — which matters when each call is a metered API
request. `predict_set_from_proba` goes further: score once, sweep offline.

**Too few calibration points raises rather than clips.** A calibration set of
size `n` can only certify `alpha >= 1/(n+1)`. Below that the module returns the
trivial all-labels set and raises `InsufficientCalibrationWarning`, instead of
silently returning a threshold that does not carry the guarantee.

## Attribution

Vendored from [tabpfn-conformal](https://github.com/ilyas-elm/tabpfn-conformal)
(Apache 2.0), where the benchmarks, figures and full method notes live.
"""


TESTS = '''"""Tests for the conformal extension.

CPU only: these use scikit-learn estimators, not TabPFN, because the conformal
machinery is model-agnostic and testing it against TabPFN would make the suite
slow without testing anything extra. Honours FAST_TEST_MODE.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tabpfn_extensions.conformal import (
    ACI,
    ConformalClassifier,
    conformal_quantile,
    coverage_by_class,
    decision_summary,
    marginal_coverage,
    route,
)
from tabpfn_extensions.conformal.calibration import InsufficientCalibrationWarning

FAST = os.environ.get("FAST_TEST_MODE") == "1"
N_SEEDS = 3 if FAST else 8
N_SAMPLES = 2000 if FAST else 6000
ALPHA = 0.1


def imbalanced(n=N_SAMPLES, minority_rate=0.02, seed=0):
    return make_classification(
        n_samples=n, n_features=12, n_informative=6, n_redundant=2,
        n_clusters_per_class=2, weights=[1 - minority_rate, minority_rate],
        flip_y=0.01, class_sep=1.0, random_state=seed,
    )


def run(method, seed, strategy="split", minority_rate=0.02):
    X, y = imbalanced(minority_rate=minority_rate, seed=seed)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    cc = ConformalClassifier(
        LogisticRegression(max_iter=1000), method=method,
        strategy=strategy, random_state=seed,
    ).fit(X_fit, y_fit)
    sets = cc.predict_set(X_test, ALPHA)
    return sets, y_test, cc


def test_quantile_uses_the_finite_sample_correction():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.5)
    assert conformal_quantile(scores, 0.5) == pytest.approx(0.3)
    assert conformal_quantile(scores, 0.1) != pytest.approx(float(np.quantile(scores, 0.9)))


def test_too_few_calibration_points_warns_and_is_conservative():
    with pytest.warns(InsufficientCalibrationWarning):
        assert conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4]), 0.1) == np.inf


def test_marginal_method_achieves_marginal_coverage():
    cov = [marginal_coverage(*run("marginal", s)[:2], run("marginal", s)[2].classes_)
           for s in range(N_SEEDS)]
    assert np.mean(cov) >= 1 - ALPHA - 0.03


def test_mondrian_covers_both_classes():
    per_class = []
    for s in range(N_SEEDS):
        sets, y_test, cc = run("mondrian", s)
        per_class.append(coverage_by_class(sets, y_test, cc.classes_))
    for k in (0, 1):
        assert np.mean([c[k] for c in per_class]) >= 1 - ALPHA - 0.05, f"class {k}"


def test_marginal_under_covers_the_minority_class():
    """The failure mode class-conditional calibration exists to fix."""
    def minority(method):
        out = []
        for s in range(N_SEEDS):
            sets, y_test, cc = run(method, s, minority_rate=0.01)
            out.append(coverage_by_class(sets, y_test, cc.classes_)[1])
        return float(np.mean(out))

    assert minority("marginal") < 1 - ALPHA
    assert minority("mondrian") > minority("marginal")


def test_cross_conformal_spends_no_labels_on_calibration():
    X, y = imbalanced(seed=0)
    split = ConformalClassifier(LogisticRegression(max_iter=1000),
                                strategy="split", random_state=0).fit(X, y)
    cross = ConformalClassifier(LogisticRegression(max_iter=1000),
                                strategy="cross", n_folds=5, random_state=0).fit(X, y)
    assert sum(cross.n_calibration_.values()) == len(y)
    assert cross.n_calibration_[1] > split.n_calibration_[1]


def test_alpha_is_a_prediction_time_argument():
    sets, _, cc = run("mondrian", 0)
    X, y = imbalanced(seed=0)
    assert cc.predict_set(X, 0.01).sum() > cc.predict_set(X, 0.2).sum()


def test_aci_updates_one_step_per_round():
    aci = ACI(alpha_target=0.05, gamma=0.02, n_classes=2)
    assert aci.update_round(1, error_rate=1.0) < 0.05
    assert aci.n_updates_[1] == 1


def test_route_respects_the_review_budget():
    sets = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [0, 0]], dtype=bool)
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.45, 0.55], [0.7, 0.3], [0.5, 0.5]])
    for k in range(6):
        assert (route(sets, proba, budget_k=k) == "review").sum() <= k
    summary = decision_summary(route(sets, proba, budget_k=2), [0, 1, 1, 0, 1], budget_k=2)
    assert summary["n_review"] == 2
'''

EXAMPLE = '''"""Conformal prediction sets for TabPFN classification.

Shows the two things this module adds over an uncalibrated probability: a
per-class coverage guarantee, and the cross-conformal strategy that spends none
of the scarce positive labels on calibration.

    python examples/conformal/imbalanced_coverage.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.conformal import (
    ConformalClassifier,
    average_set_size,
    coverage_by_class,
)

ALPHA = 0.1

X, y = make_classification(
    n_samples=2000, n_features=12, n_informative=6, weights=[0.98, 0.02],
    flip_y=0.01, random_state=0,
)
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=0
)
print(f"pool: {len(y_pool)} rows, {int(y_pool.sum())} positives")

for strategy in ("split", "cross"):
    cc = ConformalClassifier(
        TabPFNClassifier(), method="mondrian", strategy=strategy,
        n_folds=5, random_state=0,
    ).fit(X_pool, y_pool)

    sets = cc.predict_set(X_test, ALPHA)
    cov = coverage_by_class(sets, y_test, cc.classes_)
    print(
        f"{strategy:>6}: calibrated on {cc.n_calibration_[1]:>3} positives | "
        f"minority coverage {cov[1]:.3f} (target {1 - ALPHA:.2f}) | "
        f"mean set size {average_set_size(sets):.3f}"
    )

# Coverage and set size must be read together: a predictor returning every label
# has perfect coverage and no value.
'''


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    mod = OUT / "src" / "tabpfn_extensions" / "conformal"
    mod.mkdir(parents=True)

    for name in MODULES:
        text = (SRC / name).read_text()
        # Strip the original module docstring and prepend the attribution header.
        if text.startswith('"""'):
            text = text[text.index('"""', 3) + 3:].lstrip("\n")
        (mod / name).write_text(HEADER + text)
    (mod / "__init__.py").write_text(INIT)
    (mod / "README.md").write_text(MODULE_README)

    (OUT / "tests").mkdir(parents=True)
    (OUT / "tests" / "test_conformal.py").write_text(TESTS)
    (OUT / "examples" / "conformal").mkdir(parents=True)
    (OUT / "examples" / "conformal" / "imbalanced_coverage.py").write_text(EXAMPLE)

    print(f"wrote {len(MODULES) + 2} module files to {mod.relative_to(REPO)}")
    print(f"wrote {(OUT / 'tests' / 'test_conformal.py').relative_to(REPO)}")
    print(f"wrote {(OUT / 'examples' / 'conformal' / 'imbalanced_coverage.py').relative_to(REPO)}")
    print("\nTo open the PR: clone tabpfn-extensions, copy these three trees over, "
          "and run\n  FAST_TEST_MODE=1 pytest tests/test_conformal.py -v")


if __name__ == "__main__":
    build()
