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
{extra}"""

'''

# Vendoring replaces each module docstring with the header above. For most
# modules that only drops project framing. For crossconformal.py it dropped the
# approximate-validity caveat -- the one thing a user of strategy="cross" most
# needs to read -- so the shipped module carried no warning at all while PR.md
# claimed it "says so where it matters". Carried over explicitly instead, and
# scripts/verify_claims.py fails if it ever stops arriving.
CARRIED_CAVEATS = {
    "crossconformal.py": '''
Honest caveat
-------------
Pooling out-of-fold scores and then applying them to a model fitted on the full
pool gives *approximate* validity, not the exact finite-sample guarantee of
split conformal (see Vovk 2015 on cross-conformal predictors; the related CV+
of Barber et al. 2021 bounds worst-case coverage at ``1 - 2*alpha``).

This is not only theoretical. Measured across two datasets, realized coverage
sat below the level each run actually certifies in 3 of 6 dataset-alpha
combinations, by 1.5 to 2.4 points, concentrated at tight alpha -- where split
conformal was below in 0 of 6. Report empirical coverage alongside this caveat
rather than claiming the split-conformal guarantee this does not have.
''',
}

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
about specifically for TabPFN; K-fold cross-conformal spends none of the
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
| `method="mondrian"` | class-conditional calibration, one threshold per class |
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
further calls to the estimator, which matters when each call is a metered API
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

from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn_extensions.conformal import (
    ACI,
    ConformalClassifier,
    average_set_size,
    conformal_quantile,
    coverage_by_class,
    decision_summary,
    empty_set_rate,
    marginal_coverage,
    out_of_fold_proba,
    route,
)
from tabpfn_extensions.conformal.crossconformal import aligned_proba
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


# --- regression guards -----------------------------------------------------
# Each of the following was written after a mutation of the library passed the
# rest of this file. They are the tests that make the module's own docstrings
# true, so they travel with it.

class _Memoriser(BaseEstimator, ClassifierMixin):
    """Answers 1.0 on the true class of any row it was fitted on.

    A fold leak is otherwise nearly invisible: on easy data a model that trained
    on its own test rows scores them barely better than one that did not, so a
    coverage assertion will not notice. This makes the leak total.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.seen_ = {row.tobytes(): label for row, label in zip(X, y)}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
        for i, row in enumerate(X):
            label = self.seen_.get(row.tobytes())
            if label is not None:
                out[i] = 0.0
                out[i, int(np.searchsorted(self.classes_, label))] = 1.0
        return out


def test_no_row_is_scored_by_a_model_that_saw_it():
    """Cross-conformal's defining property, asserted rather than assumed."""
    X, y = imbalanced(n=600, minority_rate=0.1)
    proba = out_of_fold_proba(_Memoriser(), X, y, np.array([0, 1]),
                              n_folds=5, random_state=0)
    np.testing.assert_allclose(proba, 0.5)


def test_the_final_model_is_fitted_on_every_row():
    """Its second property: no label is spent, so none leaves the context."""
    X, y = imbalanced(n=600, minority_rate=0.1)
    cc = ConformalClassifier(_Memoriser(), strategy="cross", n_folds=5,
                             random_state=0).fit(X, y)
    seen = cc.estimator_.predict_proba(X)
    np.testing.assert_allclose(seen[np.arange(len(y)), y], 1.0)


def test_a_score_exactly_on_the_threshold_is_inside_the_set():
    """Membership is `score <= threshold`, inclusive.

    Continuous probabilities never tie, so a strict `<` passes every statistical
    test here while dropping the boundary point. Discrete scores make ties
    routine, and there it loses the guarantee.
    """
    class Grid(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p1 = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1.0 - p1, p1])

    grid = np.round(np.arange(1, 11) * 0.1, 2)
    X_cal = np.concatenate([1.0 - grid, grid]).reshape(-1, 1)
    y_cal = np.array([1] * 10 + [0] * 10)
    cc = ConformalClassifier(Grid(), method="mondrian", prefit=True).fit(X_cal, y_cal)

    # n=10, alpha=0.2 -> k = ceil(11 * 0.8) = 9 -> the 9th smallest score = 0.9.
    assert cc.quantiles(0.2)[1] == pytest.approx(0.9)
    assert cc.predict_set(np.array([[1.0 - 0.9]]), 0.2)[0, 1]
    assert not cc.predict_set(np.array([[1.0 - 0.95]]), 0.2)[0, 1]


def test_a_folds_missing_class_is_realigned_not_trusted_positionally():
    """`classes_` in another order must not be read by column position."""
    class Reversed(BaseEstimator, ClassifierMixin):
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    out = aligned_proba(Reversed(), np.zeros((3, 2)), np.array([0, 1]))
    np.testing.assert_allclose(out, np.tile([0.8, 0.2], (3, 1)))


def test_metrics_are_what_they_say():
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    y = np.array([0, 0, 1, 1])
    assert average_set_size(sets) == pytest.approx(1.0)
    assert empty_set_rate(sets) == pytest.approx(0.25)
    assert marginal_coverage(sets, y, np.array([0, 1])) == pytest.approx(0.5)


def test_classes_are_read_positionally_not_assumed_sorted():
    """`coverage_by_class` keys by position, so resolution must too."""
    classes = np.array(["legit", "fraud"])      # deliberately unsorted
    y = np.array(["fraud", "legit"])
    sets = np.array([[False, True], [True, False]])
    cov = coverage_by_class(sets, y, classes)
    assert cov["fraud"] == pytest.approx(1.0)
    assert cov["legit"] == pytest.approx(1.0)


def test_a_dataframe_with_a_non_default_index_is_taken_positionally():
    """Any filtered or merged frame has a non-default index."""
    pd = pytest.importorskip("pandas")
    X, y = imbalanced(n=600, minority_rate=0.15)
    cols = [f"f{i}" for i in range(X.shape[1])]
    ref = ConformalClassifier(LogisticRegression(max_iter=1000), strategy="cross",
                              n_folds=3, random_state=0)
    ref.fit(pd.DataFrame(X, columns=cols), y)
    base = ref.predict_set(pd.DataFrame(X, columns=cols), ALPHA)

    scrambled = pd.DataFrame(X, columns=cols,
                             index=np.random.default_rng(0).permutation(len(X)))
    other = ConformalClassifier(LogisticRegression(max_iter=1000), strategy="cross",
                                n_folds=3, random_state=0).fit(scrambled, y)
    np.testing.assert_array_equal(base, other.predict_set(scrambled, ALPHA))


def test_non_finite_probabilities_are_rejected():
    """NaN scores would otherwise yield silently empty sets, which `route`
    treats as the highest priority for human review."""
    class NaNEstimator(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)

    X, y = imbalanced(n=200, minority_rate=0.2)
    with pytest.raises(ValueError, match="non-finite"):
        ConformalClassifier(NaNEstimator(), prefit=True).fit(X, y)


def test_alpha_dict_reports_the_adapted_levels_not_the_target():
    """ACI's handoff into `predict_set`.

    If this returned `alpha_target`, the adaptation would be computed and then
    discarded, indistinguishable from the (real) finding that ACI cannot move
    the threshold at a small calibration set.
    """
    aci = ACI(alpha_target=0.05, gamma=0.1, n_classes=2)
    for _ in range(5):
        aci.update_round(1, 0.9)
    assert aci.alphas_[1] != pytest.approx(0.05)
    assert aci.alpha_dict()[1] == pytest.approx(float(aci.alphas_[1]))
    assert aci.alpha_dict()[1] != pytest.approx(aci.alpha_target)
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


PR_MD = '# Contributing `conformal` to tabpfn-extensions\n\nEverything under `contrib/tabpfn-extensions/` is **generated** by\n`scripts/build_extension_pr.py` from `src/tabpfn_conformal/`. Regenerate rather\nthan editing here, so the contribution cannot drift from the library.\n\n## Before the PR\n\nCONTRIBUTING.md asks for an issue first:\n\n> Before developing a feature / opening a PR, please open a GitHub issue and\n> describe the bug or feature request.\n\nDraft issue text is in [`ISSUE.md`](ISSUE.md).\n\n## Opening it\n\n```bash\ngit clone https://github.com/PriorLabs/tabpfn-extensions.git\ncd tabpfn-extensions && uv sync\ncp -R <this-repo>/contrib/tabpfn-extensions/src/tabpfn_extensions/conformal src/tabpfn_extensions/\ncp <this-repo>/contrib/tabpfn-extensions/tests/test_conformal.py tests/\ncp -R <this-repo>/contrib/tabpfn-extensions/examples/conformal examples/\n# their check-changelog workflow fails a PR without this; rename it to the PR number\ncp <this-repo>/contrib/tabpfn-extensions/changelog/PRNUMBER.added.md changelog/<PR>.added.md\nFAST_TEST_MODE=1 pytest tests/test_conformal.py -v\n```\n\nVerified locally: the module imports as `tabpfn_extensions.conformal` and all\n**all tests pass under `FAST_TEST_MODE=1` in under a second**, with no TabPFN import and no\nGPU.\n\n## PR description\n\n> ### Conformal prediction for classification\n>\n> `cp_missing_data` already provides conformal **regression** intervals\n> specialised to missing-data patterns. This adds the classification side:\n> prediction sets with a finite-sample, distribution-free coverage guarantee.\n>\n> **What it adds**\n>\n> - `ConformalClassifier`, scikit-learn compatible, wraps any `predict_proba`\n>   estimator\n> - `method="mondrian"`, class-conditional calibration, one threshold per class.\n>   Marginal conformal spends its error budget where the mass is, so under heavy\n>   imbalance the minority class can fall far below `1 - alpha` while the headline\n>   number looks healthy.\n> - `strategy="cross"`, K-fold cross-conformal, which spends no labelled\n>   positives on calibration\n> - `ACI`, adaptive conformal inference for drifting streams\n> - `route`, prediction sets to approve / block / review under a review budget\n>\n> **Why `strategy="cross"` is worth having here specifically.** Cross-conformal\n> normally costs K refits, which is why split conformal is the default everywhere.\n> TabPFN has no training step, so it costs K forward passes. Measured on Bank\n> Account Fraud (NeurIPS 2022) with TabPFN-3.5, comparing at an identical targeted\n> coverage level, cross-conformal reached the same level from **half the confirmed\n> fraud labels** at no cost in set width: across four datasets and nine paired\n> comparisons it was significantly wider in **zero** and significantly narrower\n> in one, at the scarcest label budget.\n>\n> **Dependencies:** none beyond numpy and scikit-learn. Nothing in the module\n> imports TabPFN, so it adds nothing to the base install.\n>\n> **Tests:** 18 tests, CPU only, `FAST_TEST_MODE` aware, about a second. They use\n> scikit-learn estimators rather than TabPFN, since the machinery is model-agnostic\n> and testing against TabPFN would slow the suite without covering anything extra.\n>\n> Nine are regression guards, each written after a deliberate mutation of the\n> library slipped past the rest: a fold leak in cross-conformal, `<` in place of\n> `<=` at the threshold, a class remapping ignored inside a fold, an off-by-one\n> in `average_set_size`, and the ACI level being discarded on handoff. Every one\n> was confirmed to fail on the mutation it guards.\n>\n> **Scope.** Prediction sets for any number of classes, `tests/test_multiclass.py`\n> covers 3 and 5, where the class-conditional argument is if anything stronger\n> (marginal leaves the worst class at 0.675 coverage against a 0.90 target;\n> Mondrian holds 0.890). Should compose with `ManyClassClassifier`. The\n> benchmarks here are binary, because the motivating problem is.\n>\n> **What it costs, measured.** Cross-conformal is approximately valid rather\n> than exactly valid (Vovk 2015; CV+ worst case `1 - 2*alpha`), and that is not\n> only theoretical. Comparing each run against the level it actually certifies\n> -- `ceil((n+1)(1-alpha))/n`, not the nominal `1-alpha` -- cross sat below its\n> own certified level in **3 of 6** dataset-alpha combinations, by 1.5 to 2.4\n> points, concentrated at tight alpha; split was below in **0 of 6**. So the\n> trade is half the labels against about two points of realized coverage, not a\n> free lunch. The module docstring carries this caveat, so it reaches anyone who\n> reads `strategy="cross"` rather than only this PR.\n> The methods are standard; this is packaging, not new statistics.\n>\n> Benchmarks, figures and full method notes:\n> https://github.com/ilyas-elm/tabpfn-conformal\n'

ISSUE_MD = "# Draft issue (post before the PR, per CONTRIBUTING.md)\n\n**Title:** Conformal prediction for classification\n\n---\n\n`cp_missing_data` provides conformal **regression** intervals specialised to\nmissing-data patterns. There is currently no conformal prediction for\n**classification** in the repo, no prediction sets, no class-conditional\ncalibration, no cross-conformal.\n\nI'd like to contribute a `conformal` module covering that. Briefly:\n\n- `ConformalClassifier`, scikit-learn compatible, wrapping any `predict_proba`\n  estimator, prediction sets with a finite-sample distribution-free guarantee\n- class-conditional (Mondrian) calibration, which is what imbalanced problems\n  need: marginal conformal can leave the minority class far below `1 - alpha`\n  while the overall number looks fine\n- K-fold cross-conformal, which spends no labelled positives on calibration.\n  This is the piece that seems worth having in *this* repo specifically;\n  cross-conformal normally costs K refits, and TabPFN has no training step.\n- adaptive conformal inference for drifting streams, and a decision layer that\n  maps prediction sets to approve / block / review under a review budget\n\nNo dependencies beyond numpy and scikit-learn; nothing in it imports TabPFN.\nTests are CPU-only, `FAST_TEST_MODE` aware, and run in under a second.\n\nWorking implementation, benchmarks on Bank Account Fraud with TabPFN-3.5, and\nthe method notes are here: https://github.com/ilyas-elm/tabpfn-conformal\n\nHappy to adjust the scope or the API before opening a PR, in particular whether\nyou'd prefer the decision layer left out, since it is more applied than the rest.\n"


# Their check-changelog workflow (scientific-python/action-towncrier-changelog)
# fails any PR without a fragment in changelog/. Categories are breaking, added,
# changed, fixed, deprecated; the filename is <PR_NUMBER>.<category>.md.
CHANGELOG_FRAGMENT = (
    "Add `tabpfn_extensions.conformal`: distribution-free prediction sets for "
    "classification, with class-conditional calibration and a cross-conformal "
    "strategy that spends no labelled positives on calibration.\n"
)


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
        (mod / name).write_text(
            HEADER.format(extra=CARRIED_CAVEATS.get(name, "")) + text
        )
    (mod / "__init__.py").write_text(INIT)
    (mod / "README.md").write_text(MODULE_README)

    (OUT / "tests").mkdir(parents=True)
    (OUT / "tests" / "test_conformal.py").write_text(TESTS)
    (OUT / "examples" / "conformal").mkdir(parents=True)
    (OUT / "examples" / "conformal" / "imbalanced_coverage.py").write_text(EXAMPLE)

    # These were hand-written once and then eaten by the rmtree above. Anything
    # living under OUT must be generated here, or it will not survive a rebuild.
    (OUT / "changelog").mkdir(parents=True, exist_ok=True)
    (OUT / "changelog" / "PRNUMBER.added.md").write_text(CHANGELOG_FRAGMENT)
    (OUT / "PR.md").write_text(PR_MD)
    (OUT / "ISSUE.md").write_text(ISSUE_MD)

    print(f"wrote {len(MODULES) + 2} module files to {mod.relative_to(REPO)}")
    print(f"wrote {(OUT / 'PR.md').relative_to(REPO)} and ISSUE.md")
    print(f"wrote {(OUT / 'tests' / 'test_conformal.py').relative_to(REPO)}")
    print(f"wrote {(OUT / 'examples' / 'conformal' / 'imbalanced_coverage.py').relative_to(REPO)}")
    print("\nTo open the PR: clone tabpfn-extensions, copy these three trees over, "
          "and run\n  FAST_TEST_MODE=1 pytest tests/test_conformal.py -v")


if __name__ == "__main__":
    build()
