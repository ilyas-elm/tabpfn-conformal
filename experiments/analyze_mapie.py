#!/usr/bin/env python3
"""Record the MAPIE head-to-head numbers the README quotes.

``tests/test_agreement_with_mapie.py`` already asserts these claims, but
deliberately with loose bounds: it says agreement exceeds 98% rather than
pinning 99.7%, so that a MAPIE release changing its aggregation does not turn
into a flaky failure. That is the right call for a test and it left the exact
figures in the README standing on nothing, which a sweep of every table cell
found.

So the figures get the same treatment as every other number here: computed by
a committed script into ``results/``, and compared against the document by
``scripts/verify_claims.py``. Rerun this when MAPIE is upgraded; if the numbers
move, the README is what is wrong.

    python experiments/analyze_mapie.py        # needs mapie, no API key

Same construction as the tests, deliberately: same fixtures, same seeds, same
splits, so the recorded numbers are the ones the tests are asserting about.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from conftest import make_imbalanced  # noqa: E402

from tabpfn_conformal import (  # noqa: E402
    ConformalClassifier,
    average_set_size,
    coverage_by_class,
    marginal_coverage,
)

ALPHAS = (0.05, 0.1, 0.2)
SEEDS = (0, 1, 2)


def cross_agreement(mapie_classification) -> dict:
    """Ours (Vovk 2015) against MAPIE's CV+ (Barber et al. 2021).

    Different constructions, so exact agreement would be suspicious. What the
    README claims is that they land in the same place.
    """
    per_alpha = {}
    for alpha in ALPHAS:
        agree, d_cov, d_size = [], [], []
        for seed in SEEDS:
            X, y = make_imbalanced(n_samples=4000, minority_rate=0.05, seed=seed)
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=seed
            )
            ours = ConformalClassifier(
                LogisticRegression(max_iter=1000), method="marginal",
                strategy="cross", n_folds=5, random_state=seed,
            ).fit(X_pool, y_pool)
            our_sets = ours.predict_set(X_test, alpha)

            theirs = mapie_classification.CrossConformalClassifier(
                estimator=LogisticRegression(max_iter=1000),
                confidence_level=1 - alpha, conformity_score="lac",
                cv=5, random_state=seed,
            )
            theirs.fit_conformalize(X_pool, y_pool)
            _, raw = theirs.predict_set(X_test)
            their_sets = np.asarray(raw).reshape(our_sets.shape)

            agree.append(float((our_sets == their_sets).all(axis=1).mean()))
            d_cov.append(abs(marginal_coverage(our_sets, y_test, ours.classes_)
                             - marginal_coverage(their_sets, y_test, ours.classes_)))
            d_size.append(abs(average_set_size(our_sets) - average_set_size(their_sets)))
        per_alpha[str(alpha)] = {
            "sets_identical": float(np.mean(agree)),
            "abs_coverage_gap": float(np.mean(d_cov)),
            "abs_set_size_gap": float(np.mean(d_size)),
        }
    return per_alpha


def class_conditional(mapie_classification) -> dict:
    """The capability gap: marginal CV+ against Mondrian cross-conformal.

    One seed and one alpha, the configuration the README tabulates.
    """
    alpha = 0.1
    X, y = make_imbalanced(n_samples=4000, minority_rate=0.04, seed=0)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    theirs = mapie_classification.CrossConformalClassifier(
        estimator=LogisticRegression(max_iter=500), confidence_level=1 - alpha,
        conformity_score="lac", cv=5, random_state=0,
    )
    theirs.fit_conformalize(X_pool, y_pool)
    _, raw = theirs.predict_set(X_test)
    their_sets = np.asarray(raw).reshape(len(X_test), 2, -1)[:, :, 0]
    their_cov = coverage_by_class(their_sets, y_test, np.array([0, 1]))

    ours = ConformalClassifier(
        LogisticRegression(max_iter=500), method="mondrian",
        strategy="cross", n_folds=5, random_state=0,
    ).fit(X_pool, y_pool)
    our_sets = ours.predict_set(X_test, alpha=alpha)
    our_cov = coverage_by_class(our_sets, y_test, ours.classes_)

    return {
        "alpha": alpha,
        "minority_rate": float(np.mean(y == 1)),
        "mapie_cv_plus_lac": {
            "minority_coverage": float(their_cov[1]),
            "majority_coverage": float(their_cov[0]),
            "mean_set_size": float(average_set_size(their_sets)),
        },
        "ours_mondrian_cross": {
            "minority_coverage": float(our_cov[1]),
            "majority_coverage": float(our_cov[0]),
            "mean_set_size": float(average_set_size(our_sets)),
        },
    }


def main() -> int:
    try:
        from mapie import classification as mapie_classification
    except ImportError:
        print("mapie is not installed; pip install -e '.[dev]'", file=sys.stderr)
        return 2

    import mapie as _m
    out = {
        "mapie_version": getattr(_m, "__version__", "unknown"),
        "note": "Same fixtures, seeds and splits as tests/test_agreement_with_mapie.py.",
        "cross_vs_cv_plus": cross_agreement(mapie_classification),
        "class_conditional": class_conditional(mapie_classification),
    }
    dest = REPO / "results" / "mapie_comparison.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")

    agree = [v["sets_identical"] for v in out["cross_vs_cv_plus"].values()]
    cc = out["class_conditional"]
    print(f"wrote {dest.relative_to(REPO)} (mapie {out['mapie_version']})")
    print(f"  cross vs CV+: {np.mean(agree):.1%} of prediction sets identical "
          f"across alpha in {{0.05, 0.1, 0.2}}")
    print(f"  worst coverage gap {max(v['abs_coverage_gap'] for v in out['cross_vs_cv_plus'].values()):.4f}, "
          f"worst set-size gap {max(v['abs_set_size_gap'] for v in out['cross_vs_cv_plus'].values()):.4f}")
    print(f"  class-conditional at {cc['minority_rate']:.1%} minority, "
          f"alpha {cc['alpha']}: MAPIE minority coverage "
          f"{cc['mapie_cv_plus_lac']['minority_coverage']:.3f}, "
          f"ours {cc['ours_mondrian_cross']['minority_coverage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
