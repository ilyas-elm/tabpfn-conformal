#!/usr/bin/env python3
"""The same thing against TabPFN-3.5, which is the point of the project.

``examples/quickstart.py`` uses a scikit-learn model so that it runs anywhere.
This is the version that matters: identical conformal code, TabPFN underneath,
and the one property that makes K-fold cross-conformal affordable, namely that
``fit`` swaps the in-context set and takes no gradient step.

Needs a free Prior Labs account and no GPU, since inference runs on their
hardware:

    pip install -e ".[experiments]"
    python -c "import tabpfn_client; tabpfn_client.init()"   # one-time login
    python examples/with_tabpfn.py

It is deliberately small, a few hundred rows, so it costs very little. Every
experiment runner in ``experiments/`` takes ``--dry-run`` to price a larger run
through ``estimate_cost`` before spending anything.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tabpfn_conformal import ConformalClassifier, average_set_size, coverage_by_class

ALPHA = 0.1
POSITIVE = 1


def main() -> int:
    try:
        from tabpfn_client import TabPFNClassifier
    except ImportError:
        print('tabpfn-client is not installed. `pip install -e ".[experiments]"`, '
              'then `python -c "import tabpfn_client; tabpfn_client.init()"`.',
              file=sys.stderr)
        return 1

    X, y = make_classification(
        n_samples=900, n_features=15, n_informative=6, weights=[0.96, 0.04],
        flip_y=0.01, random_state=0,
    )
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    print(f"pool {len(y_pool)} rows, {int(y_pool.sum())} positives; "
          f"test {len(y_test)} rows, {int(y_test.sum())} positives\n")

    for strategy in ("split", "cross"):
        cc = ConformalClassifier(
            TabPFNClassifier(), method="mondrian", strategy=strategy,
            cal_size=0.5, n_folds=3, random_state=0,
        )
        # No gradient step happens here for either strategy. That is the whole
        # reason `strategy="cross"`, which refits K times, is usable at all.
        try:
            cc.fit(X_pool, y_pool)
        except RuntimeError as exc:
            # Installed but not logged in. A traceback is the wrong way to tell
            # someone their first run needs a one-time login.
            if "token" not in str(exc).lower():
                raise
            print("No Prior Labs token found, so nothing was sent.\n\n"
                  "  python -c \"import tabpfn_client; tabpfn_client.init()\"\n\n"
                  "logs in once with a free account. To see the same library "
                  "run with no\naccount at all, use examples/quickstart.py.",
                  file=sys.stderr)
            return 1
        sets = cc.predict_set(X_test, alpha=ALPHA)
        cov = coverage_by_class(sets, y_test, cc.classes_)
        n_cal = cc.n_calibration_[POSITIVE]
        print(f"strategy={strategy!r:<8} calibrated on {n_cal:>3} positives "
              f"(tightest certifiable alpha {1 / (n_cal + 1):.4f})")
        print(f"  positive coverage {cov[POSITIVE]:.3f}   "
              f"mean set size {average_set_size(sets):.3f}")

    print("\nBoth arms spent zero gradient-trained fits. Cross calibrated on "
          "every\nlabelled positive; split spent half of them to do it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
