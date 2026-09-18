"""Synthetic imbalanced fixtures. CPU only, no TabPFN, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification


def make_imbalanced(n_samples=6000, minority_rate=0.02, seed=0, n_features=12):
    """A binary problem with a controllable minority rate.

    ``make_classification`` only approximates the requested weights, so the
    realised rate is close to but not exactly ``minority_rate``.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1.0 - minority_rate, minority_rate],
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )
    return X, y


@pytest.fixture
def imbalanced():
    return make_imbalanced()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
