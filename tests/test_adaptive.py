"""ACI: does the online level actually converge to the target error rate?"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_conformal import ACI


def test_miss_widens_and_hit_tightens():
    aci = ACI(alpha_target=0.05, gamma=0.02, n_classes=2)
    start = aci.alpha(1)
    assert aci.update(1, covered=False) < start      # missed -> lower alpha -> wider sets
    assert aci.update(1, covered=True) > aci.alphas_[1] - 1e-12


def test_classes_are_independent():
    aci = ACI(alpha_target=0.1, gamma=0.05, n_classes=2)
    before = aci.alpha(0)
    aci.update(1, covered=False)
    assert aci.alpha(0) == before, "updating class 1 moved class 0"


def test_converges_to_target_error_rate_on_a_stationary_stream():
    """The ACI guarantee: long-run error converges to alpha_target.

    Simulated with a fixed miscoverage probability; the level should drift until
    realised error sits near the target.
    """
    rng = np.random.default_rng(0)
    target = 0.10
    aci = ACI(alpha_target=target, gamma=0.02, n_classes=1, clip=(1e-4, 0.9))

    # A predictor whose miscoverage responds to the level it is given.
    for _ in range(4000):
        miss_prob = min(max(aci.alpha(0) * 1.5, 0.0), 1.0)
        aci.update(0, covered=rng.random() > miss_prob)

    assert abs(aci.empirical_error(0) - target) < 0.03


def test_adapts_when_difficulty_shifts():
    """After a regime change that causes misses, the level must fall."""
    aci = ACI(alpha_target=0.05, gamma=0.02, n_classes=1)
    for _ in range(50):
        aci.update(0, covered=True)
    easy = aci.alpha(0)
    for _ in range(50):
        aci.update(0, covered=False)
    assert aci.alpha(0) < easy


def test_level_stays_inside_clip():
    aci = ACI(alpha_target=0.05, gamma=0.5, n_classes=1, clip=(0.01, 0.2))
    for _ in range(200):
        aci.update(0, covered=False)
    assert aci.alpha(0) == pytest.approx(0.01)
    for _ in range(200):
        aci.update(0, covered=True)
    assert aci.alpha(0) == pytest.approx(0.2)


def test_update_batch_matches_sequential():
    a = ACI(alpha_target=0.1, gamma=0.03, n_classes=2)
    b = ACI(alpha_target=0.1, gamma=0.03, n_classes=2)
    y = [0, 1, 1, 0, 1]
    cov = [True, False, True, False, False]
    a.update_batch(y, cov)
    for k, c in zip(y, cov):
        b.update(k, c)
    assert a.alpha_dict() == b.alpha_dict()


def test_empirical_error_is_nan_before_any_update():
    assert np.isnan(ACI(alpha_target=0.1).empirical_error(0))


def test_record_snapshots_history():
    aci = ACI(alpha_target=0.1, n_classes=2)
    aci.update(1, covered=False)
    aci.record(month=3)
    assert aci.history_[0]["month"] == 3
    assert set(aci.history_[0]["alphas"]) == {0, 1}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha_target": 0.0}, "alpha_target must lie"),
        ({"alpha_target": 1.0}, "alpha_target must lie"),
        ({"alpha_target": 0.1, "gamma": 0}, "gamma must be positive"),
        ({"alpha_target": 0.1, "clip": (0.5, 0.1)}, "clip must be"),
    ],
)
def test_invalid_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ACI(**kwargs)


def test_unknown_class_raises():
    with pytest.raises(IndexError, match="outside"):
        ACI(alpha_target=0.1, n_classes=2).update(5, covered=True)


def test_mismatched_batch_raises():
    with pytest.raises(ValueError, match="must align"):
        ACI(alpha_target=0.1).update_batch([0, 1], [True])
