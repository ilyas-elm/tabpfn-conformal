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


def test_update_round_is_one_step_regardless_of_batch_size():
    """A round is a round: 1,400 rows must not mean 1,400 steps."""
    a = ACI(alpha_target=0.05, gamma=0.02, n_classes=2)
    before = a.alpha(1)
    a.update_round(1, error_rate=1.0)
    assert a.alpha(1) == pytest.approx(before + 0.02 * (0.05 - 1.0))
    assert a.n_updates_[1] == 1


def test_update_rounds_does_not_oscillate_on_large_batches():
    """The E3 failure, pinned.

    Per-observation updates over month-sized batches drove the level into its
    clip bounds and coverage swung between 0.517 and 1.000. One step per round
    must stay stable on the same input.
    """
    rng = np.random.default_rng(0)
    per_round = ACI(alpha_target=0.05, gamma=0.05, n_classes=2)
    per_obs = ACI(alpha_target=0.05, gamma=0.05, n_classes=2)
    trace_round, trace_obs = [], []

    for _ in range(5):                                  # five monthly batches
        y = np.ones(1400, dtype=int)
        covered = rng.random(1400) > 0.06               # ~6% miscoverage
        per_round.update_rounds(y, covered)
        per_obs.update_batch(y, covered)
        trace_round.append(per_round.alpha(1))
        trace_obs.append(per_obs.alpha(1))

    lo, hi = per_round.clip
    assert lo < per_round.alpha(1) < hi, "per-round update hit a clip bound"
    assert abs(per_round.alpha(1) - 0.05) < 0.02, "per-round level drifted far"

    # The failure mode is volatility: one step per row makes the level swing by
    # an order of magnitude more than one step per round on identical input.
    swing_round = max(trace_round) - min(trace_round)
    swing_obs = max(trace_obs) - min(trace_obs)
    assert swing_obs > 10 * swing_round, (
        f"per-observation swing {swing_obs:.4f} vs per-round {swing_round:.4f}"
    )


def test_update_rounds_tracks_a_shift_in_difficulty():
    a = ACI(alpha_target=0.05, gamma=0.1, n_classes=1)
    for _ in range(5):
        a.update_round(0, error_rate=0.05)              # on target
    steady = a.alpha(0)
    for _ in range(5):
        a.update_round(0, error_rate=0.30)              # regime gets harder
    assert a.alpha(0) < steady


def test_update_round_rejects_bad_rate():
    with pytest.raises(ValueError, match="error_rate must lie"):
        ACI(alpha_target=0.05).update_round(0, error_rate=1.5)


def test_alpha_dict_reports_the_adapted_levels_not_the_target():
    """The handoff from ACI to `predict_set` must carry the adaptation.

    `alpha_dict()` is what E3 feeds to `predict_set_from_proba` for the ACI arm.
    If it returned `alpha_target` the adaptation would be computed and then
    discarded, and every ACI result would collapse onto the frozen one —
    indistinguishable from this project's actual finding that ACI cannot move
    the threshold at 46 calibration positives. A null result has to be measured,
    not produced by a bug.
    """
    aci = ACI(alpha_target=0.05, gamma=0.1, n_classes=2)
    for _ in range(5):
        aci.update_round(1, 0.9)          # sustained under-coverage on class 1

    assert aci.alphas_[1] != pytest.approx(0.05), "fixture did not move the level"
    d = aci.alpha_dict()
    assert d[1] == pytest.approx(float(aci.alphas_[1]))
    assert d[1] != pytest.approx(aci.alpha_target)
    assert d[0] == pytest.approx(float(aci.alphas_[0]))
    assert set(d) == {0, 1}


def test_alpha_dict_drives_different_prediction_sets_than_the_target():
    """End to end: an adapted level must change what `predict_set` returns."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    from tabpfn_conformal import ConformalClassifier

    class Grid(BaseEstimator, ClassifierMixin):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p1 = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1.0 - p1, p1])

    grid = np.round(np.arange(1, 21) * 0.05, 3)
    X_cal = np.concatenate([1.0 - grid, grid]).reshape(-1, 1)
    y_cal = np.array([1] * 20 + [0] * 20)
    cc = ConformalClassifier(Grid(), method="mondrian", prefit=True).fit(X_cal, y_cal)

    # Drive the level UP, on observed over-coverage. Downward is the direction
    # that cannot show anything here: at 20 calibration points, alpha 0.05 and
    # alpha 0.0001 both land past the last order statistic, so the sets are
    # identical -- the quantization this project measures in E3, reproduced in
    # miniature.
    aci = ACI(alpha_target=0.40, gamma=0.5, n_classes=2)
    for _ in range(6):
        aci.update_round(1, 0.0)          # observed no errors at all

    assert aci.alpha_dict()[1] > 0.05
    X_test = np.linspace(0.05, 0.95, 19).reshape(-1, 1)
    assert not np.array_equal(cc.predict_set(X_test, aci.alpha_dict()),
                              cc.predict_set(X_test, 0.05))
