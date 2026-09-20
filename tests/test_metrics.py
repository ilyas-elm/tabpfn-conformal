"""The shared metric definitions, against hand-computed values.

These four functions are the vocabulary every experiment reports in, so a quiet
error here would move every number in the README at once rather than one of
them. `average_set_size` in particular produces every `set_size` in
`results/` -- E1, E2, E4, E5 and E6 all call it -- so it is pinned to an
arithmetic answer rather than to whatever it happens to return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabpfn_conformal import (
    average_set_size,
    coverage_by_class,
    empty_set_rate,
    marginal_coverage,
)

CLASSES = np.array([0, 1])

# Four sets, one of each kind: {0}, {1}, {0,1} and {}.
SETS = np.array(
    [
        [True, False],   # singleton {0}
        [False, True],   # singleton {1}
        [True, True],    # both labels
        [False, False],  # empty
    ]
)
Y = np.array([0, 0, 1, 1])  # covered, NOT covered, covered, NOT covered


def test_average_set_size_is_the_mean_number_of_labels():
    # sizes are 1, 1, 2, 0 -> mean 1.0
    assert average_set_size(SETS) == pytest.approx(1.0)
    assert average_set_size(np.ones((5, 2), dtype=bool)) == pytest.approx(2.0)
    assert average_set_size(np.zeros((5, 2), dtype=bool)) == pytest.approx(0.0)


def test_empty_set_rate_counts_empty_sets_not_full_ones():
    # Exactly one of the four rows is empty, and a different one is full.
    assert empty_set_rate(SETS) == pytest.approx(0.25)
    assert empty_set_rate(np.ones((4, 2), dtype=bool)) == pytest.approx(0.0)
    assert empty_set_rate(np.zeros((4, 2), dtype=bool)) == pytest.approx(1.0)


def test_marginal_coverage_uses_the_true_label():
    # Rows 0 and 2 contain their true label; rows 1 and 3 do not.
    assert marginal_coverage(SETS, Y, CLASSES) == pytest.approx(0.5)


def test_coverage_by_class_is_computed_within_each_class():
    cov = coverage_by_class(SETS, Y, CLASSES)
    assert cov[0] == pytest.approx(0.5)   # rows 0, 1 -> covered, not covered
    assert cov[1] == pytest.approx(0.5)   # rows 2, 3 -> covered, not covered

    # A case where the marginal number hides the minority class entirely: the
    # reason this function exists.
    sets = np.array([[True, False]] * 9 + [[True, False]])
    y = np.array([0] * 9 + [1])
    assert marginal_coverage(sets, y, CLASSES) == pytest.approx(0.9)
    by_class = coverage_by_class(sets, y, CLASSES)
    assert by_class[0] == pytest.approx(1.0)
    assert by_class[1] == pytest.approx(0.0)


def test_coverage_by_class_reports_nan_for_an_absent_class():
    sets = np.array([[True, False], [True, False]])
    cov = coverage_by_class(sets, np.array([0, 0]), CLASSES)
    assert cov[0] == pytest.approx(1.0)
    assert math.isnan(cov[1])


def test_classes_are_read_positionally_not_assumed_sorted():
    """Column k of `pred_sets` is `classes[k]`, whatever order `classes` is in.

    `coverage_by_class` keys its result by `enumerate(classes)`, so anything
    that resolved labels by sort order instead would attach each coverage to the
    wrong label. Here "legit" sorts after "fraud" but is given first, so the two
    conventions disagree and only the positional one is right.
    """
    classes = np.array(["legit", "fraud"])   # column 0 is legit, column 1 is fraud
    y = np.array(["fraud", "legit"])
    sets = np.array([[False, True], [True, False]])  # each row hits its true class
    assert marginal_coverage(sets, y, classes) == pytest.approx(1.0)

    cov = coverage_by_class(sets, y, classes)
    assert cov["fraud"] == pytest.approx(1.0)
    assert cov["legit"] == pytest.approx(1.0)

    # And a miss is attributed to the right class, not its neighbour.
    missed = np.array([[False, False], [True, False]])  # the fraud row misses
    cov = coverage_by_class(missed, y, classes)
    assert cov["fraud"] == pytest.approx(0.0)
    assert cov["legit"] == pytest.approx(1.0)


def test_a_label_absent_from_classes_is_an_error_not_a_wrong_answer():
    with pytest.raises(ValueError, match="not in classes"):
        marginal_coverage(SETS, np.array([0, 0, 1, 2]), CLASSES)
