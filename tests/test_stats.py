import numpy as np
import pytest

from hopmil.eval.stats import compare_bayesian

pytestmark = pytest.mark.slow  # baycomp import is heavy-ish; group with slow tests


def _scores(mean, n=30, noise=0.005, seed=0):
    return mean + np.random.default_rng(seed).normal(0, noise, n)


def test_orientation_a_clearly_better():
    # A (0.95) clearly beats B (0.85) -> p_a_better ~ 1, verdict "A > B".
    res = compare_bayesian(
        _scores(0.95, seed=1),
        _scores(0.85, seed=2),
        name_a="hopfield",
        name_b="mean",
        rope=0.01,
        runs=3,
    )
    assert res.p_a_better > 0.95
    assert res.verdict() == "hopfield > mean"


def test_orientation_b_clearly_better():
    res = compare_bayesian(
        _scores(0.80, seed=1),
        _scores(0.95, seed=2),
        name_a="mean",
        name_b="attention",
        rope=0.01,
        runs=3,
    )
    assert res.p_b_better > 0.95
    assert res.verdict() == "attention > mean"


def test_probabilities_sum_to_one():
    res = compare_bayesian(
        _scores(0.90, seed=1),
        _scores(0.901, seed=2),
        name_a="hopfield",
        name_b="attention",
        rope=0.01,
        runs=3,
    )
    assert res.p_a_better + res.p_rope + res.p_b_better == pytest.approx(1.0, abs=1e-6)


def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        compare_bayesian(np.zeros(10), np.zeros(9), name_a="a", name_b="b", rope=0.01, runs=2)
