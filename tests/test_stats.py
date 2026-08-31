import numpy as np
import pytest

from sism.stats import (
    Estimate, bootstrap_mean, bootstrap_paired_diff, cliffs_delta, holm,
    wilcoxon_paired, _icc2_1,
)


def test_bootstrap_mean_recovers_the_point_estimate():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    est = bootstrap_mean(x, n_boot=2000)
    assert est.point == pytest.approx(3.0)
    assert est.lo < 3.0 < est.hi


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    x = np.random.default_rng(0).normal(size=40)
    a = bootstrap_mean(x, seed=7)
    b = bootstrap_mean(x, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_paired_diff_ignores_non_finite_pairs():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, 1.0, 1.0, np.nan])
    est = bootstrap_paired_diff(a, b, n_boot=500)
    assert est.n == 2 and est.point == pytest.approx(1.0)


def test_empty_input_yields_nan_not_a_crash():
    est = bootstrap_mean(np.array([]))
    assert np.isnan(est.point) and est.n == 0


def test_holm_is_monotone_and_bounded():
    adj = holm([0.001, 0.02, 0.04, 0.5])
    assert all(0 <= p <= 1 for p in adj)
    assert adj == sorted(adj)
    assert adj[0] >= 0.001 * 4 - 1e-12


def test_holm_preserves_input_order():
    adj = holm([0.5, 0.001])
    assert adj[1] < adj[0]


def test_holm_tolerates_nan():
    adj = holm([0.01, float("nan")])
    assert np.isnan(adj[1]) and not np.isnan(adj[0])


def test_cliffs_delta_signs():
    assert cliffs_delta(np.array([5, 6, 7]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert cliffs_delta(np.array([1, 2, 3]), np.array([5, 6, 7])) == pytest.approx(-1.0)
    assert cliffs_delta(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(0.0)


def test_wilcoxon_returns_nan_on_identical_inputs():
    x = np.arange(10.0)
    w, p = wilcoxon_paired(x, x)
    assert np.isnan(w) and np.isnan(p)


def test_icc_is_one_for_perfect_agreement():
    x = np.random.default_rng(1).normal(size=30)
    assert _icc2_1(np.c_[x, x]) == pytest.approx(1.0, abs=1e-6)


def test_icc_is_low_for_unrelated_raters():
    rng = np.random.default_rng(2)
    m = np.c_[rng.normal(size=200), rng.normal(size=200)]
    assert _icc2_1(m) < 0.3
