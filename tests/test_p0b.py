"""P0b helpers: random-order cascades keep the binomial multiset, truncated-Levy pools, the shared LR null."""
import numpy as np
import pytest

import p0b_calibration as p0b
from common import hurst, stable


def test_random_cascade_has_the_binomial_values_at_every_level():
    rng = np.random.default_rng(0)
    for depth in (1, 5, 10):
        x = p0b.random_binomial_series(depth, 0.7, rng)
        det = hurst.binomial_series(depth, 0.7)
        assert np.allclose(np.sort(x), np.sort(det))               # same values, so the same h(q)
        for level in range(1, depth + 1):                          # and the same multiset of dyadic block sums
            w = 2 ** (depth - level)
            assert np.allclose(np.sort(x.reshape(-1, w).sum(1)), np.sort(det.reshape(-1, w).sum(1)))
    a, b = p0b.random_binomial_series(10, 0.7, np.random.default_rng(1)), p0b.random_binomial_series(10, 0.7, np.random.default_rng(2))
    assert not np.allclose(a, b)                                   # replicates differ


def test_trunc_levy_pool_is_signed_and_truncated():
    x = p0b.trunc_levy_pool(1.5, 20.0, 50_000, np.random.default_rng(0))
    assert len(x) == 50_000 and np.abs(x).max() <= 20.0
    assert (x < 0).mean() == pytest.approx(0.5, abs=0.02)


def test_task_seeds_do_not_depend_on_order():
    a = p0b.task_rng(7, "stable", 1, 2, 3).random(3)
    b = p0b.task_rng(7, "stable", 1, 2, 3).random(3)
    c = p0b.task_rng(7, "stable", 1, 2, 4).random(3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_shared_null_p_value_and_statistic_match_lr_test_gaussian():
    x = stable.rvs_stable(1.5, 0.0, 1.0, 0.0, 400, np.random.default_rng(3))
    lr, _ = p0b.lr_stat(x)
    out = stable.lr_test_gaussian(x, n_boot=0)
    assert lr == pytest.approx(out["lr_stat"])
    assert p0b.p_from_null(5.0, np.array([1.0, 6.0, 7.0])) == pytest.approx(3 / 4)
    assert p0b.p_from_null(9.0, np.array([1.0, 6.0, 7.0])) == pytest.approx(1 / 4)
