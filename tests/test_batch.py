"""H7.1: slope of alpha-hat on log2 B, scale exponent against 1/alpha - 1."""
import numpy as np
import pytest

from common import batch, stable


def test_stable_pool_is_invariant(rng):
    pool = stable.rvs_stable(1.5, 0.0, 1.0, 0.0, 800 * 16, rng)
    rows = batch.fit_over_batches(pool, (1, 2, 4, 8, 16), 800, symmetric=True)
    b, s, abar = batch._slopes(rows)
    assert abs(b) < 0.03
    assert abs(s - (1 / abar - 1)) < 0.06


@pytest.mark.slow
def test_truncated_levy_drifts(rng):
    # finite-variance truncation of a SaS(1.3) law: alpha-hat climbs towards 2 as B grows
    x = stable.rvs_stable(1.3, 0.0, 1.0, 0.0, 400 * 256 * 3, rng)
    x = x[np.abs(x) < 20][: 400 * 256]
    r = batch.h71_test(x, (1, 4, 16, 64, 256), 400, n_boot=3, rng=rng, symmetric=True)["summary"]
    assert r["b"] > 0.05 and r["reading"] == "drift"
    assert r["B_cross_hint"] is not None


def test_blocks_needed_matches_appendix():
    # sec. 7.5 arithmetic: about 1,600 blocks per B for a 90% half-width of 0.02
    assert 1500 <= batch.blocks_needed() <= 1700
