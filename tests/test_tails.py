"""CSN power-law fit (x_min by KS, bootstrap p_KS), Vuong tests, Hill double bootstrap, tail labels."""
import numpy as np
import pytest

from common import tails


@pytest.mark.parametrize("zeta", [1.5, 2.5])
def test_csn_recovers_zeta_on_pareto(zeta, rng):
    f = tails.fit_power_law(tails.rvs_pareto(zeta, 1.0, 5000, rng))
    assert abs(f.zeta - zeta) < 0.12 * zeta
    assert f.xmin < 2.0


def test_csn_finds_xmin_in_mixture(rng):
    body = rng.uniform(0.1, 5.0, 7000)
    tail = tails.rvs_pareto(2.0, 5.0, 3000, rng)
    f = tails.fit_power_law(np.concatenate([body, tail]))
    assert 3.0 < f.xmin < 8.0
    assert abs(f.zeta - 2.0) < 0.25


def test_bootstrap_pks_accepts_power_law(rng):
    x = np.concatenate([rng.uniform(0.1, 5.0, 7000), tails.rvs_pareto(2.0, 5.0, 3000, rng)])
    f = tails.fit_power_law(x)
    p = tails.bootstrap_pks(x, f, n_boot=100, rng=rng)
    assert p > 0.1


def _vuong_fixed_xmin(x):
    xmin = np.median(x)
    y = x[x >= xmin] / xmin
    zeta = tails.fit_power_law(x, xmin=xmin).zeta
    llpl = tails._ll_pl(y, zeta)
    _, lle = tails._fit_exponential(y)
    _, lln = tails._fit_lognormal(y)
    return tails.vuong(llpl, lle), tails.vuong(llpl, lln)


def test_vuong_prefers_lognormal_on_lognormal(rng):
    _, vs_ln = _vuong_fixed_xmin(rng.lognormal(0, 1, 20000))
    assert vs_ln["z"] < -5 and vs_ln["p"] < 1e-6


def test_vuong_prefers_exponential_on_exponential(rng):
    vs_exp, _ = _vuong_fixed_xmin(rng.exponential(1.0, 20000))
    assert vs_exp["z"] < -5


def test_vuong_on_pareto(rng):
    vs_exp, vs_ln = _vuong_fixed_xmin(tails.rvs_pareto(2.0, 1.0, 20000, rng))
    assert vs_exp["z"] > 5          # power law beats the exponential
    assert vs_ln["p"] > 0.1         # lognormal cannot be told apart (it nests the power law)


@pytest.mark.parametrize("zeta,dist", [(2.0, "pareto"), (3.0, "t")])
def test_hill_double_bootstrap(zeta, dist, rng):
    x = tails.rvs_pareto(zeta, 1.0, 20000, rng) if dist == "pareto" else np.abs(rng.standard_t(3, 20000))
    r = tails.hill_double_bootstrap(x, n_rep=100, rng=rng)
    assert abs(r["zeta_db"] - zeta) < 0.15 * zeta


def test_label_wild_on_pareto(rng):
    r = tails.read_tail(tails.rvs_pareto(1.5, 1.0, 20000, rng), n_boot=50, hill_reps=100, rng=rng)
    assert r.label == "wild"
    assert r.hill_agrees


def test_label_not_wild_on_light_tails(rng):
    for x in (rng.exponential(1.0, 20000), rng.weibull(0.5, 20000)):
        r = tails.read_tail(x, n_boot=50, hill_reps=100, rng=rng)
        assert r.label in ("mild", "slow", "truncated")


def test_two_decade_rule_makes_zeta3_undecided(rng):
    # sec. 7.1: zeta = 3 at n = 2e4 spans < 2 decades in x, so no label is given (known limit)
    r = tails.read_tail(tails.rvs_pareto(3.0, 1.0, 20000, rng), n_boot=20, hill_db=False, rng=rng)
    assert r.fit.decades_x < 2 and r.label == "undecided"


def test_read_signed_fits_both_tails(rng):
    x = tails.rvs_pareto(1.5, 1.0, 6000, rng) * rng.choice([-1, 1], 6000)
    out = tails.read_signed(x, n_boot=10, hill_db=False, rng=rng)
    assert abs(out["right"].fit.zeta - 1.5) < 0.25 and abs(out["left"].fit.zeta - 1.5) < 0.25
