"""Coupling test (within-cell residuals, permutation null, Holm) and escape-time likelihoods."""
import math

import numpy as np
import pandas as pd

from common import coupling, escape


def test_holm():
    out = coupling.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, 0.05)
    assert out["d"]["reject"] and out["a"]["reject"]
    assert not out["c"]["reject"] and not out["b"]["reject"]


def test_coupling_detects_within_cell_correlation(rng):
    recs = []
    for c in range(6):
        for i in range(12):
            e = rng.normal(size=4)
            recs.append({"cell": c, "alpha": 1.5 + 0.1 * c + 0.1 * e[0],
                         "K": 5 * c - 0.7 * e[0] + math.sqrt(0.51) * e[1],   # cell offsets must not matter
                         "alpha_esd": 3 + c + e[2], "H_dep": 0.1 * e[3]})
    res = coupling.coupling_test(pd.DataFrame(recs), ["alpha", "alpha_esd", "K", "H_dep"], ["cell"],
                                 n_perm=300, rng=rng)
    status = {tuple(v["pair"]): v["status"] for v in res["primary"]}
    assert status[("alpha", "K")] == "significant, predicted sign"
    assert status[("alpha", "alpha_esd")] == "not significant"
    assert status[("alpha", "H_dep")] == "not significant"


def test_detectable_r_decreases_with_df():
    assert coupling.detectable_r(100) < coupling.detectable_r(30) < 1


def test_censored_rate(rng):
    t = rng.exponential(50, 2000)
    r = escape.censored_rate(np.minimum(t, 200), t <= 200)
    assert abs(r["E_k"] / 50 - 1) < 0.08
    assert r["E_k_ci_lo"] < 50 < r["E_k_ci_hi"]


def test_escape_grid_recovers_power_law(rng):
    c0, a0, b0 = 2.0, -1.5, 0.5
    recs = []
    for eta in (0.05, 0.1, 0.2):
        for B in (8, 32, 128):
            Ek = math.exp(c0) * eta ** a0 * B ** b0
            t = rng.exponential(Ek, 200)
            for k, e in zip(np.minimum(t, 3 * Ek), t <= 3 * Ek):
                recs.append({"eta": eta, "B": B, "k": k, "exited": e})
    fg = escape.fit_grid(pd.DataFrame(recs), alpha_hat=1.5)
    assert abs(fg["pow_a_ln_eta"] - a0) < 3 * fg["pow_a_se"]
    assert abs(fg["pow_b_ln_B"] - b0) < 3 * fg["pow_b_se"]
    assert fg["vuong_z_pow_vs_gauss"] > 3          # the power law beats the Gaussian rival


def _checked(t, every, k_cens):
    """Exit times seen only at checks every `every` steps: (k, k_lo, exited)."""
    k = np.minimum(np.ceil(t / every) * every, k_cens)
    e = t <= k_cens
    return k, np.where(e, k - every, k), e


def test_interval_censoring_removes_the_check_floor(rng):
    t = rng.exponential(150, 3000)
    k, lo, e = _checked(t, 250, 50_000)
    r = escape.censored_rate(k, e, lo)
    assert abs(r["E_k_ic"] / 150 - 1) < 0.1 and r["E_k_ic_lo95"] < 150 < r["E_k_ic_hi95"]
    assert r["E_k"] > 1.4 * 150                       # exposure / exits reads the check times as exit times
    assert not r["floor"] and r["first_check_exits"] == int((t <= 250).sum())


def test_every_exit_at_the_first_check_gives_only_an_upper_bound(rng):
    t = rng.exponential(5, 50)
    k, lo, e = _checked(t, 250, 50_000)
    r = escape.censored_rate(k, e, lo)
    assert r["floor"] and r["E_k"] == 250 and np.isnan(r["E_k_ic"])
    assert r["E_k_ic_lo95"] == 0.0 and 5 < r["E_k_ic_hi95"] < 250


def test_interval_grid_fit_recovers_the_slopes_when_checks_are_coarse(rng):
    c0, a0, b0 = 1.5, -1.0, 0.5
    recs = []
    for eta in (0.05, 0.1, 0.2):
        for B in (8, 32, 128):
            Ek = math.exp(c0) * eta ** a0 * B ** b0           # 63 to 1,000 steps: many exits at the first check
            k, lo, e = _checked(rng.exponential(Ek, 300), 250, 20_000)
            recs += [{"eta": eta, "B": B, "k": a, "k_lo": b, "exited": c} for a, b, c in zip(k, lo, e)]
    df = pd.DataFrame(recs)
    fi = escape.fit_grid(df)
    assert fi["likelihood"] == "interval"
    assert abs(fi["pow_a_ln_eta"] - a0) < 3 * fi["pow_a_se"] and abs(fi["pow_b_ln_B"] - b0) < 3 * fi["pow_b_se"]
    fe = escape.fit_grid(df.drop(columns="k_lo"))             # exact-time likelihood: flattened by the floor
    assert fe["likelihood"] == "exact" and fe["pow_b_ln_B"] < b0 - 0.15 and fe["pow_a_ln_eta"] > a0 + 0.3
    sl = escape.slope_in_lnB(df[df.eta == 0.1])
    assert sl["likelihood"] == "interval" and abs(sl["slope"] - b0) < 3 * sl["se"]
