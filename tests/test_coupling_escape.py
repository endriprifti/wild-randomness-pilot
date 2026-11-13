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
