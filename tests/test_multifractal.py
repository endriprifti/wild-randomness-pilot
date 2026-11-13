"""Generalized dimensions of point clouds and the dimension clause of sec. 7.2."""
import math

import numpy as np

from common import multifractal


def test_d2_and_width_on_binomial_cascade(rng):
    X = multifractal.binomial_cascade_points(20000, 0.7, rng=rng)
    g = multifractal.delta_h_rep(X, rng=rng)
    D2_theory = -math.log2(0.7 ** 2 + 0.3 ** 2)
    h = multifractal.binomial_h(np.array([0.5, 4.0]), 0.7)
    assert abs(g["D2"] - D2_theory) < 0.05
    assert abs(g["delta_h_rep"] - (h[0] - h[1])) < 0.08
    assert g["D_max"] < g["log10_n"]          # the dimension half of the clause; the r range is near 2 decades


def test_uniform_square_is_monofractal(rng):
    g = multifractal.delta_h_rep(rng.uniform(size=(20000, 2)), rng=rng)
    assert abs(g["D2"] - 2.0) < 0.15
    assert abs(g["delta_h_rep"]) < 0.1


def test_dimension_clause_fails_in_high_dimension(rng):
    g = multifractal.delta_h_rep(rng.normal(size=(3000, 50)), rng=rng)
    assert not g["clause_ok"] and g["scope"] == "local" and "local_D2" in g
