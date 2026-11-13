"""Hurst estimators (DFA, Abry-Veitch wavelet), surrogates, shuffled baseline, MFDFA width."""
import numpy as np
import pytest

from common import hurst


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_dfa_and_wavelet_on_fgn(H, rng):
    x = hurst.fgn(2 ** 14, H, rng)
    assert abs(hurst.hurst_dfa(x)["H"] - H) < 0.05
    assert abs(hurst.hurst_wavelet(x)["H"] - H) < 0.05


def test_shuffled_baseline_on_levy_increments(rng):
    # iid heavy-tailed increments: no memory, H_dep near zero for both estimators
    m = hurst.memory_claim(hurst.levy_increments(2 ** 13, 1.5, rng), n_surr=39, rng=rng)
    assert not m["memory_dfa"] and not m["memory_wav"]
    assert m["H_dep_dfa"] < 0.05 and m["H_dep_wav"] < 0.05


def test_memory_claim_on_fgn(rng):
    m = hurst.memory_claim(hurst.fgn(2 ** 13, 0.8, rng), n_surr=39, rng=rng)
    assert m["memory_dfa"] and m["memory_wav"]
    assert abs(m["H_dep_dfa"] - 0.3) < 0.06 and abs(m["H_dep_wav"] - 0.3) < 0.06
    assert abs(m["H_shuf_dfa"] - 0.5) < 0.03


def test_phase_surrogate_keeps_spectrum(rng):
    x = hurst.fgn(4096, 0.7, rng) ** 3
    s = hurst.phase_surrogate(x, rng)
    A = np.abs(np.fft.rfft(x - x.mean()))
    assert np.allclose(np.abs(np.fft.rfft(s - s.mean())), A, atol=1e-9 * A.max())
    assert abs(s.mean() - x.mean()) < 1e-12
    assert not np.allclose(s, x)


def test_iaaft_keeps_marginal_and_spectrum(rng):
    x = hurst.fgn(4096, 0.7, rng) ** 3
    s = hurst.iaaft_surrogate(x, rng)
    assert np.array_equal(np.sort(s), np.sort(x))
    A = np.abs(np.fft.rfft(x))
    assert np.linalg.norm(np.abs(np.fft.rfft(s)) - A) / np.linalg.norm(A) < 0.1


def test_shuffle_surrogate_is_permutation(rng):
    x = rng.normal(size=1000)
    assert np.array_equal(np.sort(hurst.shuffle_surrogate(x, rng)), np.sort(x))


def test_mfdfa_width_on_binomial_series():
    x = hurst.binomial_series(14, 0.75)
    d = hurst.delta_h(x, 0.5, 4.0)
    th = hurst.binomial_series_h(np.array([0.5, 4.0]), 0.75)
    assert abs(d["delta_h"] - (th[0] - th[1])) < 0.06


def test_mfdfa_width_near_zero_on_fgn(rng):
    assert abs(hurst.delta_h(hurst.fgn(2 ** 14, 0.6, rng))["delta_h"]) < 0.08
