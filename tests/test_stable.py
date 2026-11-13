"""Stable laws: density against closed forms and scipy, ML fit on samples with known alpha, beta, gamma."""
import math

import numpy as np
import pytest
from scipy import stats

from common import stable


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_logpdf_closed_forms(alpha):
    # S(1,0,g,d;1) is Cauchy(d, g); S(2,0,g,d;1) is Normal(d, sqrt(2) g)
    z = np.linspace(-30, 30, 61)
    lp = stable.logpdf(z, alpha, 0.0, 1.5, 0.3)
    ref = stats.cauchy.logpdf(z, 0.3, 1.5) if alpha == 1.0 else stats.norm.logpdf(z, 0.3, 1.5 * math.sqrt(2))
    assert np.max(np.abs(np.exp(lp - ref) - 1)) < 5e-3


@pytest.mark.parametrize("alpha,beta", [(0.8, 0.0), (1.3, 0.5), (1.7, -0.3)])
def test_logpdf_matches_scipy(alpha, beta):
    from scipy.stats import levy_stable
    levy_stable.parameterization = "S1"
    z = np.array([-15, -4, -1, 0, 0.5, 2, 6, 40.0])
    ours = np.exp(stable.logpdf(z, alpha, beta, 1.0, 0.0))
    ref = levy_stable.pdf(z, alpha, beta)
    assert np.max(np.abs(ours / ref - 1)) < 0.01


def test_sampler_tail_index(rng):
    # P(|X| > x) ~ C x^-alpha: the Hill estimate on the top 1% recovers alpha = 1.5
    x = np.abs(stable.rvs_stable(1.5, 0.0, 1.0, 0.0, 400_000, rng))
    top = np.sort(x)[-4000:]
    zeta = 1 / np.mean(np.log(top / top[0]))
    assert abs(zeta - 1.5) < 0.1


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.8])
def test_fit_symmetric_alpha(alpha, rng):
    x = stable.rvs_stable(alpha, 0.0, 1.0, 0.0, 4000, rng)
    f = stable.fit_stable(x, symmetric=True)
    # Nolan's asymptotic SE at n = 4000 is about 0.05; allow three SEs
    assert abs(f.alpha - alpha) < 0.15
    assert abs(f.gamma - 1.0) < 0.1


def test_fit_skewed_all_parameters(rng):
    x = stable.rvs_stable(1.5, 0.5, 2.0, 1.0, 5000, rng)
    f = stable.fit_stable(x)
    assert abs(f.alpha - 1.5) < 0.12
    assert abs(f.beta - 0.5) < 0.25
    assert abs(f.gamma / 2.0 - 1) < 0.1
    assert abs(f.delta - 1.0) < 0.4


def test_fit_gaussian_boundary(rng):
    f = stable.fit_stable(rng.normal(0, 1, 3000))
    assert f.alpha > 1.95


@pytest.mark.slow
def test_lr_test_gaussian_rejects_stable(rng):
    x = stable.rvs_stable(1.5, 0.0, 1.0, 0.0, 2000, rng)
    r = stable.lr_test_gaussian(x, n_boot=19, rng=rng)
    assert r["p_gauss"] <= 0.05


def test_block_noise():
    v = np.arange(12, dtype=float)
    b = stable.block_noise(v, 4)
    assert np.allclose(b, [1.5 - 5.5, 5.5 - 5.5, 9.5 - 5.5])
    assert len(stable.block_noise(v, 4, max_blocks=2)) == 2


@pytest.mark.parametrize("alpha,beta", [(1.2, 0.0), (1.5, 0.0), (1.8, 0.0), (2.0, 0.0), (1.5, 0.5), (1.0, 0.3)])
def test_torch_sampler_matches_numpy_sampler(alpha, beta):
    """rvs_stable_torch (P5's injected noise, drawn on the GPU) has the distribution of rvs_stable."""
    import torch
    from scipy.stats import ks_2samp
    x_np = stable.rvs_stable(alpha, beta, 0.05, 0.1, 200_000, np.random.default_rng(1))
    g = torch.Generator().manual_seed(2)
    x_t = stable.rvs_stable_torch(alpha, beta, 0.05, 0.1, (400, 500), g, "cpu", torch.float32)
    assert x_t.shape == (400, 500) and x_t.dtype == torch.float32
    assert ks_2samp(x_np, x_t.reshape(-1).double().numpy()).pvalue > 1e-3
    again = stable.rvs_stable_torch(alpha, beta, 0.05, 0.1, (400, 500), torch.Generator().manual_seed(2), "cpu",
                                    torch.float32)
    assert torch.equal(x_t, again)                                    # reproducible from the generator's seed
