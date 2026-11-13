"""SSM arms: chunked selective scan, fractional kernel against the Mittag-Leffler impulse response,
parameter matching."""
import math

import pytest
import torch
from scipy import special

from common import ssm


def test_selective_scan_matches_naive():
    torch.manual_seed(0)
    B, L, D, N = 2, 70, 3, 4
    u, delta = torch.randn(B, L, D), torch.rand(B, L, D) * 0.5
    A = -torch.rand(D, N) - 0.1
    Bm, Cm, Dv = torch.randn(B, L, N), torch.randn(B, L, N), torch.randn(D)
    y = ssm.selective_scan(u, delta, A, Bm, Cm, Dv, chunk=16)
    assert torch.allclose(y, ssm.selective_scan_naive(u, delta, A, Bm, Cm, Dv), atol=1e-4)


def _kernel_weights(nu, lam, n_s=256, L=1000):
    k = ssm.FracKernel(1, n_s, L, nu_init=nu, lam_init=lam)
    with torch.no_grad():
        return (k.weights()[0].double(), k.r.double(), float(torch.sigmoid(k.nu_raw)[0]),
                float(torch.nn.functional.softplus(k.lam_raw)[0]))


@pytest.mark.parametrize("lam", [0.3, 1.0])
@pytest.mark.parametrize("t", [10.0, 100.0])
def test_frac_kernel_is_mittag_leffler_impulse_response_nu_half(lam, t):
    # D^nu y = -lam y + g has impulse response t^(nu-1) E_{nu,nu}(-lam t^nu);
    # for nu = 1/2, E_{1/2,1/2}(-z) = 1/sqrt(pi) - z erfcx(z)
    w, r, nu, lam_ = _kernel_weights(0.5, lam)
    approx = float((w * torch.exp(-r * t)).sum())
    z = lam_ * math.sqrt(t)
    exact = t ** -0.5 * (1 / math.sqrt(math.pi) - z * special.erfcx(z))
    assert abs(approx / exact - 1) < 0.01


def test_frac_kernel_mittag_leffler_series_nu_03():
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    w, r, nu, lam = _kernel_weights(0.3, 0.5)
    t = 10.0
    z = mpmath.mpf(-lam * t ** nu)
    ml = mpmath.nsum(lambda k: z ** k / mpmath.gamma(nu * k + nu), [0, mpmath.inf])
    exact = t ** (nu - 1) * float(ml)
    approx = float((w * torch.exp(-r * t)).sum())
    assert abs(approx / exact - 1) < 0.01


@pytest.mark.parametrize("kind", ["frac", "s4_legs", "legs", "s4d", "mamba"])
def test_width_matching(kind):
    ref = ssm.n_params(ssm.SSMModel("s4d", 1, 1, 32, 16, 2, 256, False))
    H, p = ssm.width_for_budget(kind, 1, 1, 16, 2, 256, False, ref, H_grid=range(2, 129, 2))
    assert abs(p / ref - 1) < 0.1
