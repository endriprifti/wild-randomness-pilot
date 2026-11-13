"""Stable laws in Nolan's parametrization S(alpha, beta, gamma, delta; 1).

Sampling (Chambers-Mallows-Stuck), a fast density (FFT of the characteristic function,
spliced with the tail expansion), maximum-likelihood fits of all four parameters from
McCulloch quantile starting values (Nolan 2020, sec. 4.7, p. 176; sec. 4.4, p. 166), a
likelihood-ratio test of the Gaussian against the stable law with a parametric bootstrap
(Nolan 2020, sec. 4.10.2, pp. 202-203), and the block-sum minibatch noise of sec. 7.5
(Nolan 2020, sec. 4.12, eq. 4.10, pp. 213-214).

Internally the density is computed in the continuous S0 parametrization and the fit is
reported in S1, the parametrization of eq:s2-sas-cf. For alpha = 2 the law is normal with
variance 2 gamma^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize, special, stats
from scipy.interpolate import CubicSpline

ALPHA_BOUNDS = (0.6, 2.0)


# ----------------------------------------------------------------------------- sampling
def rvs_stable(alpha: float, beta: float = 0.0, gamma: float = 1.0, delta: float = 0.0,
               size=1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Chambers-Mallows-Stuck sampler for S(alpha, beta, gamma, delta; 1) (Weron 1996 form)."""
    rng = rng if rng is not None else np.random.default_rng()
    V = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) > 1e-9:
        t = beta * math.tan(math.pi * alpha / 2)
        B = math.atan(t) / alpha
        S = (1 + t * t) ** (1 / (2 * alpha))
        X = (S * np.sin(alpha * (V + B)) / np.cos(V) ** (1 / alpha)
             * (np.cos(V - alpha * (V + B)) / W) ** ((1 - alpha) / alpha))
        return gamma * X + delta
    X = (2 / np.pi) * ((np.pi / 2 + beta * V) * np.tan(V)
                       - beta * np.log((np.pi / 2) * W * np.cos(V) / (np.pi / 2 + beta * V)))
    return gamma * X + (2 / np.pi) * beta * gamma * math.log(gamma) + delta


def s1_to_s0_delta(alpha, beta, gamma, delta1):
    if abs(alpha - 1) < 1e-9:
        return delta1 + beta * (2 / np.pi) * gamma * math.log(gamma)
    return delta1 + beta * gamma * math.tan(math.pi * alpha / 2)


def s0_to_s1_delta(alpha, beta, gamma, delta0):
    if abs(alpha - 1) < 1e-9:
        return delta0 - beta * (2 / np.pi) * gamma * math.log(gamma)
    return delta0 - beta * gamma * math.tan(math.pi * alpha / 2)


# ----------------------------------------------------------------------------- density
class _StdDensity:
    """Standardized S0 density f(x; alpha, beta) on |x| <= x_switch by FFT, tail expansion beyond."""

    N = 2 ** 16
    L = 500.0
    X_SWITCH = 50.0

    def __init__(self):
        self._cache: dict = {}
        n, L = self.N, self.L
        self.dx = 2 * L / n
        self.x = -L + self.dx * np.arange(n)
        self.du = 2 * np.pi / (n * self.dx)
        self.u = (np.arange(n) - n // 2) * self.du

    def _phi(self, alpha, beta):
        u = self.u
        au = np.abs(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            if abs(alpha - 1) > 1e-6:
                tan = math.tan(math.pi * alpha / 2)
                corr = np.where(au > 0, au ** (1 - alpha) - 1, -1.0)
                psi = au ** alpha * (1 + 1j * beta * tan * np.sign(u) * corr)
            else:
                psi = au * (1 + 1j * beta * (2 / np.pi) * np.sign(u) * np.where(au > 0, np.log(au), 0.0))
        return np.exp(-psi)

    def table(self, alpha, beta):
        key = (round(alpha, 6), round(beta, 6))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = self.N
        phi = self._phi(alpha, beta)
        # f(x_j) = (1/2pi) sum_k phi(u_k) exp(-i u_k x_j) du
        # f(x_j) = (1/2pi) sum_k phi(u_k) exp(-i u_k x_j) du with u_k = (k - n/2) du,
        # x_j = x_0 + j dx and du dx = 2pi/n, so exp(-i u_k x_j)
        # = exp(-i u_k x_0) exp(-2 pi i k j / n) (-1)^j.
        shift = np.exp(-1j * self.u * self.x[0])
        vals = np.fft.fft(phi * shift) * np.exp(1j * np.pi * np.arange(n))
        f = np.real(vals) * self.du / (2 * np.pi)
        f = np.maximum(f, 1e-300)
        mask = np.abs(self.x) <= self.X_SWITCH + 1.0
        spline = CubicSpline(self.x[mask], np.log(f[mask]))
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = spline
        return spline

    def logpdf(self, z, alpha, beta):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        switch = self.X_SWITCH if alpha >= 1.2 else 25.0  # aliasing grows as alpha falls
        inner = np.abs(z) <= switch
        if inner.any():
            out[inner] = self.table(alpha, beta)(z[inner])
        if (~inner).any():
            out[~inner] = _tail_logpdf(z[~inner], alpha, beta)
        return out


def _tail_logpdf(z, alpha, beta):
    """Tail series of the standardized density (three terms), alpha != 1.

    In S1 coordinates with rho = sqrt(1 + beta^2 tan^2(pi alpha/2)) and
    theta = arctan(beta tan(pi alpha/2)), for x -> +inf
        f(x) ~ (1/pi) sum_k (-1)^(k+1) Gamma(k alpha + 1)/k! rho^k sin(k(pi alpha/2 + theta)) x^(-k alpha - 1),
    whose first term is alpha c_alpha (1 + beta) x^(-alpha-1) (Nolan 2020, Thm 1.2); the left tail
    follows with beta -> -beta. For alpha = 2 the normal density with variance 2 is used.
    """
    if alpha >= 2.0 - 1e-9:
        return -z ** 2 / 4 - 0.5 * math.log(4 * math.pi)
    if abs(alpha - 1) < 1e-6:
        alpha = 1 + 1e-6
    tan = math.tan(math.pi * alpha / 2)
    x1 = z + beta * tan  # S0 -> S1 coordinates for the standardized law
    out = np.empty_like(z)
    for sgn in (1.0, -1.0):
        m = (np.sign(x1) == sgn) | ((x1 == 0) & (sgn > 0))
        if not m.any():
            continue
        b = sgn * beta
        rho = math.sqrt(1 + (b * tan) ** 2)
        theta = math.atan(b * tan)
        x = np.abs(x1[m])
        dens = np.zeros_like(x)
        for k in (1, 2, 3):
            dens += ((-1) ** (k + 1) * special.gamma(k * alpha + 1) / math.factorial(k) * rho ** k
                     * math.sin(k * (math.pi * alpha / 2 + theta)) * x ** (-k * alpha - 1))
        out[m] = np.log(np.maximum(dens / math.pi, 1e-300))
    return out


_DENSITY = _StdDensity()


def logpdf(x, alpha, beta, gamma, delta, param: int = 1):
    """Log density of S(alpha, beta, gamma, delta; param) for param in {0, 1}."""
    d0 = delta if param == 0 else s1_to_s0_delta(alpha, beta, gamma, delta)
    z = (np.asarray(x, dtype=float) - d0) / gamma
    if alpha >= 2.0 - 1e-9:
        return stats.norm.logpdf(z, scale=math.sqrt(2.0)) - math.log(gamma)
    return _DENSITY.logpdf(z, alpha, beta) - math.log(gamma)


# ----------------------------------------------------------------------------- fitting
@dataclass
class StableFit:
    alpha: float
    beta: float
    gamma: float
    delta: float  # S1 location
    loglik: float
    n: int
    converged: bool
    alpha_start: float
    method: str = "ml"

    def as_dict(self, prefix: str = ""):
        return {prefix + k: v for k, v in asdict(self).items()}


def quantile_start(x: np.ndarray):
    """McCulloch (1986) quantile estimates in S0: (alpha, beta, gamma, delta0)."""
    x = np.asarray(x, dtype=float)
    try:
        from scipy.stats._levy_stable import _fitstart_S0
        a, b, d0, g = _fitstart_S0(x)
        a = float(np.clip(a, ALPHA_BOUNDS[0] + 0.01, 2.0))
        return a, float(np.clip(b, -0.99, 0.99)), float(max(g, 1e-12)), float(d0)
    except Exception:
        q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        nu_a = (q95 - q05) / max(q75 - q25, 1e-300)
        a = float(np.clip(2.0 * 2.44 / max(nu_a, 2.44), 1.0, 2.0))
        return a, 0.0, float((q75 - q25) / 2.0), float(q50)


def fit_stable(x: np.ndarray, symmetric: bool = False, maxiter: int = 600) -> StableFit:
    """Maximum-likelihood fit of all four stable parameters (beta = 0 if symmetric)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    a0, b0, g0, d00 = quantile_start(x)
    if symmetric:
        b0 = 0.0
    scale = g0 if g0 > 0 else (np.std(x) or 1.0)
    xs = (x - d00) / scale  # work on a standardized copy for numerical stability

    def unpack(th):
        a = ALPHA_BOUNDS[0] + (2.0 - ALPHA_BOUNDS[0]) / (1 + np.exp(-th[0]))
        b = 0.0 if symmetric else math.tanh(th[1])
        g = math.exp(th[2])
        d = th[3]
        return a, b, g, d

    def nll(th):
        a, b, g, d = unpack(th)
        val = -np.sum(logpdf(xs, a, b, g, d, param=0))
        return val if np.isfinite(val) else 1e300

    def logit_alpha(a):
        p = (min(max(a, ALPHA_BOUNDS[0] + 1e-3), 2.0 - 1e-3) - ALPHA_BOUNDS[0]) / (2.0 - ALPHA_BOUNDS[0])
        return math.log(p / (1 - p))

    starts = [np.array([logit_alpha(a0), math.atanh(np.clip(b0, -0.95, 0.95)), 0.0, 0.0])]
    best = None
    for th0 in starts:
        res = optimize.minimize(nll, th0, method="Nelder-Mead",
                                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    a, b, g, d = unpack(best.x)
    # Gaussian boundary: compare with the alpha = 2 fit explicitly.
    gn, dn = np.std(xs) / math.sqrt(2.0), np.mean(xs)
    ll_norm = np.sum(logpdf(xs, 2.0, 0.0, gn, dn, param=0))
    ll = -best.fun
    if ll_norm > ll:
        a, b, g, d, ll = 2.0, 0.0, gn, dn, ll_norm
    gamma = g * scale
    delta0 = d * scale + d00
    delta1 = s0_to_s1_delta(a, b, gamma, delta0)
    return StableFit(alpha=float(a), beta=float(b), gamma=float(gamma), delta=float(delta1),
                     loglik=float(ll - n * math.log(scale)), n=n, converged=bool(best.success),
                     alpha_start=float(a0))


def lr_test_gaussian(x: np.ndarray, n_boot: int = 200, rng: np.random.Generator | None = None,
                     fit: StableFit | None = None) -> dict:
    """LR test of the Gaussian (alpha = 2) against the stable law, parametric bootstrap p-value.

    The Gaussian is the boundary alpha = 2 of the stable family, so the chi-square reference is
    not valid (Nolan 2020, sec. 4.10.2); the null distribution is simulated instead.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    fit = fit or fit_stable(x)
    mu, sd = float(np.mean(x)), float(np.std(x))
    ll_norm = float(np.sum(stats.norm.logpdf(x, mu, sd)))
    lr = 2 * (fit.loglik - ll_norm)
    null = []
    for _ in range(n_boot):
        y = rng.normal(mu, sd, len(x))
        fy = fit_stable(y, maxiter=300)
        null.append(2 * (fy.loglik - float(np.sum(stats.norm.logpdf(y, y.mean(), y.std())))))
    null = np.asarray(null)
    p = (1 + np.sum(null >= lr)) / (1 + len(null)) if n_boot > 0 else float("nan")
    return {"lr_stat": float(lr), "p_gauss": float(p), "n_boot": int(n_boot)}


# ----------------------------------------------------------------------------- helpers
def block_noise(values: np.ndarray, B: int, max_blocks: int | None = None) -> np.ndarray:
    """Minibatch noise at batch size B: means of disjoint blocks of B pooled values minus the pool mean."""
    v = np.asarray(values, dtype=float)
    nb = len(v) // B
    if max_blocks is not None:
        nb = min(nb, max_blocks)
    if nb < 1:
        return np.empty(0)
    blocks = v[: nb * B].reshape(nb, B).mean(axis=1)
    return blocks - v.mean()


def stable_iqr(alpha: float, n: int = 400_000, seed: int = 0) -> float:
    """Interquartile range of the standard SaS law (gamma = 1), by simulation."""
    x = rvs_stable(alpha, 0.0, 1.0, 0.0, n, np.random.default_rng(seed))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return float(q3 - q1)


def nolan_se_alpha(n: int, ref_n: int = 1000, ref_se: float = 0.100) -> float:
    """Asymptotic SE of alpha-hat at alpha = 1.5, beta = 0 scaled from Nolan (2020), p. 181."""
    return ref_se * math.sqrt(ref_n / n)
