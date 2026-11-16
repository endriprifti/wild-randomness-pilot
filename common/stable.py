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

import contextlib
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path

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


def rvs_stable_torch(alpha: float, beta: float = 0.0, gamma: float = 1.0, delta: float = 0.0, shape=(1,),
                     generator=None, device="cpu", dtype=None):
    """rvs_stable (same Chambers-Mallows-Stuck formula, same S(alpha, beta, gamma, delta; 1) form) drawn with
    torch on `device`, so that noise injected into GPU training does not wait for the CPU (P5 draws about
    six million variates per step: about 0.35 s with NumPy on 4 CPU cores). Computed in float64 so the far tail
    is not cut off, then cast to `dtype`."""
    import torch
    f64 = torch.float64
    V = (torch.rand(shape, generator=generator, device=device, dtype=f64) - 0.5) * math.pi
    W = torch.empty(shape, device=device, dtype=f64).exponential_(1.0, generator=generator)
    if abs(alpha - 1.0) > 1e-9:
        t = beta * math.tan(math.pi * alpha / 2)
        B = math.atan(t) / alpha
        S = (1 + t * t) ** (1 / (2 * alpha))
        X = (S * torch.sin(alpha * (V + B)) / torch.cos(V) ** (1 / alpha)
             * (torch.cos(V - alpha * (V + B)) / W) ** ((1 - alpha) / alpha))
        out = gamma * X + delta
    else:
        X = (2 / math.pi) * ((math.pi / 2 + beta * V) * torch.tan(V)
                             - beta * torch.log((math.pi / 2) * W * torch.cos(V) / (math.pi / 2 + beta * V)))
        out = gamma * X + (2 / math.pi) * beta * gamma * math.log(gamma) + delta
    return out.to(dtype) if dtype is not None else out


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
        # the same for every (alpha, beta): computed once (identical values to computing them in every table)
        self._shift = np.exp(-1j * self.u * self.x[0])
        self._sign = np.exp(1j * np.pi * np.arange(n))
        self.inner = np.abs(self.x) <= self.X_SWITCH + 1.0

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

    def values(self, alpha, beta) -> np.ndarray:
        """f(x_j; alpha, beta) on the whole FFT grid, floored at 1e-300."""
        phi = self._phi(alpha, beta)
        # f(x_j) = (1/2pi) sum_k phi(u_k) exp(-i u_k x_j) du with u_k = (k - n/2) du,
        # x_j = x_0 + j dx and du dx = 2pi/n, so exp(-i u_k x_j)
        # = exp(-i u_k x_0) exp(-2 pi i k j / n) (-1)^j.
        vals = np.fft.fft(phi * self._shift) * self._sign
        f = np.real(vals) * self.du / (2 * np.pi)
        return np.maximum(f, 1e-300)

    def table(self, alpha, beta):
        key = (round(alpha, 6), round(beta, 6))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        f = self.values(alpha, beta)
        spline = CubicSpline(self.x[self.inner], np.log(f[self.inner]))
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


def _lagrange4(s):
    """Weights of 4-point (cubic) Lagrange interpolation at offset s from the first of four equally spaced nodes."""
    s = np.asarray(s, dtype=float)
    return np.stack([-(s - 1) * (s - 2) * (s - 3) / 6, s * (s - 2) * (s - 3) / 2,
                     -s * (s - 1) * (s - 3) / 2, s * (s - 1) * (s - 2) / 6])


class _GridDensity:
    """The FFT density of _StdDensity, precomputed once on an (alpha, beta) grid and interpolated.

    _StdDensity computes a 65,536-point FFT and a cubic spline for every new (alpha, beta), about 9 ms, and a
    Nelder-Mead fit asks for about 300 of them: 1.6 to 2.3 s per fit, so one H7.1 test (1,608 fits) takes about
    45 minutes of one core. Here the density f on the same x nodes (|x| <= X_SWITCH + 1) is stored for alpha in
    [0.6, 2] (step 0.01) and beta in [-1, 1] (step 0.05), about 310 MB, built once and memory-mapped from a cache
    file. A density is then a cubic (4-point Lagrange) interpolation of f in alpha and in beta, and of log f
    in x, about 0.1 ms. f, not log f, is interpolated in alpha because near alpha = 2 the tail mass grows linearly
    in 2 - alpha while log f does not. Beyond |x| = X_SWITCH (25 for alpha < 1.2) the tail series is used as
    before, and for alpha < 1 with |beta| > BETA_MAX, where the density approaches a one-sided support, the exact
    FFT density. Largest |log f| difference from the exact density over |x| <= 50: 5e-5 (alpha 0.6 to 2, beta
    -0.9 to 0.9; 3e-4 at alpha 1.1, beta -0.999). tests/test_stable.py checks the fits against the exact density's.
    """

    A0, A1, DA = 0.6, 2.0, 0.01
    B0, B1, DB = -1.0, 1.0, 0.05
    BETA_MAX = 0.9          # with alpha < EXACT_BELOW_ALPHA: exact FFT density
    EXACT_BELOW_ALPHA = 1.0
    VERSION = 1

    def __init__(self, base: _StdDensity, path=None):
        self.base = base
        self.alphas = np.round(np.arange(self.A0, self.A1 + self.DA / 2, self.DA), 10)
        self.betas = np.round(np.arange(self.B0, self.B1 + self.DB / 2, self.DB), 10)
        self.idx = np.flatnonzero(base.inner)
        self.x0 = float(base.x[self.idx[0]])
        self.dx = base.dx
        self.path = Path(path) if path else self.default_path()
        self.f = self._load_or_build()

    @classmethod
    def default_path(cls) -> Path:
        b = _StdDensity
        tag = f"v{cls.VERSION}_N{b.N}_L{b.L:g}_X{b.X_SWITCH:g}_a{cls.A0:g}-{cls.A1:g}-{cls.DA:g}_b{cls.B0:g}-{cls.B1:g}-{cls.DB:g}"
        return Path(__file__).resolve().parent.parent / ".cache" / f"stable_grid_{tag}.npy"

    def _load_or_build(self):
        shape = (len(self.alphas), len(self.betas), len(self.idx))
        if self.path.exists():
            f = np.load(self.path, mmap_mode="r")
            if f.shape == shape:
                return f
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = np.empty(shape)
        for i, a in enumerate(self.alphas):
            for j, b in enumerate(self.betas):
                f[i, j] = self.base.values(float(a), float(b))[self.idx]
        tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, f)
        os.replace(tmp, self.path)
        return np.load(self.path, mmap_mode="r")

    @staticmethod
    def _stencil(v, v0, h, n):
        t = (v - v0) / h
        i0 = int(min(max(math.floor(t) - 1, 0), n - 4))
        return i0, _lagrange4(t - i0)

    def _inner(self, z, alpha, beta):
        ia, wa = self._stencil(alpha, self.A0, self.DA, len(self.alphas))
        ib, wb = self._stencil(beta, self.B0, self.DB, len(self.betas))
        t = (z - self.x0) / self.dx
        lo = max(int(math.floor(t.min())) - 2, 0)
        hi = min(int(math.ceil(t.max())) + 3, len(self.idx))
        block = np.asarray(self.f[ia:ia + 4, ib:ib + 4, lo:hi])
        row = np.tensordot(np.outer(wa, wb), block, axes=([0, 1], [0, 1]))
        lrow = np.log(np.maximum(row, 1e-300))
        j0 = np.clip(np.floor(t).astype(int) - 1 - lo, 0, hi - lo - 4)
        w = _lagrange4(t - lo - j0)
        return w[0] * lrow[j0] + w[1] * lrow[j0 + 1] + w[2] * lrow[j0 + 2] + w[3] * lrow[j0 + 3]

    def logpdf(self, z, alpha, beta):
        if alpha < self.EXACT_BELOW_ALPHA and abs(beta) > self.BETA_MAX:
            return self.base.logpdf(z, alpha, beta)
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        switch = self.base.X_SWITCH if alpha >= 1.2 else 25.0
        inner = np.abs(z) <= switch
        if inner.any():
            out[inner] = self._inner(z[inner], alpha, beta)
        if (~inner).any():
            out[~inner] = _tail_logpdf(z[~inner], alpha, beta)
        return out


_ACTIVE = _DENSITY
_GRID: _GridDensity | None = None


def set_density(name: str, path=None) -> str:
    """'fft' (default: a new FFT table for every (alpha, beta)) or 'grid' (_GridDensity, built or loaded on first
    use). Applies to logpdf, fit_stable and everything built on them in this process; returns the previous name."""
    global _ACTIVE, _GRID
    prev = density_name()
    if name == "grid":
        if _GRID is None or (path is not None and Path(path) != _GRID.path):
            _GRID = _GridDensity(_DENSITY, path)
        _ACTIVE = _GRID
    elif name == "fft":
        _ACTIVE = _DENSITY
    else:
        raise ValueError(f"unknown stable density {name!r} (fft or grid)")
    return prev


def density_name() -> str:
    return "grid" if _ACTIVE is not _DENSITY else "fft"


@contextlib.contextmanager
def using_density(name: str, path=None):
    prev = set_density(name, path)
    try:
        yield
    finally:
        set_density(prev)


def logpdf(x, alpha, beta, gamma, delta, param: int = 1):
    """Log density of S(alpha, beta, gamma, delta; param) for param in {0, 1}."""
    d0 = delta if param == 0 else s1_to_s0_delta(alpha, beta, gamma, delta)
    z = (np.asarray(x, dtype=float) - d0) / gamma
    if alpha >= 2.0 - 1e-9:
        return stats.norm.logpdf(z, scale=math.sqrt(2.0)) - math.log(gamma)
    return _ACTIVE.logpdf(z, alpha, beta) - math.log(gamma)


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


def fit_stable(x: np.ndarray, symmetric: bool = False, maxiter: int = 600, restarts: int | None = None) -> StableFit:
    """Maximum-likelihood fit of all four stable parameters (beta = 0 if symmetric).

    restarts: Nelder-Mead runs again from its best point (a fresh simplex), kept if the likelihood is higher;
    Nelder-Mead can report convergence at a point that is not a local optimum (its simplex collapses). Default:
    1 with the grid density (about 50 ms per fit), 0 with the FFT density (as in the runs before 2026-11-16)."""
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
    if restarts is None:
        restarts = 1 if density_name() == "grid" else 0
    for _ in range(restarts):
        res = optimize.minimize(nll, best.x, method="Nelder-Mead",
                                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
        if res.fun < best.fun:
            best = res
    a, b, g, d = unpack(best.x)
    # Gaussian boundary: compare with the alpha = 2 fit explicitly.
    gn, dn = np.std(xs) / math.sqrt(2.0), np.mean(xs)
    ll_norm = np.sum(logpdf(xs, 2.0, 0.0, gn, dn, param=0))
    ll = -best.fun
    if ll_norm > ll:
        a, b, g, d, ll = 2.0, 0.0, gn, dn, ll_norm
    if a >= 2.0 - 1e-9:
        b = 0.0   # the normal law has no skewness: Nelder-Mead may end at alpha = 2 with any beta (no effect there)
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
