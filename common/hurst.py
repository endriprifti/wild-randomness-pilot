"""Memory and multifractality of time series (Section 7.2).

* Hurst exponent by detrended fluctuation analysis (DFA-1; Peng et al. 1994) and by the
  Abry-Veitch wavelet estimator (Abry and Veitch 1998), over the same scale range.
* Shuffled surrogates (keep the marginal law, destroy memory): their mean H_shuf is the bias
  that heavy tails alone give the estimator; a memory claim needs the estimate outside the
  shuffled distribution, and H enters the coupling test as |H - H_shuf|.
* Phase-randomized surrogates (Theiler et al. 1992) and IAAFT surrogates (Schreiber and
  Schmitz 1996) for the multifractality null; shuffling attributes the width
  (Kantelhardt et al. 2002, sec. IV.A).
* MFDFA generalized Hurst exponents h(q) and the width Delta h = h(q_lo) - h(q_hi).
* Generators: fractional Gaussian noise (Davies-Harte), SaS Levy-motion increments, the
  binomial multifractal series of Kantelhardt et al. (2002) and its random-order version, and the
  increments of a multifractal random walk (Bacry, Delour and Muzy 2001).
* A crude nonstationarity check (Beran et al. 2013, Ch. 7): H on consecutive quarters.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import special

# ----------------------------------------------------------------------------- generators


def fgn(n: int, H: float, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    """Exact fractional Gaussian noise by circulant embedding (Davies and Harte 1987)."""
    k = np.arange(n + 1, dtype=float)
    gam = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([gam, gam[-2:0:-1]])
    lam = np.fft.fft(c).real
    if lam.min() < -1e-8:
        raise ValueError("circulant embedding is not nonnegative definite")
    lam = np.clip(lam, 0, None)
    m = len(c)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    y = np.fft.fft(np.sqrt(lam / m) * w)
    return sigma * y[:n].real


def levy_increments(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    from .stable import rvs_stable
    return rvs_stable(alpha, 0.0, 1.0, 0.0, n, rng)


def binomial_series(depth: int, a: float = 0.75) -> np.ndarray:
    """Binomial multifractal series x_k = a^{n(k)} (1-a)^{depth - n(k)}, n(k) = ones in binary k.

    For this series MFDFA gives h(q) = 1/q - ln(a^q + (1-a)^q) / (q ln 2) (Kantelhardt et al. 2002).
    """
    k = np.arange(2 ** depth)
    ones = np.zeros_like(k)
    t = k.copy()
    while t.any():
        ones += t & 1
        t >>= 1
    return a ** ones * (1 - a) ** (depth - ones)


def random_binomial_series(depth: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """Random-order binomial cascade of length 2^depth: each split gives weights a and 1-a to its two halves in
    random order. Same multiset of values at every dyadic level as binomial_series (so the same h(q))."""
    x = np.ones(1)
    for _ in range(depth):
        left = np.where(rng.random(len(x)) < 0.5, a, 1.0 - a)
        x = np.stack([x * left, x * (1.0 - left)], axis=1).reshape(-1)
    return x


def mrw_increments(n: int, lam2: float, rng: np.random.Generator, T: int | None = None) -> np.ndarray:
    """Increments of a multifractal random walk (Bacry, Delour and Muzy 2001): x_t = eps_t exp(omega_t), eps i.i.d.
    N(0, 1), omega Gaussian with Cov(omega_t, omega_{t+k}) = lam2 ln(T / (|k| + 1)) for |k| < T and 0 beyond
    (integral scale T, default n), mean -Var(omega). Volatility clustering with log-correlated magnitudes, and
    multifractal: h(q) = 1/2 + lam2 (1 - q/2) for MFDFA below T, so Delta h(q_lo, q_hi) = lam2 (q_hi - q_lo) / 2.
    omega comes from circulant embedding (negative eigenvalues, if any, set to 0)."""
    T = int(T or n)
    k = np.arange(n + 1, dtype=float)
    cov = np.where(k < T, lam2 * np.log(T / (k + 1.0)), 0.0)
    c = np.concatenate([cov, cov[-2:0:-1]])
    lam = np.clip(np.fft.fft(c).real, 0, None)
    m = len(c)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    omega = np.fft.fft(np.sqrt(lam / m) * w)[:n].real - cov[0]
    return rng.normal(size=n) * np.exp(omega)


def binomial_series_h(q: np.ndarray, a: float = 0.75) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return 1 / q - np.log(a ** q + (1 - a) ** q) / (q * math.log(2))


# ----------------------------------------------------------------------------- DFA / MFDFA
def log_scales(n: int, s_min: int = 10, s_max: int | None = None, n_scales: int = 20) -> np.ndarray:
    s_max = s_max or n // 10
    s = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(max(s_max, s_min + 1)), n_scales)).astype(int))
    return s[s >= 4]


def _segment_variances(profile: np.ndarray, s: int, order: int = 1) -> np.ndarray:
    n = len(profile)
    ns = n // s
    if ns < 1:
        return np.empty(0)
    segs = np.concatenate([profile[: ns * s].reshape(ns, s), profile[n - ns * s:].reshape(ns, s)])
    t = np.arange(s, dtype=float)
    V = np.vander(t, order + 1)
    coef, *_ = np.linalg.lstsq(V, segs.T, rcond=None)
    resid = segs.T - V @ coef
    return np.mean(resid ** 2, axis=0)


def mfdfa(x: np.ndarray, q: np.ndarray, scales: np.ndarray | None = None, order: int = 1):
    """MFDFA (Kantelhardt et al. 2002). Returns h(q), the scales and log F_q(s)."""
    x = np.asarray(x, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    profile = np.cumsum(x - x.mean())
    scales = log_scales(len(x)) if scales is None else np.asarray(scales)
    logF = np.empty((len(q), len(scales)))
    for j, s in enumerate(scales):
        F2 = _segment_variances(profile, int(s), order)
        F2 = F2[F2 > 0]
        for i, qq in enumerate(q):
            if abs(qq) < 1e-9:
                logF[i, j] = 0.5 * np.mean(np.log(F2))
            else:
                logF[i, j] = np.log(np.mean(F2 ** (qq / 2))) / qq
    ls = np.log(scales)
    h = np.array([np.polyfit(ls, logF[i], 1)[0] for i in range(len(q))])
    return h, scales, logF


def hurst_dfa(x: np.ndarray, scales: np.ndarray | None = None, order: int = 1) -> dict:
    h, sc, _ = mfdfa(x, np.array([2.0]), scales, order)
    return {"H": float(h[0]), "s_min": int(sc[0]), "s_max": int(sc[-1]),
            "decades": float(np.log10(sc[-1] / sc[0]))}


def delta_h(x: np.ndarray, q_lo: float = 0.5, q_hi: float = 4.0, scales=None) -> dict:
    h, sc, _ = mfdfa(x, np.array([q_lo, q_hi]), scales)
    return {"delta_h": float(h[0] - h[1]), "h_qlo": float(h[0]), "h_qhi": float(h[1]),
            "q_lo": q_lo, "q_hi": q_hi, "decades": float(np.log10(sc[-1] / sc[0]))}


# ----------------------------------------------------------------------------- wavelet estimator
def hurst_wavelet(x: np.ndarray, wavelet: str = "db3", j1: int = 3, j2: int | None = None) -> dict:
    """Abry-Veitch log-scale diagram for a stationary (fGn-like) series: slope = 2H - 1.

    Weighted least squares over octaves j1..j2 with the bias correction
    g_j = psi(n_j/2)/ln 2 - log2(n_j/2) and variances zeta(2, n_j/2)/ln^2 2.
    """
    import pywt
    x = np.asarray(x, dtype=float)
    w = pywt.Wavelet(wavelet)
    J = pywt.dwt_max_level(len(x), w.dec_len)
    coeffs = pywt.wavedec(x - x.mean(), w, level=J, mode="periodization")
    details = coeffs[1:][::-1]  # details[0] = finest octave j = 1
    j2 = J if j2 is None else min(j2, J)
    js, ys, vs = [], [], []
    for j in range(j1, j2 + 1):
        d = details[j - 1]
        nj = len(d)
        if nj < 4:
            break
        mu = np.mean(d ** 2)
        g = special.digamma(nj / 2) / math.log(2) - math.log2(nj / 2)
        js.append(j)
        ys.append(math.log2(mu) - g)
        vs.append(special.zeta(2, nj / 2) / math.log(2) ** 2)
    js, ys, vs = map(np.asarray, (js, ys, vs))
    if len(js) < 3:
        return {"H": float("nan"), "j1": j1, "j2": int(js[-1]) if len(js) else j1, "decades": 0.0}
    wts = 1 / vs
    S, Sx, Sy = wts.sum(), (wts * js).sum(), (wts * ys).sum()
    Sxx, Sxy = (wts * js ** 2).sum(), (wts * js * ys).sum()
    slope = (S * Sxy - Sx * Sy) / (S * Sxx - Sx ** 2)
    return {"H": float((slope + 1) / 2), "j1": int(js[0]), "j2": int(js[-1]),
            "decades": float((js[-1] - js[0]) * math.log10(2))}


# ----------------------------------------------------------------------------- surrogates
def shuffle_surrogate(x, rng):
    return rng.permutation(np.asarray(x))


def phase_surrogate(x, rng):
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean()


def iaaft_surrogate(x, rng, max_iter: int = 100, tol: float = 1e-8):
    x = np.asarray(x, dtype=float)
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    prev = None
    for _ in range(max_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(x))
        ranks = np.empty(len(y), dtype=np.intp)      # = argsort(argsort(y)), with one sort
        ranks[np.argsort(y)] = np.arange(len(y))
        y = sorted_x[ranks]
        err = np.mean((np.abs(np.fft.rfft(y)) - amp) ** 2)
        if prev is not None and abs(prev - err) < tol * (prev + 1e-300):
            break
        prev = err
    return y


SURROGATES = {"shuffle": shuffle_surrogate, "phase": phase_surrogate, "iaaft": iaaft_surrogate}


def surrogate_test(x, stat: Callable[[np.ndarray], float], kind: str = "shuffle", n_surr: int = 99,
                   rng: np.random.Generator | None = None, two_sided: bool = True) -> dict:
    """Rank test of stat(x) against its distribution on surrogates of the given kind."""
    rng = rng if rng is not None else np.random.default_rng(0)
    obs = float(stat(x))
    sur = np.array([stat(SURROGATES[kind](x, rng)) for _ in range(n_surr)])
    mean = float(np.mean(sur))
    if two_sided:
        p = (1 + np.sum(np.abs(sur - mean) >= abs(obs - mean))) / (n_surr + 1)
    else:
        p = (1 + np.sum(sur >= obs)) / (n_surr + 1)
    return {"obs": obs, "surr_mean": mean, "surr_sd": float(np.std(sur)), "p": float(p),
            "surr_q05": float(np.quantile(sur, 0.05)), "surr_q95": float(np.quantile(sur, 0.95))}


def memory_claim(x, n_surr: int = 99, rng=None, level: float = 0.05, scales=None,
                 wavelet_j1: int = 3) -> dict:
    """Both Hurst estimators with their shuffled-surrogate baselines (sec. 7.2).

    `H_dep_*` = |H - H_shuf| is the quantity that enters the coupling test and the H6.3 dose.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for name, fn in (("dfa", lambda v: hurst_dfa(v, scales)["H"]),
                     ("wav", lambda v: hurst_wavelet(v, j1=wavelet_j1)["H"])):
        t = surrogate_test(x, fn, "shuffle", n_surr, rng)
        out[f"H_{name}"] = t["obs"]
        out[f"H_shuf_{name}"] = t["surr_mean"]
        out[f"H_dep_{name}"] = abs(t["obs"] - t["surr_mean"])
        out[f"p_shuf_{name}"] = t["p"]
        out[f"memory_{name}"] = bool(t["p"] <= level)
    d = hurst_dfa(x, scales)
    out["dfa_decades"] = d["decades"]
    out["wav_decades"] = hurst_wavelet(x, j1=wavelet_j1)["decades"]
    out["two_decades"] = bool(d["decades"] >= 2 - 1e-9)
    return out


def multifractal_claim(x, q_lo=0.5, q_hi=4.0, n_surr=99, rng=None, level=0.05) -> dict:
    """Width counts only where it exceeds its IAAFT (and phase) surrogate distribution; shuffling attributes it."""
    rng = rng if rng is not None else np.random.default_rng(0)
    stat = lambda v: delta_h(v, q_lo, q_hi)["delta_h"]
    out = {"delta_h": stat(x)}
    for kind in ("phase", "iaaft", "shuffle"):
        t = surrogate_test(x, stat, kind, n_surr, rng, two_sided=False)
        out[f"dh_{kind}_mean"] = t["surr_mean"]
        out[f"dh_{kind}_q95"] = t["surr_q95"]
        out[f"p_{kind}"] = t["p"]
    out["multifractal"] = bool(out["p_iaaft"] <= level and out["p_phase"] <= level)
    # attribution: width that survives shuffling comes from the broad marginal
    out["width_from_marginal"] = bool(out["dh_shuffle_mean"] >= 0.5 * out["delta_h"])
    return out


def nonstationarity_check(x, n_parts: int = 4) -> dict:
    parts = np.array_split(np.asarray(x, dtype=float), n_parts)
    hs = [hurst_dfa(p)["H"] if len(p) >= 400 else float("nan") for p in parts]
    return {"H_parts": hs, "H_parts_range": float(np.nanmax(hs) - np.nanmin(hs)) if np.isfinite(hs).any() else float("nan")}
