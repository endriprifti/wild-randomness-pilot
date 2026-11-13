"""Tail fits of Section 7.1.

* Pareto tail: Hill/ML survival exponent zeta-hat above x_min (eq:s7-zeta-mle), with x_min
  chosen by minimum Kolmogorov-Smirnov distance (Clauset, Shalizi and Newman 2009, sec. 3.3).
* Goodness of fit: semi-parametric bootstrap p_KS (CSN sec. 4.1); p_KS <= 0.1 rejects.
* Alternatives on the same tail: log-normal, exponential, Weibull/stretched exponential
  exp(-u^w) with w free (non-nested, Vuong's z of eq:s7-vuong), and the power law with
  exponential cutoff (nested, ordinary likelihood ratio, CSN sec. 5.2).
* Hill double bootstrap (Danielsson, de Haan, Peng and de Vries 2001; Nair, Wierman and
  Zwart 2022, Thm 9.9, p. 233), used as the agreement check of sec. 7.1.
* Reading labels: wild, pre-wild, slow, mild, truncated, undecided (never reported as wild).

All densities are conditional on x >= x_min and evaluated in units y = x / x_min, which
shifts every log-likelihood by the same constant and leaves the ratios unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import optimize, special, stats


# ----------------------------------------------------------------------------- Pareto fit
@dataclass
class PowerLawFit:
    zeta: float          # survival exponent (CSN's density exponent minus one)
    xmin: float
    n_tail: int
    n: int
    ks: float
    decades_x: float     # log10(x_max / x_min)
    decades_y: float     # log10(n_tail): range of the empirical survival function

    def as_dict(self, prefix=""):
        return {prefix + k: v for k, v in asdict(self).items()}


def _candidates(xs_sorted: np.ndarray, min_tail: int, max_candidates: int) -> np.ndarray:
    n = len(xs_sorted)
    # first index of each distinct value, leaving at least min_tail points above
    _, first = np.unique(xs_sorted, return_index=True)
    first = first[first <= n - min_tail]
    if len(first) > max_candidates:
        pick = np.unique(np.linspace(0, len(first) - 1, max_candidates).round().astype(int))
        first = first[pick]
    return first


def _ks_for_start(lx: np.ndarray, csum_rev: np.ndarray, i: int, ks_points: int = 4000):
    """KS distance of the Pareto fit above lx[i]. For tails longer than ks_points the supremum is
    taken over ks_points evenly spaced order statistics (error at most about 1/ks_points)."""
    n = len(lx)
    m = n - i
    s = csum_rev[i] - m * lx[i]
    if s <= 0:
        return np.inf, np.inf
    zeta = m / s
    if m > ks_points:
        j = np.unique(np.linspace(0, m - 1, ks_points).round().astype(int))
    else:
        j = np.arange(m)
    F = 1.0 - np.exp(-zeta * (lx[i + j] - lx[i]))
    D = max(np.max(np.abs(F - j / m)), np.max(np.abs(F - (j + 1) / m)))
    return D, zeta


def fit_power_law(x, xmin: float | None = None, min_tail: int = 50,
                  max_candidates: int = 200) -> PowerLawFit:
    """CSN fit of a continuous power-law tail to positive data; x_min by minimum KS distance."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    n = len(x)
    if n < max(min_tail, 10):
        raise ValueError(f"too few positive observations ({n}) for a tail fit")
    lx = np.log(x)
    csum_rev = np.cumsum(lx[::-1])[::-1]
    if xmin is not None:
        i = int(np.searchsorted(x, xmin, side="left"))
        D, zeta = _ks_for_start(lx, csum_rev, i)
        best = (D, zeta, i)
    else:
        best = (np.inf, np.nan, 0)
        for i in _candidates(x, min_tail, max_candidates):
            D, zeta = _ks_for_start(lx, csum_rev, int(i))
            if D < best[0]:
                best = (D, zeta, int(i))
    D, zeta, i = best
    m = n - i
    return PowerLawFit(zeta=float(zeta), xmin=float(x[i]), n_tail=int(m), n=int(n), ks=float(D),
                       decades_x=float(np.log10(x[-1] / x[i])), decades_y=float(np.log10(m)))


def rvs_pareto(zeta: float, xmin: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return xmin * (1.0 - rng.uniform(size=size)) ** (-1.0 / zeta)


def _boot_chunk(args):
    x, xmin, zeta, n_tail, ks, n_rep, seed, min_tail, max_candidates = args
    rng = np.random.default_rng(seed)
    body = x[x < xmin]
    n, p_tail = len(x), n_tail / len(x)
    count = 0
    for _ in range(n_rep):
        k = max(rng.binomial(n, p_tail), min_tail)
        parts = [rvs_pareto(zeta, xmin, k, rng)]
        if n - k > 0 and len(body):
            parts.append(rng.choice(body, n - k, replace=True))
        try:
            fs = fit_power_law(np.concatenate(parts), min_tail=min_tail, max_candidates=max_candidates)
        except ValueError:
            continue
        count += fs.ks >= ks
    return count


def bootstrap_pks(x, fit: PowerLawFit, n_boot: int = 2500, rng: np.random.Generator | None = None,
                  min_tail: int = 50, max_candidates: int = 200, n_jobs: int = 1) -> float:
    """Semi-parametric bootstrap p-value of CSN sec. 4.1 (x_min re-selected in every replicate).

    n_jobs > 1 spreads the replicates over processes; seeds are drawn from `rng`, so the result
    is reproducible for a given (seed, n_jobs).
    """
    if n_boot <= 0:
        return float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n_jobs = max(1, int(n_jobs))
    sizes = [n_boot // n_jobs + (1 if i < n_boot % n_jobs else 0) for i in range(n_jobs)]
    seeds = rng.integers(0, 2**63 - 1, size=n_jobs)
    jobs = [(x, fit.xmin, fit.zeta, fit.n_tail, fit.ks, m, int(sd), min_tail, max_candidates)
            for m, sd in zip(sizes, seeds) if m > 0]
    if n_jobs == 1:
        count = sum(_boot_chunk(j) for j in jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            count = sum(ex.map(_boot_chunk, jobs))
    return float(count / n_boot)


# ----------------------------------------------------------------------------- alternatives
def _ll_pl(y, zeta):
    return np.log(zeta) - (zeta + 1) * np.log(y)


def _fit_exponential(y):
    lam = 1.0 / max(np.mean(y - 1.0), 1e-300)
    return {"lam": lam}, np.log(lam) - lam * (y - 1.0)


def _fit_lognormal(y):
    ly = np.log(y)

    def nll(th):
        mu, s = th[0], float(np.exp(np.clip(th[1], -20.0, 20.0)))
        logsf = stats.norm.logsf((0.0 - mu) / s)  # ln y_min = 0
        ll = stats.norm.logpdf((ly - mu) / s) - math.log(s) - ly - logsf
        v = -np.sum(ll)
        return v if np.isfinite(v) else 1e300

    th0 = np.array([np.mean(ly), math.log(max(np.std(ly), 1e-3))])
    best = None
    for mu0 in (th0[0], -1.0, -5.0):
        r = optimize.minimize(nll, np.array([mu0, th0[1]]), method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    mu, s = best.x[0], math.exp(best.x[1])
    ll = stats.norm.logpdf((ly - mu) / s) - math.log(s) - ly - stats.norm.logsf(-mu / s)
    return {"mu": mu, "sigma": s}, ll


def _fit_weibull(y):
    """Stretched exponential / Weibull tail conditional on y >= 1: f = w lam y^(w-1) exp(-lam (y^w - 1))."""
    ly = np.log(y)

    def ll_of(w, lam):
        with np.errstate(over="ignore", invalid="ignore"):
            return np.log(w) + np.log(lam) + (w - 1) * ly - lam * np.expm1(np.minimum(w * ly, 700.0))

    def nll(th):
        th = np.clip(th, -30.0, 30.0)
        v = -np.sum(ll_of(float(np.exp(th[0])), float(np.exp(th[1]))))
        return v if np.isfinite(v) else 1e300

    best = None
    for w0 in (0.3, 1.0, 2.0):
        lam0 = 1.0 / max(np.mean(np.exp(w0 * ly) - 1.0), 1e-12)
        r = optimize.minimize(nll, np.array([math.log(w0), math.log(lam0)]), method="Nelder-Mead",
                              options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    th = np.clip(best.x, -30.0, 30.0)
    w, lam = float(np.exp(th[0])), float(np.exp(th[1]))
    return {"w": w, "lam": lam}, ll_of(w, lam)


def _upper_gamma(s: float, x: float) -> float:
    """Gamma(s, x) for any real s and x > 0 (recurrence down from s + k > 0)."""
    if s > 0:
        return float(special.gammaincc(s, x) * special.gamma(s))
    if abs(s) < 1e-12:
        return float(special.exp1(x))
    k = int(math.floor(-s)) + 1
    s_top = s + k
    if abs(s_top) < 1e-12:
        g = float(special.exp1(x))
    else:
        g = float(special.gammaincc(s_top, x) * special.gamma(s_top))
    for j in range(k, 0, -1):
        sj = s + j - 1  # Gamma(sj, x) = (Gamma(sj+1, x) - x^sj e^-x) / sj
        g = (g - x ** sj * math.exp(-x)) / sj
    return g


def _fit_truncated(y, zeta0):
    """Power law with exponential cutoff on y >= 1: f = y^(-a) e^(-lam y) / (lam^(a-1) Gamma(1-a, lam))."""
    ly = np.log(y)
    n = len(y)
    sy, sly = float(np.sum(y)), float(np.sum(ly))

    def nll(th):
        th = np.clip(th, -30.0, 6.0)
        a, lam = 1.0 + float(np.exp(th[0])), float(np.exp(th[1]))
        try:
            G = _upper_gamma(1.0 - a, lam)
        except (OverflowError, ZeroDivisionError):
            return 1e300
        if not (G > 0 and np.isfinite(G)):
            return 1e300
        logZ = (a - 1) * math.log(lam) + math.log(G)
        return a * sly + lam * sy + n * logZ

    best = None
    for lam0 in (1e-4, 1e-2, 1e-1):
        r = optimize.minimize(nll, np.array([math.log(max(zeta0, 1e-3)), math.log(lam0)]),
                              method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    th = np.clip(best.x, -30.0, 6.0)
    a, lam = 1.0 + float(np.exp(th[0])), float(np.exp(th[1]))
    G = _upper_gamma(1.0 - a, lam)
    ll = -a * ly - lam * y - ((a - 1) * math.log(lam) + math.log(G))
    return {"a": a, "lam": lam}, ll


def vuong(ll_pl: np.ndarray, ll_alt: np.ndarray) -> dict:
    """Vuong's statistic (eq:s7-vuong). Positive z favours the power law. Two-sided p."""
    d = ll_pl - ll_alt
    n = len(d)
    R = float(np.sum(d))
    sd = float(np.std(d))
    z = R / (math.sqrt(n) * sd) if sd > 0 else 0.0
    p = float(special.erfc(abs(z) / math.sqrt(2)))
    return {"R": R, "z": z, "p": p}


# ----------------------------------------------------------------------------- Hill double bootstrap
def hill_curve(x_desc_log: np.ndarray, kmax: int):
    """gamma_k (Hill) and M_k for k = 1..kmax from log order statistics in descending order."""
    k = np.arange(1, kmax + 1)
    cs = np.cumsum(x_desc_log[:kmax])
    cs2 = np.cumsum(x_desc_log[:kmax] ** 2)
    ref = x_desc_log[1:kmax + 1]
    gam = cs / k - ref
    M = cs2 / k - 2 * ref * cs / k + ref ** 2
    return gam, M


def hill_double_bootstrap(x, eps: float = 0.1, n_rep: int = 500,
                          rng: np.random.Generator | None = None) -> dict:
    """Danielsson et al. (2001) double bootstrap for the Hill estimator's k (Nair et al. Thm 9.9)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = len(x)
    n1 = int(n ** (1 - eps))
    n2 = int(n1 ** 2 / n)
    if n2 < 20:
        return {"zeta_db": float("nan"), "k_db": 0, "n1": n1, "n2": n2}

    def kstar(m):
        acc = np.zeros(m - 1)
        for _ in range(n_rep):
            s = np.sort(np.log(rng.choice(x, m, replace=True)))[::-1]
            gam, M = hill_curve(s, m - 1)
            acc += (M - 2 * gam ** 2) ** 2
        return int(np.argmin(acc / n_rep)) + 1

    k1, k2 = kstar(n1), kstar(n2)
    if k1 < 2 or k2 < 1:
        return {"zeta_db": float("nan"), "k_db": 0, "n1": n1, "n2": n2}
    ln1, lk1 = math.log(n1), math.log(k1)
    expo = (ln1 - lk1) / ln1
    k = (k1 ** 2 / k2) * ((lk1 ** 2) / (2 * ln1 - lk1) ** 2) ** expo
    k = int(min(max(round(k), 2), n - 1))
    s = np.sort(np.log(x))[::-1]
    gam, _ = hill_curve(s, k)
    return {"zeta_db": float(1.0 / gam[-1]), "k_db": k, "n1": n1, "n2": n2}


# ----------------------------------------------------------------------------- reading the fit
LABELS = ("wild", "pre-wild", "slow", "mild", "truncated", "undecided")


@dataclass
class TailReading:
    label: str
    fit: PowerLawFit
    p_ks: float
    tests: dict = field(default_factory=dict)
    hill_db: dict = field(default_factory=dict)
    hill_agrees: bool | None = None

    def as_row(self, prefix=""):
        row = {prefix + "label": self.label, prefix + "p_ks": self.p_ks,
               prefix + "hill_agrees": self.hill_agrees}
        row.update(self.fit.as_dict(prefix))
        for name, t in self.tests.items():
            for k, v in t.items():
                row[f"{prefix}{name}_{k}"] = v
        for k, v in self.hill_db.items():
            row[prefix + k] = v
        return row


def read_tail(x, n_boot: int = 2500, level: float = 0.1, min_tail: int = 50,
              max_candidates: int = 200, hill_db: bool = True, hill_tol: float = 0.15,
              hill_eps: float = 0.1, hill_reps: int = 500, n_jobs: int = 1,
              rng: np.random.Generator | None = None) -> TailReading:
    """Full sec. 7.1 reading of one positive sample (one tail)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    fit = fit_power_law(x, min_tail=min_tail, max_candidates=max_candidates)
    p_ks = bootstrap_pks(x, fit, n_boot=n_boot, rng=rng, min_tail=min_tail,
                         max_candidates=max_candidates, n_jobs=n_jobs)
    xa = np.asarray(x, dtype=float)
    y = xa[np.isfinite(xa) & (xa >= fit.xmin)] / fit.xmin
    llpl = _ll_pl(y, fit.zeta)
    tests: dict = {}
    par_e, ll_e = _fit_exponential(y)
    tests["exp"] = {**vuong(llpl, ll_e), **par_e}
    par_ln, ll_ln = _fit_lognormal(y)
    tests["lognormal"] = {**vuong(llpl, ll_ln), **par_ln}
    par_w, ll_w = _fit_weibull(y)
    tests["weibull"] = {**vuong(llpl, ll_w), **par_w}
    par_t, ll_t = _fit_truncated(y, fit.zeta)
    lr = 2 * (float(np.sum(ll_t)) - float(np.sum(llpl)))
    tests["truncated"] = {"lr": lr, "p": float(stats.chi2.sf(max(lr, 0.0), 1)), **par_t}

    # an alternative is preferred if its Vuong (or LR) test is significant against the power law.
    # When several are, the one with the smallest AIC wins (OUR tie-break; sec. 7.1 does not rank them):
    # score = log-likelihood gain over the power law minus the alternative's number of parameters,
    # so an exponential is not relabelled "truncated" or "slow" by a two-parameter law that nests it.
    n_par = {"exp": 1, "lognormal": 2, "weibull": 2, "truncated": 2}
    preferred = []
    for name in ("lognormal", "weibull", "exp"):
        t = tests[name]
        if t["p"] <= level and t["R"] < 0:
            preferred.append((name, -t["R"] - n_par[name]))
    if tests["truncated"]["p"] <= level and lr > 0:
        preferred.append(("truncated", lr / 2 - n_par["truncated"]))
    if preferred:
        name = max(preferred, key=lambda v: v[1])[0]
        if name == "lognormal":
            label = "slow"
        elif name == "weibull":
            label = "slow" if tests["weibull"]["w"] < 1 else "mild"
        elif name == "exp":
            label = "mild"
        else:
            label = "truncated"
    elif (np.isnan(p_ks) or p_ks > level) and fit.decades_x >= 2 and fit.decades_y >= 2:
        label = "wild" if fit.zeta < 2 else "pre-wild"
    else:
        label = "undecided"

    reading = TailReading(label=label, fit=fit, p_ks=p_ks, tests=tests)
    if hill_db:
        reading.hill_db = hill_double_bootstrap(x, eps=hill_eps, n_rep=hill_reps, rng=rng)
        zdb = reading.hill_db["zeta_db"]
        reading.hill_agrees = bool(np.isfinite(zdb) and abs(zdb - fit.zeta) <= hill_tol * fit.zeta)
        # sec. 7.1: zeta-hat is accepted only where the double bootstrap agrees with it
        if label in ("wild", "pre-wild") and not reading.hill_agrees:
            reading.label = "undecided"
    return reading


def read_signed(x, **kw) -> dict:
    """Fit the two tails of signed data separately (sec. 7.1)."""
    x = np.asarray(x, dtype=float)
    out = {}
    for side, v in (("right", x[x > 0]), ("left", -x[x < 0])):
        try:
            out[side] = read_tail(v, **kw)
        except ValueError as e:
            out[side] = str(e)
    return out
