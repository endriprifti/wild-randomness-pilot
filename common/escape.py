"""Escape-time analysis of H3.3 and the escape clause of H6.1 (Section 7.5, eq:s7-escape).

Each restart either exits (barrier criterion) at step k or is right-censored at k_cens.
With an exponential (memoryless) exit law, the censored maximum-likelihood estimate of
E k_exit at a grid point is total exposure / number of exits.

Exit is only checked every check_every steps, so an exit "at k" happened somewhere in
(k_lo, k], k_lo being the previous check (0 before the first). The fits therefore use the
interval-censored likelihood S(k_lo) - S(k) for exits and S(k_cens) for censored restarts,
S(t) = exp(-t / E k). Without it, a cell where every restart exits at the first check sits at
E k = check_every exactly, a floor set by the schedule; with it, such a cell gives only an
upper bound (censored_rate reports E_k_ic_hi95). Rows without k_lo fall back to exact times.
Two models of the grid are fitted to all restarts by censored likelihood:

  heuristic (eq:s7-escape):  ln E k = c + a ln eta + b ln B,   predicted a = -(1 + J(alpha-1)), b = J(alpha-1)
  Gaussian rival (Xie et al. 2021):  ln E k = c + kappa B / eta

and compared by log-likelihood and a Vuong statistic over restarts. For the injected-noise
positive control the slope in ln B (B entering through gamma_B = B^(1/alpha - 1) gamma) should
equal alpha - 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special


def censored_rate(times: np.ndarray, exited: np.ndarray, lo: np.ndarray | None = None) -> dict:
    """E k at one grid point: exposure / exits (exact exit times), and, with lo (the check before each
    exit), the interval-censored MLE E_k_ic with a likelihood-ratio 95% interval."""
    times = np.asarray(times, float)
    exited = np.asarray(exited, bool)
    d = int(exited.sum())
    T = float(times.sum())
    if d == 0:
        out = {"exits": 0, "exposure": T, "E_k": float("inf"), "E_k_lower": T, "n": len(times)}
    else:
        out = {"exits": d, "exposure": T, "E_k": T / d, "n": len(times),
               "E_k_ci_lo": 2 * T / special.chdtri(2 * d, 0.05),
               "E_k_ci_hi": 2 * T / special.chdtri(2 * d, 0.95)}
    if lo is not None:
        out.update(interval_rate(times, exited, lo))
    return out


def interval_rate(times, exited, lo) -> dict:
    """Interval-censored exponential MLE of E k with a likelihood-ratio 95% interval (chi2_1 / 2 = 1.92).
    first_check_exits counts exits in the first interval (lo = 0); when every restart exits there, the
    likelihood keeps rising as E k -> 0 and only the upper bound is informative (floor = True)."""
    t = np.asarray(times, float)
    e = np.asarray(exited, bool)
    lo = np.where(e, np.asarray(lo, float), t)
    first = int((e & (lo <= 0)).sum())
    out = {"first_check_exits": first, "floor": bool(e.all() and first == len(e) and len(e) > 0)}
    if not e.any():
        return {**out, "E_k_ic": np.inf, "E_k_ic_lo95": np.nan, "E_k_ic_hi95": np.inf}

    def ll(logE):
        return float(np.sum(_ll_restarts(np.full(len(t), logE), t, e, lo)))
    grid = np.linspace(-6.0, 25.0, 311)                    # E k from 0.0025 to 7e10 steps
    vals = np.array([ll(g) for g in grid])
    i = int(np.argmax(vals))
    if 0 < i < len(grid) - 1:
        r = optimize.minimize_scalar(lambda g: -ll(g), bounds=(grid[i - 1], grid[i + 1]), method="bounded")
        g_hat, l_hat = float(r.x), -float(r.fun)
    else:
        g_hat, l_hat = float(grid[i]), float(vals[i])
    cut = l_hat - 1.92

    def root(a, b):
        try:
            return float(np.exp(optimize.brentq(lambda g: ll(g) - cut, a, b)))
        except ValueError:
            return np.nan
    lo95 = root(grid[0], g_hat) if ll(grid[0]) < cut else 0.0
    hi95 = root(g_hat, grid[-1]) if ll(grid[-1]) < cut else np.inf
    est = np.nan if i == 0 else float(np.exp(g_hat))      # at the lower edge: E k -> 0, no point estimate
    return {**out, "E_k_ic": est, "E_k_ic_lo95": lo95, "E_k_ic_hi95": hi95}


def _ll_restarts(logE: np.ndarray, times: np.ndarray, exited: np.ndarray, lo: np.ndarray | None = None) -> np.ndarray:
    """Log-likelihood per restart. lo None: exact exit times (density). With lo: an exit lies in (lo, t]
    (log[S(lo) - S(t)]); a censored restart contributes log S(t) either way."""
    logE = np.clip(logE, -50.0, 50.0)  # keeps the optimizer's trial points finite
    e = np.asarray(exited, float)
    if lo is None:
        return -e * logE - np.exp(-logE) * times
    rate = np.exp(-logE)
    lo = np.where(e > 0, lo, times)
    width = np.maximum(times - lo, 0.0) * rate
    with np.errstate(divide="ignore"):
        inside = -lo * rate + np.log(np.clip(-np.expm1(-width), 1e-300, None))
    return np.where(e > 0, inside, -times * rate)


def _interval(df):
    """lo per restart if the frame carries k_lo (the check before the exit), else None (exact times)."""
    if "k_lo" not in df or df["k_lo"].isna().all():
        return None
    return df["k_lo"].fillna(df["k"]).to_numpy(float)


def fit_grid(df, alpha_hat: float | None = None) -> dict:
    """df has one row per restart: eta, B, k (exit or censoring step), exited (bool)."""
    eta = df["eta"].to_numpy(float)
    B = df["B"].to_numpy(float)
    t = df["k"].to_numpy(float)
    e = df["exited"].to_numpy(bool)
    lo = _interval(df)
    if e.sum() < 3:
        return {"status": "too few exits", "exits": int(e.sum())}
    X_pow = np.column_stack([np.ones_like(eta), np.log(eta), np.log(B)])
    X_gau = np.column_stack([np.ones_like(eta), B / eta])

    def fit(X):
        def nll(th):
            v = -np.sum(_ll_restarts(X @ th, t, e, lo))
            return v if np.isfinite(v) else 1e300
        th0 = np.zeros(X.shape[1])
        th0[0] = math.log(max(t.sum() / max(e.sum(), 1), 1.0))
        r = optimize.minimize(nll, th0, method="BFGS")
        r2 = optimize.minimize(nll, r.x, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
        th = r2.x if r2.fun <= r.fun else r.x
        try:
            cov = np.linalg.inv(_hessian(nll, th))
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full_like(th, np.nan)
        return th, se, _ll_restarts(X @ th, t, e, lo)

    th_p, se_p, ll_p = fit(X_pow)
    th_g, se_g, ll_g = fit(X_gau)
    d = ll_p - ll_g
    z = float(d.sum() / (math.sqrt(len(d)) * d.std())) if d.std() > 0 else 0.0
    out = {
        "status": "ok", "likelihood": "exact" if lo is None else "interval", "n_restarts": int(len(t)),
        "exits": int(e.sum()),
        "pow_c": th_p[0], "pow_a_ln_eta": th_p[1], "pow_b_ln_B": th_p[2],
        "pow_a_se": se_p[1], "pow_b_se": se_p[2], "pow_loglik": float(ll_p.sum()),
        "gauss_c": th_g[0], "gauss_kappa": th_g[1], "gauss_kappa_se": se_g[1], "gauss_loglik": float(ll_g.sum()),
        "aic_pow": 2 * 3 - 2 * float(ll_p.sum()), "aic_gauss": 2 * 2 - 2 * float(ll_g.sum()),
        "vuong_z_pow_vs_gauss": z, "vuong_p": float(special.erfc(abs(z) / math.sqrt(2))),
    }
    if alpha_hat is not None and np.isfinite(alpha_hat):
        out["alpha_hat"] = alpha_hat
        out["pred_b_J1"] = alpha_hat - 1
        out["pred_a_J1"] = -alpha_hat
        out["b_minus_pred_in_se"] = (th_p[2] - (alpha_hat - 1)) / se_p[2] if se_p[2] > 0 else np.nan
    return out


def slope_in_lnB(df) -> dict:
    """Censored-likelihood slope of ln E k on ln B at fixed eta (and fixed g_max)."""
    B = df["B"].to_numpy(float)
    t = df["k"].to_numpy(float)
    e = df["exited"].to_numpy(bool)
    lo = _interval(df)
    if e.sum() < 3:
        return {"slope": np.nan, "se": np.nan, "exits": int(e.sum())}
    X = np.column_stack([np.ones_like(B), np.log(B)])

    def nll(th):
        v = -np.sum(_ll_restarts(X @ th, t, e, lo))
        return v if np.isfinite(v) else 1e300
    r = optimize.minimize(nll, np.array([math.log(max(t.sum() / e.sum(), 1.0)), 0.0]), method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    try:
        se = float(np.sqrt(np.linalg.inv(_hessian(nll, r.x))[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return {"slope": float(r.x[1]), "se": se, "exits": int(e.sum()),
            "likelihood": "exact" if lo is None else "interval"}


def _hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = h; ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H
