"""Escape-time analysis of H3.3 and the escape clause of H6.1 (Section 7.5, eq:s7-escape).

Each restart either exits (barrier criterion) at step k or is right-censored at k_cens.
With an exponential (memoryless) exit law, the censored maximum-likelihood estimate of
E k_exit at a grid point is total exposure / number of exits. Two models of the grid are
fitted to all restarts by censored likelihood:

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


def censored_rate(times: np.ndarray, exited: np.ndarray) -> dict:
    times = np.asarray(times, float)
    exited = np.asarray(exited, bool)
    d = int(exited.sum())
    T = float(times.sum())
    if d == 0:
        return {"exits": 0, "exposure": T, "E_k": float("inf"), "E_k_lower": T, "n": len(times)}
    return {"exits": d, "exposure": T, "E_k": T / d, "n": len(times),
            "E_k_ci_lo": 2 * T / special.chdtri(2 * d, 0.05) if d > 0 else np.nan,
            "E_k_ci_hi": 2 * T / special.chdtri(2 * d, 0.95) if d > 0 else np.nan}


def _ll_restarts(logE: np.ndarray, times: np.ndarray, exited: np.ndarray) -> np.ndarray:
    logE = np.clip(logE, -50.0, 50.0)  # keeps the optimizer's trial points finite
    return -np.asarray(exited, float) * logE - np.exp(-logE) * times


def fit_grid(df, alpha_hat: float | None = None) -> dict:
    """df has one row per restart: eta, B, k (exit or censoring step), exited (bool)."""
    eta = df["eta"].to_numpy(float)
    B = df["B"].to_numpy(float)
    t = df["k"].to_numpy(float)
    e = df["exited"].to_numpy(bool)
    if e.sum() < 3:
        return {"status": "too few exits", "exits": int(e.sum())}
    X_pow = np.column_stack([np.ones_like(eta), np.log(eta), np.log(B)])
    X_gau = np.column_stack([np.ones_like(eta), B / eta])

    def fit(X):
        def nll(th):
            v = -np.sum(_ll_restarts(X @ th, t, e))
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
        return th, se, _ll_restarts(X @ th, t, e)

    th_p, se_p, ll_p = fit(X_pow)
    th_g, se_g, ll_g = fit(X_gau)
    d = ll_p - ll_g
    z = float(d.sum() / (math.sqrt(len(d)) * d.std())) if d.std() > 0 else 0.0
    out = {
        "status": "ok", "n_restarts": int(len(t)), "exits": int(e.sum()),
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
    if e.sum() < 3:
        return {"slope": np.nan, "se": np.nan, "exits": int(e.sum())}
    X = np.column_stack([np.ones_like(B), np.log(B)])

    def nll(th):
        v = -np.sum(_ll_restarts(X @ th, t, e))
        return v if np.isfinite(v) else 1e300
    r = optimize.minimize(nll, np.array([math.log(max(t.sum() / e.sum(), 1.0)), 0.0]), method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    try:
        se = float(np.sqrt(np.linalg.inv(_hessian(nll, r.x))[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return {"slope": float(r.x[1]), "se": se, "exits": int(e.sum())}


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
