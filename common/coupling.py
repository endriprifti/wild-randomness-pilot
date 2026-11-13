"""Coupling statistic of Section 7.4 (eq:s7-partial).

Residuals within design cells, e_ri = Y_ri - mean of quantity i in cell c(r), give the partial
correlation matrix C. Each C_ij is tested against a null from permuting residuals
independently within cells; all pairs form one Holm family at familywise level 0.05
(the registered family has fifteen pairs). Primary pairs and signs (Endri, 2026-11-05):
(alpha, alpha_ESD) positive, (alpha, K) negative, (alpha, H_grad) negative, with H_grad
entered as |H_grad - H_shuf|; (alpha_ESD, K) is the named secondary pair. The leading
eigenvalue of C against the same null is a secondary check.

a_N belongs to a family (a sweep over N); its pairs use family means of the other residuals.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

PRIMARY = {("alpha", "alpha_esd"): +1, ("alpha", "K"): -1, ("alpha", "H_dep"): -1}
SECONDARY = {("alpha_esd", "K"): +1}  # sign not registered in sec. 7.4; reported only


def residuals(df: pd.DataFrame, quantities: list[str], cell_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(cell_cols)[quantities].transform("mean")
    return df[quantities] - g


def holm(pvals: dict, alpha: float = 0.05, m: int | None = None) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = m or len(items)
    out, stop = {}, False
    for k, (key, p) in enumerate(items):
        thr = alpha / (m - k)
        if stop or p > thr:
            stop = True
            out[key] = {"p": p, "holm_threshold": thr, "reject": False}
        else:
            out[key] = {"p": p, "holm_threshold": thr, "reject": True}
    return out


def coupling_test(df: pd.DataFrame, quantities: list[str], cell_cols: list[str], n_perm: int = 2000,
                  family_col: str | None = None, family_quantity: str | None = None,
                  m_family: int = 15, rng: np.random.Generator | None = None) -> dict:
    """Partial correlations, within-cell permutation p-values, Holm, primary-pair verdicts."""
    rng = rng if rng is not None else np.random.default_rng(0)
    d = df.dropna(subset=quantities).reset_index(drop=True)
    E = residuals(d, quantities, cell_cols).to_numpy(float)
    cells = d.groupby(cell_cols).ngroup().to_numpy()
    k = len(quantities)
    C = np.corrcoef(E, rowvar=False) if len(d) > 2 else np.full((k, k), np.nan)
    lead = float(np.max(np.linalg.eigvalsh(C))) if np.isfinite(C).all() else np.nan

    idx_by_cell = [np.where(cells == c)[0] for c in np.unique(cells)]
    null_C = np.empty((n_perm, k, k))
    null_lead = np.empty(n_perm)
    for b in range(n_perm):
        Ep = E.copy()
        for j in range(k):
            for ix in idx_by_cell:
                Ep[ix, j] = E[rng.permutation(ix), j]
        Cb = np.corrcoef(Ep, rowvar=False)
        null_C[b] = Cb
        null_lead[b] = np.max(np.linalg.eigvalsh(Cb))

    pairs, pvals = {}, {}
    for i, j in itertools.combinations(range(k), 2):
        obs = C[i, j]
        p = (1 + np.sum(np.abs(null_C[:, i, j]) >= abs(obs))) / (n_perm + 1)
        key = (quantities[i], quantities[j])
        pairs[key] = {"r": float(obs), "p_perm": float(p)}
        pvals[key] = float(p)

    if family_col and family_quantity and family_quantity in df.columns:
        fam = d.assign(**{f"e_{q}": E[:, n] for n, q in enumerate(quantities)})
        fm = fam.groupby(family_col)[[f"e_{q}" for q in quantities]].mean()
        aN = df.groupby(family_col)[family_quantity].first().reindex(fm.index)
        for q in quantities:
            x, y = fm[f"e_{q}"].to_numpy(), aN.to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4:
                continue
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            null = np.array([np.corrcoef(x[ok], rng.permutation(y[ok]))[0, 1] for _ in range(n_perm)])
            p = (1 + np.sum(np.abs(null) >= abs(r))) / (n_perm + 1)
            key = (q, family_quantity)
            pairs[key] = {"r": r, "p_perm": float(p), "level": "family", "n_families": int(ok.sum())}
            pvals[key] = float(p)

    H = holm(pvals, 0.05, m=max(m_family, len(pvals)))
    for key in pairs:
        pairs[key].update(H[key])
    verdict = []
    for key, sign in PRIMARY.items():
        k2 = key if key in pairs else (key[1], key[0])
        if k2 not in pairs:
            verdict.append({"pair": key, "status": "not measured"})
            continue
        r = pairs[k2]
        if r["reject"]:
            status = "significant, predicted sign" if np.sign(r["r"]) == sign else "significant, WRONG sign"
        else:
            status = "not significant"
        verdict.append({"pair": key, "predicted_sign": sign, "r": r["r"], "p": r["p_perm"], "status": status})
    p_lead = float((1 + np.sum(null_lead >= lead)) / (n_perm + 1))
    return {"quantities": quantities, "C": C, "pairs": pairs, "primary": verdict,
            "leading_eigenvalue": lead, "p_leading": p_lead, "n_runs": int(len(d)),
            "n_cells": int(len(idx_by_cell)), "m_family": int(max(m_family, len(pvals)))}


def detectable_r(df_within: int, alpha: float = 0.05 / 15, power: float = 0.8) -> float:
    """Smallest |r| detectable at the first Holm step (Fisher z, two-sided)."""
    from scipy import stats
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(z / np.sqrt(max(df_within - 3, 1))))
