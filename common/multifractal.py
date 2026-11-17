"""Generalized dimensions of point clouds and the width Delta h_rep (Sections 7.2 and 7.4).

Generalized correlation sums Z_q(r) = mean_i p_i(r)^(q-1), with p_i(r) the fraction of other
points within r of point i, give tau(q) as the slope of log Z_q(r) on log r, D_q = tau(q)/(q-1)
(D_1 from mean_i log p_i(r)), and h(q) = d tau / d q (the Legendre variable). Delta h_rep is
h(q_lo) - h(q_hi) over a preregistered q range; negative q is unstable (Falconer 2014, p. 304).

Dimension clause (sec. 7.2): global D_q claims only where two decades of r are available,
roughly D_q below log10 n with n points; above that the code reports a local dimension at a
stated scale and marks Delta h_rep as local. The width that P1 and P4 report (reported_width,
2026-11-17) is NaN when the clause fails or the width is negative (h(q) must not increase in q);
delta_h_rep's own number stays in their dh_rep_raw column.
"""

from __future__ import annotations

import math

import numpy as np


def _neighbour_counts(X: np.ndarray, radii: np.ndarray, chunk: int = 1024, device: str | None = None,
                      centres: np.ndarray | None = None):
    """counts[c, k] = #{j != c : |x_c - x_j| <= radii[k]} for centre points c (all points by default)."""
    if centres is not None:
        n_all = len(X)
        order = np.concatenate([centres, np.setdiff1d(np.arange(n_all), centres)])
        X = X[order]
        n_c = len(centres)
    else:
        n_c = len(X)
    n = len(X)
    K = len(radii)
    counts = np.zeros((n_c, K), dtype=np.int64)
    try:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        rt = torch.as_tensor(radii, dtype=torch.float32, device=dev)
        for s in range(0, n_c, chunk):
            d = torch.cdist(Xt[s:min(s + chunk, n_c)], Xt)
            idx = torch.arange(s, min(s + chunk, n_c), device=dev)
            d[torch.arange(len(idx), device=dev), idx] = float("inf")
            b = torch.searchsorted(rt, d.contiguous())  # bin of each distance
            oh = torch.zeros(len(idx), K + 1, dtype=torch.int64, device=dev)
            oh.scatter_add_(1, b, torch.ones_like(b))
            counts[s:s + len(idx)] = torch.cumsum(oh[:, :K], dim=1).cpu().numpy()
        return counts
    except ImportError:
        pass
    for s in range(0, n_c, chunk):
        d = np.sqrt(((X[s:min(s + chunk, n_c), None, :] - X[None, :, :]) ** 2).sum(-1))
        for r, i in enumerate(range(s, min(s + chunk, n_c))):
            d[r, i] = np.inf
        b = np.searchsorted(radii, d)
        for r in range(len(b)):
            counts[s + r] = np.cumsum(np.bincount(b[r], minlength=K + 1)[:K])
    return counts


def choose_radii(X: np.ndarray, n_radii: int = 32, p_lo: float | None = None, p_hi: float = 0.1,
                 rng: np.random.Generator | None = None, n_pairs: int = 200_000) -> np.ndarray:
    """Log-spaced radii between the random-pair distance quantiles p_lo (default 1/n) and p_hi."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(X)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    # pair distances in chunks: the same values, without three (n_pairs x dim) arrays at once (10 GB at dim 2,048)
    d = np.concatenate([np.sqrt(((X[i[s:s + 8192]] - X[j[s:s + 8192]]) ** 2).sum(-1))
                        for s in range(0, len(i), 8192)]) if len(i) else np.empty(0)
    p_lo = p_lo if p_lo is not None else min(1.0 / n, p_hi / 100)
    lo, hi = np.quantile(d, [p_lo, p_hi])
    lo = max(lo, 1e-12)
    return np.logspace(np.log10(lo), np.log10(hi), n_radii)


def generalized_dimensions(X: np.ndarray, q: np.ndarray, radii: np.ndarray | None = None,
                           min_neighbours: int = 5, max_mean_p: float = 0.1, device=None,
                           n_centres: int | None = None, rng: np.random.Generator | None = None) -> dict:
    """tau(q), D_q and h(q) of a point cloud from generalized correlation sums.

    p_i(r) counts point i itself, (count_i + 1)/n, so that every p_i > 0 (needed for q < 1);
    the fit uses only radii at which the 10th percentile of neighbour counts is at least
    `min_neighbours` (the self-count is then a small bias) and the mean p is at most `max_mean_p`
    (before saturation).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    q = np.asarray(q, dtype=float)
    radii = choose_radii(X, p_hi=max_mean_p) if radii is None else np.asarray(radii)
    centres = None
    if n_centres is not None and n_centres < n:
        centres = (rng or np.random.default_rng(0)).choice(n, n_centres, replace=False)
    counts = _neighbour_counts(X, radii, device=device, centres=centres)
    p = (counts + 1.0) / n
    lr = np.log(radii)
    ok = (np.quantile(counts, 0.1, axis=0) >= min_neighbours) & (p.mean(axis=0) <= max_mean_p)
    if ok.sum() < 3:
        ok = np.zeros(len(radii), bool)
        ok[np.argsort(np.abs(p.mean(axis=0) - max_mean_p / 10))[:3]] = True
    lr_fit = lr[ok]
    pk = p[:, ok]
    tau = np.empty(len(q))
    d1 = np.nan
    for k, qq in enumerate(q):
        if abs(qq - 1) < 1e-9:
            d1 = np.polyfit(lr_fit, np.mean(np.log(pk), axis=0), 1)[0]
            tau[k] = 0.0
        else:
            tau[k] = np.polyfit(lr_fit, np.log(np.mean(pk ** (qq - 1), axis=0)), 1)[0]
    is1 = np.abs(q - 1) < 1e-9
    D = np.where(is1, d1, tau / np.where(is1, 1.0, q - 1))
    h = np.gradient(tau, q) if len(q) > 1 else np.full(1, np.nan)
    decades = float(np.log10(radii[ok][-1] / radii[ok][0]))
    return {"q": q, "tau": tau, "D": D, "h": h, "radii": radii[ok], "decades_r": decades, "n": n}


def delta_h_rep(X: np.ndarray, q_lo: float = 0.5, q_hi: float = 4.0, n_q: int = 15,
                device=None, max_points: int | None = None, n_centres: int | None = 5000,
                rng: np.random.Generator | None = None) -> dict:
    """Delta h_rep of a representation point cloud with the dimension clause applied."""
    rng = rng if rng is not None else np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    if max_points is not None and len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    n = len(X)
    q = np.linspace(q_lo, q_hi, n_q)
    q = np.unique(np.concatenate([q, [1.0, 2.0]]))
    gd = generalized_dimensions(X, q, device=device, n_centres=n_centres, rng=rng)
    i_lo, i_hi = int(np.argmin(np.abs(q - q_lo))), int(np.argmin(np.abs(q - q_hi)))
    D2 = float(gd["D"][int(np.argmin(np.abs(q - 2.0)))])
    D_max = float(np.nanmax(gd["D"]))
    clause_ok = bool(D_max < math.log10(n) and gd["decades_r"] >= 2 - 1e-9)
    out = {"n_points": n, "D2": D2, "D_max": D_max, "log10_n": math.log10(n),
           "decades_r": gd["decades_r"], "clause_ok": clause_ok,
           "h_qlo": float(gd["h"][i_lo]), "h_qhi": float(gd["h"][i_hi]),
           "delta_h_rep": float(gd["h"][i_lo] - gd["h"][i_hi]), "q_lo": q_lo, "q_hi": q_hi,
           "scope": "global" if clause_ok else "local"}
    if not clause_ok:
        # local dimension at a stated scale: slope of log C(r) over the smallest third of radii
        r = gd["radii"]
        k = max(3, len(r) // 3)
        gl = generalized_dimensions(X, np.array([2.0]), radii=r[:k], device=device, n_centres=n_centres, rng=rng)
        out["local_D2"] = float(gl["D"][0])
        out["local_scale"] = float(np.exp(np.mean(np.log(r[:k]))))
    return out


def reported_width(d: dict) -> tuple[float, str]:
    """The Delta h_rep to report from delta_h_rep's result, and why it is NaN ("" when it is not): NaN when the
    dimension clause fails (D_max >= log10 n or fewer than two decades of r: no global D_q claim) or the width
    is negative (h(q) decreases in q for a measure, so a negative width is an estimation failure)."""
    w = float(d.get("delta_h_rep", np.nan))
    if not d.get("clause_ok", False):
        return float("nan"), (f"dimension clause fails (D_max {float(d.get('D_max', np.nan)):.2f} vs log10 n "
                              f"{float(d.get('log10_n', np.nan)):.2f}, {float(d.get('decades_r', np.nan)):.2f} decades "
                              f"of r): local only")
    if not np.isfinite(w):
        return float("nan"), "no width"
    if w < 0:
        return float("nan"), f"negative width {w:.3g}"
    return w, ""


# ----------------------------------------------------------------------------- cascades
def binomial_cascade_points(n: int, p: float = 0.7, depth: int = 16, dim: int = 1,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """n points sampled from a binomial multiplicative cascade on [0,1]^dim (independent axes if dim > 1)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    pts = np.zeros((n, dim))
    for d in range(dim):
        bits = (rng.uniform(size=(n, depth)) > p).astype(float)  # 0 = left (prob p)
        w = 0.5 ** np.arange(1, depth + 1)
        pts[:, d] = bits @ w + rng.uniform(size=n) * 0.5 ** depth
    return pts


def binomial_tau(q, p: float = 0.7, dim: int = 1):
    q = np.asarray(q, dtype=float)
    return -dim * np.log2(p ** q + (1 - p) ** q)


def binomial_h(q, p: float = 0.7, dim: int = 1):
    q = np.asarray(q, dtype=float)
    a, b = p ** q, (1 - p) ** q
    return -dim * (a * math.log(p) + b * math.log(1 - p)) / ((a + b) * math.log(2))
