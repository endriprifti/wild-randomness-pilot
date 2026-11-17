"""Batch-size test of H7.1 (Section 7.5, eq:s7-batch, tolerances).

From a pool of per-example gradient projections at one checkpoint and direction, minibatch
noise at each B is the mean of disjoint blocks of B pooled values minus the pool mean (the
block-sum estimator; Nolan 2020, sec. 4.12). A stable law is fitted at each B, giving
alpha-hat(B) and gamma-hat_B; b is the OLS slope of alpha-hat on log2 B (change per doubling).
Its 90% interval comes from a bootstrap that resamples the pool, so the positive correlation
of the alpha-hat(B) across B (one pool) is kept.

Batch order (2026-11-17): the blocks are consecutive values of the pool, so they are random batches only if the
pool is in a random order. P1's pools are consecutive chunks of the same documents, whose neighbours are
correlated; the i.i.d. bootstrap then describes random batches while the main fit described batches of
neighbouring chunks, and b fell outside its own interval. With order_seed the main fit first puts the pool in a
fixed random order (pool_order), as SGD draws random batches. The bootstrap is unchanged: it draws the pool
i.i.d. with replacement, so the order of the pool does not change its distribution, and its replicates are the
same with or without order_seed (P1 recomputes the main fit of saved checkpoints and keeps their bootstrap).

Preregistered reading (thresholds ours, sec. 7.5): invariance if the 90% interval lies inside
+-0.03; drift towards 2 if it lies above 0.05; otherwise undecided. The scale exponent is the
slope of log gamma-hat_B on log B, compared with 1/alpha-hat - 1.

The summary's "reading" is the slope alone. A flat slope is also what a Gaussian gives (alpha-hat = 2 at every B),
so the H7.1 verdict (common/h71_verdict.py) adds the level of alpha-hat and the scale exponent: invariant (wild),
flat at 2 (mild), drift or undecided, recomputed from the summary columns written here.
"""

from __future__ import annotations

import math

import numpy as np

from .stable import block_noise, fit_stable


def pool_order(n: int, seed) -> np.ndarray:
    """A fixed random order of n pool values; seed is an int or a list of ints (np.random.SeedSequence entropy)."""
    return np.random.default_rng(np.random.SeedSequence(seed)).permutation(int(n))


def shuffled(pool, seed) -> np.ndarray:
    """The pool in the fixed random order of `seed` (rows, for a 2-D pool), so that its blocks are random batches."""
    pool = np.asarray(pool)
    return pool[pool_order(len(pool), seed)]


def fit_over_batches(pool: np.ndarray, batch_sizes, n_blocks: int, symmetric: bool = False) -> list[dict]:
    rows = []
    for B in batch_sizes:
        noise = block_noise(pool, int(B), max_blocks=n_blocks)
        if len(noise) < 20:
            rows.append({"B": int(B), "n_blocks": len(noise), "alpha": np.nan, "gamma": np.nan, "beta": np.nan})
            continue
        f = fit_stable(noise, symmetric=symmetric)
        rows.append({"B": int(B), "n_blocks": len(noise), "alpha": f.alpha, "beta": f.beta,
                     "gamma": f.gamma, "delta": f.delta, "loglik": f.loglik})
    return rows


def _slopes(rows):
    B = np.array([r["B"] for r in rows], float)
    a = np.array([r["alpha"] for r in rows], float)
    g = np.array([r["gamma"] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(g) & (g > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    b = np.polyfit(np.log2(B[ok]), a[ok], 1)[0]
    s = np.polyfit(np.log(B[ok]), np.log(g[ok]), 1)[0]
    abar = float(np.mean(a[ok]))
    return float(b), float(s), abar


def h71_bootstrap(pool: np.ndarray, batch_sizes, n_blocks: int, n_boot: int, rng: np.random.Generator,
                  symmetric: bool = False) -> tuple[list[float], list[float]]:
    """n_boot resamples of the pool: the slope b and the scale gap s - (1/alpha-bar - 1) of each (finite ones only).
    Split into chunks with their own generators, the replicates can run in separate processes (P1)."""
    pool = np.asarray(pool, dtype=float)
    boots_b, boots_gap = [], []
    for _ in range(n_boot):
        res = rng.choice(pool, len(pool), replace=True)
        rb = fit_over_batches(res, batch_sizes, n_blocks, symmetric)
        bb, ss, aa = _slopes(rb)
        if np.isfinite(bb):
            boots_b.append(bb)
            boots_gap.append(ss - (1 / aa - 1))
    return boots_b, boots_gap


def h71_summary(rows: list[dict], boots_b, boots_gap, n_pool: int, batch_sizes, n_blocks: int,
                level: float = 0.90, inv_tol: float = 0.03, drift_tol: float = 0.05) -> dict:
    """The H7.1 summary from the per-B fits of the pool and the bootstrap replicates."""
    b, s, abar = _slopes(rows)
    need = n_blocks * max(batch_sizes)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    if len(boots_b):
        b_lo, b_hi = np.quantile(boots_b, [lo_q, hi_q])
        g_lo, g_hi = np.quantile(boots_gap, [lo_q, hi_q])
    else:
        b_lo = b_hi = g_lo = g_hi = np.nan
    if np.isfinite(b_lo) and b_lo >= -inv_tol and b_hi <= inv_tol:
        reading = "invariant"
    elif np.isfinite(b_lo) and b_lo > drift_tol:
        reading = "drift"
    else:
        reading = "undecided"
    summary = {
        "b": b, "b_lo": float(b_lo), "b_hi": float(b_hi), "reading": reading,
        "scale_exponent": s, "alpha_mean": abar, "predicted_scale_exponent": (1 / abar - 1) if abar else np.nan,
        "scale_gap_lo": float(g_lo), "scale_gap_hi": float(g_hi),
        "scale_exponent_excludes_prediction": bool(np.isfinite(g_lo) and (g_lo > 0 or g_hi < 0)),
        "n_pool": int(n_pool), "n_pool_needed": int(need), "pool_sufficient": bool(n_pool >= need),
        "n_boot": len(boots_b), "n_blocks": n_blocks,
    }
    # truncated-Levy hint: first B at which alpha-hat exceeds alpha-hat(B_min) by 0.2
    a = [r["alpha"] for r in rows]
    cross = next((r["B"] for r in rows if np.isfinite(r["alpha"]) and np.isfinite(a[0]) and r["alpha"] - a[0] > 0.2), None)
    summary["B_cross_hint"] = cross
    return summary


def h71_test(pool: np.ndarray, batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024), n_blocks: int = 1600,
             n_boot: int = 200, level: float = 0.90, inv_tol: float = 0.03, drift_tol: float = 0.05,
             rng: np.random.Generator | None = None, symmetric: bool = False, order_seed=None) -> dict:
    """The H7.1 test of one pool. order_seed: the main fit's blocks come from the pool in that fixed random order
    (for a pool whose neighbours may be correlated); None keeps the pool's order (pools drawn i.i.d., as in P0,
    P0b and P4's control, whose results stay as they were)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    pool = np.asarray(pool, dtype=float)
    rows = fit_over_batches(pool if order_seed is None else shuffled(pool, order_seed), batch_sizes, n_blocks, symmetric)
    boots_b, boots_gap = h71_bootstrap(pool, batch_sizes, n_blocks, n_boot, rng, symmetric)
    summary = h71_summary(rows, boots_b, boots_gap, len(pool), batch_sizes, n_blocks, level, inv_tol, drift_tol)
    return {"per_B": rows, "summary": summary}


def blocks_needed(target_halfwidth: float = 0.02, se_at_ref: float = 0.100, ref_n: int = 1000,
                  batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024), z: float = 1.645) -> int:
    """Blocks per B for a 90% half-width of the slope b, independent estimates (sec. 7.5 arithmetic)."""
    x = np.log2(np.asarray(batch_sizes, float))
    sxx = float(np.sum((x - x.mean()) ** 2))
    # half-width = z * SE(alpha-hat) / sqrt(sxx), SE = se_at_ref * sqrt(ref_n / n)
    se_needed = target_halfwidth * math.sqrt(sxx) / z
    return int(math.ceil(ref_n * (se_at_ref / se_needed) ** 2))
