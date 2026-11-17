"""H_link (exploratory; added 2026-11-17 after the first results, fixed before computing it): the Noah variable's
magnitudes carry Joseph-type memory (volatility clustering), and the series are multifractal.

For a series x, or the columns of a matrix (one P4 run: one column per fixed random direction; every statistic is
then the mean over the columns, and each surrogate replicate replaces every column by its own surrogate):
  memory  H of the magnitudes m = |x - median(x)| by DFA-1 and by the Abry-Veitch wavelet estimator, against the
          same statistic on surrogates of x: shuffled (keep the marginal, destroy all memory; the pass rule) and
          phase-randomized (keep the spectrum of x, destroy nonlinear dependence such as volatility clustering;
          reported). One-sided p = (1 + #{surrogate >= observed}) / (n_surr + 1).
  width   MFDFA Delta h = h(q_lo) - h(q_hi) of x against IAAFT surrogates of x (keep marginal and spectrum),
          one-sided p as above.
Pass: p_shuf < level for DFA and for the wavelet estimator, each over at least min_decades decades of scale,
AND p_iaaft < level.

Scales: DFA and MFDFA on s_min ... n/10 (common/hurst.log_scales), wavelet octaves j1 ... the coarsest with at
least four coefficients. Main row s_min = 8 (Endri, 2026-11-17, before any H_link value was computed: at the s7.2
range 10 ... n/10 the shortest series, P4's transformer after warm-up, 9,064 steps, spans only 1.96 decades);
sensitivity row s_min = 10 ... n/10, on the same surrogates, where such series fail the two-decade clause
(decades_flag). pass_ignoring_decades (reported only) drops that clause.

Calibration series (synthetic(); parameters ours, fixed before the calibration ran):
  levy     i.i.d. SaS, alpha = 1.5                         Noah only        expected: fail
  fgn      fractional Gaussian noise, H = 0.75             Joseph only      expected: fail
  mrw      multifractal random walk increments,            both             expected: pass
           lam2 = 0.05, integral scale = n (Delta h = 0.0875 in theory)
  cascade  random-order binomial cascade, a = 0.75         both             expected: pass
           (length 2^k >= n, cut to n; Delta h = 0.439 in theory)
"""

from __future__ import annotations

import math

import numpy as np

from . import hurst

TEST_DEFAULTS = {"n_surr": 99, "s_min": 8, "s_min_sensitivity": 10, "wavelet_j1": 3, "q_lo": 0.5, "q_hi": 4.0,
                 "level": 0.05, "min_decades": 2.0}
SYNTHETIC = {"levy": {"alpha": 1.5}, "fgn": {"H": 0.75}, "mrw": {"lam2": 0.05}, "cascade": {"a": 0.75}}
EXPECTED = {"levy": "fail (Noah only)", "fgn": "fail (Joseph only)", "mrw": "pass (both)", "cascade": "pass (both)"}


def synthetic(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind not in SYNTHETIC:
        raise ValueError(f"unknown synthetic series {kind!r}")
    p = SYNTHETIC[kind]
    if kind == "levy":
        return hurst.levy_increments(n, p["alpha"], rng)
    if kind == "fgn":
        return hurst.fgn(n, p["H"], rng)
    if kind == "mrw":
        return hurst.mrw_increments(n, p["lam2"], rng)
    return hurst.random_binomial_series(int(math.ceil(math.log2(n))), p["a"], rng)[:n]


def magnitudes(v: np.ndarray) -> np.ndarray:
    return np.abs(v - np.median(v))


def _p_greater(obs: float, sur: np.ndarray) -> float:
    sur = np.asarray(sur, float)
    sur = sur[np.isfinite(sur)]
    if not np.isfinite(obs) or not len(sur):
        return float("nan")
    return float((1 + np.sum(sur >= obs)) / (len(sur) + 1))


def _stat(f, s: np.ndarray, *a) -> float:
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(f(s, *a)) if len(s) else float("nan")


def hlink_rows(X, rng: np.random.Generator, n_surr: int = 99, s_min: int = 8, s_min_sensitivity: int | None = 10,
               wavelet_j1: int = 3, q_lo: float = 0.5, q_hi: float = 4.0, level: float = 0.05,
               min_decades: float = 2.0) -> list[dict]:
    """The H_link test of one series (1-D) or of the columns of one matrix (statistics averaged over columns).

    One row per scale range: "main" (DFA and MFDFA on s_min ... n/10) and, unless s_min_sensitivity is None,
    "sensitivity" (s_min_sensitivity ... n/10, the s7.2 range), both on the same surrogates; the wavelet estimator
    has no s_min (octaves wavelet_j1 ... ) and is the same in both rows.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = X[np.isfinite(X).all(axis=1)]
    n, d = X.shape
    roles = [("main", int(s_min))]
    if s_min_sensitivity is not None and int(s_min_sensitivity) != int(s_min):
        roles.append(("sensitivity", int(s_min_sensitivity)))
    scales = {r: hurst.log_scales(n, sm) for r, sm in roles}
    dec_wav = float(hurst.hurst_wavelet(magnitudes(X[:, 0]), j1=wavelet_j1)["decades"])

    def stats(cols, which):
        """Column means: H_dfa of the magnitudes per scale range, H_wav of the magnitudes, Delta h per range."""
        out = {}
        if "mem" in which:
            mags = [magnitudes(v) for v in cols]
            out["wav"] = np.mean([hurst.hurst_wavelet(m, j1=wavelet_j1)["H"] for m in mags])
            for r, _ in roles:
                out[f"dfa_{r}"] = np.mean([hurst.hurst_dfa(m, scales[r])["H"] for m in mags])
        if "width" in which:
            for r, _ in roles:
                out[f"width_{r}"] = np.mean([hurst.delta_h(v, q_lo, q_hi, scales[r])["delta_h"] for v in cols])
        return out

    cols = [X[:, j] for j in range(d)]
    obs = stats(cols, ("mem", "width"))
    sur = {"shuf": [], "phase": [], "iaaft": []}
    for _ in range(n_surr):
        sur["shuf"].append(stats([hurst.shuffle_surrogate(v, rng) for v in cols], ("mem",)))
        sur["phase"].append(stats([hurst.phase_surrogate(v, rng) for v in cols], ("mem",)))
        sur["iaaft"].append(stats([hurst.iaaft_surrogate(v, rng) for v in cols], ("width",)))

    rows = []
    for r, sm in roles:
        sc = scales[r]
        dec_dfa = float(np.log10(sc[-1] / sc[0])) if len(sc) else 0.0
        out = {"scales": r, "n": int(n), "n_series": int(d), "n_surr": int(n_surr),
               "s_min": int(sc[0]) if len(sc) else None, "s_max": int(sc[-1]) if len(sc) else None,
               "decades_dfa": dec_dfa, "decades_wav": dec_wav,
               "H_abs_dfa": float(obs[f"dfa_{r}"]), "H_abs_wav": float(obs["wav"])}
        for est, k in (("dfa", f"dfa_{r}"), ("wav", "wav")):
            for kind in ("shuf", "phase"):
                s = np.asarray([z[k] for z in sur[kind]], float)
                out[f"H_abs_{est}_{kind}_mean"] = _stat(np.mean, s)
                out[f"H_abs_{est}_{kind}_q95"] = _stat(np.quantile, s, 0.95)
                out[f"p_{kind}_{est}"] = _p_greater(obs[k], s)
        w = np.asarray([z[f"width_{r}"] for z in sur["iaaft"]], float)
        out.update({"delta_h": float(obs[f"width_{r}"]), "delta_h_iaaft_mean": _stat(np.mean, w),
                    "delta_h_iaaft_q95": _stat(np.quantile, w, 0.95), "p_iaaft": _p_greater(obs[f"width_{r}"], w),
                    "q_lo": q_lo, "q_hi": q_hi})
        ok_dec = {"dfa": dec_dfa >= min_decades - 1e-9, "wav": dec_wav >= min_decades - 1e-9}
        out["two_decades"] = bool(ok_dec["dfa"] and ok_dec["wav"])
        out["decades_flag"] = ("" if out["two_decades"] else
                               f"below {min_decades:g} decades (DFA {dec_dfa:.2f}, wavelet {dec_wav:.2f})")
        p_mem = all(out[f"p_shuf_{e}"] < level for e in ("dfa", "wav"))
        out["memory_pass"] = bool(out["two_decades"] and p_mem)
        out["memory_beyond_linear"] = bool(all(out[f"p_phase_{e}"] < level for e in ("dfa", "wav")))  # reported
        out["width_pass"] = bool(out["p_iaaft"] < level)
        out["pass"] = bool(out["memory_pass"] and out["width_pass"])
        out["pass_ignoring_decades"] = bool(p_mem and out["width_pass"])                              # reported
        rows.append(out)
    return rows


def hlink_test(X, rng: np.random.Generator, **kw) -> dict:
    """The main row only (scales s_min ... n/10)."""
    kw.setdefault("s_min_sensitivity", None)
    return hlink_rows(X, rng, **kw)[0]
