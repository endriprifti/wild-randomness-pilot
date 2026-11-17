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

Confound controls (Endri, 2026-11-17, fixed before any H_link value on real data was computed; reported beside the
main rows, the pass rule unchanged):
  documents  P1's series joins 16 context windows ("documents") of 2,048 tokens, 2,047 surprisals each. Against 99
             within-document shuffles (tokens shuffled inside each document, documents in place: each keeps its
             level and spread, clustering inside documents is gone): p_doc_dfa, p_doc_wav (H of the magnitudes)
             and p_doc_width (MFDFA width); beyond_documents = all three < level, over two decades.
  detrended  P4's projections contain the full-batch gradient, which drifts as the loss falls (eta is constant
             after warm-up). Extra rows, scales = "detrended": the same test on x minus its centred moving median
             of DETREND_WIDTH = 201 steps (detrend()). Why 201: a median of x removes the level (the full-batch
             gradient) and keeps the spread, where volatility clustering lives, so the window can be short; 201
             steps follow a drift with periods of 3,000 steps or more to within 1% (1,000 steps: 6%), the median's
             own noise is about 1.25 sd / sqrt(201) = 0.08 sd (under 1% of the variance), the window is odd so
             centred, fixed in steps for every run, and shorter than the largest DFA scale (906): linear memory
             slower than about 200 steps is removed with the drift, the memory of the magnitudes is not.

Calibration series (synthetic(); parameters ours, fixed before the calibration ran):
  levy     i.i.d. SaS, alpha = 1.5                         Noah only        expected: fail
  fgn      fractional Gaussian noise, H = 0.75             Joseph only      expected: fail
  mrw      multifractal random walk increments,            both             expected: pass
           lam2 = 0.05, integral scale = n (Delta h = 0.0875 in theory)
  cascade  random-order binomial cascade, a = 0.75         both             expected: pass
           (length 2^k >= n, cut to n; Delta h = 0.439 in theory)
Control calibration (synthetic_control(); parameters ours, fixed before it ran):
  docs_shift  16 documents of 2,047 i.i.d. values (standardized Gamma(2), skewed like surprisal), document level
              ~ N(0, 0.5^2) and log spread ~ N(0, 0.3^2) in within-document sd    beyond_documents: fail
  mrw_docs    MRW (as mrw) of 16 x 2,047 values, cut into 16 documents             beyond_documents: pass
  iid_drift   i.i.d. N(0, 1) plus a slow drift (drift())                          detrended: fail (drift alone)
  fgn_drift   fGn H = 0.75 plus the drift                                          detrended: fail (Joseph only)
  mrw_drift   MRW plus the drift                                                   detrended: pass
  drift: exp(-t / 500) + sum_k c_k cos(2 pi k t / n + phi_k), k = 1..3, c_k ~ N(0, (0.5 / k)^2), phi_k uniform,
  scaled to the sd of the series it is added to (as large as the noise; a fast relaxation of 500 steps and
  periods n, n/2, n/3).
"""

from __future__ import annotations

import math

import numpy as np

from . import hurst

TEST_DEFAULTS = {"n_surr": 99, "s_min": 8, "s_min_sensitivity": 10, "wavelet_j1": 3, "q_lo": 0.5, "q_hi": 4.0,
                 "level": 0.05, "min_decades": 2.0}
SYNTHETIC = {"levy": {"alpha": 1.5}, "fgn": {"H": 0.75}, "mrw": {"lam2": 0.05}, "cascade": {"a": 0.75}}
EXPECTED = {"levy": "fail (Noah only)", "fgn": "fail (Joseph only)", "mrw": "pass (both)", "cascade": "pass (both)"}
DOC_LEN = 2047                  # P1: 16 windows of 2,048 tokens, 2,047 surprisals each (config P1 surprisal_len - 1)
N_DOCS = 16
DETREND_WIDTH = 201             # P4 detrended rows: centred moving median, steps (why: module docstring)
CONTROLS = {"doc_len": DOC_LEN, "n_docs": N_DOCS, "detrend_width": DETREND_WIDTH}
CONTROL_SYNTHETIC = {"docs_shift": {"level_sd": 0.5, "log_spread_sd": 0.3, "gamma_shape": 2.0},
                     "mrw_docs": {"lam2": 0.05}, "iid_drift": {}, "fgn_drift": {"H": 0.75}, "mrw_drift": {"lam2": 0.05},
                     "drift": {"tau": 500.0, "periods": [1, 2, 3], "amp_scale": 0.5, "sd_ratio": 1.0}}
CONTROL_N = {"docs_shift": N_DOCS * DOC_LEN, "mrw_docs": N_DOCS * DOC_LEN, "iid_drift": 9064, "fgn_drift": 9064,
             "mrw_drift": 9064}
CONTROL_EXPECTED = {"docs_shift": "beyond_documents: fail", "mrw_docs": "beyond_documents: pass",
                    "iid_drift": "detrended: fail (drift alone)", "fgn_drift": "detrended: fail (Joseph only)",
                    "mrw_drift": "detrended: pass"}


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


def drift(n: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """A slow drift: a relaxation exp(-t / tau) plus cosines of periods n / k, scaled to standard deviation sd."""
    p = CONTROL_SYNTHETIC["drift"]
    t = np.arange(n, dtype=float)
    g = np.exp(-t / p["tau"])
    for k in p["periods"]:
        g = g + rng.normal(0.0, p["amp_scale"] / k) * np.cos(2 * np.pi * k * t / n + rng.uniform(0, 2 * np.pi))
    g = g - g.mean()
    return sd * p["sd_ratio"] * g / g.std()


def synthetic_control(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A series of the control calibration and how to test it: {"doc_len": ...} or {"detrend": True}."""
    if kind not in CONTROL_N:
        raise ValueError(f"unknown control series {kind!r}")
    n = CONTROL_N[kind]
    if kind == "docs_shift":
        p = CONTROL_SYNTHETIC[kind]
        k = p["gamma_shape"]
        z = (rng.gamma(k, size=(N_DOCS, DOC_LEN)) - k) / np.sqrt(k)
        level = rng.normal(0.0, p["level_sd"], size=(N_DOCS, 1))
        spread = np.exp(rng.normal(0.0, p["log_spread_sd"], size=(N_DOCS, 1)))
        return (level + spread * z).ravel(), {"doc_len": DOC_LEN}
    if kind == "mrw_docs":
        return hurst.mrw_increments(n, CONTROL_SYNTHETIC[kind]["lam2"], rng), {"doc_len": DOC_LEN}
    if kind == "iid_drift":
        x = rng.normal(size=n)
    elif kind == "fgn_drift":
        x = hurst.fgn(n, CONTROL_SYNTHETIC[kind]["H"], rng)
    else:
        x = hurst.mrw_increments(n, CONTROL_SYNTHETIC[kind]["lam2"], rng)
    return x + drift(n, float(np.std(x)), rng), {"detrend": True}


def within_doc_shuffle(v: np.ndarray, doc_len: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the values inside each document of doc_len values (a shorter last part is its own document)."""
    v = np.asarray(v)
    k = len(v) // doc_len
    out = np.empty_like(v)
    out[:k * doc_len] = rng.permuted(v[:k * doc_len].reshape(k, doc_len), axis=1).ravel()
    if len(v) > k * doc_len:
        out[k * doc_len:] = rng.permutation(v[k * doc_len:])
    return out


def detrend(X, width: int = DETREND_WIDTH) -> np.ndarray:
    """x minus its centred moving median of `width` steps (per column; reflected at the ends)."""
    from scipy import ndimage
    X = np.asarray(X, dtype=float)
    size = (width,) if X.ndim == 1 else (width, 1)
    return X - ndimage.median_filter(X, size=size, mode="reflect")


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
               min_decades: float = 2.0, doc_len: int | None = None) -> list[dict]:
    """The H_link test of one series (1-D) or of the columns of one matrix (statistics averaged over columns).

    One row per scale range: "main" (DFA and MFDFA on s_min ... n/10) and, unless s_min_sensitivity is None,
    "sensitivity" (s_min_sensitivity ... n/10, the s7.2 range), both on the same surrogates; the wavelet estimator
    has no s_min (octaves wavelet_j1 ... ) and is the same in both rows. With doc_len, the documents control too:
    n_surr within-document shuffles, drawn after the other surrogates (the main columns do not depend on it).
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
    if doc_len:
        sur["doc"] = [stats([within_doc_shuffle(v, doc_len, rng) for v in cols], ("mem", "width"))
                      for _ in range(n_surr)]

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
        if doc_len:
            for est, k in (("dfa", f"dfa_{r}"), ("wav", "wav")):
                s = np.asarray([z[k] for z in sur["doc"]], float)
                out[f"H_abs_{est}_doc_mean"] = _stat(np.mean, s)
                out[f"H_abs_{est}_doc_q95"] = _stat(np.quantile, s, 0.95)
                out[f"p_doc_{est}"] = _p_greater(obs[k], s)
            w = np.asarray([z[f"width_{r}"] for z in sur["doc"]], float)
            out.update({"doc_len": int(doc_len), "delta_h_doc_mean": _stat(np.mean, w),
                        "delta_h_doc_q95": _stat(np.quantile, w, 0.95),
                        "p_doc_width": _p_greater(obs[f"width_{r}"], w)})
            out["memory_beyond_documents"] = bool(out["two_decades"] and out["p_doc_dfa"] < level
                                                  and out["p_doc_wav"] < level)
            out["width_beyond_documents"] = bool(out["p_doc_width"] < level)
            out["beyond_documents"] = bool(out["memory_beyond_documents"] and out["width_beyond_documents"])
        rows.append(out)
    return rows


def hlink_test(X, rng: np.random.Generator, **kw) -> dict:
    """The main row only (scales s_min ... n/10)."""
    kw.setdefault("s_min_sensitivity", None)
    return hlink_rows(X, rng, **kw)[0]
