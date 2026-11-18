"""P6 (added 2026-11-18, fixed before any P6 run): the CPU side of the factorial Noah-Joseph test.

numpy, scipy and pandas only. Two parts:

1. Tasks for the CPU pool of p6_noah_joseph.py (run_p6_task, in spawned processes): P1's statistics unchanged
   (common/p1_tasks.run_task: grad, esd, act, dh, htok, h71_fit, h71_boot) and the H_link rows of h_link.run_unit
   (unchanged: P1 treatment for token-surprisal series, P4 treatment for the per-step gradient projections).

2. The pre-stated analysis (FAMILY, analyse): per outcome, the linear model
       Y = seed block + b_N n + b_J j + b_NJ n j + e,
   n and j the Noah and Joseph levels coded -1/2 and +1/2 at the corners (ordinal scores scaled to [-1/2, 1/2] with
   three levels), seeds as blocks (the cells of one seed share their random numbers, p6_streams.py). Each term is
   tested two-sided by Freedman-Lane permutation (residuals of the model without the term, permuted within seed
   blocks, statistic t, 9,999 permutations); the OLS t-test is reported beside it. The confirmatory family is the
   14 tests below at the 14M corners, final checkpoint, Holm at 0.05:
       Joseph -> Noah   K, alpha                            b_J (predicted: K up, alpha down) and b_NJ
       Noah -> Joseph   H_tok excess, magnitude memory and   b_N (predicted: up) and b_NJ
                        width of the token surprisals; magnitude memory and width of the gradient series
   A test passes (a link) if its Holm-adjusted p < 0.05; with the predicted sign "as predicted", with the other
   "opposite to the prediction" (both are links). A negative N or N x J term on a token-surprisal outcome is read
   as "dilution" and not as a link unless the same term on the variant part's H_tok excess (rank noise removed) is
   negative too at p < 0.05: the rank part is i.i.d. in the data and its spread grows with the Zipf tail, so it
   lowers the measured memory of the whole series where the data have memory (p6 power check, README). No test passing is reported as "no link
   detected", with each effect's 95% interval and the effect detectable with 80% power.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

LEVEL = 0.05
N_PERM = 9999
STAGE_PRIMARY = 1.0
SIZE_PRIMARY = "14m"

# (outcome, column, direction it belongs to, the main-effect term tested, predicted sign of that term)
FAMILY = [
    ("K", "K", "Joseph->Noah", "J", +1),
    ("alpha", "alpha", "Joseph->Noah", "J", -1),
    ("Htok_excess", "Htok_excess", "Noah->Joseph", "N", +1),
    ("Habs_tok_excess", "tok_main_Habs_excess", "Noah->Joseph", "N", +1),
    ("dh_tok_excess", "tok_main_dh_excess", "Noah->Joseph", "N", +1),
    ("Habs_grad_excess", "grad_main_Habs_excess", "Noah->Joseph", "N", +1),
    ("dh_grad_excess", "grad_main_dh_excess", "Noah->Joseph", "N", +1),
]
# outcomes on the token surprisals, whose rank part (i.i.d. in the data, its spread growing with the Zipf tail)
# dilutes the memory of the whole series: a negative N or N x J term on them is a link only if the same term on the
# variant part's H_tok excess (REFEREE: the part that carries the data's memory) is negative at p < 0.05 too
TOKEN_OUTCOMES = {"Htok_excess", "Habs_tok_excess", "dh_tok_excess"}
REFEREE = "variant_Htok_excess"
# outcomes reported beside the family (no correction): the other P1 quantities, the own effects
SECONDARY = ["K_max", "Kurt0", "n_outlier_channels", "act_zeta", "alpha_esd", "zeta_right", "zeta_left",
             "h71_mean_alpha_hat", "h71_slope", "variant_Htok_excess", "rank_Htok_excess", "eval_loss",
             "loss_gap_oracle", "tok_main_pass", "grad_main_pass"]


# ----------------------------------------------------------------------------- pool tasks
def _write_stats(ck_dir: Path, name: str, kind: str, result, err, t0: float) -> dict:
    from common import p1_tasks
    rec = {"name": name, "kind": kind, "result": result, "error": err,
           "seconds": round(time.perf_counter() - t0, 3), "pid": os.getpid(),
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    path = p1_tasks.stats_path(ck_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    p1_tasks.atomic_json(path, rec)
    return rec


def run_p6_task(spec: dict) -> dict:
    """P1's tasks as they are; kind "hlink": h_link.run_unit on spec["unit"] with spec["test"]."""
    from common import p1_tasks
    if spec["kind"] in p1_tasks.KINDS:
        return p1_tasks.run_task(spec)
    t0 = time.perf_counter()
    err = None
    try:
        import h_link                       # sets CUDA_VISIBLE_DEVICES = "" (this process has no GPU anyway)
        result = h_link.run_unit(spec["unit"], spec["test"])
    except p1_tasks.CODE_ERRORS:
        raise
    except Exception as e:  # noqa: BLE001
        result, err = None, f"{type(e).__name__}: {str(e)[:300]}"
    return _write_stats(Path(spec["dir"]), spec["name"], spec["kind"], result, err, t0)


# ----------------------------------------------------------------------------- rows of one run
def hlink_columns(prefix: str, rec: dict | None) -> dict:
    """Excess magnitude memory and width (over the shuffled and IAAFT surrogate means) and the pass flags of each
    H_link row (main, sensitivity, standardized / detrended / rescaled) of one series."""
    out = {}
    res = (rec or {}).get("result") or {}
    for r in res.get("rows", []):
        sc = r.get("scales")
        p = f"{prefix}_{sc}"
        out[f"{p}_Habs_excess"] = r["H_abs_dfa"] - r["H_abs_dfa_shuf_mean"]
        out[f"{p}_Habs_wav_excess"] = r["H_abs_wav"] - r["H_abs_wav_shuf_mean"]
        out[f"{p}_dh_excess"] = r["delta_h"] - r["delta_h_iaaft_mean"]
        out[f"{p}_pass"] = bool(r["pass"])
        out[f"{p}_memory_beyond_linear"] = bool(r["memory_beyond_linear"])
        for k in ("beyond_documents", "decades_dfa"):
            if k in r:
                out[f"{p}_{k}"] = r[k]
    return out


def checkpoint_row(result: dict, stats: dict, extra: dict) -> dict:
    """One checkpoint's row: P1's assembled row plus the P6 columns."""
    row = dict(result["row"])
    if "Htok_H_dfa" in row and "Htok_H_shuf_dfa" in row:
        row["Htok_excess"] = row["Htok_H_dfa"] - row["Htok_H_shuf_dfa"]
    for part in ("variant", "rank"):
        s = (stats.get(f"htok_{part}") or {}).get("result") or {}
        if s:
            row[f"{part}_Htok_excess"] = s["H_dfa"] - s["H_shuf_dfa"]
    row.update(hlink_columns("tok", stats.get("hlink_tok")))
    summ = [s for s in result.get("h71_summ", []) if str(s.get("direction", "")).startswith("random")]
    if summ:
        row["h71_mean_alpha_hat"] = float(np.nanmean([s.get("alpha_mean", np.nan) for s in summ]))
        row["h71_slope"] = float(np.nanmean([s.get("b", np.nan) for s in summ]))
        row["h71_verdicts"] = ";".join(str(s.get("verdict", "")) for s in summ)
    row.update(extra)
    if "eval_loss" in row and "oracle_floor" in row:
        row["loss_gap_oracle"] = row["eval_loss"] - row["oracle_floor"]
    return row


# ----------------------------------------------------------------------------- the factorial test
def codes(scores: np.ndarray) -> np.ndarray:
    """Ordinal scores to [-1/2, 1/2] (two levels: -1/2, +1/2)."""
    s = np.asarray(scores, float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) - 0.5 if hi > lo else np.zeros_like(s)


def design(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """[intercept, seed blocks (all but the first), n, j, n j] and the block labels."""
    n = codes(df["noah_score"].to_numpy())
    j = codes(df["joseph_score"].to_numpy())
    blocks = df["seed"].to_numpy()
    cols, names = [np.ones(len(df))], ["intercept"]
    for b in sorted(set(blocks))[1:]:
        cols.append((blocks == b).astype(float))
        names.append(f"seed{b}")
    cols += [n, j, n * j]
    names += ["N", "J", "NJ"]
    return np.column_stack(cols), names, blocks


def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    s2 = float(resid @ resid) / df if df > 0 else np.nan
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * s2, 0, None))
    return beta, se, df, resid


def _t(X, y, k):
    beta, se, _, _ = _ols(X, y)
    return beta[k] / se[k] if se[k] > 0 else (np.inf if beta[k] != 0 else 0.0)


def within_block_perm(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(blocks))
    out = idx.copy()
    for b in np.unique(blocks):
        m = idx[blocks == b]
        out[m] = rng.permutation(m)
    return out


def term_test(df: pd.DataFrame, col: str, term: str, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """Freedman-Lane permutation test (two-sided, t statistic, residuals permuted within seed blocks) and the OLS
    t-test of one term, with the estimate, its 95% interval and the effect detectable with 80% power at the
    family's first Holm threshold."""
    d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
    y = pd.to_numeric(d[col]).to_numpy(float)
    out = {"outcome": col, "term": term, "n_runs": int(len(d))}
    if len(d) < 6 or d["noah_score"].nunique() < 2 or d["joseph_score"].nunique() < 2:
        return {**out, "note": "not testable (too few runs or levels)"}
    X, names, blocks = design(d)
    k = names.index(term)
    beta, se, dof, _ = _ols(X, y)
    if dof <= 0 or not np.isfinite(se[k]):
        return {**out, "note": "not testable (no residual degrees of freedom)"}
    t_obs = beta[k] / se[k] if se[k] > 0 else np.inf
    Xr = np.delete(X, k, axis=1)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fit_r = Xr @ br
    e_r = y - fit_r
    ge = 0
    for _ in range(n_perm):
        ys = fit_r + e_r[within_block_perm(blocks, rng)]
        ge += abs(_t(X, ys, k)) >= abs(t_obs) - 1e-12
    p_perm = (1 + ge) / (n_perm + 1)
    p_t = float(2 * sstats.t.sf(abs(t_obs), dof))
    tq = float(sstats.t.ppf(0.975, dof))
    first = LEVEL / (2 * len(FAMILY))
    mde = (float(sstats.t.ppf(1 - first / 2, dof)) + float(sstats.t.ppf(0.80, dof))) * se[k]
    return {**out, "estimate": float(beta[k]), "se": float(se[k]), "ci_lo": float(beta[k] - tq * se[k]),
            "ci_hi": float(beta[k] + tq * se[k]), "t": float(t_obs), "df": int(dof), "p_perm": float(p_perm),
            "p_t": p_t, "mde_80": float(mde), "n_perm": int(n_perm)}


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def family_tests(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = N_PERM) -> pd.DataFrame:
    """The 14 confirmatory tests on the rows given (the 14M corners at the final checkpoint), Holm-adjusted, with
    the reading of each."""
    rows = []
    for name, col, direction, main, sign in FAMILY:
        for term in (main, "NJ"):
            if col not in df.columns:
                rows.append({"outcome": col, "term": term, "name": name, "direction": direction,
                             "note": "not measured"})
                continue
            r = term_test(df, col, term, rng, n_perm)
            r.update(name=name, direction=direction, predicted_sign=(sign if term == main else 0))
            if name in TOKEN_OUTCOMES and REFEREE in df.columns:
                v = term_test(df, REFEREE, term, rng, n_perm)
                r.update(variant_estimate=v.get("estimate"), variant_p_perm=v.get("p_perm"))
            rows.append(r)
    out = pd.DataFrame(rows)
    ok = out["p_perm"].notna() if "p_perm" in out else pd.Series(False, index=out.index)
    out["p_holm"] = np.nan
    if ok.any():
        # untestable tests keep their place in the family (p = 1), so the correction never gets weaker
        p = out["p_perm"].where(ok, 1.0).tolist()
        out["p_holm"] = holm(p)
    out["reading"] = [reading(r) for r in out.to_dict("records")]
    return out


def reading(r: dict) -> str:
    if r.get("note"):
        return r["note"]
    if not (r.get("p_holm", 1.0) < LEVEL):
        return (f"no link detected (estimate {r['estimate']:+.3g}, 95% CI {r['ci_lo']:+.3g} .. {r['ci_hi']:+.3g}; "
                f"detectable with 80% power: |effect| >= {r['mde_80']:.3g})")
    sgn = int(np.sign(r["estimate"]))
    if r.get("name") in TOKEN_OUTCOMES and r["term"] in ("N", "NJ") and sgn < 0:
        ve, vp = r.get("variant_estimate"), r.get("variant_p_perm")
        if not (ve is not None and vp is not None and ve < 0 and vp < LEVEL):
            return "dilution by the rank noise (not a link: the variant part does not show it)"
        if r["term"] == "NJ":
            return "link (interaction; also in the variant part)"
        return "link, opposite to the prediction (also in the variant part)"
    if r["term"] == "NJ" or not r.get("predicted_sign"):
        return "link (interaction)"
    return "link as predicted" if sgn == r["predicted_sign"] else "link, opposite to the prediction"


def verdict(tests: pd.DataFrame) -> str:
    links = tests[tests["reading"].str.startswith("link")]
    if links.empty:
        return "No Noah-Joseph link detected in either direction (14 tests, Holm at 0.05)."
    parts = [f"{r['direction']}: {r['name']} {r['term']} ({r['reading']}, Holm p {r['p_holm']:.3g})"
             for r in links.to_dict("records")]
    return "Link(s): " + "; ".join(parts)


# ----------------------------------------------------------------------------- manipulation checks
def manipulation_checks(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = N_PERM) -> pd.DataFrame:
    """Reported beside the family, not corrected: the streams passed; the models learnt the streams (eval loss
    within 0.15 nats of the oracle floor and below the topic-blind floor by at least half the topic information);
    the own effects: Joseph on the variant part's memory (positive control of the memory measures) and Noah on the
    tail-sensitive quantities."""
    out = []
    if df.empty:
        return pd.DataFrame([{"check": "no runs in the confirmatory set", "value": False, "detail": ""}])
    if "stream_ok" in df:
        out.append({"check": "every stream passed its checks", "value": bool(df["stream_ok"].all()),
                    "n": int(len(df))})
    if {"eval_loss", "oracle_floor", "topic_blind_floor", "topic_information"} <= set(df.columns):
        gap = df["eval_loss"] - df["oracle_floor"]
        used = df["topic_blind_floor"] - df["eval_loss"] >= 0.5 * df["topic_information"]
        out.append({"check": "eval loss within 0.15 nats of the oracle floor", "value": bool((gap <= 0.15).all()),
                    "detail": f"gap {gap.min():.3f} .. {gap.max():.3f} nats"})
        out.append({"check": "the models use the topic (loss below the topic-blind floor by >= half the topic "
                             "information)", "value": bool(used.all()), "detail": f"{int(used.sum())}/{len(df)}"})
    for col, term, what in (("variant_Htok_excess", "J", "Joseph -> memory of the variant part (positive control)"),
                            ("Htok_excess", "J", "Joseph -> H_tok excess"),
                            ("tok_main_Habs_excess", "J", "Joseph -> magnitude memory of the token surprisals"),
                            ("alpha_esd", "N", "Noah -> alpha_ESD"), ("K", "N", "Noah -> K"),
                            ("alpha", "N", "Noah -> alpha")):
        if col in df.columns:
            r = term_test(df, col, term, rng, n_perm)
            out.append({"check": what, "value": (r.get("p_perm", 1.0) < LEVEL),
                        "detail": (f"estimate {r['estimate']:+.3g}, p {r['p_perm']:.3g}" if "estimate" in r
                                   else r.get("note"))})
    return pd.DataFrame(out)


def to_markdown(tests: pd.DataFrame, checks: pd.DataFrame, header: str) -> str:
    lines = [f"# P6 factorial Noah-Joseph test\n", header, "", f"**{verdict(tests)}**", "",
             "| direction | outcome | term | estimate | 95% CI | p (perm) | p (t) | Holm p | MDE 80% | reading |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tests.to_dict("records"):
        if r.get("note") and "estimate" not in r or not np.isfinite(r.get("estimate", np.nan)):
            lines.append(f"| {r.get('direction')} | {r.get('name')} | {r['term']} | | | | | | | {r.get('reading')} |")
            continue
        lines.append(f"| {r['direction']} | {r['name']} | {r['term']} | {r['estimate']:+.4g} | "
                     f"{r['ci_lo']:+.3g} .. {r['ci_hi']:+.3g} | {r['p_perm']:.4f} | {r['p_t']:.4f} | "
                     f"{r['p_holm']:.4f} | {r['mde_80']:.3g} | {r['reading']} |")
    lines += ["", "## Manipulation checks (reported, not corrected)", "", "| check | passed | detail |", "|---|---|---|"]
    for r in checks.to_dict("records"):
        lines.append(f"| {r['check']} | {r['value']} | {r.get('detail', '')} |")
    return "\n".join(lines) + "\n"


def secondary_tests(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = 999) -> pd.DataFrame:
    """Every outcome of FAMILY and SECONDARY, each term, per size and stage (all levels, ordinal scores):
    exploratory, unadjusted."""
    rows = []
    cols = [c for _, c, *_ in FAMILY] + SECONDARY
    for (size, stage), d in df.groupby(["size", "stage"]):
        for col in cols:
            if col not in d.columns or d[col].dtype == object:
                continue
            for term in ("N", "J", "NJ"):
                r = term_test(d, col, term, rng, n_perm)
                rows.append({"size": size, "stage": stage, **r})
    return pd.DataFrame(rows)


def json_load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
