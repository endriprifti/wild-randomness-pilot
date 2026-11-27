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


def design(df: pd.DataFrame, covariates=(), terms=("N", "J", "NJ")) -> tuple[np.ndarray, list[str], np.ndarray]:
    """[intercept, seed blocks (all but the first), n, j, n j] and the block labels. P6c (2026-11-27) adds two
    options for its exploratory list, neither used by P6 or P6b: `terms`, a subset of the factorial terms (the simple
    effect of Joseph within one Noah level: ("J",)), and `covariates`, columns of df added after the terms."""
    n = codes(df["noah_score"].to_numpy())
    j = codes(df["joseph_score"].to_numpy())
    blocks = df["seed"].to_numpy()
    cols, names = [np.ones(len(df))], ["intercept"]
    for b in sorted(set(blocks))[1:]:
        cols.append((blocks == b).astype(float))
        names.append(f"seed{b}")
    eff = {"N": n, "J": j, "NJ": n * j}
    cols += [eff[t] for t in terms]
    names += list(terms)
    for c in covariates:
        cols.append(pd.to_numeric(df[c]).to_numpy(float))
        names.append(str(c))
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


REL_TOL = 1e-10      # residuals this small against the outcome's magnitude are rounding, not variance


def no_variation(y: np.ndarray, resid: np.ndarray) -> str | None:
    """The note of an outcome that cannot be tested because nothing varies: the same value in every run, or a fit
    with no residual variance left; None otherwise."""
    tol = REL_TOL * float(np.max(np.abs(y)))
    if float(np.ptp(y)) <= tol:
        return "not testable (the same value in every run)"
    if float(np.sqrt(np.mean(resid ** 2))) <= tol:
        return "not testable (no residual variance: the model fits every run exactly)"
    return None


def within_block_perm(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(blocks))
    out = idx.copy()
    for b in np.unique(blocks):
        m = idx[blocks == b]
        out[m] = rng.permutation(m)
    return out


def term_test(df: pd.DataFrame, col: str, term: str, rng: np.random.Generator, n_perm: int = N_PERM,
              m_family: int | None = None, covariates=(), terms=("N", "J", "NJ")) -> dict:
    """Freedman-Lane permutation test (two-sided, t statistic, residuals permuted within seed blocks) and the OLS
    t-test of one term, with the estimate, its 95% interval and the effect detectable with 80% power at the
    family's first Holm threshold (P6's 14 tests; P6c passes m_family = 4, its family's size). covariates, terms:
    design()'s options (P6c's exploratory list only); the defaults are P6's model."""
    d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
    for c in covariates:
        d = d[np.isfinite(pd.to_numeric(d[c], errors="coerce"))]
    y = pd.to_numeric(d[col]).to_numpy(float)
    out = {"outcome": col, "term": term, "n_runs": int(len(d))}
    factors = [f for f, used in (("noah_score", {"N", "NJ"} & set(terms)), ("joseph_score", {"J", "NJ"} & set(terms)))
               if used]
    if len(d) < 6 or any(d[f].nunique() < 2 for f in factors):
        return {**out, "note": "not testable (too few runs or levels)"}
    X, names, blocks = design(d, covariates, terms)
    k = names.index(term)
    beta, se, dof, resid = _ols(X, y)
    if dof <= 0 or not np.isfinite(se[k]):
        return {**out, "note": "not testable (no residual degrees of freedom)"}
    note = no_variation(y, resid)
    if note:
        # (2026-11-21) an outcome with the same value in every run (or fitted exactly by the model) has no residual
        # variance: t is 0/0, and the permutation p and t-test p it gave (0.001, 0) meant nothing. The permutations
        # are still drawn, so every later test in the same random stream keeps the numbers it had before
        for _ in range(n_perm):
            within_block_perm(blocks, rng)
        return {**out, "note": note}
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
    first = LEVEL / (m_family or 2 * len(FAMILY))
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
    note = r.get("note")
    if isinstance(note, str) and note:
        # (2026-11-21) a string only: in a table where some test has a note, the others carry NaN in that column,
        # which is truthy, and their readings came out blank (NaN). A family without notes reads as before
        return note
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


def to_markdown(tests: pd.DataFrame, checks: pd.DataFrame, header: str, title: str = "P6",
                verdict_text: str | None = None) -> str:
    lines = [f"# {title} factorial Noah-Joseph test\n", header, "", f"**{verdict_text or verdict(tests)}**", "",
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


# ----------------------------------------------------------------------------- P6b (2026-11-21)
# P6b, a replication of the small design with new seeds (Endri, 2026-11-21, after P6's small design was analysed and
# before any P6b run; README "P6b"). Its analysis is P6's, on the complete seeds only, plus one rule fixed in advance:
# if the positive control (Joseph's effect on the memory of the variant part, at the final checkpoint, the
# manipulation check below) does not reach p < 0.05, the Noah -> Joseph tests that would read "no link detected" read
# "uninformative" instead (the memory measures did not show the effect that is known to be there). A test that
# passes Holm keeps its reading, marked "positive control missed". The Joseph -> Noah tests are read as usual. P6's
# own analysis does not use any of this.
POSITIVE_CONTROL = "Joseph -> memory of the variant part (positive control)"     # a check of manipulation_checks
NOAH_TO_JOSEPH = "Noah->Joseph"


def positive_control(checks: pd.DataFrame) -> dict:
    """{"passed": bool, "detail": str} of the positive control in manipulation_checks' output; missing or not testable
    counts as not passed."""
    if checks.empty or "check" not in checks:
        return {"passed": False, "detail": "not in the manipulation checks"}
    r = checks[checks["check"] == POSITIVE_CONTROL]
    if r.empty:
        return {"passed": False, "detail": "not measured"}
    r = r.iloc[0]
    detail = r.get("detail")
    return {"passed": bool(r["value"]), "detail": str(detail) if isinstance(detail, str) and detail else ""}


def positive_control_rule(tests: pd.DataFrame, pc: dict) -> pd.DataFrame:
    """P6b's readings: family_tests' output with the rule applied (a copy; p-values and Holm unchanged)."""
    out = tests.copy()
    out["positive_control_passed"] = bool(pc["passed"])
    if pc["passed"]:
        return out
    why = f"the positive control missed: {pc['detail']}" if pc["detail"] else "the positive control missed"
    new = []
    for r in out.to_dict("records"):
        reading = str(r["reading"])
        if r.get("direction") == NOAH_TO_JOSEPH:
            if reading.startswith("no link detected"):
                reading = f"uninformative ({why}) " + reading[len("no link detected "):]
            elif reading.startswith(("link", "dilution")):
                reading = f"{reading}; positive control missed"
        new.append(reading)
    out["reading"] = new
    return out


def verdict_p6b(tests: pd.DataFrame, pc: dict) -> str:
    """P6b's verdict line: P6's when the positive control passed; otherwise the Joseph -> Noah tests as usual and the
    Noah -> Joseph tests uninformative unless one of them passed."""
    if pc["passed"]:
        return verdict(tests) + " The positive control passed."
    links = tests[tests["reading"].str.startswith("link")]
    jn = len(tests[tests["direction"] != NOAH_TO_JOSEPH])
    nj = len(tests) - jn
    why = f"the positive control missed ({pc['detail']})" if pc["detail"] else "the positive control missed"
    parts = [f"{r['direction']}: {r['name']} {r['term']} ({r['reading']}, Holm p {r['p_holm']:.3g})"
             for r in links.to_dict("records")]
    head = ("Link(s): " + "; ".join(parts) + ". ") if parts else ""
    jn_links = links[links["direction"] != NOAH_TO_JOSEPH]
    jn_text = "" if len(jn_links) else f"No Joseph -> Noah link detected ({jn} tests). "
    return (f"{head}{jn_text}The Noah -> Joseph tests ({nj}) that detected no link are uninformative: {why}. "
            f"({len(tests)} tests, Holm at 0.05.)")


def complete_seeds(units: list[dict], design: dict) -> tuple[list[int], list[int]]:
    """Seeds of `design` with a finished run in every cell of the confirmatory set (sizes x Noah x Joseph of the
    design; P6b: the four 14M corners), and the seeds with some but not all of them. units: finished runs, each
    with key, size, noah, joseph, seed."""
    cells = {(s, n, j) for s in design["sizes"] for n in design["noah"] for j in design["joseph"]}
    have: dict[int, set] = {}
    for u in units:
        have.setdefault(int(u["seed"]), set()).add((u["size"], u["noah"], u["joseph"]))
    complete = sorted(s for s, c in have.items() if cells <= c)
    partial = sorted(s for s, c in have.items() if not cells <= c)
    return complete, partial


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
    out = pd.DataFrame(rows)
    for c in ("df", "n_perm"):         # whole numbers in the csv beside the rows that are not testable
        if c in out:
            out[c] = out[c].astype("Int64")
    return out


# ----------------------------------------------------------------------------- P6c (2026-11-27)
# P6c (Endri, 2026-11-27, after P6b was analysed and before any P6c run; README "P6c"): a confirmatory test of
# Joseph's effect on alpha_ESD, which P6 and P6b showed outside their family (P6 -0.030, p 0.040; P6b -0.023, p 0.008,
# both at the final checkpoint), and a replication on new seeds of P6b's one confirmatory link (Noah -> width of the
# gradient series beyond its IAAFT surrogates, opposite to P6's prediction). Four tests at the 14M corners, final
# checkpoint, with P6's model and Freedman-Lane permutations (9,999), Holm over the four at 0.05. Reported beside P6
# and P6b, never pooled with them.
# (test, outcome, column, term, predicted sign, gate, what a pass with the predicted sign confirms)
FAMILY_P6C = [
    ("C1", "alpha_ESD (median over the 24 matrices)", "alpha_esd", "J", -1, "joseph",
     "Joseph lowers alpha_ESD"),
    ("C2", "alpha_ESD of the attention input projections (median over the query_key_value matrices)",
     "alpha_esd_qkv", "J", -1, "joseph", "Joseph lowers alpha_ESD of the attention input projections"),
    ("C3", "alpha_ESD (median over the 24 matrices)", "alpha_esd", "NJ", +1, "joseph",
     "the Zipf law weakens Joseph's effect on alpha_ESD"),
    ("C4", "width of the gradient series beyond its IAAFT surrogates (main rows)", "grad_main_dh_excess", "N", -1,
     "noah", "P6b's link replicates (Noah lowers the width of the gradient series beyond its IAAFT surrogates)"),
]
# the gates, fixed with the family: a manipulation check of manipulation_checks at the final checkpoint, passed if its
# permutation p < 0.05. If a test's gate misses, the test reads "uninformative" unless it passes Holm; a test that
# passes keeps its reading, marked "<gate> missed" (P6b's rule for a passing test under a missed positive control)
P6C_GATES = {"joseph": ("Joseph gate", POSITIVE_CONTROL),
             "noah": ("Noah gate", "Noah -> K")}
# C4 is reported with its two parts beside it (Noah's effect on each, as in P6b's brief): the observed width and the
# mean width of its IAAFT surrogates (grad_main_dh_excess = observed - IAAFT mean)
C4_PARTS = [("observed", "grad_main_dh"), ("iaaft_mean", "grad_main_dh_iaaft_mean")]
# alpha_ESD by matrix type (GPT-NeoX names; P1's 24 matrices of the 14M model: six layers of four)
P6C_TYPES = {"qkv": "attention.query_key_value", "attn_out": "attention.dense", "mlp_in": "mlp.dense_h_to_4h",
             "mlp_out": "mlp.dense_4h_to_h"}
P6C_EARLY_STAGES = [0.01, 0.1, 0.5]


def esd_columns(esd_stats: dict | None) -> dict:
    """P6c's alpha_ESD columns of one checkpoint from its stats/esd.json (the fits P1's task made; no new
    measurement): alpha_esd_<type> (median over that type's matrices; alpha_esd_qkv is C2's outcome), the number of
    matrices of each type with a finite fit, alpha_esd_layer<i> (median over layer i's matrices) and the medians over
    all matrices of the fits' cutoff (esd_xmin), tail count (esd_n_tail) and KS distance (esd_ks)."""
    rows = ((esd_stats or {}).get("result") or {}).get("rows") or []
    out: dict = {}

    def med(vals):
        v = [float(x) for x in vals if isfinite(x)]
        return float(np.median(v)) if v else np.nan
    for typ, pat in P6C_TYPES.items():
        a = [r.get("alpha_esd") for r in rows if pat in str(r.get("layer", ""))]
        out[f"alpha_esd_{typ}"] = med(a)
        out[f"esd_n_{typ}"] = int(sum(isfinite(x) for x in a))
    layers: dict[int, list] = {}
    for r in rows:
        w = str(r.get("layer", "")).split(".")
        if "layers" in w and w.index("layers") + 1 < len(w) and w[w.index("layers") + 1].isdigit():
            layers.setdefault(int(w[w.index("layers") + 1]), []).append(r.get("alpha_esd"))
    for i in sorted(layers):
        out[f"alpha_esd_layer{i}"] = med(layers[i])
    for k in ("esd_xmin", "esd_n_tail", "esd_ks"):
        out[f"{k}_median"] = med(r.get(k) for r in rows)
    return out


def c4_part_columns(grad_stats: dict | None) -> dict:
    """The two parts of C4's outcome from the gradient series' H_link main row (hlink/stats/hlink_grad.json): the
    observed width (grad_main_dh) and the mean width of its IAAFT surrogates (grad_main_dh_iaaft_mean)."""
    res = (grad_stats or {}).get("result") or {}
    for r in res.get("rows", []):
        if r.get("scales") == "main":
            return {"grad_main_dh": r["delta_h"], "grad_main_dh_iaaft_mean": r["delta_h_iaaft_mean"]}
    return {}


def named_check(checks: pd.DataFrame, name: str) -> dict:
    """{"passed", "detail"} of one manipulation check (positive_control's logic, for any check); missing or not
    testable counts as not passed."""
    if checks.empty or "check" not in checks:
        return {"passed": False, "detail": "not in the manipulation checks"}
    r = checks[checks["check"] == name]
    if r.empty:
        return {"passed": False, "detail": "not measured"}
    r = r.iloc[0]
    detail = r.get("detail")
    return {"passed": bool(r["value"]), "detail": str(detail) if isinstance(detail, str) and detail else ""}


def p6c_gates(checks: pd.DataFrame) -> dict:
    return {g: {**named_check(checks, check), "label": label, "check": check}
            for g, (label, check) in P6C_GATES.items()}


def family_p6c(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = N_PERM) -> pd.DataFrame:
    """P6c's four tests on the rows given (the 14M corners at the final checkpoint), Holm over the four, with C4's
    two parts beside it. The readings need the gates: p6c_readings."""
    m = len(FAMILY_P6C)
    rows = []
    for test, name, col, term, sign, gate, effect in FAMILY_P6C:
        base = {"test": test, "name": name, "gate": gate, "predicted_sign": sign, "confirms": effect}
        if col not in df.columns:
            rows.append({"outcome": col, "term": term, **base, "note": "not measured"})
            continue
        r = {**term_test(df, col, term, rng, n_perm, m_family=m), **base}
        if test == "C4":
            for part, pc in C4_PARTS:
                v = term_test(df, pc, term, rng, n_perm, m_family=m) if pc in df.columns else {}
                r.update({f"{part}_estimate": v.get("estimate"), f"{part}_ci_lo": v.get("ci_lo"),
                          f"{part}_ci_hi": v.get("ci_hi"), f"{part}_p_perm": v.get("p_perm")})
        rows.append(r)
    out = pd.DataFrame(rows)
    ok = out["p_perm"].notna() if "p_perm" in out else pd.Series(False, index=out.index)
    out["p_holm"] = np.nan
    if ok.any():
        out["p_holm"] = holm(out["p_perm"].where(ok, 1.0).tolist())     # untestable tests keep their place (p = 1)
    return out


def reading_p6c(r: dict, gate: dict) -> str:
    """P6c's readings, fixed 2026-11-27: "confirmed: <effect>" (a pass with the predicted sign), "effect opposite to
    the prediction" (a pass with the other sign), "no effect detected" (no pass, gate passed) or "uninformative" (no
    pass, gate missed), the last two with the estimate, its 95% interval and the effect detectable with 80% power."""
    note = r.get("note")
    if isinstance(note, str) and note:
        return note
    ci = (f"estimate {r['estimate']:+.3g}, 95% CI {r['ci_lo']:+.3g} .. {r['ci_hi']:+.3g}; detectable with 80% "
          f"power: |effect| >= {r['mde_80']:.3g}")
    if r.get("p_holm", 1.0) < LEVEL:
        sgn = int(np.sign(r["estimate"]))
        text = f"confirmed: {r['confirms']}" if sgn == int(r["predicted_sign"]) else "effect opposite to the prediction"
        return text if gate["passed"] else f"{text}; {gate['label']} missed"
    if gate["passed"]:
        return f"no effect detected ({ci})"
    why = f"the {gate['label']} missed: {gate['detail']}" if gate["detail"] else f"the {gate['label']} missed"
    return f"uninformative ({why}) ({ci})"


def p6c_readings(tests: pd.DataFrame, gates: dict) -> pd.DataFrame:
    out = tests.copy()
    out["gate_passed"] = [bool(gates[g]["passed"]) for g in out["gate"]]
    out["reading"] = [reading_p6c(r, gates[r["gate"]]) for r in out.to_dict("records")]
    return out


def verdict_p6c(tests: pd.DataFrame, gates: dict) -> str:
    parts = []
    for r in tests.to_dict("records"):
        head = str(r["reading"]).split(" (")[0]
        hp = f", Holm p {r['p_holm']:.3g}" if isfinite(r.get("p_perm")) else ""
        parts.append(f"{r['test']} {head}{hp}")
    g = "; ".join(f"{v['label']} {'passed' if v['passed'] else 'missed'}" for v in gates.values())
    return "; ".join(parts) + f". {g}. ({len(tests)} tests, Holm at 0.05.)"


def exploratory_p6c(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = 999, size: str = SIZE_PRIMARY,
                    final: float = STAGE_PRIMARY) -> pd.DataFrame:
    """P6c's exploratory list (proposal section 5), uncorrected, on the corners of `size`: the simple effects of
    Joseph within each Noah level (C1's and C2's outcomes); C1 to C3 at 1, 10 and 50% of training; alpha_ESD by
    matrix type and by layer (N, J, NJ); C1 to C3 with the loss gap to the oracle as a covariate; the fits' cutoff,
    tail count and KS distance (N, J, NJ); C4's outcome and its two parts (N, J, NJ). All at the final checkpoint
    unless a stage is named. The shuffled and phase-randomized parts of C4 are the step-1 decomposition script's."""
    m = len(FAMILY_P6C)
    d0 = df[df["size"] == size]
    fin = d0[np.isclose(d0["stage"], final)]
    rows = []

    def add(what, stage, d, col, term, **kw):
        if col not in d.columns:
            return
        rows.append({"what": what, "stage": stage, **term_test(d, col, term, rng, n_perm, m_family=m, **kw)})
    for col in ("alpha_esd", "alpha_esd_qkv"):
        for noah in sorted(fin["noah"].unique()):
            add(f"simple effect of Joseph at Noah = {noah}", final, fin[fin["noah"] == noah], col, "J", terms=("J",))
    for stage in P6C_EARLY_STAGES:
        ds = d0[np.isclose(d0["stage"], stage)]
        for test, _, col, term, *_ in FAMILY_P6C[:3] if len(ds) else []:
            add(f"{test} at {stage:.0%} of training", stage, ds, col, term)
    cols = [c for c in fin.columns if c.startswith("alpha_esd_") and not c.startswith("alpha_esd_layer")]
    cols += sorted((c for c in fin.columns if c.startswith("alpha_esd_layer")), key=lambda c: int(c[15:]))
    for col in cols:
        for term in ("N", "J", "NJ"):
            add("alpha_ESD by matrix type" if "layer" not in col else "alpha_ESD by layer", final, fin, col, term)
    for test, _, col, term, *_ in FAMILY_P6C[:3]:
        if "loss_gap_oracle" in fin.columns:
            add(f"{test} with the loss gap to the oracle as a covariate", final, fin, col, term,
                covariates=("loss_gap_oracle",))
    for col in ("esd_xmin_median", "esd_n_tail_median", "esd_ks_median"):
        for term in ("N", "J", "NJ"):
            add("alpha_ESD fits: cutoff, tail count, KS distance", final, fin, col, term)
    for col in ("grad_main_dh_excess", "grad_main_dh", "grad_main_dh_iaaft_mean"):
        for term in ("N", "J", "NJ"):
            add("C4's outcome and its parts (observed width, IAAFT mean)", final, fin, col, term)
    out = pd.DataFrame(rows)
    for c in ("df", "n_perm"):
        if c in out:
            out[c] = out[c].astype("Int64")
    return out


def to_markdown_p6c(tests: pd.DataFrame, checks: pd.DataFrame, header: str, verdict_text: str) -> str:
    lines = ["# P6c: a confirmatory test of Joseph's effect on alpha_ESD, with a replication of P6b's one link\n",
             header, "", f"**{verdict_text}**", "",
             "| test | outcome | term | predicted | estimate | 95% CI | p (perm) | p (t) | Holm p | MDE 80% "
             "| reading |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tests.to_dict("records"):
        sign = "+" if r["predicted_sign"] > 0 else "-"
        if not isfinite(r.get("estimate")):
            lines.append(f"| {r['test']} | {r['name']} | {r['term']} | {sign} | | | | | | | {r['reading']} |")
            continue
        lines.append(f"| {r['test']} | {r['name']} | {r['term']} | {sign} | {r['estimate']:+.4g} | "
                     f"{r['ci_lo']:+.3g} .. {r['ci_hi']:+.3g} | {r['p_perm']:.4f} | {r['p_t']:.4f} | "
                     f"{r['p_holm']:.4f} | {r['mde_80']:.3g} | {r['reading']} |")
    c4 = tests[tests["test"] == "C4"]
    if len(c4) and isfinite(c4.iloc[0].get("observed_estimate")):
        r = c4.iloc[0]
        lines += ["", "C4's two parts (Noah's effect; not tests of the family): "
                  + "; ".join(f"{lab} {r[f'{p}_estimate']:+.4g} (95% CI {r[f'{p}_ci_lo']:+.3g} .. "
                              f"{r[f'{p}_ci_hi']:+.3g}, p {r[f'{p}_p_perm']:.4f})"
                              for p, lab in (("observed", "observed width"), ("iaaft_mean", "IAAFT mean"))
                              if isfinite(r.get(f"{p}_estimate")))]
    lines += ["", "## Manipulation checks and gates (reported, not corrected)", "", "| check | passed | detail |",
              "|---|---|---|"]
    for r in checks.to_dict("records"):
        lines.append(f"| {r['check']} | {r['value']} | {r.get('detail', '')} |")
    return "\n".join(lines) + "\n"


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
