"""P7 analysis (fixed 2026-11-27, before any P7 run): the confirmatory family of P7 (README section "P7").

numpy, scipy and pandas only (CPU; no torch). Reads
  results/p7_transformer_sgd/                 phase0.json, runs.csv (p7_transformer_sgd.py), stopped.json if any
  results/h_link_p7_transformer_sgd/p4_grad.csv
                                              h_link.py --p1 --p4 results/p7_transformer_sgd (P4's H_link code)
and, where they exist (exploratory only):
  results/surrogate_decomposition/P7_units.csv   the width decomposition (surrogate_decomposition.py --design P7)
  results/p4_transformer/runs.csv, results/h_link/p4_grad.csv (folder p4_transformer)   the AdamW twins in P4

The confirmatory family (fixed now): the four plain arms of each seed (unclipped, fixed thresholds at targets 1, 10
and 50%), scored -1.5, -0.5, +0.5, +1.5; per outcome, least squares with seed blocks and the score (the trend), tested
two-sided by permuting the four arm labels within each seed (9,999 draws; statistic: the slope); Holm over the two
tests at 0.05. Outcomes from h_link.py's main rows:
  T1  H(|x|) by DFA                                  H_abs_dfa                           predicted: rises with clipping
  T2  width beyond the IAAFT surrogates              delta_h - delta_h_iaaft_mean        predicted: rises with clipping
Gates (fixed now):
  - Phase 0 passed (phase0.json status "chosen"); a Phase 1 run that diverged is reported and left out.
  - clipping manipulation: the realized clipped fractions are ordered (means over seeds: unclipped < 1% < 10% < 50%),
    the 10% arm clips at least 3% of steps and the 50% arm 25 to 95% (amended 2026-11-27, before any P7 run, from
    25 to 75%: under SGD clipping slows the steps, so the realized fraction exceeds the target); if not, the tests
    that do not pass read "uninformative (clipping manipulation missed)".
  - width beyond the surrogates (T2): a T2 that passes with a rise reads as width beyond the surrogates only if the
    observed delta_h also rises (same model, slope positive, 95% t-interval above 0); otherwise "the surrogates'
    baseline moved, not the width".
Readings (fixed now): a rise in T1 or T2 (T2 with its gate) -> "ResNet-18's clipping pattern appears in the
transformer under SGD" ("in memory and width" for both); a test that passes with a fall -> "clipping effect opposite to
ResNet-18's"; neither passes -> "no clipping effect detected under SGD", with the intervals and the detectable effects
(the optimizer swap alone does not bring back ResNet-18's pattern at this power). Pass counts of the full H_link rule
are reported by arm, not tested.

Choices of ours, stated before any run: only seeds with all four plain arms saved enter the tests (a STOP or a crash
leaves whole seeds out, not single arms); a diverged run or a non-finite outcome is left out of its seed; the rises
and intervals are also given over the whole range of scores (3 x the slope); the detectable rise (80% power, two-sided
t at 0.05, as the proposal's power numbers) and the same at Holm's first threshold (0.025) are both reported.

Outputs (results/p7_transformer_sgd/analysis/[smoke/]): tests.csv (T1, T2 and the delta_h gate), gates.csv,
arms.csv (per arm: runs, divergences, realized clipped fraction, rate, outcomes, H_link pass counts), exploratory.csv
(uncorrected: each arm minus unclipped, matched against plain 10%, the trend on the detrended and rescaled rows, the
width decomposition, alpha-hat and zeta-hat, the losses), twins.csv (each SGD run against its AdamW twin in P4),
summary.json and summary.md (the reading).

Commands (in the pilot folder inside the venv; the queue's job p7_analysis runs the first):
  python p7_stats.py
  python p7_stats.py --smoke --runs <smoke folder> --hlink <its h_link p4_grad.csv>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import config
from common import runtime
from common import units as units_mod

ROOT = Path(__file__).resolve().parent
PLAIN = ["0.0", "0.01", "0.1", "0.5"]           # the targets of the four plain arms, as config.py's score keys
TESTS = [("T1", "H(|x|) by DFA", "H_abs_dfa"),
         ("T2", "width beyond the IAAFT surrogates", "width_excess")]
GATE = ("dh_gate", "observed width (delta_h)", "delta_h")
PATTERN = "ResNet-18's clipping pattern appears in the transformer under SGD"
OPPOSITE = "clipping effect opposite to ResNet-18's"
BASELINE = "the surrogates' baseline moved, not the width"
NONE = "no clipping effect detected under SGD"
UNINFORMATIVE = "uninformative (clipping manipulation missed)"
NO_POWER = "the optimizer swap alone does not bring back ResNet-18's pattern at this power"
HLINK_OUT = ["H_abs_dfa", "width_excess", "delta_h", "delta_h_iaaft_mean", "delta_h_iaaft_q95", "p_iaaft",
             "H_abs_dfa_shuf_mean", "H_abs_dfa_phase_mean", "memory_pass", "width_pass", "pass"]
RUN_OUT = ["p_clip_realized", "lr", "lr_scale", "mean_update_len", "alpha", "zeta", "train_loss", "test_loss",
           "ood_loss", "gap", "final_train_loss_ema"]


# ----------------------------------------------------------------------------- arms and tables
def arm_label(rule, p, matched) -> str:
    """P4's tag without the model and seed: fixed0.1_matched."""
    return f"{rule}{'' if p is None or (isinstance(p, float) and math.isnan(p)) else p}" \
           f"{'_matched' if _bool(matched) else ''}"


def _bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x) and not (isinstance(x, float) and math.isnan(x))


def _num(x) -> float:
    """A float of a csv cell (True/False -> 1/0, anything else not a number -> nan)."""
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _key(p) -> str:
    return str(float(p))


def load_runs(runs_dir: Path) -> pd.DataFrame:
    f = runs_dir / "runs.csv"
    if not f.exists() or f.stat().st_size == 0:
        return pd.DataFrame(columns=["seed", "rule", "p_clip_target", "matched", "diverged"])
    d = pd.read_csv(f)
    d["matched"] = d["matched"].map(_bool)
    d["diverged"] = d["diverged"].map(_bool)
    d["arm"] = [arm_label(r, p, m) for r, p, m in zip(d["rule"], d["p_clip_target"], d["matched"])]
    d["tag"] = [f"transformer_s{int(s)}_{a}" for s, a in zip(d["seed"], d["arm"])]
    return d


def load_hlink(path: Path, folder: str | None = None) -> pd.DataFrame:
    """h_link.py's p4_grad.csv rows (all four scales), with width_excess = delta_h - delta_h_iaaft_mean."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["tag", "scales"])
    d = pd.read_csv(path)
    if folder is not None:
        d = d[d["folder"] == folder]
    for c in ("memory_pass", "width_pass", "pass", "matched", "diverged"):
        if c in d:
            d[c] = d[c].map(_bool)
    d["width_excess"] = d["delta_h"] - d["delta_h_iaaft_mean"]
    return d


def merged(runs: pd.DataFrame, hl: pd.DataFrame, scales: str = "main") -> pd.DataFrame:
    """One row per Phase 1 run: runs.csv's columns and h_link's columns on the rows `scales`."""
    h = hl[hl["scales"] == scales][["tag"] + [c for c in HLINK_OUT if c in hl.columns]]
    return runs.merge(h, on="tag", how="left")


def complete_seeds(runs: pd.DataFrame) -> list[int]:
    """Seeds with all four plain arms saved (diverged or not)."""
    out = []
    for s, g in runs.groupby("seed"):
        plain = g[(g["rule"] == "fixed") & (~g["matched"])]
        if set(PLAIN) <= {_key(p) for p in plain["p_clip_target"]}:
            out.append(int(s))
    return sorted(out)


def plain_rows(d: pd.DataFrame, seeds: list[int], scores: dict) -> pd.DataFrame:
    x = d[(d["rule"] == "fixed") & (~d["matched"]) & d["seed"].isin(seeds)].copy()
    x = x[x["p_clip_target"].map(_key).isin(PLAIN)]
    x["score"] = x["p_clip_target"].map(lambda p: float(scores[_key(p)]))
    return x


# ----------------------------------------------------------------------------- the trend test
def _centered(score: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    c = score.astype(float).copy()
    for b in np.unique(blocks):
        m = blocks == b
        c[m] -= c[m].mean()
    return c


def trend_fit(y: np.ndarray, score: np.ndarray, blocks: np.ndarray) -> dict:
    """Least squares y = seed block + b score + e: the slope, its standard error and residual degrees of freedom.
    With the blocks as fixed effects the slope is the within-seed one: sum(c y) / sum(c^2), c = score centred in its
    seed."""
    c = _centered(score, blocks)
    sxx = float(c @ c)
    if sxx <= 0:
        return {"slope": np.nan, "se": np.nan, "df": 0}
    b = float(c @ y) / sxx
    yc = y - np.array([y[blocks == k].mean() for k in blocks])
    resid = yc - b * c
    dof = len(y) - len(np.unique(blocks)) - 1
    s2 = float(resid @ resid) / dof if dof > 0 else np.nan
    return {"slope": b, "se": math.sqrt(s2 / sxx) if dof > 0 else np.nan, "df": int(dof)}


def perm_p(y: np.ndarray, score: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    """Two-sided permutation p of the slope: the arm labels (scores) permuted within each seed."""
    c = _centered(score, blocks)
    sxx = float(c @ c)
    obs = abs(float(c @ y) / sxx)
    idx = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    ge = 0
    perm = score.astype(float).copy()
    for _ in range(n_perm):
        for m in idx:
            perm[m] = score[m][rng.permutation(len(m))]
        cp = _centered(perm, blocks)
        ge += abs(float(cp @ y) / sxx) >= obs - 1e-12
    return (1 + ge) / (n_perm + 1)


def trend_test(d: pd.DataFrame, col: str, rng: np.random.Generator | None, n_perm: int, level: float,
               power: float, span: float, n_family: int) -> dict:
    """The trend of one outcome on the plain arms: slope per unit of score, its rise over the whole range of scores
    (span x slope), 95% t-interval, t-test p, permutation p (when rng is given) and the rise detectable with the given
    power at two-sided `level` and at Holm's first threshold (level / n_family)."""
    x = d[np.isfinite(pd.to_numeric(d[col], errors="coerce"))] if col in d else d.iloc[:0]
    y = pd.to_numeric(x[col]).to_numpy(float) if len(x) else np.array([])
    out = {"outcome": col, "n_runs": int(len(x)), "n_seeds": int(x["seed"].nunique()) if len(x) else 0}
    if len(x) < 4 or x["score"].nunique() < 2:
        return {**out, "note": "not testable (too few runs or arms)"}
    score, blocks = x["score"].to_numpy(float), x["seed"].to_numpy()
    f = trend_fit(y, score, blocks)
    if f["df"] <= 0 or not np.isfinite(f["se"]) or f["se"] <= 0:
        return {**out, "slope": f["slope"], "note": "not testable (no residual variance or degrees of freedom)"}
    tq = float(sstats.t.ppf(0.975, f["df"]))
    t = f["slope"] / f["se"]
    zp = float(sstats.t.ppf(power, f["df"]))
    res = {**out, "slope": f["slope"], "se": f["se"], "df": f["df"], "t": t,
           "ci_lo": f["slope"] - tq * f["se"], "ci_hi": f["slope"] + tq * f["se"],
           "rise": span * f["slope"], "rise_ci_lo": span * (f["slope"] - tq * f["se"]),
           "rise_ci_hi": span * (f["slope"] + tq * f["se"]),
           "p_t": float(2 * sstats.t.sf(abs(t), f["df"])),
           "mde_rise": span * f["se"] * (float(sstats.t.ppf(1 - level / 2, f["df"])) + zp),
           "mde_rise_holm1": span * f["se"] * (float(sstats.t.ppf(1 - level / n_family / 2, f["df"])) + zp)}
    if rng is not None:
        res["p_perm"] = perm_p(y, score, blocks, rng, n_perm)
        res["n_perm"] = int(n_perm)
    return res


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    m, run, adj = len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


# ----------------------------------------------------------------------------- gates and readings
def manipulation_gate(d: pd.DataFrame, rule: dict) -> dict:
    """Realized clipped fractions of the plain arms (means over the runs in the tests)."""
    means = {k: float(np.nanmean(d.loc[d["p_clip_target"].map(_key) == k, "p_clip_realized"]))
             if (d["p_clip_target"].map(_key) == k).any() else np.nan for k in PLAIN}
    v = [means[k] for k in PLAIN]
    ordered = all(np.isfinite(v)) and all(a < b for a, b in zip(v, v[1:]))
    lo, hi = rule["range_50"]
    ok10 = bool(np.isfinite(means["0.1"]) and means["0.1"] >= rule["min_10"])
    ok50 = bool(np.isfinite(means["0.5"]) and lo <= means["0.5"] <= hi)
    return {"means": means, "ordered": bool(ordered), "ok_10": ok10, "ok_50": ok50,
            "passed": bool(ordered and ok10 and ok50)}


def read_test(r: dict, level: float, manipulation_ok: bool, gate: dict | None) -> tuple[str, str]:
    """(direction, reading) of one test. direction: rise, fall, baseline, none or untestable."""
    if r.get("note"):
        return "untestable", r["note"]
    p = r.get("p_holm")
    if p is not None and np.isfinite(p) and p < level:
        if r["slope"] > 0:
            if r["test"] == "T2" and not (gate and gate.get("slope", 0) > 0 and gate.get("ci_lo", -1) > 0):
                return "baseline", BASELINE
            return "rise", "rise with clipping (ResNet-18's direction)"
        return "fall", OPPOSITE
    if not manipulation_ok:
        return "none", UNINFORMATIVE
    return "none", NONE


def overall_reading(tests: list[dict], manipulation_ok: bool) -> str:
    rises = [t["test"] for t in tests if t["direction"] == "rise"]
    parts = []
    if rises:
        parts.append(PATTERN + (" in memory and width" if len(rises) == 2 else f" ({rises[0]})"))
    parts += [f"{t['test']}: {OPPOSITE}" for t in tests if t["direction"] == "fall"]
    parts += [f"T2: {BASELINE}" for t in tests if t["direction"] == "baseline"]
    untestable = [f"{t['test']}: {t['reading']}" for t in tests if t["direction"] == "untestable"]
    if parts:
        return "; ".join(parts + untestable)
    if len(untestable) == len(tests):
        return "no test: " + "; ".join(untestable)
    if not manipulation_ok:
        return "; ".join([UNINFORMATIVE] + untestable)
    iv = "; ".join(f"{t['test']} rise {t.get('rise', np.nan):+.4f} [{t.get('rise_ci_lo', np.nan):+.4f}, "
                   f"{t.get('rise_ci_hi', np.nan):+.4f}], detectable {t.get('mde_rise', np.nan):.4f}"
                   for t in tests if t["direction"] != "untestable")
    return "; ".join([f"{NONE}: {NO_POWER} ({iv})"] + untestable)


# ----------------------------------------------------------------------------- exploratory
def paired(d: pd.DataFrame, a: str, b: str, col: str) -> dict:
    """Arm a minus arm b, paired by seed: mean, 95% t-interval, t-test p (uncorrected)."""
    x = d[d["arm"] == a].set_index("seed")[col]
    y = d[d["arm"] == b].set_index("seed")[col]
    diff = (pd.to_numeric(x, errors="coerce") - pd.to_numeric(y, errors="coerce")).dropna()
    out = {"contrast": f"{a} - {b}", "outcome": col, "n_seeds": int(len(diff))}
    if len(diff) < 2:
        return {**out, "mean": float(diff.mean()) if len(diff) else np.nan}
    m, s = float(diff.mean()), float(diff.std(ddof=1))
    tq = float(sstats.t.ppf(0.975, len(diff) - 1))
    se = s / math.sqrt(len(diff))
    p = float(2 * sstats.t.sf(abs(m / se), len(diff) - 1)) if se > 0 else np.nan
    return {**out, "mean": m, "ci_lo": m - tq * se, "ci_hi": m + tq * se, "p_t": p}


def arms_table(runs: pd.DataFrame, main: pd.DataFrame, seeds: list[int], decomp: pd.DataFrame | None) -> list[dict]:
    rows = []
    for arm, g in main[main["seed"].isin(seeds)].groupby("arm", sort=False):
        ok = g[~g["diverged"]]
        r = {"arm": arm, "n_runs": int(len(g)), "n_diverged": int(g["diverged"].sum()),
             "seeds": " ".join(str(int(s)) for s in sorted(g["seed"]))}
        for c in RUN_OUT + ["H_abs_dfa", "width_excess", "delta_h", "delta_h_iaaft_mean", "H_abs_dfa_shuf_mean"]:
            if c in ok:
                v = pd.to_numeric(ok[c], errors="coerce")
                r[f"{c}_mean"] = float(v.mean()) if v.notna().any() else np.nan
                if c in ("H_abs_dfa", "width_excess", "delta_h", "p_clip_realized"):
                    r[f"{c}_median"] = float(v.median()) if v.notna().any() else np.nan
        for c in ("memory_pass", "width_pass", "pass"):
            if c in ok:
                r[f"n_{c}"] = int(ok[c].fillna(False).astype(bool).sum())
        if decomp is not None and len(decomp):
            dd = decomp[decomp["tag"].isin(ok["tag"])]
            for c in ("dh", "shuf_mean", "phase_mean", "iaaft_mean", "kurt"):
                if c in dd:
                    r[f"decomp_{c}_mean"] = float(pd.to_numeric(dd[c], errors="coerce").mean()) if len(dd) else np.nan
        rows.append(r)
    return rows


def exploratory(runs, hl, main, seeds, scores, an, decomp) -> list[dict]:
    rows = []
    ok = main[main["seed"].isin(seeds) & ~main["diverged"]]
    ref = "fixed0.0"
    outs = ["H_abs_dfa", "width_excess", "delta_h", "delta_h_iaaft_mean", "alpha", "zeta", "test_loss", "train_loss",
            "p_clip_realized"]
    for arm in ["fixed0.01", "fixed0.1", "fixed0.5", "fixed0.1_matched"]:
        for c in outs:
            if c in ok:
                rows.append({"part": "arm minus unclipped", **paired(ok, arm, ref, c)})
    for c in outs:
        if c in ok:
            rows.append({"part": "matched against plain 10%", **paired(ok, "fixed0.1_matched", "fixed0.1", c)})
    span = float(max(scores.values()) - min(scores.values()))
    for sc in ("detrended", "rescaled", "sensitivity"):
        d = plain_rows(merged(runs, hl, sc), seeds, scores)
        d = d[~d["diverged"]]
        for _, name, col in TESTS + [GATE]:
            r = trend_test(d, col, None, 0, an["alpha"], an["power"], span, len(TESTS))
            rows.append({"part": f"trend on the {sc} rows", "contrast": "trend (plain arms)", **r})
    pl = plain_rows(main, seeds, scores)
    pl = pl[~pl["diverged"]]
    for c in ("alpha", "zeta", "test_loss", "train_loss", "delta_h_iaaft_mean", "H_abs_dfa_shuf_mean"):
        if c in pl:
            r = trend_test(pl, c, None, 0, an["alpha"], an["power"], span, len(TESTS))
            rows.append({"part": "trend of other outcomes", "contrast": "trend (plain arms)", **r})
    if decomp is not None and len(decomp):
        dd = pl.merge(decomp[["tag"] + [c for c in ("dh", "shuf_mean", "phase_mean", "iaaft_mean") if c in decomp]]
                      .rename(columns={c: f"decomp_{c}" for c in ("dh", "shuf_mean", "phase_mean", "iaaft_mean")}),
                      on="tag", how="inner")
        for c in ("decomp_dh", "decomp_shuf_mean", "decomp_phase_mean", "decomp_iaaft_mean"):
            if c in dd:
                r = trend_test(dd, c, None, 0, an["alpha"], an["power"], span, len(TESTS))
                rows.append({"part": "width decomposition (trend)", "contrast": "trend (plain arms)", **r})
    return rows


def twins(main: pd.DataFrame, p4_runs: pd.DataFrame | None, p4_hl: pd.DataFrame | None) -> list[dict]:
    """Each SGD run against its AdamW twin in P4 (same seed and arm: the same initial weights, data order and
    directions)."""
    if p4_runs is None or not len(p4_runs):
        return []
    tw = p4_runs.copy()
    if p4_hl is not None and len(p4_hl):
        tw = merged(tw, p4_hl, "main")
    cols = ["H_abs_dfa", "width_excess", "delta_h", "delta_h_iaaft_mean", "alpha", "zeta", "test_loss",
            "p_clip_realized", "pass"]
    out = []
    for _, r in main.iterrows():
        t = tw[tw["tag"] == r["tag"]]
        if not len(t):
            continue
        t = t.iloc[0]
        row = {"tag": r["tag"], "seed": int(r["seed"]), "arm": r["arm"], "diverged_sgd": bool(r["diverged"]),
               "diverged_adamw": bool(t["diverged"])}
        for c in cols:
            a, b = _num(r.get(c)), _num(t.get(c))
            row[f"{c}_sgd"], row[f"{c}_adamw"] = a, b
            row[f"{c}_diff"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        out.append(row)
    return out


# ----------------------------------------------------------------------------- the analysis
def analyse(runs_dir: Path, hlink: Path, out: Path, an: dict, scores: dict, p4_runs_path: Path | None = None,
            p4_hlink: Path | None = None, decomp_path: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = dt.datetime.now(dt.timezone.utc)
    phase0 = _json(runs_dir / "phase0.json")
    stopped = _json(runs_dir / "stopped.json")
    summary = {"utc": t0.isoformat(timespec="seconds"), "runs": str(runs_dir), "hlink": str(hlink),
               "phase0": {k: (phase0 or {}).get(k) for k in ("status", "chosen_lr", "extension", "reading")},
               "stopped": stopped}
    if not phase0 or phase0.get("status") != "chosen":
        summary["reading"] = (phase0 or {}).get("reading") or "Phase 0 not decided: no tests"
        summary["gates"] = {"phase0": False}
        _write(out, summary, [], [], [], [], [])
        return summary
    runs = load_runs(runs_dir)
    hl = load_hlink(hlink)
    main = merged(runs, hl, "main")
    seeds = complete_seeds(runs)
    span = float(max(scores.values()) - min(scores.values()))
    d = plain_rows(main, seeds, scores)
    diverged = d[d["diverged"]]["tag"].tolist()
    all_div = main[main["diverged"]]["tag"].tolist()
    d = d[~d["diverged"]]
    missing = d[d["H_abs_dfa"].isna()]["tag"].tolist() if "H_abs_dfa" in d else d["tag"].tolist()
    man = manipulation_gate(d, an["manipulation"])
    tests = []
    for name, what, col in TESTS:
        rng = units_mod.rng_for(units_mod.seed_for(config.SEED, "p7_stats", name))
        r = trend_test(d, col, rng, int(an["n_perm"]), an["alpha"], an["power"], span, len(TESTS))
        tests.append({"test": name, "name": what, "predicted": "rise", **r})
    ok = [np.isfinite(t.get("p_perm", np.nan)) for t in tests]
    ph = holm([t["p_perm"] if o else 1.0 for t, o in zip(tests, ok)])
    for t, o, p in zip(tests, ok, ph):
        t["p_holm"] = p if o else np.nan
    gate = trend_test(d, GATE[2], None, 0, an["alpha"], an["power"], span, len(TESTS))
    gate = {"test": GATE[0], "name": GATE[1], **gate}
    for t in tests:
        t["direction"], t["reading"] = read_test(t, an["alpha"], man["passed"], gate)
    gates = [
        {"gate": "Phase 0 passed", "value": True, "detail": f"chosen rate {phase0.get('chosen_lr')}"},
        {"gate": "no Phase 1 run diverged", "value": not all_div,
         "detail": ("diverged (left out): " + " ".join(all_div)) if all_div else "none diverged"},
        {"gate": "clipping manipulation", "value": man["passed"],
         "detail": "realized means " + ", ".join(f"{k}: {v:.4f}" for k, v in man["means"].items())
                   + f"; ordered {man['ordered']}, 10% >= {an['manipulation']['min_10']} {man['ok_10']}, "
                     f"50% in {an['manipulation']['range_50']} {man['ok_50']}"},
        {"gate": "observed width rises (for T2)", "value": bool(gate.get("slope", 0) > 0 and gate.get("ci_lo", -1) > 0),
         "detail": f"delta_h slope {gate.get('slope', np.nan):+.4g}, 95% [{gate.get('ci_lo', np.nan):+.4g}, "
                   f"{gate.get('ci_hi', np.nan):+.4g}]"},
    ]
    p4_runs = load_runs(p4_runs_path) if p4_runs_path and (p4_runs_path / "runs.csv").exists() else None
    if p4_runs is not None:
        p4_runs = p4_runs[p4_runs["model"] == "transformer"] if "model" in p4_runs else p4_runs
    p4_hl = load_hlink(p4_hlink, folder=p4_runs_path.name) if p4_hlink and p4_runs_path else None
    decomp = pd.read_csv(decomp_path) if decomp_path and Path(decomp_path).exists() else None
    if decomp is not None and "tag" not in decomp and "key" in decomp:
        decomp["tag"] = decomp["key"].str.split("/").str[-1]
    arms = arms_table(runs, main, seeds, decomp)
    expl = exploratory(runs, hl, main, seeds, scores, an, decomp)
    tw = twins(main[main["seed"].isin(seeds)], p4_runs, p4_hl)
    summary.update({
        "seeds_complete": seeds, "seeds_planned": sorted(int(s) for s in runs["seed"].unique()) if len(runs) else [],
        "n_runs_tests": int(len(d)), "diverged_left_out": diverged, "missing_hlink": missing,
        "gates": {g["gate"]: g["value"] for g in gates}, "manipulation": man,
        "tests": [{k: t.get(k) for k in ("test", "name", "slope", "rise", "rise_ci_lo", "rise_ci_hi", "p_perm",
                                         "p_t", "p_holm", "mde_rise", "mde_rise_holm1", "direction", "reading",
                                         "note")} for t in tests],
        "dh_gate": {k: gate.get(k) for k in ("slope", "ci_lo", "ci_hi", "rise")},
        "reading": overall_reading(tests, man["passed"]),
        "pass_counts_by_arm": {a["arm"]: {k: a.get(f"n_{k}") for k in ("memory_pass", "width_pass", "pass")}
                               | {"n": a["n_runs"] - a["n_diverged"]} for a in arms},
        "twins": len(tw), "decomposition": bool(decomp is not None and len(decomp)),
    })
    _write(out, summary, tests + [gate], gates, arms, expl, tw)
    return summary


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(out: Path, summary, tests, gates, arms, expl, tw) -> None:
    runtime.write_csv(out / "tests.csv", tests)
    runtime.write_csv(out / "gates.csv", gates)
    runtime.write_csv(out / "arms.csv", arms)
    runtime.write_csv(out / "exploratory.csv", expl)
    runtime.write_csv(out / "twins.csv", tw)
    units_mod.atomic_json(out / "summary.json", _clean(summary))
    (out / "summary.md").write_text(to_markdown(summary, tests, gates, arms))


def to_markdown(summary, tests, gates, arms) -> str:
    L = ["# P7: P4's transformer arm under SGD with momentum", "",
         f"Computed {summary['utc']} from `{summary['runs']}` and `{summary['hlink']}`.", "",
         f"**Reading:** {summary['reading']}", ""]
    if not tests:
        return "\n".join(L) + "\n"
    L += [f"Seeds in the tests: {summary.get('seeds_complete')}; runs {summary.get('n_runs_tests')}; "
          f"phase 0 rate {summary['phase0'].get('chosen_lr')}.", "",
          "| test | outcome | rise over the range | 95% CI | p (perm) | p (t) | Holm p | detectable (80%) | reading |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        f = lambda k, s="{:+.4f}": (s.format(t[k]) if isinstance(t.get(k), (int, float)) and np.isfinite(t[k]) else "")
        L.append(f"| {t['test']} | {t['name']} | {f('rise')} | {f('rise_ci_lo')} .. {f('rise_ci_hi')} | "
                 f"{f('p_perm', '{:.4f}')} | {f('p_t', '{:.4f}')} | {f('p_holm', '{:.4f}')} | "
                 f"{f('mde_rise', '{:.4f}')} | {t.get('reading', t.get('note', ''))} |")
    L += ["", "| gate | holds | detail |", "|---|---|---|"]
    L += [f"| {g['gate']} | {g['value']} | {g['detail']} |" for g in gates]
    L += ["", "| arm | runs | diverged | clipped (mean) | H(\\|x\\|) | width excess | memory / width / H_link pass |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        g = lambda k: f"{a[k]:.4f}" if isinstance(a.get(k), float) and np.isfinite(a[k]) else ""
        L.append(f"| {a['arm']} | {a['n_runs']} | {a['n_diverged']} | {g('p_clip_realized_mean')} | "
                 f"{g('H_abs_dfa_median')} | {g('width_excess_median')} | {a.get('n_memory_pass', '')} / "
                 f"{a.get('n_width_pass', '')} / {a.get('n_pass', '')} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="config.py's smoke settings (199 permutations)")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--runs", default=None, help="P7's folder (default results/p7_transformer_sgd[/smoke])")
    ap.add_argument("--hlink", default=None, help="h_link.py's p4_grad.csv for P7 "
                                                  "(default results/h_link_p7_transformer_sgd/p4_grad.csv)")
    ap.add_argument("--p4", default=None, help="P4's transformer folder for the twins (default results/p4_transformer)")
    ap.add_argument("--p4-hlink", default=None, help="h_link.py's p4_grad.csv with P4's transformer rows "
                                                     "(default results/h_link/p4_grad.csv)")
    ap.add_argument("--decomposition", default=None,
                    help="surrogate_decomposition.py's P7 units (default results/surrogate_decomposition/P7_units.csv)")
    ap.add_argument("--out", default=None, help="default <runs>/analysis")
    args = ap.parse_args(argv)
    res = Path(args.results)
    p7 = config.get("p7_transformer_sgd", smoke=args.smoke)
    runs_dir = Path(args.runs) if args.runs else res / "p7_transformer_sgd" / ("smoke" if args.smoke else "")
    hlink = Path(args.hlink) if args.hlink else res / "h_link_p7_transformer_sgd" / "p4_grad.csv"
    out = Path(args.out) if args.out else runs_dir / "analysis"
    s = analyse(runs_dir, hlink, out, p7["analysis"], p7["analysis"]["scores"],
                p4_runs_path=Path(args.p4) if args.p4 else res / "p4_transformer",
                p4_hlink=Path(args.p4_hlink) if args.p4_hlink else res / "h_link" / "p4_grad.csv",
                decomp_path=Path(args.decomposition) if args.decomposition
                else res / "surrogate_decomposition" / "P7_units.csv")
    print(f"P7 analysis -> {out}\n{s['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
