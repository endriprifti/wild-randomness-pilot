"""The surrogate decomposition of the gradient series' width ("step 1", procedure fixed 2026-11-27 before any value was
computed; docs/surrogate_decomposition.md; results brief 11). Exploratory, outside every confirmatory family.

Splits the baseline H_link's width part compares against (the IAAFT surrogates, which keep both the marginal and
the spectrum) into its two parts: per run, the observed MFDFA width Delta h (q = 0.5 and 4, main scales 8 to n/10,
mean over the 8 directions, as H_link's main rows), the mean, sd, 95% quantile and p of 99 shuffled surrogates (keep
the marginal only) and 99 phase-randomized surrogates (keep the spectrum only, Gaussian marginal), and the excess
kurtosis of the marginal. The IAAFT mean and 95% quantile come from H_link's own files; they are recomputed (99
surrogates) only on the check runs (CHECK), where they must agree with the file within Monte Carlo error, and the
observed width must reproduce the file's delta_h in every run. Random numbers:
units.rng_for(units.seed_for(config.SEED, "step1_decomposition", key)), key "<design>/<run or tag>"; per surrogate
draw the order is shuffled, phase-randomized, then IAAFT (check runs only). Run on the same series, this script gives
brief 11's numbers (reference/surrogate_decomposition_brief11.csv, the 242 runs of P6, P6b, P4 transformer and P4
ResNet-18: --compare checks them).

Designs (paths relative to --results, default results/; the warm-up steps H_link cut are cut here too):
  P6   p6_small/14m_s*/runs/<run>/hgrad.npy, H_link values from .../hlink/stats/hlink_grad.json (warm-up 81)
  P6b  p6b_small/ (same layout)
  P6c  p6c_small/ (same layout; added 2026-11-27 for P6c's listed decomposition)
  P4T  p4_transformer/hgrad_series/<tag>.npy, H_link values from h_link_p4_transformer/p4_grad.csv if it exists,
       else h_link/p4_grad.csv (rows of folder p4_transformer; warm-up 91)
  P4R  p4_resnet18/hgrad_series/<tag>.npy, H_link values from h_link/p4_grad.csv (folder p4_resnet18; warm-up 234)
  P7   p7_transformer_sgd/hgrad_series/<tag>.npy, H_link values from h_link_p7_transformer_sgd/p4_grad.csv
       (added 2026-11-27 for P7's listed decomposition)

Analysis (from the saved units; analyse() is the brief's analyse.py with its prints kept as tables):
  P6, P6b, P6c (each separately, never pooled): P6's model (seed blocks, Noah and Joseph coded +-1/2, their
    product), OLS, the Noah effect with its 95% interval on Delta h, the shuffled, phase-randomized and IAAFT means
    and the kurtosis; no correction. Reading (brief 9, decision 6): the baseline effect is assigned to the marginal if
    the shuffled surrogates' width shows a Noah effect of the IAAFT mean's sign with its interval excluding 0 and the
    phase-randomized surrogates' does not; to the spectrum in the reverse case; otherwise both are reported as found.
  P4T, P4R, P7 (each separately): per arm, the medians over seeds of Delta h and the three surrogate means, and the
    runs whose Delta h exceeds each kind's 95% quantile. Reading: over the fixed arms with targets 1 to 50% (plain and
    matched), the width is beyond the marginal and beyond the spectrum taken one at a time if Delta h exceeds the
    shuffled and the phase-randomized 95% quantiles in at least as many runs as the IAAFT one; otherwise the kind
    that keeps the width is named. Guard (Endri, 2026-11-27; procedure.md applies the rule to ResNet-18 and describes
    the transformer by counts and medians): the rule is applied only where Delta h exceeds the IAAFT 95% quantile in
    at least half of those runs; otherwise the reading is "no reading: the width exceeds the IAAFT quantile in only
    n of N runs", with the counts and the arm medians of Delta h and of the phase-randomized mean. No number changes.

Outputs (--out, default results/surrogate_decomposition/): units/<design>__<run>.json (one per run; a rerun
computes only the missing ones), <design>_units.csv (one row per run), and from every design saved: units.csv,
iaaft_check.csv, noah_effects.csv, cells.csv, arms.csv, readings.csv, report.txt.

Commands (in the pilot folder inside the venv; CPU only, nice 19, one thread per worker):
  python surrogate_decomposition.py --list                                  # runs found and saved per design
  python surrogate_decomposition.py --designs P6,P6b,P4T,P4R --compare reference/surrogate_decomposition_brief11.csv
  python surrogate_decomposition.py --designs P6c                           # after P6c
  python surrogate_decomposition.py --designs P7                            # after P7's h_link (the queue's job)
  python surrogate_decomposition.py --analyse-only                          # the tables from the saved runs
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from common import hurst  # noqa: E402
from common import units as units_mod  # noqa: E402

N_SURR = 99
P6_DESIGNS = {"P6": "p6_small", "P6b": "p6b_small", "P6c": "p6c_small"}
P4_DESIGNS = {  # design: (series folder, H_link csv candidates in order, H_link folder name)
    "P4T": ("p4_transformer", ["h_link_p4_transformer/p4_grad.csv", "h_link/p4_grad.csv"], "p4_transformer"),
    "P4R": ("p4_resnet18", ["h_link/p4_grad.csv"], "p4_resnet18"),
    "P7": ("p7_transformer_sgd", ["h_link_p7_transformer_sgd/p4_grad.csv"], "p7_transformer_sgd"),
}
DESIGNS = list(P6_DESIGNS) + list(P4_DESIGNS)
# runs whose IAAFT surrogates are recomputed (the check): brief 11's nine, and for P6c and P7 (added 2026-11-27)
# the same choice: the four corners of the first seed, seed 0's unclipped and fixed 10% runs
CHECK = {"P6/14m_light_H0.5_s1", "P6b/14m_light_H0.5_s4", "P6b/14m_light_H0.9_s4", "P6b/14m_zipf1.0_H0.5_s4",
         "P6b/14m_zipf1.0_H0.9_s4", "P4T/transformer_s0_fixed0.0", "P4T/transformer_s0_fixed0.1",
         "P4R/resnet18_s0_fixed0.0", "P4R/resnet18_s0_fixed0.1",
         "P6c/14m_light_H0.5_s15", "P6c/14m_light_H0.9_s15", "P6c/14m_zipf1.0_H0.5_s15", "P6c/14m_zipf1.0_H0.9_s15",
         "P7/transformer_s0_fixed0.0", "P7/transformer_s0_fixed0.1"}
QUANTITIES = ["dh", "shuf_mean", "phase_mean", "iaaft_file", "kurt"]


# ----------------------------------------------------------------------------- the runs of each design
def specs(results: Path, designs: list[str]) -> list[dict]:
    """One spec per run: key, series path (relative to results), warm-up cut and H_link's values from its files."""
    out = []
    for des in designs:
        if des in P6_DESIGNS:
            root = results / P6_DESIGNS[des]
            for f in sorted(glob.glob(str(root / "14m_s*" / "runs" / "*" / "hlink" / "stats" / "hlink_grad.json"))):
                r = json.loads(Path(f).read_text())["result"]
                run = r["folder"]
                seedf = Path(f).parts[-6]
                main = [x for x in r["rows"] if x["scales"] == "main"][0]
                _, noah, joseph, s = run.split("_")
                out.append(dict(design=des, key=f"{des}/{run}",
                                path=f"{P6_DESIGNS[des]}/{seedf}/runs/{run}/hgrad.npy", warm=int(r["warm"]),
                                noah=noah, joseph=joseph, seed=int(s[1:]), ref_dh=main["delta_h"],
                                ref_iaaft=main["delta_h_iaaft_mean"], ref_iaaft_q95=main["delta_h_iaaft_q95"],
                                ref_p_iaaft=main["p_iaaft"]))
        elif des in P4_DESIGNS:
            sdir, refs, folder = P4_DESIGNS[des]
            ref = next((results / c for c in refs if (results / c).exists()), None)
            if ref is None:
                continue
            d = pd.read_csv(ref)
            d = d[(d.scales == "main") & (d.folder == folder)]
            for _, r in d.iterrows():
                out.append(dict(design=des, key=f"{des}/{r.tag}", path=f"{sdir}/hgrad_series/{r.tag}.npy",
                                warm=int(r.warm), seed=int(r.seed), rule=r.rule,
                                p_clip_target=(None if pd.isna(r.p_clip_target) else float(r.p_clip_target)),
                                matched=str(r.matched).lower() == "true", ref_dh=float(r.delta_h),
                                ref_iaaft=float(r.delta_h_iaaft_mean), ref_iaaft_q95=float(r.delta_h_iaaft_q95),
                                ref_p_iaaft=float(r.p_iaaft)))
        else:
            raise SystemExit(f"unknown design {des!r}; designs: {', '.join(DESIGNS)}")
    return out


# ----------------------------------------------------------------------------- one run
def width(cols, scales) -> float:
    return float(np.mean([hurst.delta_h(v, 0.5, 4.0, scales)["delta_h"] for v in cols]))


def unit_file(out: Path, key: str) -> Path:
    return out / "units" / (key.replace("/", "__") + ".json")


def compute(spec: dict, results: Path, n_surr: int = N_SURR) -> dict:
    """The decomposition of one run (procedure.md; the brief's decomp.run, paths relative to results)."""
    t0 = time.perf_counter()
    X = np.asarray(np.load(results / spec["path"]), dtype=float)
    warm = spec["warm"]
    X = X[warm:] if len(X) > warm + 50 else X
    X = X[np.isfinite(X).all(axis=1)]
    n, d = X.shape
    cols = [X[:, j] for j in range(d)]
    scales = hurst.log_scales(n, 8)
    rng = units_mod.rng_for(units_mod.seed_for(config.SEED, "step1_decomposition", spec["key"]))
    obs = width(cols, scales)
    do_iaaft = spec["key"] in CHECK
    sur = {"shuf": [], "phase": []} | ({"iaaft": []} if do_iaaft else {})
    for _ in range(n_surr):
        sur["shuf"].append(width([hurst.shuffle_surrogate(v, rng) for v in cols], scales))
        sur["phase"].append(width([hurst.phase_surrogate(v, rng) for v in cols], scales))
        if do_iaaft:
            sur["iaaft"].append(width([hurst.iaaft_surrogate(v, rng) for v in cols], scales))
    kurt = float(np.mean([((v - v.mean()) ** 4).mean() / v.var() ** 2 - 3 for v in cols]))
    res = {**spec, "n": n, "dh": obs, "kurt": kurt, "seconds": round(time.perf_counter() - t0, 1)}
    for k, v in sur.items():
        v = np.asarray(v)
        res[f"{k}_mean"] = float(v.mean())
        res[f"{k}_sd"] = float(v.std(ddof=1))
        res[f"{k}_q95"] = float(np.quantile(v, 0.95))
        res[f"{k}_p"] = float((1 + (v >= obs).sum()) / (n_surr + 1))
    return res


def _work(task):
    spec, results, out, n_surr, nice = task
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass
    f = unit_file(out, spec["key"])
    if f.exists():
        return spec["key"], "saved before"
    res = compute(spec, results, n_surr)
    units_mod.atomic_json(f, res)
    return spec["key"], res["seconds"]


def run_all(S: list[dict], results: Path, out: Path, workers: int, nice: int, n_surr: int = N_SURR) -> None:
    todo = [s for s in S if not unit_file(out, s["key"]).exists()]
    missing = [s["key"] for s in todo if not (results / s["path"]).exists()]
    if missing:
        print(f"series missing for {len(missing)} run(s), left out: {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""), flush=True)
    todo = [s for s in todo if (results / s["path"]).exists()]
    print(f"{len(S)} runs, {len(S) - len(todo) - len(missing)} saved before, {len(todo)} to compute "
          f"({workers} worker(s))", flush=True)
    (out / "units").mkdir(parents=True, exist_ok=True)
    tasks = [(s, results, out, n_surr, nice) for s in todo]
    if workers <= 0:
        it = map(_work, tasks)
    else:
        pool = get_context("spawn").Pool(workers)
        it = pool.imap_unordered(_work, tasks)
    for i, (key, sec) in enumerate(it):
        if i % 10 == 0 or i == len(tasks) - 1:
            print(f"  {i + 1}/{len(tasks)} {key}: {sec} s", flush=True)
    if workers > 0:
        pool.close()
        pool.join()


def load_units(out: Path, designs: list[str] | None = None) -> pd.DataFrame:
    rows = [json.loads(f.read_text()) for f in sorted((out / "units").glob("*.json"))]
    U = pd.DataFrame(rows)
    if len(U) and designs:
        U = U[U.design.isin(designs)].reset_index(drop=True)
    if len(U):
        U["iaaft_file"] = U.ref_iaaft
    return U


# ----------------------------------------------------------------------------- the analysis (analyse.py)
def fit(d: pd.DataFrame, y: str, extra=()):
    """The brief's confound.fit: seed-block OLS of y on N, J, NJ (coded +-1/2) and optional centred covariates;
    (estimate, se, two-sided t p) per term, and the simple effects of J under light and under Zipf."""
    seeds = sorted(d.seed.unique())
    n = np.where(d.noah == "zipf1.0", .5, -.5)
    j = np.where(d.joseph == "H0.9", .5, -.5)
    cols = [(d.seed == s).astype(float).values for s in seeds] + [n, j, n * j] + \
           [d[c].values - d[c].mean() for c in extra]
    X = np.column_stack(cols)
    Y = d[y].values.astype(float)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ b
    df = len(Y) - X.shape[1]
    s2 = r @ r / df
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    k = len(seeds)
    out = {}
    for i, name in enumerate(["N", "J", "NJ"] + list(extra)):
        t = b[k + i] / se[k + i]
        out[name] = (b[k + i], se[k + i], 2 * sstats.t.sf(abs(t), df))
    for lab, sgn in (("J|light", -1), ("J|zipf", 1)):
        c = np.zeros(X.shape[1])
        c[k + 1] = 1
        c[k + 2] = sgn * .5
        est = c @ b
        s = np.sqrt(c @ cov @ c)
        out[lab] = (est, s, 2 * sstats.t.sf(abs(est / s), df))
    return out, df


def p4_arm(d: pd.DataFrame) -> pd.Series:
    arm = np.where(d.rule == "fixed", "fixed " + d.p_clip_target.map(lambda x: f"{100 * x:g}%"), d.rule)
    arm = pd.Series(arm, index=d.index)
    arm[(d.rule == "fixed") & (d.p_clip_target == 0)] = "unclipped"
    return arm


def analyse(U: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    """The tables and the text of the brief's analyse.py."""
    L, T = [], {}
    L.append(f"runs {U.design.value_counts().to_dict()}")
    L.append(f"max |dh - file| {(U.dh - U.ref_dh).abs().max()}")
    c = U[U.iaaft_mean.notna()] if "iaaft_mean" in U else U.iloc[:0]
    chk = []
    L.append("IAAFT check (recomputed - file, in MC SE):")
    for _, r in c.iterrows():
        z = (r.iaaft_mean - r.ref_iaaft) / (r.iaaft_sd * np.sqrt(2 / N_SURR))
        chk.append({"key": r.key, "iaaft_mean": r.iaaft_mean, "iaaft_file": r.ref_iaaft, "z": z})
        L.append("  %-32s %.5f %.5f  z %.2f" % (r.key, r.iaaft_mean, r.ref_iaaft, z))
    T["iaaft_check"] = pd.DataFrame(chk)
    eff, cells, reads, arms = [], [], [], []
    for des in [x for x in P6_DESIGNS if x in set(U.design)]:
        d = U[U.design == des].reset_index(drop=True)
        L.append(f"===== {des} ({len(d)} runs): Noah effects, OLS with seed blocks")
        est = {}
        for y in QUANTITIES:
            o, df = fit(d, y)
            e, se, p = o["N"]
            t = sstats.t.ppf(0.975, df)
            est[y] = (e, e - t * se, e + t * se)
            eff.append({"design": des, "quantity": y, "N": e, "N_se": se, "N_ci_lo": e - t * se,
                        "N_ci_hi": e + t * se, "N_p": p, "J": o["J"][0], "J_p": o["J"][2], "df": df})
            L.append(f"  {y:11s} N {e:+.4f} (SE {se:.4f}, 95% {e - t * se:+.4f} to {e + t * se:+.4f}, p {p:.2g})"
                     f"   J {o['J'][0]:+.4f} (p {o['J'][2]:.2g})")
        g = d.groupby(["noah", "joseph"])[QUANTITIES].mean()
        L.append(g.round(4).to_string())
        cells.append(g.reset_index().assign(design=des))
        n_s, n_p, n_i = int((d.dh > d.shuf_q95).sum()), int((d.dh > d.phase_q95).sum()), \
            int((d.dh > d.ref_iaaft_q95).sum())
        L.append(f"  runs with dh > q95: shuffled {n_s}  phase {n_p}  IAAFT {n_i}")
        reads.append({"design": des, "reading": p6_reading(est), "runs": len(d), "dh_gt_shuf_q95": n_s,
                      "dh_gt_phase_q95": n_p, "dh_gt_iaaft_q95": n_i})
    for des in [x for x in ("P4R", "P4T", "P7") if x in set(U.design)]:
        d = U[U.design == des].copy()
        d["arm"] = p4_arm(d)
        d["gt_shuf"] = d.dh > d.shuf_q95
        d["gt_phase"] = d.dh > d.phase_q95
        d["gt_iaaft"] = d.dh > d.ref_iaaft_q95
        order = ["unclipped"] + [f"fixed {x:g}%" for x in (0.01, 0.1, 1, 10, 50, 100)] + ["autoclip", "zclip", "agc"]
        g = d.groupby(["arm", "matched"]).agg(dh=("dh", "median"), shuf=("shuf_mean", "median"),
                                              phase=("phase_mean", "median"), iaaft=("iaaft_file", "median"),
                                              kurt=("kurt", "median"), n_shuf=("gt_shuf", "sum"),
                                              n_phase=("gt_phase", "sum"), n_iaaft=("gt_iaaft", "sum"))
        g = g.reindex([(a, m) for a in order for m in (False, True) if (a, m) in g.index])
        L.append(f"===== {des}")
        L.append(g.round(4).to_string())
        arms.append(g.reset_index().assign(design=des))
        sel = d[(d.rule == "fixed") & d.p_clip_target.isin([0.01, 0.1, 0.5])]
        L.append(f"  fixed 1-50% ({len(sel)} runs): dh > q95 of shuffled {int(sel.gt_shuf.sum())}, phase "
                 f"{int(sel.gt_phase.sum())}, IAAFT {int(sel.gt_iaaft.sum())}")
        un = d[d.arm == "unclipped"]
        L.append(f"  unclipped: shuffled {int(un.gt_shuf.sum())}, phase {int(un.gt_phase.sum())}, IAAFT "
                 f"{int(un.gt_iaaft.sum())}")
        L.append(f"  all runs: shuffled {int(d.gt_shuf.sum())}, phase {int(d.gt_phase.sum())}, IAAFT "
                 f"{int(d.gt_iaaft.sum())} of {len(d)}")
        reads.append({"design": des, "reading": p4_reading(sel),
                      "runs": len(sel), "dh_gt_shuf_q95": int(sel.gt_shuf.sum()),
                      "dh_gt_phase_q95": int(sel.gt_phase.sum()), "dh_gt_iaaft_q95": int(sel.gt_iaaft.sum())})
    T["noah_effects"] = pd.DataFrame(eff)
    T["cells"] = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    T["arms"] = pd.concat(arms, ignore_index=True) if arms else pd.DataFrame()
    T["readings"] = pd.DataFrame(reads)
    L.append("===== readings (procedure.md)")
    L += [f"  {r['design']}: {r['reading']}" for r in reads]
    return T, "\n".join(L) + "\n"


def _excludes_zero_with_sign(e, sign) -> bool:
    est, lo, hi = e
    return (lo > 0 or hi < 0) and np.sign(est) == sign


def p6_reading(est: dict) -> str:
    """Brief 9, decision 6: where the Noah effect on the IAAFT baseline sits (marginal, spectrum or both)."""
    sign = np.sign(est["iaaft_file"][0])
    shuf = _excludes_zero_with_sign(est["shuf_mean"], sign)
    phase = _excludes_zero_with_sign(est["phase_mean"], sign)
    f = lambda k: f"{est[k][0]:+.4f} [{est[k][1]:+.4f}, {est[k][2]:+.4f}]"
    found = f"(IAAFT mean {f('iaaft_file')}, shuffled {f('shuf_mean')}, phase-randomized {f('phase_mean')})"
    if shuf and not phase:
        return f"the baseline effect sits in the marginal {found}"
    if phase and not shuf:
        return f"the baseline effect sits in the spectrum {found}"
    return f"not assigned: both effects as found {found}"


ARM_ORDER = [f"fixed {x:g}%" for x in (1, 10, 50)]


def p4_reading(sel: pd.DataFrame) -> str:
    """The reading over the fixed arms with targets 1 to 50% (plain and matched; `sel` has arm, matched, dh,
    phase_mean and the gt_* flags). Guard (2026-11-27): only if Delta h exceeds the IAAFT 95% quantile in at least
    half of these runs is the rule applied: beyond the marginal and beyond the spectrum, one at a time, if Delta h
    exceeds the shuffled and the phase-randomized 95% quantiles in at least as many runs as the IAAFT one. Otherwise
    no reading, with the counts and the arm medians of Delta h and of the phase-randomized mean."""
    n, n_shuf, n_phase, n_iaaft = len(sel), int(sel.gt_shuf.sum()), int(sel.gt_phase.sum()), int(sel.gt_iaaft.sum())
    counts = f"(dh > q95: shuffled {n_shuf}, phase-randomized {n_phase}, IAAFT {n_iaaft})"
    if n == 0:
        return "no reading: no runs at the fixed targets 1 to 50%"
    if 2 * n_iaaft < n:
        g = sel.groupby(["arm", "matched"])[["dh", "phase_mean"]].median()
        keys = [(a, m) for a in ARM_ORDER for m in (False, True) if (a, m) in g.index]
        med = "; ".join(f"{a}{' matched' if m else ''} {g.loc[(a, m), 'dh']:.4f} / {g.loc[(a, m), 'phase_mean']:.4f}"
                        for a, m in keys)
        return (f"no reading: the width exceeds the IAAFT quantile in only {n_iaaft} of {n} runs {counts}; "
                f"medians by arm, dh / phase-randomized mean: {med}")
    if n_shuf >= n_iaaft and n_phase >= n_iaaft:
        return f"width beyond the marginal and beyond the spectrum taken one at a time {counts}"
    keep = [k for k, n in (("the marginal (shuffled)", n_shuf), ("the spectrum (phase-randomized)", n_phase))
            if n < n_iaaft]
    return f"the width is kept by {' and '.join(keep)} {counts}"


def compare(U: pd.DataFrame, ref: Path) -> tuple[pd.DataFrame, str]:
    """Each saved run against the reference table (the brief's step1_units.csv): largest absolute differences."""
    R = pd.read_csv(ref)
    m = U.merge(R, on="key", suffixes=("", "_ref"))
    cols = ["dh", "kurt", "shuf_mean", "shuf_q95", "shuf_p", "phase_mean", "phase_q95", "phase_p", "iaaft_mean",
            "iaaft_q95", "ref_dh", "ref_iaaft", "n"]
    rows = []
    for c in cols:
        if c in m and f"{c}_ref" in m:
            a, b = pd.to_numeric(m[c], errors="coerce"), pd.to_numeric(m[f"{c}_ref"], errors="coerce")
            both = a.notna() & b.notna()
            rows.append({"column": c, "runs": int(both.sum()),
                         "max_abs_diff": float((a[both] - b[both]).abs().max()) if both.any() else np.nan})
    out = pd.DataFrame(rows)
    txt = (f"compared with {ref}: {len(m)} of {len(R)} reference runs saved here; "
           f"largest |difference| {out['max_abs_diff'].max():.3g}\n" + out.to_string(index=False) + "\n")
    return out, txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", default="P6,P6b,P4T,P4R", help=f"comma-separated, of {','.join(DESIGNS)}")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None, help="default <results>/surrogate_decomposition")
    ap.add_argument("--workers", type=int, default=2, help="processes (0: in this process)")
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--n-surr", type=int, default=N_SURR, help=argparse.SUPPRESS)       # tests
    ap.add_argument("--list", action="store_true", help="runs found and saved per design; compute nothing")
    ap.add_argument("--analyse-only", action="store_true", help="the tables from the saved runs; compute nothing")
    ap.add_argument("--compare", default=None, help="a reference table (reference/surrogate_decomposition_brief11.csv)")
    args = ap.parse_args(argv)
    results = Path(args.results).resolve()
    out = Path(args.out) if args.out else results / "surrogate_decomposition"
    designs = [d for d in args.designs.split(",") if d]
    S = specs(results, designs)
    if args.list:
        for des in designs:
            s = [x for x in S if x["design"] == des]
            print(f"{des}: {len(s)} runs found, {sum((results / x['path']).exists() for x in s)} series present, "
                  f"{sum(unit_file(out, x['key']).exists() for x in s)} saved")
        return 0
    if not args.analyse_only:
        run_all(S, results, out, args.workers, args.nice, args.n_surr)
    U = load_units(out)
    if not len(U):
        print("no saved runs")
        return 0
    cols = [c for c in U.columns if c != "iaaft_file"] + ["iaaft_file"]
    for des in designs:
        d = U[U.design == des]
        if len(d):
            d[cols].to_csv(out / f"{des}_units.csv", index=False)
    U[cols].to_csv(out / "units.csv", index=False)
    T, txt = analyse(U)
    for name, t in T.items():
        t.to_csv(out / f"{name}.csv", index=False)
    if args.compare:
        c, ctxt = compare(U, Path(args.compare))
        c.to_csv(out / "compare.csv", index=False)
        txt += "===== " + ctxt
    (out / "report.txt").write_text(txt)
    print(txt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
