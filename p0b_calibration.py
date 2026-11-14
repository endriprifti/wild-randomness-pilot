"""P0b: three calibrations that P0 left thin (CPU only, separate from the cloud run).

  cascades  MFDFA width test on binomial cascade SERIES, 20 replicates per (p, n) (P0 had one).
            P0's series is the deterministic binomial series, so its replicates would be identical; here each
            replicate is a random-order binomial cascade (every split gives p and 1-p to its two halves in
            random order). Its values are the same multiset at every dyadic level, so its h(q) is the
            deterministic series' h(q) (Kantelhardt et al. 2002). p = 0.5 is the monofractal control (fGn,
            H = 0.5), as in P0.
  h71       H7.1 on truncated-Levy pools (signed SaS alpha = 1.5 values with |x| <= cutoff), where alpha-hat
            drifts to 2 once B is large enough to see the cutoff. Where that happens depends on the cutoff, and
            the test can only see a drift that falls inside B = 8 ... 1024: with P0's cutoff 20 alpha-hat is
            already about 1.8 at B = 8 and 2 from B = 64, so the slope b is only about 0.03 per doubling; with
            cutoff 100 the drift spans the whole range (b about 0.07); with 500 it starts near B = 512. So the
            check runs at several cutoffs. Per cutoff, as P0's block check on stable pools: 200 replicate pools
            give the spread of b (alpha-hat on log2 B), and test_reps further pools get the full preregistered
            test (h71_test, bootstrap 90% interval, n_boot = 200) and its reading.
  stable    the direct stable fit (fit_stable) and the bootstrap LR test against the Gaussian
            (lr_test_gaussian) on SaS alpha = 1.2, 1.5, 1.8, 2.0 at n = 10^3, 10^4, 10^5, 200 replicates
            each: bias, SD and RMSE of alpha-hat (against Nolan's asymptotic SE), the share of fits at the
            boundary alpha = 2, and the test's rejection rate (size at alpha = 2, power below).
            The LR statistic is location-scale invariant, so its parametric-bootstrap null distribution
            depends on n only: it is simulated once per n (lr_null_draws Gaussian samples) and shared by all
            replicates, instead of 200 fresh null fits per replicate (which would take about 45 h on 6
            cores). lr_check_reps replicates per cell also run lr_test_gaussian itself (n_boot = 200), to
            show the shared null gives the same p-values.

Every task gets its own seed from (seed, part, indices), so results do not depend on --jobs or on the order in
which the worker processes finish.

Outputs (results/p0b_calibration/[smoke/]): cascade_series.csv, cascade_series_summary.csv, h71_trunc.csv,
h71_trunc_tests.csv, h71_trunc_summary.csv, stable_fit.csv, stable_fit_summary.csv, stable_lr_null.csv, stable_lr_check.csv, run.json

Commands:
  python p0b_calibration.py --jobs 6                 # all three parts on 6 worker processes
  python p0b_calibration.py --jobs 6 --parts stable  # one part
  python p0b_calibration.py --smoke
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one thread per worker process; --jobs sets the parallelism

import concurrent.futures as cf
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from scipy import stats

import config
from common import hurst, runtime, stable
from common.batch import _slopes, fit_over_batches, h71_test

PART_CODES = {"cascades": 1, "h71": 2, "stable": 3, "null": 4, "check": 5}


def task_rng(seed: int, part: str, *idx: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), PART_CODES[part], *[int(i) for i in idx]])


# ----------------------------------------------------------------------------- samplers
def random_binomial_series(depth: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """Random-order binomial cascade of length 2^depth: each split gives weights a and 1-a to its two halves in
    random order. Same multiset of values at every dyadic level as hurst.binomial_series (so the same h(q))."""
    x = np.ones(1)
    for _ in range(depth):
        left = np.where(rng.random(len(x)) < 0.5, a, 1.0 - a)
        x = np.stack([x * left, x * (1.0 - left)], axis=1).reshape(-1)
    return x


def trunc_levy_pool(alpha: float, cutoff: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n signed SaS(alpha, gamma = 1) values with |x| <= cutoff (abrupt truncation, Mantegna-Stanley)."""
    out = np.empty(0)
    while len(out) < n:
        x = stable.rvs_stable(alpha, 0.0, 1.0, 0.0, 2 * (n - len(out)) + 100, rng)
        out = np.concatenate([out, x[np.abs(x) <= cutoff]])
    return out[:n]


def lr_stat(x: np.ndarray, fit=None, maxiter: int = 600) -> tuple[float, object]:
    """LR statistic of lr_test_gaussian: 2 (stable loglik - Gaussian loglik at the MLE)."""
    fit = fit or stable.fit_stable(x, maxiter=maxiter)
    ll_norm = float(np.sum(stats.norm.logpdf(x, np.mean(x), np.std(x))))
    return 2 * (fit.loglik - ll_norm), fit


def p_from_null(lr: float, null: np.ndarray) -> float:
    """Bootstrap p-value as in lr_test_gaussian: (1 + #{null >= lr}) / (1 + M)."""
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= lr)) / (1 + len(null)))


# ----------------------------------------------------------------------------- worker tasks
def t_cascade(seed, i_p, p, i_n, n, r, S):
    rng = task_rng(seed, "cascades", i_p, i_n, r)
    depth = int(np.log2(n))
    s = random_binomial_series(depth, p, rng) if p != 0.5 else hurst.fgn(2 ** depth, 0.5, rng)
    mc = hurst.multifractal_claim(s, S["q_lo"], S["q_hi"], n_surr=S["n_surr"], rng=rng)
    th = hurst.binomial_series_h(np.array([S["q_lo"], S["q_hi"]]), p) if p != 0.5 else np.zeros(2)
    return {"p": p, "n": 2 ** depth, "replicate": r, "delta_h_true": float(th[0] - th[1]),
            "truth": "multifractal" if p != 0.5 else "monofractal", **mc}


def t_h71_rep(seed, r, hc, Bs, i_c=0, cutoff=None):
    rng = task_rng(seed, "h71", 0, i_c, r)
    cutoff = cutoff if cutoff is not None else hc["cutoffs"][0]
    pool = trunc_levy_pool(hc["alpha"], cutoff, hc["n_blocks"] * max(Bs), rng)
    rows = fit_over_batches(pool, Bs, hc["n_blocks"])
    b, s, abar = _slopes(rows)
    a = [x["alpha"] for x in rows]
    cross = next((x["B"] for x in rows if np.isfinite(x["alpha"]) and np.isfinite(a[0]) and x["alpha"] - a[0] > 0.2),
                 None)
    return {"cutoff": cutoff, "replicate": r, "b": b, "scale_exponent": s, "alpha_mean": abar,
            "pred_scale_exponent": 1 / abar - 1,
            "B_cross_hint": cross, **{f"alpha_B{x['B']}": x["alpha"] for x in rows}}


def t_h71_test(seed, r, hc, Bs, tol, i_c, cutoff):
    rng = task_rng(seed, "h71", 1, i_c, r)
    pool = trunc_levy_pool(hc["alpha"], cutoff, hc["n_blocks"] * max(Bs), rng)
    t = h71_test(pool, Bs, hc["n_blocks"], n_boot=hc["test_n_boot"], level=tol["level"], inv_tol=tol["inv_tol"],
                 drift_tol=tol["drift_tol"], rng=rng)
    return {"cutoff": cutoff, "test": r, **t["summary"], **{f"alpha_B{x['B']}": x["alpha"] for x in t["per_B"]}}


def t_stable(seed, i_a, alpha, i_n, n, r):
    rng = task_rng(seed, "stable", i_a, i_n, r)
    x = stable.rvs_stable(alpha, 0.0, 1.0, 0.0, n, rng)
    t0 = time.perf_counter()
    lr, f = lr_stat(x)
    return {"alpha_true": alpha, "n": n, "replicate": r, "alpha_hat": f.alpha, "beta_hat": f.beta,
            "gamma_hat": f.gamma, "converged": f.converged, "alpha_start": f.alpha_start, "lr_stat": lr,
            "fit_seconds": time.perf_counter() - t0}


def t_null(seed, i_n, n, j):
    rng = task_rng(seed, "null", i_n, j)
    y = rng.normal(0.0, 1.0, n)
    lr, _ = lr_stat(y, maxiter=300)        # lr_test_gaussian fits its null samples with maxiter = 300
    return {"n": n, "draw": j, "lr_stat": lr}


def t_check(seed, i_a, alpha, i_n, n, r, n_boot):
    """lr_test_gaussian itself on replicate r of cell (alpha, n): the same sample as t_stable draws."""
    x = stable.rvs_stable(alpha, 0.0, 1.0, 0.0, n, task_rng(seed, "stable", i_a, i_n, r))
    out = stable.lr_test_gaussian(x, n_boot=n_boot, rng=task_rng(seed, "check", i_a, i_n, r))
    return {"alpha_true": alpha, "n": n, "replicate": r, "lr_stat_function": out["lr_stat"],
            "p_function": out["p_gauss"], "n_boot": out["n_boot"]}


# ----------------------------------------------------------------------------- summaries
def summarize_cascades(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["p", "n"]).agg(
        truth=("truth", "first"), replicates=("replicate", "size"), delta_h_true=("delta_h_true", "first"),
        delta_h_mean=("delta_h", "mean"), delta_h_sd=("delta_h", "std"),
        claim_rate=("multifractal", "mean"), width_from_marginal_rate=("width_from_marginal", "mean"),
        p_iaaft_median=("p_iaaft", "median"), p_phase_median=("p_phase", "median")).reset_index()


def summarize_h71(reps: pd.DataFrame, tests: pd.DataFrame, tol: dict) -> list[dict]:
    """One row per cutoff: spread of b over the replicate pools, and the full test's readings."""
    cuts = sorted(set(reps.get("cutoff", pd.Series(dtype=float))) | set(tests.get("cutoff", pd.Series(dtype=float))))
    out = []
    for c in cuts:
        row = {"cutoff": c}
        g = reps[reps.cutoff == c] if len(reps) else reps
        if len(g):
            b = g.b.dropna()
            row.update({"replicates": int(len(g)), "b_mean": float(b.mean()), "b_q05": float(b.quantile(0.05)),
                        "b_q95": float(b.quantile(0.95)),
                        "share_b_above_drift_tol": float((b > tol["drift_tol"]).mean()),
                        "share_b_inside_inv_tol": float((b.abs() <= tol["inv_tol"]).mean()),
                        **{k: float(g[k].mean()) for k in g.columns if k.startswith("alpha_B")}})
        t = tests[tests.cutoff == c] if len(tests) else tests
        if len(t):
            row.update({"tests": int(len(t)), "test_readings": t.reading.value_counts().to_dict(),
                        "test_drift_rate": float((t.reading == "drift").mean())})
        out.append(row)
    return out


def summarize_stable(df: pd.DataFrame, levels=(0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for (a, n), g in df.groupby(["alpha_true", "n"]):
        err = g.alpha_hat - a
        row = {"alpha_true": a, "n": n, "replicates": len(g), "alpha_hat_mean": g.alpha_hat.mean(),
               "bias": err.mean(), "sd": g.alpha_hat.std(), "rmse": float(np.sqrt(np.mean(err ** 2))),
               "nolan_se": stable.nolan_se_alpha(n) if a < 2 else np.nan,
               "share_at_boundary_2": float((g.alpha_hat >= 2.0 - 1e-6).mean()),
               "share_converged": float(g.converged.mean()), "fit_seconds_mean": g.fit_seconds.mean()}
        for lv in levels:
            row[f"reject_gauss_{lv:g}"] = float((g.p_gauss <= lv).mean())
        row["reading"] = "size (should be near the level)" if a >= 2 else "power"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--parts", default="cascades,h71,stable", help="comma-separated subset of cascades,h71,stable")
    ap.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)),
                    help="worker processes (default: every CPU this process may use)")
    args = ap.parse_args()
    args.device = args.device or "cpu"
    cfg = config.get("p0b_calibration", args.smoke)
    run = runtime.start_run("p0b_calibration", cfg, args)
    parts = [p for p in args.parts.split(",") if p]
    bad = set(parts) - {"cascades", "h71", "stable"}
    if bad:
        raise SystemExit(f"unknown part(s) {sorted(bad)}")
    seed, S, tol = run.seed, cfg["scale"], cfg["h71_tolerances"]
    Bs = cfg["batch_sizes_h71"] if not run.smoke else cfg["h71"]["batch_sizes"]
    sc, hc, stc = cfg["cascades"], cfg["h71"], cfg["stable"]

    # every task with an estimated cost (CPU seconds measured on a 4-core cloud CPU); the longest go first so
    # that the few long H7.1 tests do not finish last on their own
    tasks = []   # (cost, part, fn, args)
    if "cascades" in parts:
        for i_p, p in enumerate(sc["p"]):
            for i_n, n in enumerate(sc["n_values"]):
                for r in range(sc["replicates"]):
                    tasks.append((44.0 * n / 65536, "cascades", t_cascade, (seed, i_p, p, i_n, n, r, S)))
    if "h71" in parts:
        for i_c, c in enumerate(hc["cutoffs"]):
            for r in range(hc["replicates"]):
                tasks.append((len(Bs) * 1.7, "h71", t_h71_rep, (seed, r, hc, Bs, i_c, c)))
            for r in range(hc["test_reps"]):
                tasks.append(((hc["test_n_boot"] + 1) * len(Bs) * 1.7, "h71_test", t_h71_test,
                              (seed, r, hc, Bs, tol, i_c, c)))
    if "stable" in parts:
        for i_a, a in enumerate(stc["alphas"]):
            for i_n, n in enumerate(stc["n_values"]):
                for r in range(stc["replicates"]):
                    tasks.append((2.0, "stable", t_stable, (seed, i_a, a, i_n, n, r)))
                for r in range(min(stc["lr_check_reps"], stc["replicates"])):
                    tasks.append(((stc["lr_check_n_boot"] + 1) * 2.0, "check", t_check,
                                  (seed, i_a, a, i_n, n, r, stc["lr_check_n_boot"])))
        for i_n, n in enumerate(stc["n_values"]):
            for j in range(stc["lr_null_draws"]):
                tasks.append((1.8, "null", t_null, (seed, i_n, n, j)))
    tasks.sort(key=lambda t: -t[0])
    est = sum(t[0] for t in tasks)
    run.note(f"{len(tasks)} tasks, about {est / 3600:.1f} CPU-hours (estimate from a 4-core cloud CPU), "
             f"{args.jobs} worker processes: roughly {est / 3600 / max(1, args.jobs):.1f} h")

    results: dict = {}
    remaining = {}
    for t in tasks:
        remaining[t[1]] = remaining.get(t[1], 0) + 1
    total = dict(remaining)
    extra: dict = {"parts": parts, "jobs": args.jobs, "tasks": total}

    def finish_part(part):
        """Write a part's files once all of its tasks are in (h71: replicates and tests; stable: fits, null, checks)."""
        rows = results.get(part, [])
        if part == "cascades":
            df = pd.DataFrame(rows).sort_values(["p", "n", "replicate"])
            runtime.write_csv(run.out_dir / "cascade_series.csv", df.to_dict("records"))
            summ = summarize_cascades(df)
            runtime.write_csv(run.out_dir / "cascade_series_summary.csv", summ.to_dict("records"))
            extra["cascades"] = summ[["p", "n", "claim_rate", "delta_h_mean", "delta_h_true"]].to_dict("records")
            run.note("cascades written")
        elif part in ("h71", "h71_test") and remaining.get("h71", 0) == 0 and remaining.get("h71_test", 0) == 0:
            reps = (pd.DataFrame(results["h71"]).sort_values(["cutoff", "replicate"]) if results.get("h71")
                    else pd.DataFrame())
            tests = (pd.DataFrame(results["h71_test"]).sort_values(["cutoff", "test"]) if results.get("h71_test")
                     else pd.DataFrame())
            runtime.write_csv(run.out_dir / "h71_trunc.csv", reps.to_dict("records"))
            runtime.write_csv(run.out_dir / "h71_trunc_tests.csv", tests.to_dict("records"))
            extra["h71_trunc"] = summarize_h71(reps, tests, tol)
            runtime.write_csv(run.out_dir / "h71_trunc_summary.csv",
                              [{k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in extra["h71_trunc"]])
            run.note("h71 written")
        elif part in ("stable", "null", "check") and all(remaining.get(k, 0) == 0 for k in ("stable", "null", "check")):
            df = pd.DataFrame(results["stable"]).sort_values(["alpha_true", "n", "replicate"])
            null = pd.DataFrame(results["null"]).sort_values(["n", "draw"])
            nulls = {n: g.lr_stat.to_numpy() for n, g in null.groupby("n")}
            df["p_gauss"] = [p_from_null(lr, nulls[n]) for lr, n in zip(df.lr_stat, df.n)]
            runtime.write_csv(run.out_dir / "stable_fit.csv", df.to_dict("records"))
            runtime.write_csv(run.out_dir / "stable_lr_null.csv", null.to_dict("records"))
            summ = summarize_stable(df)
            runtime.write_csv(run.out_dir / "stable_fit_summary.csv", summ.to_dict("records"))
            extra["stable"] = summ.drop(columns=["reading"]).to_dict("records")
            if results.get("check"):
                chk = pd.DataFrame(results["check"]).merge(df[["alpha_true", "n", "replicate", "p_gauss", "lr_stat"]],
                                                           on=["alpha_true", "n", "replicate"])
                chk = chk.rename(columns={"p_gauss": "p_shared_null"})
                runtime.write_csv(run.out_dir / "stable_lr_check.csv", chk.to_dict("records"))
                dp = (chk.p_function - chk.p_shared_null).abs()
                extra["lr_check"] = {"pairs": int(len(chk)),
                                     "max_abs_lr_difference": float((chk.lr_stat_function - chk.lr_stat).abs().max()),
                                     "median_abs_p_difference": float(dp.median()),
                                     "max_abs_p_difference": float(dp.max()),
                                     "same_decision_at_0.05": float(((chk.p_function <= 0.05)
                                                                     == (chk.p_shared_null <= 0.05)).mean())}
            run.note("stable written")

    t_start = time.perf_counter()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *a): (part, cost) for cost, part, fn, a in tasks}
        for fut in cf.as_completed(futs):
            part, cost = futs[fut]
            results.setdefault(part, []).append(fut.result())
            remaining[part] -= 1
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                run.note(f"{done}/{len(tasks)} tasks done after {(time.perf_counter() - t_start) / 3600:.2f} h")
            if remaining[part] == 0:
                finish_part(part)
    runtime.finish_run(run, extra=extra)


if __name__ == "__main__":
    main()
