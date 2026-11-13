"""P0: calibrate the Section 7 rules on samples of known state (CPU only).

Outputs (results/p0_calibration/[smoke/]):
  tails.csv              one row per (law, n, replicate): label and every fitted quantity
  tails_confusion.csv    true state x assigned label counts, per n
  hurst.csv              DFA and wavelet H with shuffled-surrogate baselines on fGn and SaS Levy increments
  hurst_summary.csv      bias, spread and memory-claim rates per series and n
  cascades.csv           Delta h_rep and D_q on binomial cascade point clouds; MFDFA width on series
  block_check.csv        slope b of alpha-hat on log2 B on SaS pools (no true drift): spread of b
  run.json               config, git commit, GPU, wall-clock

Command: python p0_calibration.py [--smoke]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from common import hurst, multifractal, runtime, stable, tails
from common.batch import fit_over_batches, _slopes, h71_test


def sample_law(name: str, n: int, rng, cutoff: float) -> np.ndarray:
    kind, par = name.split("_")
    par = float(par)
    if kind == "sas":
        return np.abs(stable.rvs_stable(par, 0.0, 1.0, 0.0, n, rng))
    if kind == "pareto":
        return tails.rvs_pareto(par, 1.0, n, rng)
    if kind == "lognormal":
        return rng.lognormal(0.0, par, n)
    if kind == "weibull":
        return rng.weibull(par, n)
    if kind == "trunclevy":
        out = np.empty(0)
        while len(out) < n:
            x = stable.rvs_stable(par, 0.0, 1.0, 0.0, 2 * n, rng)
            out = np.concatenate([out, np.abs(x[np.abs(x) <= cutoff])])
        return out[:n]
    raise ValueError(name)


def run_tails(cfg, run, rng):
    rows = []
    T = cfg["tails"]
    for law, truth in cfg["tail_laws"].items():
        for n in cfg["n_values"]:
            for r in range(cfg["replicates"]):
                x = sample_law(law, n, rng, cfg["trunc_levy_cutoff"])
                try:
                    rd = tails.read_tail(x, n_boot=cfg["n_boot"], level=T["level"], min_tail=min(T["min_tail"], n // 10),
                                         max_candidates=T["max_candidates"], hill_eps=T["hill_eps"],
                                         hill_reps=T["hill_reps"], hill_tol=T["hill_tol"], n_jobs=T["n_jobs"], rng=rng)
                    row = rd.as_row()
                except ValueError as e:
                    row = {"label": "undecided", "error": str(e)}
                row.update({"law": law, "true_state": truth, "n": n, "replicate": r})
                rows.append(row)
            run.note(f"tails {law} n={n} done")
    df = pd.DataFrame(rows)
    conf = (df.groupby(["n", "true_state", "label"]).size().unstack(fill_value=0)
            .reindex(columns=list(tails.LABELS), fill_value=0).reset_index())
    return df, conf


def run_hurst(cfg, run, rng):
    rows = []
    S = cfg["scale"]
    for name, H in cfg["hurst_series"].items():
        for n in cfg["n_values"]:
            for r in range(cfg["replicates"]):
                if name.startswith("fgn"):
                    x = hurst.fgn(n, H, rng)
                    truth = H
                else:
                    x = hurst.levy_increments(n, float(name.split("_")[1]), rng)
                    truth = 0.5  # independent increments: no memory (self-similarity 1/alpha is not memory)
                m = hurst.memory_claim(x, n_surr=S["n_surr"], rng=rng, wavelet_j1=S["wavelet_j1"])
                m.update({"series": name, "n": n, "replicate": r, "H_true": truth,
                          "memory_true": bool(name.startswith("fgn") and H != 0.5)})
                rows.append(m)
            run.note(f"hurst {name} n={n} done")
    df = pd.DataFrame(rows)
    summ = df.groupby(["series", "n"]).agg(
        H_dfa_mean=("H_dfa", "mean"), H_dfa_sd=("H_dfa", "std"), H_wav_mean=("H_wav", "mean"),
        H_wav_sd=("H_wav", "std"), H_shuf_dfa=("H_shuf_dfa", "mean"), H_shuf_wav=("H_shuf_wav", "mean"),
        memory_rate_dfa=("memory_dfa", "mean"), memory_rate_wav=("memory_wav", "mean"),
        two_decades=("two_decades", "mean"), H_true=("H_true", "first")).reset_index()
    return df, summ


def run_cascades(cfg, run, rng):
    rows = []
    S = cfg["scale"]
    for p in cfg["cascade_p"]:
        for n in cfg["n_values"]:
            for r in range(max(1, cfg["replicates"] // 10)):
                X = multifractal.binomial_cascade_points(n, p, cfg["cascade_depth"], 1, rng)
                d = multifractal.delta_h_rep(X, S["q_lo"], S["q_hi"], rng=rng)
                th = multifractal.binomial_h(np.array([S["q_lo"], S["q_hi"]]), p)
                d.update({"kind": "points", "p": p, "n": n, "replicate": r, "delta_h_true": float(th[0] - th[1]),
                          "D2_true": float(multifractal.binomial_tau(2.0, p))})
                rows.append(d)
            # series: binomial multifractal series vs fGn (monofractal) with surrogate tests
            depth = int(np.log2(n))
            s = hurst.binomial_series(depth, p) if p != 0.5 else hurst.fgn(2 ** depth, 0.5, rng)
            mc = hurst.multifractal_claim(s, S["q_lo"], S["q_hi"], n_surr=S["n_surr"], rng=rng)
            th = hurst.binomial_series_h(np.array([S["q_lo"], S["q_hi"]]), p) if p != 0.5 else np.zeros(2)
            mc.update({"kind": "series", "p": p, "n": 2 ** depth, "delta_h_true": float(th[0] - th[1])})
            rows.append(mc)
        run.note(f"cascade p={p} done")
    return pd.DataFrame(rows)


def run_block_check(cfg, run, rng):
    bc = cfg["block_check"]
    Bs = cfg["batch_sizes_h71"] if not run.smoke else [8, 16, 32]
    rows = []
    for r in range(bc["replicates"]):
        pool = stable.rvs_stable(bc["alpha"], 0.0, 1.0, 0.0, bc["n_blocks"] * max(Bs), rng)
        per_b = fit_over_batches(pool, Bs, bc["n_blocks"])
        b, s, abar = _slopes(per_b)
        rows.append({"replicate": r, "b": b, "scale_exponent": s, "alpha_mean": abar,
                     "pred_scale_exponent": 1 / abar - 1})
    df = pd.DataFrame(rows)
    tol = cfg["h71_tolerances"]
    summ = {"b_q05": float(df.b.quantile(0.05)), "b_q95": float(df.b.quantile(0.95)),
            "halfwidth_90": float((df.b.quantile(0.95) - df.b.quantile(0.05)) / 2),
            "target_halfwidth": 0.02, "inv_tol": tol["inv_tol"]}
    # does the pool bootstrap reproduce that width?
    boots = []
    for r in range(bc["bootstrap_reps"]):
        pool = stable.rvs_stable(bc["alpha"], 0.0, 1.0, 0.0, bc["n_blocks"] * max(Bs), rng)
        t = h71_test(pool, Bs, bc["n_blocks"], n_boot=tol["n_boot"] if not run.smoke else 3, rng=rng)
        boots.append({**t["summary"], "replicate": r})
    return df, summ, pd.DataFrame(boots)


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--parts", default="tails,hurst,cascades,block", help="comma-separated subset")
    args = ap.parse_args()
    cfg = config.get("p0_calibration", args.smoke)
    run = runtime.start_run("p0_calibration", cfg, args)
    rng = runtime.rng(run.seed)
    parts = args.parts.split(",")
    extra = {}
    if "tails" in parts:
        df, conf = run_tails(cfg, run, rng)
        runtime.write_csv(run.out_dir / "tails.csv", df.to_dict("records"))
        runtime.write_csv(run.out_dir / "tails_confusion.csv", conf.to_dict("records"))
        extra["tail_accuracy"] = float((df.label == df.true_state).mean())
    if "hurst" in parts:
        df, summ = run_hurst(cfg, run, rng)
        runtime.write_csv(run.out_dir / "hurst.csv", df.to_dict("records"))
        runtime.write_csv(run.out_dir / "hurst_summary.csv", summ.to_dict("records"))
    if "cascades" in parts:
        runtime.write_csv(run.out_dir / "cascades.csv", run_cascades(cfg, run, rng).to_dict("records"))
    if "block" in parts:
        df, summ, boots = run_block_check(cfg, run, rng)
        runtime.write_csv(run.out_dir / "block_check.csv", df.to_dict("records"))
        runtime.write_csv(run.out_dir / "block_check_bootstrap.csv", boots.to_dict("records"))
        extra["block_check"] = summ
    runtime.finish_run(run, extra=extra)


if __name__ == "__main__":
    main()
