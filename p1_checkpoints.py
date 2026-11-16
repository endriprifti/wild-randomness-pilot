"""P1: measurements on public checkpoints, no training (Pythia, PolyPythias, OLMo).

Per checkpoint (log-spaced steps):
  alpha, beta, gamma  stable fit to minibatch gradient noise at B = coupling_B (per-example
                      gradients of the last block, projected on fixed random directions by JVP)
  zeta (both tails)   CSN fit of the same noise, with the sec. 7.1 reading label
  alpha_ESD           median over layers of the ESD density exponent (+ per-layer CSV)
  K, Kurt0, outliers  activations of every layer on a fixed probe set, with zeta of |activation|
  Delta h_rep         width of the final-layer representations of the probe set (dimension clause)
  H_tok               EXPLORATORY: Hurst exponent of the per-token loss series along long
                      documents, with its shuffled baseline. This is NOT H_grad: checkpoints carry
                      no per-step gradient series, so H_grad comes only from P4 (appendix, P1).
At the final step: eval loss for a_N (one fit per suite and seed = one family).
H7.1 transformer arm: at 1, 10, 50, 100% of training, pools of n_blocks * max(B) examples for
Pythia-14M (1,600 blocks) and coarse sizes (200 blocks), random and Hessian-block directions.
Finally an exploratory coupling matrix across seeds within each (suite, size) cell.

Suites (config.py): pythia (seed 0, 14M-1.4B); polypythia = pythia-{size}-seed1..9 (initialization and
batch composition vary jointly; run-to-run variance); polypythia_data / polypythia_weight = the 160M
data-seed1-3 and weight-seed1-3 runs (data order alone, initialization alone); olmo = OLMo 2 1B with its
stage-1 checkpoints; olmo7b = OLMo 2 7B, OPTIONAL and never run by default (L40S, last job).
Every model is loaded at a commit pinned in pins.json; checkpoints.csv records commit and commit date. For
pythia-14m/31m (re-created with the standard-Pile models on 24 February 2026; the deduplicated ones moved to
pythia-14m-deduped / pythia-31m-deduped) two checks: dedup_check flags a commit whose model.safetensors has the
same LFS sha256 as the same revision of the -deduped repository (Hub API metadata, no download), and
name_switch_check flags a commit dated before 24 February 2026. --pin-only runs both for every pinned name and
writes name_switch_check.csv.
Loss-spike check (PolyPythias 410M seeds 3 and 4 had loss spikes), two passes:
  (a) probe loss only, at every 1,000-step checkpoint, for seeds 3 and 4 and one stable seed; a jump is a
      loss increase, net of the stable seed's change, with robust z > 3.5 over the run's intervals;
  (b) alpha, alpha_ESD and K for all nine 410M seeds at every 1,000-step checkpoint within 5,000 steps
      either side of each jump; a flagged seed's change counts as a jump if it exceeds every other seed's
      and is more than 3.5 robust z above them. spike_check.csv holds both passes and the jump steps.
Each checkpoint is deleted from the Hugging Face cache after it is read (--keep-cache keeps them).

Outputs (results/p1_checkpoints/[smoke/]): checkpoints.csv, esd_layers.csv, activations_layers.csv,
h71_per_B.csv, h71_summary.csv, scaling.csv, seed_variance.csv, spike_check.csv, coupling_pairs.csv,
coupling.json, run.json; with --pin-only: name_switch_check.csv and run.json

Commands:
  python p1_checkpoints.py --pin-only            # resolve and record every commit first (CPU is enough);
                                                 # run.json then gives the download volume of each pass
  python p1_checkpoints.py --suite pythia
  python p1_checkpoints.py --suite polypythia,polypythia_data,polypythia_weight   # includes both spike passes
  python p1_checkpoints.py --suite polypythia --spike-only                      # only the spike passes
  python p1_checkpoints.py --suite olmo           # OLMo 2 1B, fits the RTX 5090
  python p1_checkpoints.py --suite olmo7b         # OPTIONAL, last: OLMo 2 7B on the L40S
  python p1_checkpoints.py --smoke
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time

import numpy as np
import pandas as pd
import torch

import config
from common import batch as batchmod
from common import coupling, gradients, hf, hurst, lm, multifractal, runtime, spectra, stable, tails


def suite_models(cfg, suite):
    s = cfg["suites"][suite]
    for size in s["sizes"]:
        for seed in s["seeds"]:
            repo = s["repo"].format(size=size, seed=seed)
            yield size, seed, repo, s.get("variant", "base")


def default_suites(cfg) -> list[str]:
    return [k for k, v in cfg["suites"].items() if not v.get("optional")]


def wanted_steps(cfg, suite: str, size: str, steps_avail: dict) -> list[int]:
    """Steps of the main grid for one model (the loss-spike passes choose their own)."""
    s = cfg["suites"][suite]
    final = max(steps_avail)
    if "step_fracs" in s:   # OLMo: fractions of the last pretraining checkpoint
        wanted = [max(1, int(round(f * final))) for f in s["step_fracs"]]
    else:
        wanted = list(cfg["steps"])
    h71 = cfg["h71"]
    if size in h71["full_res_sizes"] + h71["coarse_sizes"] and suite == "pythia":
        wanted += [int(round(st * final)) for st in h71.get("stages", cfg["stages"])]
    return sorted(set(wanted))


def scan_grid(sc: dict) -> list[int]:
    lo, hi = sc["scan_range"]
    return list(range(lo, hi + 1, sc["scan_every"]))


def scan_steps(avail: dict, sc: dict) -> list[int]:
    """Pass (a) steps: every released checkpoint on the scan grid (multiples of scan_every in scan_range)."""
    lo, hi = sc["scan_range"]
    return [s for s in sorted(avail) if lo <= s <= hi and s % sc["scan_every"] == 0]


def loss_jumps(df: pd.DataFrame, sc: dict) -> list[dict]:
    """Pass (a). For each flagged seed, the change in probe loss between consecutive scan checkpoints minus
    the reference seeds' mean change over the same interval (this removes the common downward trend);
    robust z = (excess - median) / (1.4826 MAD) over all intervals of that seed. A jump is a loss
    increase with z > sc['robust_z']."""
    if not len(df):
        return []
    d = df[(df.suite == sc["suite"]) & (df["size"] == sc["size"])]
    piv = d.pivot_table(index="step", columns="seed", values="probe_loss", aggfunc="first").sort_index()
    steps = piv.index.to_list()
    diffs = piv.diff()
    refs = [r for r in sc["reference_seeds"] if r in diffs.columns]
    ref = diffs[refs].mean(axis=1) if refs else pd.Series(0.0, index=diffs.index)
    out = []
    for s in sc["flagged_seeds"]:
        if s not in diffs.columns:
            continue
        excess = (diffs[s] - ref).iloc[1:]
        ok = excess.dropna().to_numpy(float)
        med = float(np.median(ok)) if len(ok) else np.nan
        mad = float(1.4826 * np.median(np.abs(ok - med))) if len(ok) else np.nan
        for i in range(1, len(steps)):
            e = excess.iloc[i - 1]
            z = (e - med) / mad if np.isfinite(e) and np.isfinite(mad) and mad > 0 else np.nan
            out.append({"pass": "a", "quantity": "probe_loss", "seed": s, "step_from": steps[i - 1], "step_to": steps[i],
                        "value_from": float(piv.iloc[i - 1][s]), "value_to": float(piv.iloc[i][s]),
                        "dq": float(diffs[s].iloc[i]), "reference_dq": float(ref.iloc[i]), "excess_dq": float(e),
                        "robust_z": float(z), "jump": bool(np.isfinite(z) and z > sc["robust_z"])})
    return out


def merge_windows(jumps: list[tuple[int, int]], half: int) -> list[tuple[int, int, list]]:
    """[step_from - half, step_to + half] around each jump; overlapping windows merge."""
    out: list = []
    for a, b in sorted(jumps):
        lo, hi = a - half, b + half
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi), out[-1][2] + [(a, b)])
        else:
            out.append((lo, hi, [(a, b)]))
    return out


def name_switch_flag(cfg, repo: str, commit_date: str | None) -> str:
    ns = cfg.get("name_switch") or {}
    if repo not in ns.get("repos", []):
        return ""
    if not commit_date:
        return "unknown: no commit date"
    if commit_date[:10] < ns["date"]:
        return "CHECK: commit predates the name switch (may be the deduplicated model)"
    return "after switch"


def dedup_content_flag(cfg, repo: str, rev: str, commit: str | None, lookup) -> dict:
    """Content check of the name switch, independent of dates: the LFS sha256 of the weights at the pinned commit
    against the same revision (step branch) of repo + '-deduped'. lookup(repo, revision) -> (commit,
    {filename: sha256}) reads Hub metadata only. Flagged ("CHECK") only if the files are identical; files that
    differ are not byte-identical, which is all a hash can say. {} for repos without a name switch."""
    ns = cfg.get("name_switch") or {}
    if repo not in ns.get("repos", []):
        return {}
    other = repo + ns.get("deduped_suffix", "-deduped")
    out = {"dedup_repo": other, "dedup_revision": rev}
    try:
        _, mine = lookup(repo, commit or rev)
        out["dedup_commit"], theirs = lookup(other, rev)
    except Exception as e:  # noqa: BLE001  a missing branch or no network: report, do not stop the run
        return {**out, "dedup_check": f"unknown: Hub lookup failed ({type(e).__name__}: {str(e)[:100]})"}
    fname, same = hf.compare_weights(mine, theirs, ns.get("content_files", hf.WEIGHTS_FILES))
    if fname is None:
        return {**out, "dedup_check": f"unknown: no weights file in common (pinned: {sorted(mine) or 'none'}, "
                                      f"{other}: {sorted(theirs) or 'none'})"}
    out.update(weights_file=fname, weights_sha256=mine[fname], dedup_sha256=theirs[fname])
    out["dedup_check"] = (f"CHECK: {fname} identical to {other}@{rev} (the deduplicated model)" if same
                          else f"differs from {other}@{rev}")
    return out


def content_check(cfg, repo: str, rev: str, pins, lookup) -> dict:
    """dedup_content_flag with its result cached in the pin entry (pins.json: deduped_check), so a committed
    pins.json carries the evidence and later runs reuse it. Lookup failures are not cached."""
    entry = pins.entry(repo, rev)
    if entry and "deduped_check" in entry:
        return dict(entry["deduped_check"])
    res = dedup_content_flag(cfg, repo, rev, (entry or {}).get("commit"), lookup)
    if res and entry and not res["dedup_check"].startswith("unknown"):
        res["dedup_checked_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        entry["deduped_check"] = dict(res)
        pins.save()
    return res


def spike_check(df: pd.DataFrame, sc: dict) -> list[dict]:
    """Pass (b). Per quantity, |change| between consecutive steps of each flagged seed against the other
    seeds of the same size over the same interval: robust z = (|dq| - median) / (1.4826 MAD) of the
    other seeds' |dq|; a jump needs |dq| above every other seed and z > sc['robust_z']."""
    d = df[(df.suite == sc["suite"]) & (df["size"] == sc["size"])]
    flagged = list(sc["flagged_seeds"])
    out = []
    for q in sc["quantities"]:
        if q not in d or d[q].notna().sum() == 0:
            continue
        piv = d.pivot_table(index="step", columns="seed", values=q, aggfunc="first").sort_index()
        steps = piv.index.to_list()
        diffs = piv.diff()
        for i in range(1, len(steps)):
            r = diffs.iloc[i]
            others = np.abs(r.drop(labels=flagged, errors="ignore").dropna().to_numpy(float))
            med = float(np.median(others)) if len(others) else np.nan
            mad = float(1.4826 * np.median(np.abs(others - med))) if len(others) else np.nan
            for s in flagged:
                if s not in r.index or not np.isfinite(r[s]):
                    continue
                a = abs(float(r[s]))
                z = (a - med) / mad if np.isfinite(mad) and mad > 0 else np.nan
                jump = bool(len(others) >= 3 and a > others.max() and np.isfinite(z) and z > sc["robust_z"])
                out.append({"quantity": q, "seed": s, "step_from": steps[i - 1], "step_to": steps[i],
                            "value_from": float(piv.iloc[i - 1][s]), "value_to": float(piv.iloc[i][s]),
                            "dq": float(r[s]), "others_n": int(len(others)), "others_median_abs_dq": med,
                            "others_max_abs_dq": float(others.max()) if len(others) else np.nan,
                            "robust_z": z, "jump": jump})
    return out


SEED_VAR_QS = ["probe_loss", "alpha", "zeta_right", "zeta_left", "alpha_esd", "K", "Kurt0", "H_dep", "dh_rep", "eval_loss"]


def seed_variance(df: pd.DataFrame) -> list[dict]:
    """Spread across seeds per (variant, size, step): joint (init and batch composition), data order
    alone, initialization alone. Only (size, step) pairs that more than one variant reached."""
    d = df[df.variant.isin(["joint", "data", "weight"])]
    out = []
    for (size, step), g in d.groupby(["size", "step"]):
        if g.variant.nunique() < 2:
            continue
        for q in [q for q in SEED_VAR_QS if q in g]:
            sds = {}
            for var, h in g.groupby("variant"):
                v = h[q].dropna().to_numpy(float)
                sds[var] = (len(v), float(np.mean(v)) if len(v) else np.nan,
                            float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan)
            ref = sds.get("joint", (0, np.nan, np.nan))[2]
            for var, (n, m, sd) in sds.items():
                out.append({"size": size, "step": step, "quantity": q, "variant": var, "n_seeds": n,
                            "mean": m, "sd": sd, "sd_ratio_to_joint": sd / ref if ref and np.isfinite(ref) else np.nan})
    return out


def jvp_projections(model, ids, params, dirs, device, batch):
    outs = []
    for s in range(0, len(ids), batch):
        x = ids[s:s + batch].to(device)
        outs.append(gradients.per_example_projections(model, lm.per_example_lm_loss, x, params, dirs).cpu())
    return torch.cat(outs).numpy()


def directions(model, ids_hess, device, n_random, n_hessian, seed):
    names = lm.grad_param_names(model)
    params = {n: p for n, p in model.named_parameters() if n in names}
    gen = torch.Generator().manual_seed(seed)
    dirs = [("random", i, d) for i, d in enumerate(gradients.random_directions(params, n_random, gen))]
    if n_hessian > 0:
        hvp, p0 = gradients.hvp_fn(model, lm.per_example_lm_loss, ids_hess.to(device), params)
        vals, vecs = gradients.top_eigvecs(hvp, p0, k=n_hessian, iters=20, generator=gen)
        dirs += [("hessian", i, v) for i, v in enumerate(vecs)]
    return params, dirs


def noise_quantities(values: np.ndarray, B: int, n_blocks: int, T: dict, rng) -> dict:
    noise = stable.block_noise(values, B, max_blocks=n_blocks)
    out = {"n_blocks": len(noise)}
    if len(noise) < 20:
        return out
    f = stable.fit_stable(noise)
    out.update({"alpha": f.alpha, "beta": f.beta, "gamma": f.gamma})
    for side, v in (("right", noise[noise > 0]), ("left", -noise[noise < 0])):
        try:
            rd = tails.read_tail(v, n_boot=T["n_boot"], min_tail=min(T["min_tail"], max(10, len(v) // 10)),
                                 max_candidates=T["max_candidates"], hill_reps=T["hill_reps"],
                                 hill_tol=T["hill_tol"], hill_eps=T["hill_eps"], n_jobs=T["n_jobs"], rng=rng)
            out[f"zeta_{side}"] = rd.fit.zeta
            out[f"tail_label_{side}"] = rd.label
            out[f"tail_reason_{side}"] = rd.reason
            out[f"p_ks_{side}"] = rd.p_ks
        except ValueError:
            pass
    return out


def gpu_lock(args):
    """One P1 job on the GPU at a time (an exclusive lock on logs/.p1_gpu.lock, held until the process ends).
    run_lane.sh runs P1 suites next to run_all.sh; the lock keeps them from overlapping each other or run_all.sh's
    p1_olmo7b (31.5 GiB by memory_check.py, which with a 17.3 GiB suite would exceed the L40S's 44 GiB): the later
    job waits. Not taken under --smoke, --pin-only or --device cpu. Returns (open lock file, seconds waited)."""
    if args.smoke or args.pin_only or (args.device or "").startswith("cpu"):
        return None, 0.0
    try:
        import fcntl
    except ImportError:
        return None, 0.0
    path = runtime.PILOT_ROOT / "logs" / ".p1_gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    t0 = time.perf_counter()
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[p1_checkpoints] another P1 job holds {path}; waiting for it to finish", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f, time.perf_counter() - t0


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--suite", default=None, help="comma-separated suites from config.py (default: every suite "
                                                  "not marked optional, so olmo7b runs only when named)")
    ap.add_argument("--pin-only", action="store_true", help="resolve and record the commit of every model and "
                                                            "revision in pins.json, then stop (no measurement)")
    ap.add_argument("--sizes", default=None, help="comma-separated subset of sizes")
    ap.add_argument("--spike-only", action="store_true", help="run only the two loss-spike passes")
    ap.add_argument("--keep-cache", action="store_true", help="keep every checkpoint in the Hugging Face cache "
                                                              "(default: delete each one after it is read; "
                                                              "--smoke always keeps them)")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: "
                                               "grad,esd,act,dh,htok,h71,scaling,spike,coupling")
    ap.add_argument("--jvp-batch", type=int, default=None,
                    help="sequences per JVP micro-batch (default: the suite's jvp_batch in config.py, else 32)")
    args = ap.parse_args()
    cfg = config.get("p1_checkpoints", args.smoke)
    _held, waited = gpu_lock(args)               # before CUDA starts: a waiting job holds no GPU memory; the lock
                                                 # lasts as long as this process (and _held)
    run = runtime.start_run("p1_checkpoints", cfg, args)
    if waited > 1:
        run.note(f"waited {waited / 3600:.2f} h for the P1 GPU lock (logs/.p1_gpu.lock); progress.txt's elapsed "
                 f"hours for this job include the wait")
    rng = runtime.rng(run.seed)
    dev = run.device
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    skip = set(filter(None, args.skip.split(",")))
    T = cfg["tails"]
    S = cfg["scale"]
    suites = args.suite.split(",") if args.suite else default_suites(cfg)
    unknown = [x for x in suites if x not in cfg["suites"]]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; config.py has {list(cfg['suites'])}")
    h71cfg = cfg["h71"]
    sc = cfg.get("spike_check") or {}
    h71_Bs = h71cfg.get("batch_sizes", cfg["batch_sizes_h71"])
    stages = h71cfg.get("stages", cfg["stages"])

    rows, esd_rows, act_rows, h71_rows, h71_summ, scale_rows = [], [], [], [], [], []
    data_cache: dict = {}

    def read_checkpoint(suite, size, seed, repo, variant, step, rev, final, skip_here, tag):
        """Load one checkpoint, measure everything not in skip_here, append and return its row."""
        model, tok, src = hf.load_causal_lm(repo, rev, run, dtype=dtype, device=dev)
        # micro-batches: sequences per no-grad forward (losses, hidden states) and per JVP; olmo7b sets smaller
        # ones so that 27 GiB of float32 weights plus activations stay well under the L40S's 44 GiB
        eb = int(cfg["suites"][suite].get("eval_batch", 4))
        jb = int(args.jvp_batch or cfg["suites"][suite].get("jvp_batch", 32))
        pin = (hf.PINS.entry(repo, rev) or {}) if src == "hub" else {}
        key = type(tok).__name__ + str(getattr(tok, "name_or_path", ""))
        if key not in data_cache:  # same examples for every checkpoint of a tokenizer
            need_pool = cfg["coupling_blocks"] * cfg["coupling_B"]
            n_h71 = h71cfg.get("n_blocks", config.H71["n_blocks"]) * max(h71_Bs)
            n_pool = max(need_pool, n_h71 if h71cfg["full_res_sizes"] else 0)
            pool_texts, _ = hf.load_texts(cfg["pool_dataset"], n=max(64, n_pool * cfg["example_len"] // 400 + 64),
                                          run=run, seed=run.seed)
            pool = hf.tokenize_stream(pool_texts, tok, cfg["example_len"], n_pool)
            ev_texts, _ = hf.load_texts(cfg["eval_dataset"], n=max(16, cfg["eval_seqs"] * cfg["eval_len"] // 400 + 16),
                                        run=run, seed=run.seed + 1)
            probe = hf.tokenize_stream(ev_texts, tok, cfg["probe_len"], cfg["probe_seqs"])
            evals = hf.tokenize_stream(ev_texts, tok, cfg["eval_len"], cfg["eval_seqs"])
            sur = hf.tokenize_stream(ev_texts[::-1], tok, cfg["surprisal_len"], cfg["surprisal_docs"])
            data_cache[key] = (pool, probe, evals, sur)
        pool, probe, evals, sur = data_cache[key]
        vocab = model.get_output_embeddings().weight.shape[0]
        pool_v, probe_v, evals_v, sur_v = (t.clamp(max=vocab - 1) for t in (pool, probe, evals, sur))
        row = {"suite": suite, "variant": variant, "size": size, "seed": seed, "repo": repo, "step": step,
               "revision": rev, "commit": pin.get("commit"), "commit_date": pin.get("commit_date"),
               "name_switch_check": name_switch_flag(cfg, repo, pin.get("commit_date")) if src == "hub" else "",
               "source": src, "pass": tag,
               "loss_spike_seed": bool(suite == sc.get("suite") and size == sc.get("size")
                                       and seed in sc.get("flagged_seeds", [])),
               "n_params": sum(p.numel() for p in model.parameters()),
               "n_nonembed": lm.non_embedding_params(model)}
        if src == "hub":
            row.update(content_check(cfg, repo, rev, hf.PINS, hf.hub_lfs))
        for col in ("dedup_check", "name_switch_check"):
            if str(row.get(col, "")).startswith("CHECK"):
                run.note(f"{repo}@{rev} ({row['commit']}): {col} {row[col]}")
        frac = step / final if final else 1.0
        row["probe_loss"] = float(lm.token_losses(model, probe_v, dev, batch=eb).mean())   # locates loss spikes

        if "esd" not in skip_here:
            a_esd, lrows = spectra.model_alpha_esd(model, rng=rng)
            row["alpha_esd"] = a_esd
            esd_rows.extend({**r, "repo": repo, "step": step} for r in lrows)
        if "act" not in skip_here or "dh" not in skip_here:
            hs = lm.hidden_states(model, probe_v, dev, batch=eb)
            if "act" not in skip_here:
                ks = []
                for li, H in enumerate(hs):
                    st = spectra.activation_stats(H)
                    act_rows.append({**st, "layer": li, "repo": repo, "step": step})
                    ks.append(st)
                row["K"] = float(np.median([s["K"] for s in ks]))
                row["K_max"] = float(np.max([s["K"] for s in ks]))
                row["Kurt0"] = float(np.median([s["Kurt0"] for s in ks]))
                row["n_outlier_channels"] = int(np.sum([s["n_outlier_channels"] for s in ks]))
                row["act_zeta"] = float(np.nanmedian([s.get("act_zeta", np.nan) for s in ks]))
            if "dh" not in skip_here:
                X = hs[-1].numpy()
                if len(X) > cfg["dh_points"]:
                    X = X[rng.choice(len(X), cfg["dh_points"], replace=False)]
                d = multifractal.delta_h_rep(X, S["q_lo"], S["q_hi"], rng=rng,
                                             n_centres=min(5000, len(X)))
                row.update({"dh_rep": d["delta_h_rep"], "dh_scope": d["scope"], "dh_D2": d["D2"],
                            "dh_clause_ok": d["clause_ok"], "dh_decades": d["decades_r"]})
            del hs
        if "htok" not in skip_here:
            series = lm.token_losses(model, sur_v, dev, batch=eb).reshape(-1).numpy()
            m = hurst.memory_claim(series, n_surr=S["n_surr"], rng=rng, wavelet_j1=S["wavelet_j1"])
            row.update({f"Htok_{k}": v for k, v in m.items()})
            row["H_dep"] = m["H_dep_dfa"]  # exploratory stand-in in the coupling matrix
        if "grad" not in skip_here:
            B = cfg["coupling_B"]
            n = cfg["coupling_blocks"] * B
            params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev, cfg["n_random_dirs"], 0, run.seed)
            P = jvp_projections(model, pool_v[:n], params, [d for _, _, d in dirs], dev, jb)
            q = noise_quantities(P[:, 0], B, cfg["coupling_blocks"], T, rng)
            row.update(q)
        # H7.1 at the four training stages
        h71_size = size in h71cfg["full_res_sizes"] or size in h71cfg["coarse_sizes"]
        stage_hit = any(abs(frac - st) < 0.02 or (st >= 0.999 and step == final) for st in stages)
        if "h71" not in skip_here and h71_size and stage_hit and suite == "pythia":
            nb = h71cfg.get("n_blocks", config.H71["n_blocks"]) if size in h71cfg["full_res_sizes"] else h71cfg["coarse_blocks"]
            n_pool = nb * max(h71_Bs)
            params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev,
                                      cfg["n_random_dirs"], cfg["n_hessian_dirs"], run.seed)
            P = jvp_projections(model, pool_v[:n_pool], params, [d for _, _, d in dirs], dev, jb)
            for j, (kind, i, _) in enumerate(dirs):
                res = batchmod.h71_test(P[:, j], h71_Bs, nb, n_boot=h71cfg.get("n_boot", config.H71["n_boot"]),
                                        rng=rng)
                meta = {"repo": repo, "commit": row["commit"], "size": size, "step": step, "stage": round(frac, 3),
                        "direction": f"{kind}{i}"}
                h71_rows.extend({**meta, **r} for r in res["per_B"])
                h71_summ.append({**meta, **res["summary"]})
        if step == final and "scaling" not in skip_here:
            row["eval_loss"] = float(lm.token_losses(model, evals_v, dev, batch=eb).mean())
        rows.append(row)
        run.note(f"{repo}@{rev} [{tag}]: done")
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        if src == "hub" and not (args.keep_cache or run.smoke):   # smoke re-reads small checkpoints: keep them
            hf.purge_revision(repo, pin.get("commit"), run)   # disk holds about one checkpoint at a time
        reads[tag] = reads.get(tag, 0) + 1
        return row


    def available(suite, repo):
        """step -> branch of one repo (offline stand-in: the configured steps plus the spike scan grid)."""
        fallback = list(cfg["steps"])
        if sc and suite == sc["suite"]:
            fallback += scan_grid(sc)
        return hf.list_steps(repo, run, fallback, branch_regex=cfg["suites"][suite].get("branch_regex", r"^step(\d+)$"))

    def pin_all(repo, revs, tag):
        for rev in revs:
            try:
                hf.PINS.resolve(repo, rev, run)
                pinned[tag].append(hf.PINS.key(repo, rev))
            except Exception as e:  # noqa: BLE001  report every missing name, then fail
                run.note(f"PIN FAILED {repo}@{rev}: {type(e).__name__}: {str(e)[:160]}")

    pinned: dict = {"grid": [], "spike_a": []}
    reads: dict = {}

    # ---- main grid
    if not args.spike_only:
        for suite in suites:
            for size, seed, repo, variant in suite_models(cfg, suite):
                if args.sizes and size not in args.sizes.split(","):
                    continue
                steps_avail = available(suite, repo)
                final = max(steps_avail)
                chosen = hf.nearest_steps(steps_avail, wanted_steps(cfg, suite, size, steps_avail))
                if args.pin_only:
                    pin_all(repo, [rev for _, rev in chosen], "grid")
                    continue
                for step, rev in chosen:
                    read_checkpoint(suite, size, seed, repo, variant, step, rev, final, skip, "grid")

    # ---- loss-spike check, pass (a): probe loss at every scan step, flagged seeds and the stable reference
    spike_rows, windows = [], []
    do_spike = bool(sc) and "spike" not in skip and sc["suite"] in suites and (
        not args.sizes or sc["size"] in args.sizes.split(","))
    if do_spike:
        variant = cfg["suites"][sc["suite"]].get("variant", "base")
        repo_of = {seed: repo for size, seed, repo, _ in suite_models(cfg, sc["suite"]) if size == sc["size"]}
        skip_a = skip | {"esd", "act", "dh", "htok", "grad", "h71", "scaling"}
        for seed in list(sc["flagged_seeds"]) + list(sc["reference_seeds"]):
            repo = repo_of[seed]
            avail = available(sc["suite"], repo)
            steps = scan_steps(avail, sc)
            if args.pin_only:
                pin_all(repo, [avail[st] for st in steps], "spike_a")
                continue
            for st in steps:
                read_checkpoint(sc["suite"], sc["size"], seed, repo, variant, st, avail[st], max(avail), skip_a, "spike_a")
        if not args.pin_only:
            df_a = pd.DataFrame([r for r in rows if r["pass"] == "spike_a"])
            a_out = loss_jumps(df_a, sc)
            spike_rows += a_out
            jumps = [(r["step_from"], r["step_to"]) for r in a_out if r["jump"]]
            if not jumps and run.smoke and sc.get("smoke_force_top") and a_out:
                top = max(a_out, key=lambda r: r["robust_z"] if np.isfinite(r["robust_z"]) else -np.inf)
                jumps = [(top["step_from"], top["step_to"])]
                run.note(f"smoke: no loss jump found; pass (b) runs around the largest z ({top['step_to']}) to exercise the code")
            windows = merge_windows(jumps, sc["window"])

            # ---- pass (b): alpha, alpha_ESD and K for every seed of that size around each jump
            skip_b = skip | {"dh", "htok", "h71", "scaling"}
            for seed, repo in repo_of.items():
                avail = available(sc["suite"], repo)
                for lo, hi, js in windows:
                    for st in [x for x in scan_steps(avail, sc) if lo <= x <= hi]:
                        row = read_checkpoint(sc["suite"], sc["size"], seed, repo, variant, st, avail[st], max(avail),
                                              skip_b, "spike_b")
                        row["spike_window"] = f"{lo}-{hi}"
            df_b = pd.DataFrame([r for r in rows if r["pass"] == "spike_b"])
            for lo, hi, js in windows:
                w = df_b[df_b.spike_window == f"{lo}-{hi}"] if len(df_b) else df_b
                if not len(w):
                    continue
                for r in spike_check(w, sc):
                    spike_rows.append({"pass": "b", "window": f"{lo}-{hi}",
                                       "loss_jump_steps": ";".join(f"{a}-{b}" for a, b in js), **r})

    extra = {}
    if args.pin_only:
        failed = [m for m in run.notes if m.startswith("PIN FAILED")]
        # name switch: content check (cached in pins.json) and date check for every pinned pythia-14m/31m name
        ns_rows = []
        for k in dict.fromkeys(k for keys in pinned.values() for k in keys):
            repo, rev = k.rsplit("@", 1)
            if repo not in (cfg.get("name_switch") or {}).get("repos", []):
                continue
            entry = hf.PINS.data.get(k) or {}
            r = {"repo": repo, "revision": rev, "commit": entry.get("commit"), "commit_date": entry.get("commit_date"),
                 "name_switch_check": name_switch_flag(cfg, repo, entry.get("commit_date")),
                 **content_check(cfg, repo, rev, hf.PINS, hf.hub_lfs)}
            ns_rows.append(r)
            for col in ("dedup_check", "name_switch_check"):
                if str(r.get(col, "")).startswith("CHECK"):
                    run.note(f"{k} ({r['commit']}): {col} {r[col]}")
        if ns_rows:
            runtime.write_csv(run.out_dir / "name_switch_check.csv", ns_rows)
            def verdict(v):  # CHECK / differs / unknown / after switch
                v = str(v)
                return "differs" if v.startswith("differs") else v.split(":")[0]
            extra["name_switch"] = {col: pd.Series([verdict(r.get(col, "")) for r in ns_rows]).value_counts().to_dict()
                                    for col in ("dedup_check", "name_switch_check")}
            run.note(f"name switch, {len(ns_rows)} pinned names: content {extra['name_switch']['dedup_check']}, "
                     f"date {extra['name_switch']['name_switch_check']}")
        extra["pins_resolved"] = len(run.pins)
        extra["download"] = {}
        for tag, keys in pinned.items():
            sizes = [(hf.PINS.data.get(k) or {}).get("weights_bytes") for k in dict.fromkeys(keys)]
            known = [b for b in sizes if b]
            extra["download"][tag] = {"checkpoints": len(sizes), "with_size": len(known),
                                      "GB": round(sum(known) / 1e9, 1)}
            run.note(f"download for {tag}: {len(sizes)} checkpoints, {sum(known) / 1e9:.1f} GB "
                     f"({len(known)} with a known size)")
        runtime.finish_run(run, extra=extra)
        if failed:
            raise SystemExit(f"{len(failed)} name(s) could not be pinned; see run.json notes")
        return

    df = pd.DataFrame(rows)
    runtime.write_csv(run.out_dir / "checkpoints.csv", rows)
    runtime.write_csv(run.out_dir / "esd_layers.csv", esd_rows)
    runtime.write_csv(run.out_dir / "activations_layers.csv", act_rows)
    runtime.write_csv(run.out_dir / "h71_per_B.csv", h71_rows)
    runtime.write_csv(run.out_dir / "h71_summary.csv", h71_summ)
    if not len(df):
        extra["n_checkpoints"] = 0
        runtime.finish_run(run, extra=extra)
        return
    df["family"] = df["suite"] + ":" + df["seed"].astype(str)
    full = df[df["pass"] == "grid"]   # the spike passes enter only spike_check.csv

    # a_N per family (suite, seed) from final eval losses; a fit needs at least three sizes
    if "eval_loss" in full:
        fin = full.dropna(subset=["eval_loss"])
        for (suite, seed), g in fin.groupby(["suite", "seed"]):
            if g["size"].nunique() < 3:
                continue
            fit = lm.fit_scaling(g["n_nonembed"].to_numpy(), g["eval_loss"].to_numpy())
            scale_rows.append({"suite": suite, "seed": seed, "family": f"{suite}:{seed}", "n_sizes": len(g), **fit})
        runtime.write_csv(run.out_dir / "scaling.csv", scale_rows)

    # spread across seeds: joint vs data order alone vs initialization alone (160M)
    sv = seed_variance(full)
    runtime.write_csv(run.out_dir / "seed_variance.csv", sv)

    # loss-spike seeds: pass (a) loss jumps, pass (b) do alpha, alpha_ESD or K jump around them?
    if do_spike:
        runtime.write_csv(run.out_dir / "spike_check.csv", spike_rows)
        pick = ("pass", "quantity", "seed", "step_from", "step_to", "robust_z")
        extra["spike_check"] = {"suite": sc["suite"], "size": sc["size"], "flagged_seeds": sc["flagged_seeds"],
                                "reference_seeds": sc["reference_seeds"],
                                "loss_jumps": [{k: r[k] for k in pick} for r in spike_rows if r["pass"] == "a" and r["jump"]],
                                "windows": [f"{lo}-{hi}" for lo, hi, _ in windows],
                                "quantity_jumps": [{k: r[k] for k in pick} for r in spike_rows
                                                   if r["pass"] == "b" and r["jump"]]}

    # exploratory coupling across seeds within (suite, size) cells (final checkpoints); a_N per family
    if "coupling" not in skip:
        fin = full[full.step == full.groupby("repo").step.transform("max")].copy()
        sc_df = pd.DataFrame(scale_rows)
        if len(sc_df):
            fin = fin.merge(sc_df[["family", "a_N"]], on="family", how="left")
        fin = fin[fin.groupby(["suite", "size"]).repo.transform("size") >= 2]   # a one-run cell adds nothing
        qs = [q for q in ["alpha", "alpha_esd", "K", "H_dep", "dh_rep"] if q in fin and fin[q].notna().sum() >= 4]
        if len(qs) >= 2 and len(fin) and fin.groupby(["suite", "size"]).size().max() >= 3:
            res = coupling.coupling_test(fin, qs, ["suite", "size"], n_perm=200 if run.smoke else 5000,
                                         family_col="family", family_quantity="a_N" if "a_N" in fin else None, rng=rng)
            pair_rows = [{"q1": k[0], "q2": k[1], **v} for k, v in res["pairs"].items()]
            runtime.write_csv(run.out_dir / "coupling_pairs.csv", pair_rows)
            (run.out_dir / "coupling.json").write_text(json.dumps(runtime.to_jsonable(
                {k: v for k, v in res.items() if k != "pairs"}), indent=2))
            extra["coupling_primary"] = res["primary"]
        else:
            run.note("coupling matrix skipped: needs >= 3 seeds in one (suite, size) cell and >= 2 quantities")
    extra["n_checkpoints"] = len(rows)
    extra["checkpoints_read"] = reads
    runtime.finish_run(run, extra=extra)


if __name__ == "__main__":
    main()
