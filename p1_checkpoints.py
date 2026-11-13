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
Finally an exploratory coupling matrix across seeds within each size (PolyPythias).

Outputs (results/p1_checkpoints/[smoke/]): checkpoints.csv, esd_layers.csv, activations_layers.csv,
h71_per_B.csv, h71_summary.csv, scaling.csv, coupling_pairs.csv, coupling.json, run.json

Commands:
  python p1_checkpoints.py --suite pythia
  python p1_checkpoints.py --suite polypythia
  python p1_checkpoints.py --suite olmo           # needs ~48 GB for OLMo-7B
  python p1_checkpoints.py --smoke
"""

from __future__ import annotations

import json
import math

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
            yield size, seed, repo


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
            out[f"p_ks_{side}"] = rd.p_ks
        except ValueError:
            pass
    return out


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--suite", default=None, help="pythia | polypythia | olmo (default: all in config)")
    ap.add_argument("--sizes", default=None, help="comma-separated subset of sizes")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: grad,esd,act,dh,htok,h71,scaling,coupling")
    ap.add_argument("--jvp-batch", type=int, default=32)
    args = ap.parse_args()
    cfg = config.get("p1_checkpoints", args.smoke)
    run = runtime.start_run("p1_checkpoints", cfg, args)
    rng = runtime.rng(run.seed)
    dev = run.device
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    skip = set(filter(None, args.skip.split(",")))
    T = cfg["tails"]
    S = cfg["scale"]
    suites = [args.suite] if args.suite else list(cfg["suites"])
    h71cfg = cfg["h71"]
    h71_Bs = h71cfg.get("batch_sizes", cfg["batch_sizes_h71"])
    stages = h71cfg.get("stages", cfg["stages"])

    rows, esd_rows, act_rows, h71_rows, h71_summ, scale_rows = [], [], [], [], [], []
    data_cache: dict = {}

    for suite in suites:
        for size, seed, repo in suite_models(cfg, suite):
            if args.sizes and size not in args.sizes.split(","):
                continue
            # revisions available for this repo
            steps_avail = hf.list_steps(repo, run, cfg["steps"])
            final = max(steps_avail)
            wanted = list(cfg["steps"])
            if size in h71cfg["full_res_sizes"] + h71cfg["coarse_sizes"]:
                wanted += [int(round(st * final)) for st in stages]
            chosen = hf.nearest_steps(steps_avail, sorted(set(wanted)))
            for step, rev in chosen:
                model, tok, src = hf.load_causal_lm(repo, rev, run, dtype=dtype, device=dev)
                key = type(tok).__name__ + str(getattr(tok, "name_or_path", ""))
                if key not in data_cache:  # same examples for every checkpoint of a tokenizer
                    need_pool = cfg["coupling_blocks"] * cfg["coupling_B"]
                    n_h71 = h71cfg.get("n_blocks", config.H71["n_blocks"]) * max(h71_Bs)
                    n_pool = max(need_pool, n_h71 if h71cfg["full_res_sizes"] else 0)
                    pool_texts, _ = hf.load_texts(*cfg["pool_dataset"], n=max(64, n_pool * cfg["example_len"] // 400 + 64),
                                                  run=run, seed=run.seed)
                    pool = hf.tokenize_stream(pool_texts, tok, cfg["example_len"], n_pool)
                    ev_texts, _ = hf.load_texts(*cfg["eval_dataset"], n=max(16, cfg["eval_seqs"] * cfg["eval_len"] // 400 + 16),
                                                run=run, seed=run.seed + 1)
                    probe = hf.tokenize_stream(ev_texts, tok, cfg["probe_len"], cfg["probe_seqs"])
                    evals = hf.tokenize_stream(ev_texts, tok, cfg["eval_len"], cfg["eval_seqs"])
                    sur = hf.tokenize_stream(ev_texts[::-1], tok, cfg["surprisal_len"], cfg["surprisal_docs"])
                    data_cache[key] = (pool, probe, evals, sur)
                pool, probe, evals, sur = data_cache[key]
                vocab = model.get_output_embeddings().weight.shape[0]
                pool_v, probe_v, evals_v, sur_v = (t.clamp(max=vocab - 1) for t in (pool, probe, evals, sur))
                row = {"suite": suite, "size": size, "seed": seed, "repo": repo, "step": step, "revision": rev,
                       "source": src, "n_params": sum(p.numel() for p in model.parameters()),
                       "n_nonembed": lm.non_embedding_params(model)}
                frac = step / final if final else 1.0

                if "esd" not in skip:
                    a_esd, lrows = spectra.model_alpha_esd(model, rng=rng)
                    row["alpha_esd"] = a_esd
                    esd_rows += [{**r, "repo": repo, "step": step} for r in lrows]
                if "act" not in skip or "dh" not in skip:
                    hs = lm.hidden_states(model, probe_v, dev)
                    if "act" not in skip:
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
                    if "dh" not in skip:
                        X = hs[-1].numpy()
                        if len(X) > cfg["dh_points"]:
                            X = X[rng.choice(len(X), cfg["dh_points"], replace=False)]
                        d = multifractal.delta_h_rep(X, S["q_lo"], S["q_hi"], rng=rng,
                                                     n_centres=min(5000, len(X)))
                        row.update({"dh_rep": d["delta_h_rep"], "dh_scope": d["scope"], "dh_D2": d["D2"],
                                    "dh_clause_ok": d["clause_ok"], "dh_decades": d["decades_r"]})
                    del hs
                if "htok" not in skip:
                    series = lm.token_losses(model, sur_v, dev).reshape(-1).numpy()
                    m = hurst.memory_claim(series, n_surr=S["n_surr"], rng=rng, wavelet_j1=S["wavelet_j1"])
                    row.update({f"Htok_{k}": v for k, v in m.items()})
                    row["H_dep"] = m["H_dep_dfa"]  # exploratory stand-in in the coupling matrix
                if "grad" not in skip:
                    B = cfg["coupling_B"]
                    n = cfg["coupling_blocks"] * B
                    params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev, cfg["n_random_dirs"], 0, run.seed)
                    P = jvp_projections(model, pool_v[:n], params, [d for _, _, d in dirs], dev, args.jvp_batch)
                    q = noise_quantities(P[:, 0], B, cfg["coupling_blocks"], T, rng)
                    row.update(q)
                # H7.1 at the four training stages
                h71_size = size in h71cfg["full_res_sizes"] or size in h71cfg["coarse_sizes"]
                stage_hit = any(abs(frac - st) < 0.02 or (st >= 0.999 and step == final) for st in stages)
                if "h71" not in skip and h71_size and stage_hit and seed in (0, 1) and suite in ("pythia",):
                    nb = h71cfg.get("n_blocks", config.H71["n_blocks"]) if size in h71cfg["full_res_sizes"] else h71cfg["coarse_blocks"]
                    n_pool = nb * max(h71_Bs)
                    params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev,
                                              cfg["n_random_dirs"], cfg["n_hessian_dirs"], run.seed)
                    P = jvp_projections(model, pool_v[:n_pool], params, [d for _, _, d in dirs], dev, args.jvp_batch)
                    for j, (kind, i, _) in enumerate(dirs):
                        res = batchmod.h71_test(P[:, j], h71_Bs, nb, n_boot=h71cfg.get("n_boot", config.H71["n_boot"]),
                                                rng=rng)
                        meta = {"repo": repo, "size": size, "step": step, "stage": round(frac, 3),
                                "direction": f"{kind}{i}"}
                        h71_rows += [{**meta, **r} for r in res["per_B"]]
                        h71_summ.append({**meta, **res["summary"]})
                if step == final and "scaling" not in skip:
                    row["eval_loss"] = float(lm.token_losses(model, evals_v, dev).mean())
                rows.append(row)
                run.note(f"{repo}@{rev}: done")
                del model
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    runtime.write_csv(run.out_dir / "checkpoints.csv", rows)
    runtime.write_csv(run.out_dir / "esd_layers.csv", esd_rows)
    runtime.write_csv(run.out_dir / "activations_layers.csv", act_rows)
    runtime.write_csv(run.out_dir / "h71_per_B.csv", h71_rows)
    runtime.write_csv(run.out_dir / "h71_summary.csv", h71_summ)

    # a_N per family (suite, seed) from final eval losses
    extra = {}
    if "eval_loss" in df:
        fin = df.dropna(subset=["eval_loss"])
        for (suite, seed), g in fin.groupby(["suite", "seed"]):
            fit = lm.fit_scaling(g["n_nonembed"].to_numpy(), g["eval_loss"].to_numpy())
            scale_rows.append({"suite": suite, "seed": seed, "n_sizes": len(g), **fit})
        runtime.write_csv(run.out_dir / "scaling.csv", scale_rows)

    # exploratory coupling across seeds within sizes (final checkpoints)
    if "coupling" not in skip and len(df):
        fin = df[df.step == df.groupby("repo").step.transform("max")].copy()
        sc = pd.DataFrame(scale_rows)
        if len(sc):
            fin = fin.merge(sc[["suite", "seed", "a_N"]], on=["suite", "seed"], how="left")
        qs = [q for q in ["alpha", "alpha_esd", "K", "H_dep", "dh_rep"] if q in fin and fin[q].notna().sum() >= 4]
        if len(qs) >= 2 and fin.groupby("size").size().max() >= 3:
            res = coupling.coupling_test(fin, qs, ["size"], n_perm=200 if run.smoke else 5000,
                                         family_col="seed", family_quantity="a_N" if "a_N" in fin else None, rng=rng)
            pair_rows = [{"q1": k[0], "q2": k[1], **v} for k, v in res["pairs"].items()]
            runtime.write_csv(run.out_dir / "coupling_pairs.csv", pair_rows)
            (run.out_dir / "coupling.json").write_text(json.dumps(runtime.to_jsonable(
                {k: v for k, v in res.items() if k != "pairs"}), indent=2))
            extra["coupling_primary"] = res["primary"]
        else:
            run.note("coupling matrix skipped: needs >= 3 seeds of one size and >= 2 quantities")
    extra["n_checkpoints"] = len(rows)
    runtime.finish_run(run, extra=extra)


if __name__ == "__main__":
    main()
