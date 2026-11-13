"""P4: H6.1 threshold sweep, the convolutional control of H7.1, and H_grad.

Models: a transformer of Pythia-14M shape on FineWeb-Edu, and ResNet-18 on CIFAR-10 (the
near-Gaussian control). Arms: fixed thresholds set so that a fraction p_clip of steps is
clipped (p_clip = 0 ... 1), AutoClip, ZClip and AGC; with matched_update_length every arm is
rerun with its learning rate rescaled so that its mean update length equals the unclipped
run's (same model and seed). Five seeds.

Per run: final train/test loss and the gap G, OOD loss (WikiText-2 for the transformer, CIFAR-10
with Gaussian pixel noise for ResNet-18), divergences, realized p_clip, mean update length,
Lambda_max (power iteration), Hessian trace (Hutchinson), adaptive sharpness, and the coupling
quantities at the final checkpoint: alpha and zeta of unclipped minibatch noise at the training
batch size, alpha_ESD, K, Delta h_rep and H_grad (DFA and wavelet on per-step projections of the
unclipped minibatch gradient on fixed random directions over the constant-eta phase), entered as
|H_grad - H_shuf|.

H7.1 control: on one designated run, checkpoints at 1, 10, 50, 100% of training; per-example
gradient projections (JVP, BatchNorm in eval mode) on n_blocks * max(B) augmented draws.
CIFAR-10 has 50,000 images, so at B > 31 the 1,600 disjoint blocks reuse base images through
augmentation; base-image ids are saved so the analysis can check the effect.

Outputs (results/p4_clipping/[smoke/]): runs.csv, hgrad_series/*.npy, h71_per_B.csv,
h71_summary.csv, run.json

Commands:
  python p4_clipping.py --model transformer
  python p4_clipping.py --model resnet18
  python p4_clipping.py --smoke
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from common import batch as batchmod
from common import clipping, gradients, hf, hurst, lm, multifractal, runtime, spectra, stable, tails


# ----------------------------------------------------------------------------- data and models
def transformer_data(cfg, run):
    c = cfg["transformer"]
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(c["tokenizer"], revision=hf.PINS.resolve(c["tokenizer"], None, run))
    except Exception as e:
        if not run.offline_ok:
            raise
        run.fallback(f"tokenizer {c['tokenizer']}: byte tokenizer ({type(e).__name__})")
        tok = hf.ByteTokenizer(512)
    n_train = c["tokens"] // c["seq_len"]
    n_eval = c["eval_seqs"]
    texts, src = hf.load_texts(*c["dataset"], n=max(64, (n_train + n_eval) * c["seq_len"] // 300 + 64), run=run, seed=run.seed)
    ids = hf.tokenize_stream(texts, tok, c["seq_len"], n_train + n_eval)
    ood_texts, _ = hf.load_texts(*c["ood_dataset"], n=2000, run=run, min_chars=1, seed=run.seed + 7)
    ood = hf.tokenize_stream(["\n".join(ood_texts)], tok, c["seq_len"], n_eval)
    return {"train": ids[:-n_eval], "test": ids[-n_eval:], "ood": ood, "vocab": tokenizer_vocab(tok), "source": src}


def tokenizer_vocab(tok) -> int:
    """Number of token IDs the tokenizer can emit. len(tok) counts the added tokens; tok.vocab_size does not
    (Pythia: vocab_size 50,254 but len 50,277, and the added whitespace tokens 50,254-50,276 occur in web text)."""
    try:
        n = len(tok)
    except TypeError:
        n = 0
    return int(max(n, getattr(tok, "vocab_size", 0) or 0))


def check_token_ids(data: dict, vocab_size: int) -> None:
    """Fail on the CPU with a readable message, not with a CUDA device-side assert, if a token ID has no row
    in the embedding and logits."""
    for split in ("train", "test", "ood"):
        x = data.get(split)
        if torch.is_tensor(x) and x.numel():
            top = int(x.max())
            if top >= vocab_size:
                raise ValueError(f"token ID {top} in the {split} split is outside the model's vocabulary of "
                                 f"{vocab_size}: the tokenizer emits {data.get('vocab')} IDs (len(tok)); "
                                 f"build the model with vocab_size >= len(tok)")


def build_transformer(cfg, vocab, seed):
    neox = dict(getattr(hf, cfg["transformer"]["neox"]))
    neox["vocab_size"] = max(int(neox["vocab_size"]), int(vocab))   # config slots, or every ID the tokenizer emits
    neox["max_position_embeddings"] = max(neox.get("max_position_embeddings", 2048), cfg["transformer"]["seq_len"])
    return hf.neox_from_config(neox, seed=seed)


def cifar_data(cfg, run):
    c = cfg["resnet18"]
    try:
        Xtr, ytr, Xte, yte, src = hf.load_cifar10(run, str(runtime.PILOT_ROOT / "data"),
                                                  c.get("limit_train"), c.get("limit_test"))
        Xtr = torch.tensor(Xtr).permute(0, 3, 1, 2).float() / 255
        Xte = torch.tensor(Xte).permute(0, 3, 1, 2).float() / 255
        ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    except RuntimeError as e:
        if not run.offline_ok:
            raise
        run.fallback(f"CIFAR-10: random images ({type(e).__name__})")
        Xtr, ytr = torch.rand(c.get("limit_train", 512), 3, 32, 32), torch.randint(0, 10, (c.get("limit_train", 512),))
        Xte, yte = torch.rand(c.get("limit_test", 128), 3, 32, 32), torch.randint(0, 10, (c.get("limit_test", 128),))
        src = "fallback"
    if "limit_train" in c:
        Xtr, ytr, Xte, yte = Xtr[:c["limit_train"]], ytr[:c["limit_train"]], Xte[:c["limit_test"]], yte[:c["limit_test"]]
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    norm = lambda x: (x - mean) / std
    g = torch.Generator().manual_seed(run.seed + 11)
    Xood = (Xte + c["ood_noise_sigma"] * torch.randn(Xte.shape, generator=g)).clamp(0, 1)
    return {"train": (norm(Xtr), ytr), "test": (norm(Xte), yte), "ood": (norm(Xood), yte), "source": src}


def augment(x, g):
    """Random crop (pad 4) and horizontal flip, per image."""
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), generator=g)
    j = torch.randint(0, 9, (n,), generator=g)
    flip = torch.rand(n, generator=g) < 0.5
    out = torch.stack([xp[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32] for k in range(n)])
    out[flip] = out[flip].flip(-1)
    return out


def build_resnet(seed):
    import torchvision
    torch.manual_seed(seed)
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def resnet_pel(run, batch):
    x, y = batch
    return F.cross_entropy(run(x), y, reduction="none")


# ----------------------------------------------------------------------------- training
def flat_dirs(model, k, seed, device):
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    dirs = []
    for _ in range(k):
        d = [torch.randn(p.shape, generator=g).to(device) for p in params]
        n = math.sqrt(sum(float((t ** 2).sum()) for t in d))
        dirs.append([t / n for t in d])
    return dirs


def train(kind, cfg, data, rule, p_clip, seed, lr_scale, device, run, save_stages=None, ref_norms=None):
    runtime.set_all_seeds(seed)
    is_lm = kind == "transformer"
    c = cfg["transformer"] if is_lm else cfg["resnet18"]
    model = build_transformer(cfg, data["vocab"], seed) if is_lm else build_resnet(seed)
    if is_lm:
        check_token_ids(data, model.config.vocab_size)
    model = model.to(device)
    model.train()
    lr = c["lr"] * lr_scale
    if is_lm:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"], betas=(0.9, 0.95))
        n_train = len(data["train"])
        steps = max(1, n_train // c["batch"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=c["momentum"], weight_decay=c["weight_decay"])
        n_train = len(data["train"][0])
        steps = max(1, c["epochs"] * n_train // c["batch"])
    warm = max(1, int(c["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))  # then constant eta
    clip = clipping.Clipper(rule, p_clip, cfg["calib_steps"], cfg.get("autoclip_pct", 10.0), cfg.get("zclip_z", 2.5),
                            cfg.get("zclip_beta", 0.97), cfg.get("agc_lambda", 0.01))
    if rule == "fixed" and ref_norms is not None and len(ref_norms):
        # g_max from the unclipped run of the same model and seed (post-warmup norms), so that about
        # a fraction p_clip of steps is clipped over the whole run; p_clip = 1 clips every step
        clip.g_max = 0.5 * float(np.min(ref_norms)) if p_clip >= 1.0 else float(np.quantile(ref_norms, 1 - p_clip))
    dirs = flat_dirs(model, cfg["n_hgrad_dirs"], seed + 999, device)
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_train, generator=g)
    proj, upd, losses = [], [], []
    stage_ckpts = {}
    diverged = False
    loss0 = None
    t0 = time.perf_counter()
    for step in range(steps):
        lo = (step * c["batch"]) % n_train
        idx = order[lo:lo + c["batch"]]
        if len(idx) < c["batch"]:
            order = torch.randperm(n_train, generator=g)
            idx = order[:c["batch"]]
        if is_lm:
            x = data["train"][idx].to(device).long()
            loss = lm.per_example_lm_loss(lambda **kw: model(**kw), x).mean()
        else:
            x = augment(data["train"][0][idx], g).to(device)
            y = data["train"][1][idx].to(device)
            loss = F.cross_entropy(model(x), y)
        lval = loss.item()
        if not math.isfinite(lval) or (loss0 is not None and step > warm and lval > cfg.get("diverge_factor", 3.0) * loss0):
            diverged = True
            break
        loss0 = loss0 if loss0 is not None else lval
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # unclipped minibatch-gradient projections for H_grad (before clipping)
        proj.append([sum(float((p.grad * d).sum()) for p, d in zip(params, dv) if p.grad is not None) for dv in dirs])
        clip(model)
        before = [p.detach().clone() for p in params]
        opt.step()
        sched.step()
        upd.append(math.sqrt(sum(float(((p.detach() - b) ** 2).sum()) for p, b in zip(params, before))))
        losses.append(lval)
        if save_stages:
            for st in save_stages:
                if step + 1 == max(1, int(round(st * steps))):
                    stage_ckpts[st] = copy.deepcopy(model.state_dict())
    return model, {"steps": steps, "warmup_steps": warm, "diverged": diverged, "p_clip_realized": clip.stats.p_clip,
                   "mean_update_len": float(np.mean(upd)) if upd else float("nan"), "lr": lr,
                   "final_train_loss_ema": float(np.mean(losses[-max(1, len(losses) // 20):])) if losses else float("nan"),
                   "wall_s": time.perf_counter() - t0, "g_max": clip.g_max,
                   "norms_post_warmup": clip.stats.norms[warm:]}, np.array(proj), stage_ckpts


# ----------------------------------------------------------------------------- measurement
@torch.no_grad()
def mean_loss(kind, model, X, device, bs=64):
    model.eval()
    tot, n, err = 0.0, 0, 0
    if kind == "transformer":
        for s in range(0, len(X), bs):
            l = lm.per_example_lm_loss(lambda **kw: model(**kw), X[s:s + bs].to(device).long())
            tot += float(l.sum()); n += len(l)
        model.train()
        return tot / n, float("nan")
    Xs, ys = X
    for s in range(0, len(Xs), bs):
        out = model(Xs[s:s + bs].to(device))
        y = ys[s:s + bs].to(device)
        tot += float(F.cross_entropy(out, y, reduction="sum")); n += len(y)
        err += int((out.argmax(-1) != y).sum())
    model.train()
    return tot / n, err / n


def measure(kind, model, data, cfg, run, rng, device, proj, warm):
    M = cfg["measure"]
    T, S = cfg["tails"], cfg["scale"]
    out = {}
    is_lm = kind == "transformer"
    pel = lm.per_example_lm_loss if is_lm else resnet_pel
    tr = data["train"][:len(data["test"])] if is_lm else (data["train"][0][:len(data["test"][0])], data["train"][1][:len(data["test"][0])])
    out["train_loss"], out["train_err"] = mean_loss(kind, model, tr, device)
    out["test_loss"], out["test_err"] = mean_loss(kind, model, data["test"], device)
    out["gap"] = out["test_loss"] - out["train_loss"]
    out["ood_loss"], out["ood_err"] = mean_loss(kind, model, data["ood"], device)
    model.eval()
    hb = M["hessian_batch"]
    hbatch = data["train"][:hb].to(device).long() if is_lm else (data["train"][0][:hb].to(device), data["train"][1][:hb].to(device))
    params = gradients.select_params(model, None)
    hvp, p0 = gradients.hvp_fn(model, pel, hbatch, params)
    gen = torch.Generator().manual_seed(run.seed)
    vals, _ = gradients.top_eigvecs(hvp, p0, 1, M["power_iters"], gen)
    out["lambda_max"] = vals[0]
    out["hessian_trace"] = gradients.hutchinson_trace(hvp, p0, M["trace_samples"], gen)
    out["adaptive_sharpness"] = gradients.adaptive_sharpness(model, pel, hbatch, M["sharp_rho"], M["sharp_steps"])
    # alpha / zeta of unclipped minibatch noise at the training batch size, one random direction
    B = cfg["transformer"]["batch"] if is_lm else cfg["resnet18"]["batch"]
    n_avail = len(data["train"]) if is_lm else len(data["train"][0])
    n_pool = min(M["alpha_pool_blocks"] * B, n_avail)
    sel = gradients.select_params(model, None)
    d = gradients.random_directions(sel, 1, gen)
    vals_p = []
    bs = 32
    for s in range(0, n_pool, bs):
        b = data["train"][s:s + bs].to(device).long() if is_lm else (data["train"][0][s:s + bs].to(device), data["train"][1][s:s + bs].to(device))
        vals_p.append(gradients.per_example_projections(model, pel, b, sel, d)[:, 0].cpu())
    v = torch.cat(vals_p).numpy()
    noise = stable.block_noise(v, B)
    out["alpha_blocks"] = len(noise)
    if len(noise) >= 20:
        f = stable.fit_stable(noise)
        out.update({"alpha": f.alpha, "beta": f.beta, "gamma": f.gamma})
        try:
            rd = tails.read_tail(np.abs(noise), n_boot=T["n_boot"], min_tail=min(T["min_tail"], max(10, len(noise) // 10)),
                                 hill_reps=T["hill_reps"], n_jobs=T["n_jobs"], rng=rng)
            out.update({"zeta": rd.fit.zeta, "tail_label": rd.label, "p_ks": rd.p_ks})
        except ValueError:
            pass
    out["alpha_esd"], _ = spectra.model_alpha_esd(model, min_dim=16, rng=rng)
    # activations / representations on a probe set
    feats = []
    if is_lm:
        hs = lm.hidden_states(model, data["test"][:max(2, M["probe_points"] // cfg["transformer"]["seq_len"] + 1)], device)
        st = [spectra.activation_stats(h) for h in hs]
        out["K"] = float(np.median([s["K"] for s in st]))
        X = hs[-1].numpy()
    else:
        acts = {}
        hook = model.avgpool.register_forward_hook(lambda m, i, o: acts.setdefault("f", []).append(o.flatten(1).cpu()))
        with torch.no_grad():
            for s in range(0, min(M["probe_points"], len(data["test"][0])), 128):
                model(data["test"][0][s:s + 128].to(device))
        hook.remove()
        X = torch.cat(acts["f"]).numpy()
        out["K"] = spectra.activation_stats(torch.tensor(X))["K"]
    X = X[rng.choice(len(X), min(len(X), M["probe_points"]), replace=False)]
    dh = multifractal.delta_h_rep(X, S["q_lo"], S["q_hi"], rng=rng, n_centres=min(5000, len(X)))
    out.update({"dh_rep": dh["delta_h_rep"], "dh_scope": dh["scope"]})
    # H_grad over the constant-eta phase
    series = proj[warm:, :] if len(proj) > warm + 50 else proj
    if len(series) >= 10 * S["dfa_s_min"]:
        sc = hurst.log_scales(len(series), S["dfa_s_min"])
        hs_ = [hurst.memory_claim(series[:, j], n_surr=S["n_surr"], rng=rng, scales=sc,
                                  wavelet_j1=S["wavelet_j1"] if len(series) >= 2 ** 10 else 1) for j in range(series.shape[1])]
        for k in ("H_dfa", "H_shuf_dfa", "H_dep_dfa", "H_wav", "H_shuf_wav", "H_dep_wav", "dfa_decades"):
            out[f"Hgrad_{k}"] = float(np.mean([h[k] for h in hs_]))
        out["H_dep"] = out["Hgrad_H_dep_dfa"]
        out["Hgrad_len"] = len(series)
    model.train()
    return out


def h71_control(kind, cfg, data, stage_ckpts, seed, device, run, rng):
    hc = cfg["h71"]
    Bs = hc["batch_sizes"]
    n_pool = hc["n_blocks"] * max(Bs)
    rows, summ = [], []
    model = build_resnet(seed).to(device)
    X, Y = data["train"]
    g = torch.Generator().manual_seed(seed + 5)
    for st, sd in sorted(stage_ckpts.items()):
        model.load_state_dict(sd)
        model.eval()
        params = gradients.select_params(model, [hc["layer"]])
        gen = torch.Generator().manual_seed(run.seed)
        dirs = [("random", i, d) for i, d in enumerate(gradients.random_directions(params, hc["n_random_dirs"], gen))]
        if hc["n_hessian_dirs"]:
            hvp, p0 = gradients.hvp_fn(model, resnet_pel, (X[:256].to(device), Y[:256].to(device)), params)
            _, vecs = gradients.top_eigvecs(hvp, p0, hc["n_hessian_dirs"], 20, gen)
            dirs += [("hessian", i, v) for i, v in enumerate(vecs)]
        P, base = [], []
        bs = 256
        for s in range(0, n_pool, bs):
            m = min(bs, n_pool - s)
            ids = torch.randint(0, len(X), (m,), generator=g)
            xb = augment(X[ids], g).to(device)
            P.append(gradients.per_example_projections(model, resnet_pel, (xb, Y[ids].to(device)), params, [d for _, _, d in dirs]).cpu())
            base.append(ids)
        P = torch.cat(P).numpy()
        base = torch.cat(base).numpy()
        np.save(run.out_dir / f"h71_pool_stage{st}.npy", P)
        np.save(run.out_dir / f"h71_baseids_stage{st}.npy", base)
        for j, (kd, i, _) in enumerate(dirs):
            res = batchmod.h71_test(P[:, j], Bs, hc["n_blocks"], n_boot=hc["n_boot"], rng=rng)
            meta = {"model": kind, "stage": st, "direction": f"{kd}{i}"}
            rows += [{**meta, **r} for r in res["per_B"]]
            summ.append({**meta, **res["summary"]})
    return rows, summ


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--model", default=None, help="transformer | resnet18 (default: both)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--no-h71", action="store_true")
    args = ap.parse_args()
    cfg = config.get("p4_clipping", args.smoke)
    run = runtime.start_run("p4_clipping", cfg, args)
    rng = runtime.rng(run.seed)
    dev = run.device
    kinds = [args.model] if args.model else cfg["models"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    arms = [("fixed", p) for p in cfg["p_clip"]] + [(r, None) for r in cfg["adaptive"]]
    rows, h71_rows, h71_summ = [], [], []
    (run.out_dir / "hgrad_series").mkdir(exist_ok=True)
    for kind in kinds:
        data = transformer_data(cfg, run) if kind == "transformer" else cifar_data(cfg, run)
        for seed in seeds:
            ref_len, ref_norms = None, None
            for matched in ([False, True] if cfg["matched_update_length"] else [False]):
                for rule, p in arms:
                    if matched and rule == "fixed" and p == 0.0:
                        continue  # the unclipped run is the reference
                    scale = 1.0
                    if matched:
                        nominal = next(r for r in rows if r["model"] == kind and r["seed"] == seed and r["rule"] == rule
                                       and r.get("p_clip_target") == p and not r["matched"])
                        scale = ref_len / nominal["mean_update_len"] if nominal["mean_update_len"] > 0 else 1.0
                    hc = cfg["h71"]["run"]
                    want_h71 = (not args.no_h71 and not matched and kind == hc["model"] and seed == hc["seed"]
                                and ((rule == "fixed" and p == 0.0) if hc["rule"] == "none" else rule == hc["rule"]))
                    model, info, proj, ckpts = train(kind, cfg, data, rule if rule != "fixed" or p else "none",
                                                     p or 0.0, seed, scale, dev, run,
                                                     save_stages=cfg["stages"] if want_h71 else None,
                                                     ref_norms=ref_norms)
                    norms = info.pop("norms_post_warmup")
                    tag = f"{kind}_s{seed}_{rule}{'' if p is None else p}{'_matched' if matched else ''}"
                    np.save(run.out_dir / "hgrad_series" / f"{tag}.npy", proj)
                    row = {"model": kind, "seed": seed, "rule": rule, "p_clip_target": p, "matched": matched,
                           "lr_scale": scale, "data_source": data["source"], **info}
                    if not info["diverged"]:
                        row.update(measure(kind, model, data, cfg, run, rng, dev, proj, info["warmup_steps"]))
                    rows.append(row)
                    if rule == "fixed" and p == 0.0 and not matched:
                        ref_len, ref_norms = info["mean_update_len"], norms
                    run.note(f"{tag}: steps={info['steps']} p_clip={info['p_clip_realized']:.3g} "
                             f"diverged={info['diverged']} test_loss={row.get('test_loss', float('nan')):.4f}")
                    if want_h71 and ckpts:
                        r1, r2 = h71_control(kind, cfg, data, ckpts, seed, dev, run, rng)
                        h71_rows += r1
                        h71_summ += r2
                    del model
    runtime.write_csv(run.out_dir / "runs.csv", rows)
    runtime.write_csv(run.out_dir / "h71_per_B.csv", h71_rows)
    runtime.write_csv(run.out_dir / "h71_summary.csv", h71_summ)
    runtime.finish_run(run, extra={"n_runs": len(rows)})


if __name__ == "__main__":
    main()
