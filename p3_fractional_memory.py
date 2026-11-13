"""P3: H4.1, accuracy against state size n_s at matched parameters.

Arms: fractional block (modes tied to (nu, lambda)), S4 with HiPPO-LegS initialization, S4D,
Mamba (pure-PyTorch reference scan), optional frozen HiPPO-LegS. For every n_s the channel
width of each arm is set so that its parameter count matches the reference arm (S4D at H_ref),
and every arm gets the same learning-rate grid (equal tuning budget), steps and seeds.

Tasks:
  powerlaw     y_t = sum_k kappa_k g_{t-k}, kappa_k ~ (k+1)^-(1+nu*), lags 1..L (power-law memory)
  exponential  kappa_k ~ exp(-k / (tau_frac L))                        (exponential memory)
  lra_image    Long Range Arena image task: CIFAR-10 in grayscale as 1,024-step sequences
Accuracy is R^2 on held-out sequences for the two regression tasks and classification
accuracy for lra_image. H4.1 is refuted if the baseline accuracy-state curves lie on or above
the fractional block's on the power-law task, or if any advantage does not depend on task type.

Outputs (results/p3_fractional_memory/[smoke/]): runs.csv (one row per run incl. tuning trials,
with params, FLOPs proxy, peak memory and wall-clock), curves.csv (best accuracy per arm, task,
n_s, seed), states_to_target.csv, run.json

Commands:
  python p3_fractional_memory.py
  python p3_fractional_memory.py --tasks powerlaw,exponential      # drop LRA first if time runs short
  python p3_fractional_memory.py --smoke
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from common import hf, runtime, ssm


def memory_task(kind, L, n, cfg_t, rng):
    g = rng.standard_normal((n, L)).astype(np.float32)
    k = np.arange(L, dtype=np.float64)
    if kind == "powerlaw":
        kappa = (k + 1) ** (-(1 + cfg_t["nu_star"]))
    else:
        kappa = np.exp(-k / (cfg_t["tau_frac"] * L))
    kappa /= np.sqrt((kappa ** 2).sum())
    y = np.real(np.fft.irfft(np.fft.rfft(g, 2 * L) * np.fft.rfft(kappa, 2 * L), 2 * L))[:, :L]
    return torch.tensor(g[..., None]), torch.tensor(y[..., None].astype(np.float32))


def lra_image(cfg_t, run, rng):
    """CIFAR-10 as grayscale sequences (LRA image). The Hugging Face copy first, then torchvision."""
    def to_seq(imgs):
        x = torch.tensor(np.asarray(imgs), dtype=torch.float32)
        if x.ndim == 4 and x.shape[-1] == 3:
            x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
        x = x.reshape(len(x), -1, 1) / 255.0
        return (x - 0.5) / 0.25
    try:
        Xtr, ytr, Xte, yte, src = hf.load_cifar10(run, str(runtime.PILOT_ROOT / "data"),
                                                  cfg_t.get("limit_train"), cfg_t.get("limit_test"))
        Xtr, ytr, Xte, yte = to_seq(Xtr), torch.tensor(ytr), to_seq(Xte), torch.tensor(yte)
    except RuntimeError as e:
        if not run.offline_ok:
            raise
        run.fallback(f"CIFAR-10 for LRA image: random images with random labels (smoke only; {str(e)[:160]})")
        n_tr, n_te = cfg_t.get("limit_train", 256), cfg_t.get("limit_test", 64)
        Xtr = torch.randn(n_tr, 1024, 1); ytr = torch.randint(0, 10, (n_tr,))
        Xte = torch.randn(n_te, 1024, 1); yte = torch.randint(0, 10, (n_te,))
        src = "fallback"
    if "limit_train" in cfg_t:
        Xtr, ytr = Xtr[:cfg_t["limit_train"]], ytr[:cfg_t["limit_train"]]
        Xte, yte = Xte[:cfg_t["limit_test"]], yte[:cfg_t["limit_test"]]
    return (Xtr, ytr, Xte, yte), src


def evaluate(model, X, Y, task, device, batch=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            preds.append(model(X[s:s + batch].to(device)).float().cpu())
    P = torch.cat(preds)
    model.train()
    if task == "lra_image":
        return float((P.argmax(-1) == Y).float().mean())
    resid = ((P - Y) ** 2).sum()
    tot = ((Y - Y.mean()) ** 2).sum()
    return float(1 - resid / tot)


def train_one(arch, task, n_s, H, data, cfg, lr, seed, device):
    runtime.set_all_seeds(seed)
    Xtr, Ytr, Xte, Yte = data
    L = Xtr.shape[1]
    cls = task == "lra_image"
    model = ssm.SSMModel(arch, 1, 10 if cls else 1, H, n_s, cfg["n_layers"], L, pool=cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    steps = cfg["steps"] if not cls else cfg["lra_epochs"] * max(1, len(Xtr) // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    diverged = False
    for step in range(steps):
        idx = torch.randint(0, len(Xtr), (cfg["batch"],), generator=g)
        x, y = Xtr[idx].to(device), Ytr[idx].to(device)
        out = model(x)
        loss = F.cross_entropy(out, y) if cls else F.mse_loss(out, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    wall = time.time() - t0
    acc = evaluate(model, Xte, Yte, task, device) if not diverged else float("nan")
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
    return {"acc": acc, "params": ssm.n_params(model), "steps": steps, "wall_s": wall, "peak_mem_mb": peak,
            "diverged": diverged, "tokens_seen": steps * cfg["batch"] * L,
            "flops_proxy": 6 * ssm.n_params(model) * steps * cfg["batch"] * L}


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--archs", default=None)
    args = ap.parse_args()
    cfg = config.get("p3_fractional_memory", args.smoke)
    run = runtime.start_run("p3_fractional_memory", cfg, args)
    rng = runtime.rng(run.seed)
    dev = run.device
    tasks = args.tasks.split(",") if args.tasks else list(cfg["tasks"])
    archs = args.archs.split(",") if args.archs else cfg["archs"]
    rows = []
    for task in tasks:
        ct = cfg["tasks"][task]
        if task == "lra_image":
            data, src = lra_image(ct, run, rng)
        else:
            Xtr, Ytr = memory_task(task, ct["L"], ct["n_train"], ct, rng)
            Xte, Yte = memory_task(task, ct["L"], ct["n_test"], ct, rng)
            data, src = (Xtr, Ytr, Xte, Yte), "synthetic"
        L = data[0].shape[1]
        cls = task == "lra_image"
        def widths_for(n_s):
            ref = ssm.SSMModel(cfg["ref_arch"], 1, 10 if cls else 1, cfg["H_ref"], n_s, cfg["n_layers"], L, pool=cls)
            target = ssm.n_params(ref)
            return target, {a: ssm.width_for_budget(a, 1, 10 if cls else 1, n_s, cfg["n_layers"], L, cls, target)
                            for a in archs}

        # equal tuning budget: the same LR grid for every arm, on the tuning seed at the tuning state size
        best_lr = {}
        n_tune = cfg["tune_n_states"]
        target_t, widths_t = widths_for(n_tune)
        for arch in archs:
            trials = []
            for lr in cfg["lr_grid"]:
                r = train_one(arch, task, n_tune, widths_t[arch][0], data, cfg, lr, cfg["tune_seed"], dev)
                rows.append({"task": task, "arch": arch, "n_s": n_tune, "H": widths_t[arch][0], "lr": lr,
                             "seed": cfg["tune_seed"], "phase": "tune", "target_params": target_t,
                             "data_source": src, **r})
                trials.append((r["acc"] if np.isfinite(r["acc"]) else -np.inf, lr))
            best_lr[arch] = max(trials)[1]
        for n_s in cfg["n_states"]:
            target, widths = widths_for(n_s)
            for arch in archs:
                H = widths[arch][0]
                for seed in cfg["seeds"]:
                    r = train_one(arch, task, n_s, H, data, cfg, best_lr[arch], seed, dev)
                    rows.append({"task": task, "arch": arch, "n_s": n_s, "H": H, "lr": best_lr[arch], "seed": seed,
                                 "phase": "main", "target_params": target, "data_source": src, **r})
                run.note(f"{task} n_s={n_s} {arch}: H={H}, params={widths[arch][1]} (target {target}), lr={best_lr[arch]}")
    df = pd.DataFrame(rows)
    runtime.write_csv(run.out_dir / "runs.csv", rows)
    main_runs = df[df.phase == "main"]
    curves = main_runs.groupby(["task", "arch", "n_s"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                             params=("params", "mean"), n=("acc", "size")).reset_index()
    runtime.write_csv(run.out_dir / "curves.csv", curves.to_dict("records"))
    tgt = []
    for (task, arch), g in curves.groupby(["task", "arch"]):
        thr = cfg["target_accuracy"][task]
        hit = g[g.acc_mean >= thr].n_s
        tgt.append({"task": task, "arch": arch, "target_acc": thr, "min_n_s": int(hit.min()) if len(hit) else None})
    runtime.write_csv(run.out_dir / "states_to_target.csv", tgt)
    runtime.finish_run(run, extra={"n_runs": len(rows)})


if __name__ == "__main__":
    main()
