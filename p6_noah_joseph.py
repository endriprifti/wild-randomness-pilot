"""P6 (added 2026-11-18, fixed before any P6 run): a factorial test of a causal Noah-Joseph link.

Identical small GPT-NeoX models (Pythia-14M shape; Pythia-31M shape in the full design) are trained on synthetic
token streams in which the two effects are set independently (p6_streams.py):
  Noah    the rank law of the tokens: light-tailed control (exponential, entropy-matched to Zipf 1.0), Zipf 1.25
          (full design only), Zipf 1.0
  Joseph  the H of the fGn that drives the topic-switching hazard: 0.5 (no memory), 0.7 (full design only), 0.9
Every cell gets the same tokens (2^28), the same model, optimizer, schedule and batch, and, per seed, the same random
numbers (p6_streams.py: common random numbers; seeds are blocks in the analysis). Every stream is checked before a
model sees it (rank law, variants, switch rate, H of the fGn and of the switch counts); a failed check stops the job.

Training (OURS, the same in every cell): AdamW (0.9, 0.95), lr 1e-3, weight decay 0.1, warm-up 1% then constant
(P4's schedule), 16 sequences of 2,048 tokens per step (32,768 tokens; 8,192 steps), no clipping, bf16 autocast with
float32 weights, each window of the stream used once in a fixed random order. The loss is the mean next-token NLL,
computed in chunks of 4,096 tokens with recomputation (the same value as one pass, less memory). Every step saves
the unclipped minibatch gradient projected on 8 fixed random directions (P4's H_grad series, "hgrad").

Measurements at P1's training stages (1, 10, 50, 100%), with P1's code and settings (config.py p1_checkpoints, the
coarse H7.1 resolution of 200 blocks): K, K_max, Kurt0 and outlier channels on the probe set, act_zeta, alpha and
the CSN tails of the minibatch noise at B = 8, H7.1 (alpha-hat over B = 8 ... 1,024 with the bootstrap and the
verdict), alpha_ESD, Delta h_rep, eval loss, and H_tok (P1's memory_claim) on the token surprisals of 64
consecutive held-out windows of 2,048 tokens (P1 takes 16; 64 because at 16 the surprisal memory barely moves with
the data's H, p6 power check in the README). The surprisal is also split into its rank part -log p(rank) and its
variant part -log p(variant | rank), from the same logits: the data's memory is in the variant part only, and the
rank part's noise grows with the Zipf tail (it dilutes the memory of the whole series). H_link (h_link.run_unit,
unchanged) on the whole token-surprisal series (P1 treatment: main, sensitivity, standardized rows, window
control) at every stage and on the hgrad series after training (P4 treatment: main, sensitivity, detrended,
rescaled). All CPU statistics run in a process pool next to the GPU (P1's StatPool and p1_tasks).

The pre-stated predictions, tests and pass rules are in p6_stats.py (14 tests, Holm at 0.05) and in
s7-appendix-pilot-design.tex (P6, 2026-11-18).

Outputs (--out; one folder per job): runs/<run>/ with stream_check_train.json and stream_check_heldout.json,
train_log.csv (loss per step), hgrad.npy (steps x 8), checkpoints/<stage>/ (P1's layout: samples/, stats/,
row.json, result.json), hlink/; units/<run>.json per finished run; progress.json (run_queue.py --status);
timings.csv. --analyse --from <folder> collects every job folder below it and writes checkpoints.csv, family.csv,
manipulation_checks.csv, secondary.csv and summary.md into --out.

Commands (in the pilot folder inside the venv):
  python p6_noah_joseph.py --plan                       # the runs of both designs and the cost model
  python p6_noah_joseph.py --timing --size 14m --steps 300 --micro-batch 2 --loss-chunk 1024 --eval-batch 1 \
      --jvp-batch 8 --hvp-batch 4 --out results/p6_timing_laptop        # 4 GB laptop GPU (2.0 GiB by --memory)
  python p6_noah_joseph.py --memory [--sizes 14m] [batch flags]         # GPU memory of each step (meta device)
  python p6_noah_joseph.py --design small --seeds 1 --out results/p6_small/14m_s1    # one job of the small design
  python p6_noah_joseph.py --analyse --from results/p6_small --out results/p6_small/analysis
  setsid nohup nice -n 10 taskset -c 0-10 python p6_after_queue.py > logs/p6_after_queue.out 2>&1 < /dev/null &
                                    # the L40S: the small design through run_queue.py once the running queue ends
  python p6_noah_joseph.py --smoke --out results/p6_smoke   # tiny model and streams on the CPU (minutes)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import copy
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import config
import p1_checkpoints
import p6_stats
import p6_streams
from common import gradients, hf, hlink, lm, p1_tasks, runtime, spectra, stable
from common import units as units_mod

TEST = "p6_noah_joseph"
ROOT = Path(__file__).resolve().parent

SIZES = {"14m": dict(hf.PYTHIA_14M),
         "31m": dict(hf.PYTHIA_14M, hidden_size=256, num_attention_heads=8, intermediate_size=1024)}

TRAIN = {"tokens": 2 ** 28, "batch_seqs": 16, "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95],
         "warmup_frac": 0.01, "precision": "bf16", "loss_chunk": 4096, "micro_batch": 16, "n_hgrad_dirs": 8,
         "diverge_factor": 3.0, "stages": list(config.TRAINING_STAGES)}
MEASURE = {"surprisal_docs": 64,          # held-out windows of seq_len tokens for H_tok and H_link (P1: 16)
           "h71_blocks": "coarse",        # P1's coarse resolution (200 blocks) at every size and stage
           "hlink_stages": "all",
           # sequences per micro-batch of the Hessian-vector products for H7.1's Hessian direction (P1: one batch
           # of hessian_batch = 64). The same H v up to float summation order (common/gradients.hvp_fn: the loss is a
           # mean over examples), at 3.9 instead of 15.4 GiB (14M, 64 x 128 tokens; --memory). Added 2026-11-18,
           # before any P6 run.
           "hvp_batch": 16}
DESIGNS = {
    # the confirmatory family is tested on the 14M corners at the final stage in both designs
    # small: H7.1 at the 50% and final stages only (the other measurements at all four), to stay near 12 L40S hours
    "small": {"sizes": ["14m"], "noah": ["light", "zipf1.0"], "joseph": ["H0.5", "H0.9"], "seeds": [1, 2, 3, 4, 5],
              "h71_stages": [0.5, 1.0]},
    "full": {"sizes": ["14m", "31m"], "noah": ["light", "zipf1.25", "zipf1.0"], "joseph": ["H0.5", "H0.7", "H0.9"],
             "seeds": [1, 2, 3, 4, 5], "h71_stages": list(config.TRAINING_STAGES)},
}
BUDGET_HOURS = {"small": 12.0, "full": 80.0}      # --budget-hours in the queue files

SMOKE_STREAM = dict(copy.deepcopy(p6_streams.P6), vocab=512, K=4, R=128, block=16, p_bar=1 / 32, seq_len=128,
                    heldout_windows=400)
SMOKE_TRAIN = dict(TRAIN, tokens=24 * 4 * 128, batch_seqs=4, micro_batch=2, loss_chunk=96, n_hgrad_dirs=2,
                   stages=[0.5, 1.0], precision="fp32")
SMOKE_MEASURE = dict(MEASURE, surprisal_docs=4, hvp_batch=4)      # smoke hessian_batch 8: two micro-batches
SMOKE_DESIGN = {"sizes": ["tiny"], "noah": ["light", "zipf1.0"], "joseph": ["H0.5", "H0.9"], "seeds": [1, 2],
                "h71_stages": [1.0]}

# GPU -> L40S speed ratio range for these small models (training and JVP passes are memory-bound: logits of 50,304
# per token). RTX 2050: 112 GB/s, about 12 bf16 TFLOPS with float32 accumulation, 6 float32 TFLOPS; L40S: 864 GB/s,
# 362 and 92. Bandwidth alone gives 7.7x, compute 15-30x; small kernels and launch overhead keep it near the low
# end. OURS, a prior only: the first L40S job of the queue measures it (p6_timing_*) and --budget-hours stops the
# runs if the measured hours exceed the budget.
L40S_RATIO = {"RTX 2050": (6.0, 15.0)}
L40S_RATIO_DEFAULT = (4.0, 20.0)
PRICE_PER_HOUR = 0.69


# ----------------------------------------------------------------------------- settings
def settings(smoke: bool, design: dict | None = None) -> dict:
    p1 = config.get("p1_checkpoints", smoke)
    P = SMOKE_STREAM if smoke else p6_streams.P6
    tr = SMOKE_TRAIN if smoke else TRAIN
    me = SMOKE_MEASURE if smoke else MEASURE
    m = {k: p1[k] for k in ("example_len", "n_random_dirs", "n_hessian_dirs", "hessian_batch", "coupling_B",
                            "coupling_blocks", "probe_seqs", "probe_len", "dh_points", "eval_seqs", "eval_len",
                            "tails", "scale", "batch_sizes_h71")}
    m["surprisal_docs"] = int(me["surprisal_docs"])
    m["hvp_batch"] = int(me["hvp_batch"])
    m["surprisal_len"] = int(P["seq_len"])
    m["h71"] = {"coarse_blocks": int(p1["h71"]["coarse_blocks"]),
                "batch_sizes": [int(b) for b in p1["h71"].get("batch_sizes", p1["batch_sizes_h71"])],
                "n_boot": int(p1["h71"].get("n_boot", config.H71["n_boot"]))}
    m["hlink_test"] = dict(hlink.TEST_DEFAULTS, **({"n_surr": 9} if smoke else {}))
    m["hlink_stages"] = me["hlink_stages"]
    m["h71_stages"] = [float(x) for x in (design or (SMOKE_DESIGN if smoke else DESIGNS["full"]))["h71_stages"]]
    return {"stream": P, "train": tr, "measure": m, "smoke": smoke, "seed": config.SEED}


def size_cfg(size: str) -> dict:
    return dict(hf.TINY_NEOX) if size == "tiny" else dict(SIZES[size])


def steps_of(st: dict) -> int:
    return int(st["train"]["tokens"]) // (int(st["train"]["batch_seqs"]) * int(st["stream"]["seq_len"]))


def stage_steps(st: dict) -> dict[float, int]:
    n = steps_of(st)
    return {float(s): max(1, int(round(s * n))) for s in st["train"]["stages"]}


def run_key(size: str, noah: str, joseph: str, seed: int) -> str:
    return f"{size}_{noah}_{joseph}_s{seed}"


def plan(design: dict, sizes=None, seeds=None) -> list[dict]:
    out = []
    for size in design["sizes"]:
        if sizes and size not in sizes:
            continue
        for seed in design["seeds"]:
            if seeds and seed not in seeds:
                continue
            for noah in design["noah"]:
                for joseph in design["joseph"]:
                    out.append({"key": run_key(size, noah, joseph, seed), "size": size, "noah": noah,
                                "joseph": joseph, "seed": int(seed)})
    return out


def init_seed(seed: int) -> int:
    """The model's initial weights: the seed only (the same in every cell of that seed)."""
    return int(np.random.SeedSequence(units_mod.seed_for(config.SEED, "p6 init", int(seed))).generate_state(1)[0])


# ----------------------------------------------------------------------------- model, loss, gradient series
def build_model(size: str, seed: int, vocab: int):
    cfg = dict(size_cfg(size), vocab_size=int(vocab))
    model = hf.neox_from_config(cfg, seed=init_seed(seed))
    return model


def set_attention(model, impl: str) -> None:
    """sdpa while training (memory-efficient kernels); eager for the measurements, as P1 (torch.func JVPs)."""
    model.config._attn_implementation = impl


def lm_loss(model, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """Mean next-token NLL of a batch, the output projection and softmax in chunks of `chunk` tokens, recomputed in
    the backward pass (torch.utils.checkpoint): the logits of 50,304 per token are never all in memory."""
    h = model.gpt_neox(input_ids=x).last_hidden_state[:, :-1]
    h = h.reshape(-1, h.shape[-1])
    tgt = x[:, 1:].reshape(-1)
    head = model.get_output_embeddings()

    def part(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="sum")

    tot = h.new_zeros((), dtype=torch.float32)
    for s in range(0, len(tgt), chunk):
        hc, tc = h[s:s + chunk], tgt[s:s + chunk]
        tot = tot + (checkpoint(part, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else part(hc, tc))
    return tot / len(tgt)


def directions(model, ids_hess, device, n_random: int, n_hessian: int, seed: int, hvp_batch: int | None):
    """P1's directions (p1_checkpoints.directions: random directions, then the top Hessian eigenvectors by power
    iteration), with the Hessian-vector products on micro-batches of at most hvp_batch sequences. With hvp_batch
    covering the batch (or None) it is P1's function itself."""
    if not hvp_batch or n_hessian <= 0 or hvp_batch >= len(ids_hess):
        return p1_checkpoints.directions(model, ids_hess, device, n_random, n_hessian, seed)
    names = lm.grad_param_names(model)
    params = {n: p for n, p in model.named_parameters() if n in names}
    gen = torch.Generator().manual_seed(seed)
    dirs = [("random", i, d) for i, d in enumerate(gradients.random_directions(params, n_random, gen))]
    hvp, p0 = gradients.hvp_fn(model, lm.per_example_lm_loss, ids_hess.to(device), params, chunk=int(hvp_batch))
    vals, vecs = gradients.top_eigvecs(hvp, p0, k=n_hessian, iters=20, generator=gen)
    dirs += [("hessian", i, v) for i, v in enumerate(vecs)]
    return params, dirs


def flat_directions(params, k: int, seed: int, device) -> torch.Tensor:
    """k unit directions in the space of all parameters (as P4's flat_dirs), as one k x P matrix."""
    g = torch.Generator().manual_seed(int(seed))
    P = sum(p.numel() for p in params)
    D = torch.randn(k, P, generator=g)
    D /= D.norm(dim=1, keepdim=True)
    return D.to(device)


def flat_grad(params) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


@torch.no_grad()
def surprisal_parts(model, ids: torch.Tensor, device, K: int, batch: int = 4) -> tuple[np.ndarray, ...]:
    """Per-token (whole, rank part, variant part) surprisal [n_seqs, len - 1]: token = (rank - 1) K + variant, so
    log p(rank) = logsumexp over the rank's K tokens and the variant part is the rest."""
    outs = [[], [], []]
    for s in range(0, len(ids), batch):
        tok, rank = _surprisal_batch(model, ids[s:s + batch].to(device).long(), K)
        for o, v in zip(outs, (tok, rank, tok - rank)):
            o.append(v.cpu())
    return tuple(torch.cat(o).numpy() for o in outs)


def _surprisal_batch(model, x: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(whole, rank part) surprisal of one micro-batch, on its device (also run on the meta device by --memory)."""
    lp = torch.log_softmax(model(input_ids=x).logits.float()[:, :-1], dim=-1)
    y = x[:, 1:]
    tok = -lp.gather(-1, y[..., None])[..., 0]
    V = lp.shape[-1]
    lr = torch.logsumexp(lp.reshape(*lp.shape[:-1], V // K, K), dim=-1)
    rank = -lr.gather(-1, (y // K)[..., None])[..., 0]
    return tok, rank


# ----------------------------------------------------------------------------- the CPU pool
class P6Pool(p1_checkpoints.StatPool):
    """P1's pool, with p6_stats.run_p6_task (P1's tasks unchanged, plus the H_link units)."""

    def _dispatch(self) -> None:   # under self.cv, as StatPool._dispatch
        cpus = self._allowed()
        slots = self.n_fixed or len(cpus)
        while self.pending and len(self.running) < slots:
            spec = {**self.pending.popleft(), "cpus": cpus}
            self.running[self._executor().submit(p6_stats.run_p6_task, spec)] = (spec, time.monotonic())


class Progress(p1_checkpoints.Progress):
    def say(self, msg: str) -> None:
        print(f"{p1_checkpoints.utc()} [p6] {msg}", flush=True)


# ----------------------------------------------------------------------------- measurement at one stage
def heldout_sets(held: p6_streams.Stream, m: dict) -> dict:
    lay = p6_streams.heldout_layout(held.P, m)
    w0, n = lay["pool"]
    pool = held.examples(w0, n, int(m["example_len"]))[: lay["n_pool"]]
    w0, n = lay["eval"]
    evals = held.windows(range(w0, w0 + n))[: int(m["eval_seqs"])]
    w0, n = lay["surprisal"]
    sur = held.windows(range(w0, w0 + n))
    w0, n = lay["probe"]
    probe = held.examples(w0, n, int(m["probe_len"]))[: int(m["probe_seqs"])]
    t = lambda a: torch.as_tensor(a, dtype=torch.long)     # noqa: E731
    return {"pool": t(pool), "eval": t(evals), "surprisal": t(sur), "probe": t(probe), "layout": lay}


def gpu_pass(model, c: dict, sets: dict, st: dict, dev: str, prog, eb: int = 4, jb: int = 32,
             h71_pool: int | None = None, hb: int | None = None) -> dict:
    """P1's GPU pass (p1_checkpoints.py gpu_pass) on a model in memory: samples/ and row.json in P1's layout, the
    CPU tasks in P1's task specs (p1_tasks.assemble reads them unchanged), plus the P6 series. h71_pool limits the
    H7.1 pool (timing only); hb: sequences per micro-batch of the Hessian-vector products (default
    MEASURE["hvp_batch"]). Returns the record; timings in record["seconds"]."""
    m = st["measure"]
    d = c["dir"]
    if d.exists():
        shutil.rmtree(d)
    sd = d / "samples"
    sd.mkdir(parents=True)
    was_training = model.training
    model.eval()
    set_attention(model, "eager")
    T, S = m["tails"], m["scale"]
    seconds: dict = {}
    t_all = time.perf_counter()

    @contextlib.contextmanager
    def timed(what):
        t = time.perf_counter()
        yield
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        seconds[what] = time.perf_counter() - t
        prog.record(c["key"], what, "gpu", seconds[what])

    def task(name, kind, params, **kw):
        return {"name": name, "kind": kind, "params": params, "key": c["key"], "dir": str(d),
                "seed": p1_tasks.seed_for(st["seed"], c["key"], name), **kw}

    row = {"suite": "p6", "variant": f"{c['noah']}|{c['joseph']}", "size": c["size"], "seed": c["seed"],
           "repo": c["run"], "step": c["step"], "revision": f"stage{c['stage']}", "source": "trained",
           "pass": "p6", "n_params": sum(p.numel() for p in model.parameters()),
           "n_nonembed": lm.non_embedding_params(model)}
    tasks: list = []
    record = {"row": row, "tasks": tasks}
    probe, evals, sur, pool = sets["probe"], sets["eval"], sets["surprisal"], sets["pool"]
    with timed("probe_loss"):
        row["probe_loss"] = float(lm.token_losses(model, probe, dev, batch=eb).mean())
    with timed("esd"):
        esds = spectra.layer_esds(model)
        p1_tasks.atomic_npz(sd / "esd.npz", names=np.array([n for n, _, _ in esds]),
                            shapes=np.array([s for _, s, _ in esds]),
                            **{f"ev_{i}": ev for i, (_, _, ev) in enumerate(esds)})
    tasks.append(task("esd", "esd", {}))
    with timed("hidden states and activation moments"):
        hs = lm.hidden_states(model, probe, dev, batch=eb)
        moments, subs = [], {}
        for li, H in enumerate(hs):
            Hd = H.to(dev)
            moments.append(spectra.activation_moments(Hd))
            sub = spectra.activation_tail_sample(Hd)
            if sub is not None:
                subs[f"sub_{li}"] = sub
            del Hd
        p1_tasks.atomic_npz(sd / "act.npz", **subs)
        record["act_moments"] = moments
        X = hs[-1].numpy()
        ck_rng = p1_tasks.rng_for(p1_tasks.seed_for(st["seed"], c["key"], "gpu pass"))
        if len(X) > m["dh_points"]:
            X = X[ck_rng.choice(len(X), m["dh_points"], replace=False)]
        p1_tasks.atomic_npy(sd / "dh.npy", np.ascontiguousarray(X, dtype=np.float32))
        del hs
    tasks.append(task("act", "act", {"n_layers": len(moments)}))
    tasks.append(task("dh", "dh", {"q_lo": S["q_lo"], "q_hi": S["q_hi"]}))
    with timed("token surprisals (whole, rank and variant parts)"):
        whole, rank, var = surprisal_parts(model, sur, dev, int(st["stream"]["K"]), batch=eb)
        p1_tasks.atomic_npy(sd / "htok.npy", whole.reshape(-1))
        for part, arr in (("rank", rank), ("variant", var)):
            pd_ = d / "parts" / part / "samples"
            pd_.mkdir(parents=True, exist_ok=True)
            p1_tasks.atomic_npy(pd_ / "htok.npy", arr.reshape(-1))
        row["surprisal_mean"] = float(whole.mean())
        row["surprisal_rank_mean"] = float(rank.mean())
        row["surprisal_variant_mean"] = float(var.mean())
    hp = {"n_surr": S["n_surr"], "wavelet_j1": S["wavelet_j1"]}
    tasks.append(task("htok", "htok", hp))
    for part in ("rank", "variant"):
        tasks.append(task(f"htok_{part}", "htok", hp, samples_dir=str(d / "parts" / part)))
    if m["hlink_stages"] == "all" or float(c["stage"]) == 1.0:
        n_docs, doc_len = int(sur.shape[0]), int(sur.shape[1]) - 1
        unit = {"source": "p1", "folder": c["run"], "key": f"{c['run']}/stage{c['stage']}",
                "path": str(sd / "htok.npy"), "stamp": "", "warm": 0, "n_docs": n_docs, "doc_len": doc_len,
                "eos_cuts": None, "group": c["size"]}
        tasks.append(task("hlink_tok", "hlink", {}, unit=unit, test=m["hlink_test"]))
    with timed("gradient projections (JVP, coupling)"):
        B = int(m["coupling_B"])
        n = int(m["coupling_blocks"]) * B
        params, dirs = p1_checkpoints.directions(model, pool[:m["hessian_batch"]], dev, m["n_random_dirs"], 0,
                                                 st["seed"])
        P = p1_checkpoints.jvp_projections(model, pool[:n], params, [d_ for _, _, d_ in dirs], dev, jb)
        p1_tasks.atomic_npy(sd / "grad.npy", P)
    tasks.append(task("grad", "grad", {"B": B, "n_blocks": int(m["coupling_blocks"]), "T": T,
                                       "order_seed": p1_tasks.order_seed(st["seed"])}))
    h = m["h71"]
    nb = int(h["coarse_blocks"])
    Bs = [int(b) for b in h["batch_sizes"]]
    n_pool = nb * max(Bs)
    n_take = min(n_pool, int(h71_pool)) if h71_pool else n_pool
    if not h71_pool and not any(abs(float(c["stage"]) - x) < 1e-9 for x in m["h71_stages"]):
        n_take = 0
    if n_take:
        with timed(f"H7.1 projections ({n_take} examples)"):
            params, dirs = directions(model, pool[:m["hessian_batch"]], dev, m["n_random_dirs"],
                                      m["n_hessian_dirs"], st["seed"], int(hb or m["hvp_batch"]))
            P = p1_checkpoints.jvp_projections(model, pool[:n_take], params, [d_ for _, _, d_ in dirs], dev, jb)
    record["h71_examples_timed"] = int(n_take)
    if n_take and n_take == n_pool:
        p1_tasks.atomic_npy(sd / "h71.npy", P)
        names = [f"{kind}{i}" for kind, i, _ in dirs]
        n_boot = int(h["n_boot"])
        chunk = 25
        chunks = [min(chunk, n_boot - s0) for s0 in range(0, n_boot, chunk)]
        record["h71"] = {"directions": names, "stage": c["stage"], "frac": c["stage"], "nb": nb, "Bs": Bs,
                         "n_pool": int(len(P)), "n_boot": n_boot, "n_chunks": len(chunks)}
        for j, dname in enumerate(names):
            tasks.append(task(f"h71_fit.{dname}", "h71_fit", {"j": j, "Bs": Bs, "nb": nb,
                                                              "order_seed": p1_tasks.order_seed(st["seed"])}))
            for ci, nbt in enumerate(chunks):
                tasks.append(task(f"h71_boot.{dname}.{ci}", "h71_boot", {"j": j, "Bs": Bs, "nb": nb, "n_boot": nbt}))
    with timed("eval loss"):
        row["eval_loss"] = float(lm.token_losses(model, evals, dev, batch=eb).mean())
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    set_attention(model, "sdpa")
    model.train(was_training)
    row["gpu_pass_seconds"] = round(time.perf_counter() - t_all, 1)
    record["seconds"] = seconds
    p1_tasks.atomic_json(d / "row.json", record)
    return record


# ----------------------------------------------------------------------------- one run
class StreamCheckFailed(SystemExit):
    pass


def stream_checks(run_dir: Path, streams: dict, enforce: bool, prog) -> dict:
    out = {}
    for role, s in streams.items():
        f = run_dir / f"stream_check_{role}.json"
        rec = p6_stats.json_load(f) if f.exists() else None
        if rec is None or rec.get("n_tokens") != s.n_tokens:
            rec = p6_streams.check_stream(s)
            p1_tasks.atomic_json(f, rec)
            prog.say(f"{run_dir.name}: {role} stream checked in {rec['seconds']:.0f} s: "
                     + ("ok" if rec["ok"] else f"FAILED {rec['failed']}"))
        out[role] = rec
        if enforce and not rec["ok"]:
            raise StreamCheckFailed(f"{run_dir.name}: the {role} stream failed its checks {rec['failed']} "
                                    f"({f}); no model is trained on it")
    return out


def train_run(c: dict, st: dict, args, run, pool: P6Pool | None, prog: Progress, measure=None) -> dict:
    """Train one run with its measurements at every stage; resumable from the last stage reached. measure: None for
    P6's measurements (gpu_pass, its CPU tasks in `pool`); the feasibility check passes outlier_pass instead, which
    also measures step 0 (the training is the same)."""
    tr, P = st["train"], st["stream"]
    rd = Path(c["dir"])
    rd.mkdir(parents=True, exist_ok=True)
    dev = run.device
    n_steps = steps_of(st)
    B, L = int(tr["batch_seqs"]), int(P["seq_len"])
    t0 = time.perf_counter()
    train = p6_streams.Stream(c["noah"], c["joseph"], c["seed"], "train", n_steps * B, P)
    held = p6_streams.Stream(c["noah"], c["joseph"], c["seed"], "heldout", int(P["heldout_windows"]), P)
    prog.say(f"{c['key']}: streams built in {time.perf_counter() - t0:.0f} s")
    checks = stream_checks(rd, {"train": train, "heldout": held}, enforce=not st["smoke"], prog=prog)
    sets = heldout_sets(held, st["measure"])
    model = build_model(c["size"], c["seed"], P["vocab"]).to(dev)
    set_attention(model, "sdpa")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"],
                            betas=tuple(tr["betas"]))
    warm = max(1, int(tr["warmup_frac"] * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    params = [p for p in model.parameters() if p.requires_grad]
    D = flat_directions(params, int(tr["n_hgrad_dirs"]), init_seed(c["seed"]) + 999, dev)
    order = units_mod.rng_for(units_mod.seed_for(config.SEED, "p6 order", c["seed"])).permutation(train.n_windows)
    stages = stage_steps(st)
    state_f = rd / "train_state.pt"
    start, proj, losses = 0, [], []
    if state_f.exists():
        s = torch.load(state_f, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start, proj, losses = int(s["step"]), list(s["proj"]), list(s["losses"])
        prog.say(f"{c['key']}: resumed at step {start} of {n_steps}")
    bf16 = tr["precision"] == "bf16" and dev.startswith("cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    mb = int(args.micro_batch or tr["micro_batch"])
    chunk = int(args.loss_chunk or tr["loss_chunk"])
    loader = cf.ThreadPoolExecutor(1)
    fetch = lambda i: torch.as_tensor(train.windows(order[i * B:(i + 1) * B]), dtype=torch.long)   # noqa: E731
    nxt = loader.submit(fetch, start) if start < n_steps else None
    if measure is not None and start == 0:
        ck0 = {**c, "key": f"{c['key']}/stage0", "stage": 0.0, "step": 0, "dir": rd / "checkpoints" / "stage0"}
        if not (ck0["dir"] / "row.json").exists():
            measure(model, ck0, sets, dev)
    diverged = False
    loss0 = losses[0] if losses else None
    proj_gpu: list = []
    t_train = time.perf_counter()
    for step in range(start, n_steps):
        X = nxt.result()
        nxt = loader.submit(fetch, step + 1) if step + 1 < n_steps else None
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for xb in X.split(mb):
            with amp:
                loss = lm_loss(model, xb.to(dev, non_blocking=True), chunk) * (len(xb) / len(X))
            loss.backward()
            tot += loss.detach()
        proj_gpu.append(D @ flat_grad(params))
        lval = float(tot)
        losses.append(lval)
        loss0 = lval if loss0 is None else loss0
        if not math.isfinite(lval) or (step > warm and lval > tr["diverge_factor"] * loss0):
            diverged = True
            prog.say(f"{c['key']}: diverged at step {step} (loss {lval:.3g}, first {loss0:.3g})")
            break
        opt.step()
        sched.step()
        done = step + 1
        if done % 500 == 0:
            el = time.perf_counter() - t_train
            prog.say(f"{c['key']}: step {done}/{n_steps}, loss {lval:.4f}, "
                     f"{(done - start) * B * L / el:,.0f} tokens/s")
        for stage, s_step in stages.items():
            if done == s_step:
                proj += torch.stack(proj_gpu).cpu().tolist()
                proj_gpu = []
                ck = {**c, "key": f"{c['key']}/stage{stage:g}", "stage": stage, "step": done,
                      "dir": rd / "checkpoints" / f"stage{stage:g}"}
                if (ck["dir"] / "row.json").exists():
                    continue
                if measure is None:
                    rec = gpu_pass(model, ck, sets, st, dev, prog, eb=args.eval_batch, jb=args.jvp_batch,
                                   hb=args.hvp_batch)
                else:
                    rec = measure(model, ck, sets, dev)
                for t in rec["tasks"]:
                    pool.submit(t)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": done, "proj": proj, "losses": losses}, state_f.with_suffix(".tmp"))
                os.replace(state_f.with_suffix(".tmp"), state_f)
                prog.say(f"{c['key']}: stage {stage:g} (step {done}) measured in {rec['row']['gpu_pass_seconds']:.0f} "
                         f"s; {len(rec['tasks'])} CPU tasks queued")
    loader.shutdown(wait=False)
    if proj_gpu:
        proj += torch.stack(proj_gpu).cpu().tolist()
    hg = np.asarray(proj, dtype=np.float64)
    p1_tasks.atomic_npy(rd / "hgrad.npy", hg)
    pd.DataFrame({"step": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(rd / "train_log.csv", index=False)
    if pool is not None and not diverged and len(hg) > warm + 50:
        unit = {"source": "p4", "folder": c["run"], "key": f"{c['run']}/hgrad", "path": str(rd / "hgrad.npy"),
                "stamp": "", "warm": warm, "tag": c["key"], "group": c["size"]}
        pool.submit({"name": "hlink_grad", "kind": "hlink", "key": c["key"], "dir": str(rd / "hlink"),
                     "seed": p1_tasks.seed_for(st["seed"], c["key"], "hlink_grad"), "params": {},
                     "unit": unit, "test": st["measure"]["hlink_test"]})
    del model, opt
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    if state_f.exists() and not args.keep_state:
        state_f.unlink()
    fl = p6_streams.floors(c["noah"], P)
    return {**{k: c[k] for k in ("key", "size", "noah", "joseph", "seed")}, "steps": n_steps, "warmup_steps": warm,
            "diverged": diverged, "final_train_loss": float(np.mean(losses[-max(1, len(losses) // 20):])),
            "stream_ok": bool(all(v["ok"] for v in checks.values())), **fl,
            "noah_score": P["noah"][c["noah"]]["score"], "joseph_score": P["joseph"][c["joseph"]]["score"],
            "train_hours": (time.perf_counter() - t0) / 3600}


def resubmit_missing(run_dir: Path, pool: P6Pool) -> int:
    """Tasks of row.json records (and the hgrad H_link unit) whose stats are missing: a job stopped before its pool
    finished."""
    n = 0
    for rj in sorted(run_dir.glob("checkpoints/*/row.json")):
        rec = json.loads(rj.read_text())
        for t in rec["tasks"]:
            if not p1_tasks.stats_path(t["dir"], t["name"]).exists():
                pool.submit(t)
                n += 1
    return n


def assemble_run(run_dir: Path, unit: dict) -> list[dict]:
    """Rows of one run: every stage's P1 row (p1_tasks.assemble) with the P6 columns and the hgrad H_link."""
    rows = []
    grad = p6_stats.json_load(p1_tasks.stats_path(run_dir / "hlink", "hlink_grad"))
    for rj in sorted(run_dir.glob("checkpoints/*/row.json")):
        d = rj.parent
        rec = json.loads(rj.read_text())
        stats = {t["name"]: p6_stats.json_load(p1_tasks.stats_path(d, t["name"])) for t in rec["tasks"]}
        res = p1_tasks.assemble(rec, stats, p1_checkpoints.h71_kw())
        p1_tasks.atomic_json(d / "result.json", res)
        stage = float(d.name.replace("stage", ""))
        extra = {k: unit.get(k) for k in ("noah", "joseph", "noah_score", "joseph_score", "diverged", "stream_ok",
                                          "oracle_floor", "topic_blind_floor", "topic_information",
                                          "rank_entropy")}
        extra.update(key=unit["key"], stage=stage)
        row = p6_stats.checkpoint_row(res, stats, extra)
        if grad:
            row.update(p6_stats.hlink_columns("grad", grad))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- timing and the cost model
def gpu_name(dev: str) -> str:
    return torch.cuda.get_device_name(0) if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"


def ratio_range(name: str) -> tuple[float, float, bool]:
    if "L40S" in name:
        return 1.0, 1.0, True
    for k, v in L40S_RATIO.items():
        if k in name:
            return v[0], v[1], False
    return L40S_RATIO_DEFAULT[0], L40S_RATIO_DEFAULT[1], False


SIZE_FACTOR_31M = (1.4, 2.3)   # 31M / 14M time: FLOPs per token 2.3x, the logits traffic (most of it) the same


def component_hours(t: dict) -> dict:
    """L40S hours per run component {size: {"train", "stage", "h71"}: [lo, hi]} from a timing.json: the timed GPU
    divided by its L40S ratio range (1 on the L40S); 31M from a 14M timing by SIZE_FACTOR_31M."""
    lo_r, hi_r = t["l40s_ratio"]
    here = {"train": t["train_seconds_per_run"] / 3600, "stage": t["measure_seconds_per_stage_without_h71"] / 3600,
            "h71": t["h71_seconds_full_estimate"] / 3600}
    out = {t["size"]: {k: [v / hi_r, v / lo_r] for k, v in here.items()}}
    if t["size"] == "14m":
        out["31m"] = {k: [v[0] * SIZE_FACTOR_31M[0], v[1] * SIZE_FACTOR_31M[1]] for k, v in out["14m"].items()}
    return out


def design_hours(design: dict, comp: dict, n_stages: int) -> list[float] | None:
    """[lo, hi] L40S hours of a design, one run after another (jobs side by side on the GPU can only shorten it)."""
    tot = np.zeros(2)
    n_h71 = len(design["h71_stages"])
    for r in plan(design):
        c = comp.get(r["size"])
        if c is None:
            return None
        tot += np.asarray(c["train"]) + n_stages * np.asarray(c["stage"]) + n_h71 * np.asarray(c["h71"])
    return [round(float(tot[0]), 2), round(float(tot[1]), 2)]


def cost_table(comp: dict, n_stages: int) -> dict:
    out = {}
    for name, d in DESIGNS.items():
        h = design_hours(d, comp, n_stages)
        if h is None:
            continue
        out[name] = {"runs": len(plan(d)), "h71_stages": d["h71_stages"], "gpu_hours": h,
                     "cost_usd": [round(h[0] * PRICE_PER_HOUR, 2), round(h[1] * PRICE_PER_HOUR, 2)]}
    return out


def timing(args, run, st: dict, prog: Progress) -> dict:
    """A few hundred training steps of one cell and one measurement pass (H7.1 on --timing-pool examples, scaled to
    the full pool), with peak GPU memory; writes timing.json and prints the per-run and per-design estimates."""
    dev = run.device
    P, tr = st["stream"], st["train"]
    size = args.size or ("tiny" if st["smoke"] else "14m")
    noah, joseph = args.noah, args.joseph
    B, L = int(tr["batch_seqs"]), int(P["seq_len"])
    n_steps = steps_of(st)
    t0 = time.perf_counter()
    train = p6_streams.Stream(noah, joseph, 1, "train", n_steps * B, P)
    held = p6_streams.Stream(noah, joseph, 1, "heldout", int(P["heldout_windows"]), P)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    every = 64 if not st["smoke"] else 1
    p6_streams.token_counts(train, every)
    t_check = (time.perf_counter() - t0) * every + 10.0        # + the latent checks (about 10 s, CPU)
    sets = heldout_sets(held, st["measure"])
    model = build_model(size, 1, P["vocab"]).to(dev)
    set_attention(model, "sdpa")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"], betas=tuple(tr["betas"]))
    params = [p for p in model.parameters() if p.requires_grad]
    D = flat_directions(params, int(tr["n_hgrad_dirs"]), 999, dev)
    bf16 = tr["precision"] == "bf16" and dev.startswith("cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    mb = int(args.micro_batch or tr["micro_batch"])
    chunk = int(args.loss_chunk or tr["loss_chunk"])
    cuda = dev.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    warm_steps = min(10, max(1, args.steps // 10))
    times = []
    for step in range(warm_steps + args.steps):
        if cuda:
            torch.cuda.synchronize()
        ts = time.perf_counter()
        X = torch.as_tensor(train.windows(range(step * B, (step + 1) * B)), dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for xb in X.split(mb):
            with amp:
                loss = lm_loss(model, xb.to(dev), chunk) * (len(xb) / len(X))
            loss.backward()
            tot += loss.detach()
        _ = D @ flat_grad(params)
        opt.step()
        lval = float(tot)
        if cuda:
            torch.cuda.synchronize()
        if step >= warm_steps:
            times.append(time.perf_counter() - ts)
        if (step + 1) % 50 == 0:
            prog.say(f"timing: step {step + 1}/{warm_steps + args.steps}, loss {lval:.3f}, "
                     f"{np.mean(times[-50:]) if times else float('nan'):.3f} s/step")
    peak_train = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float("nan")
    step_s = float(np.median(times))
    del opt
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    c = {"key": "timing", "run": "timing", "size": size, "noah": noah, "joseph": joseph, "seed": 1, "stage": 1.0,
         "step": n_steps, "dir": run.out_dir / "timing_pass"}
    h = st["measure"]["h71"]
    n_pool = int(h["coarse_blocks"]) * max(h["batch_sizes"])
    take = min(n_pool, int(args.timing_pool))
    rec = gpu_pass(model, c, sets, st, dev, prog, eb=args.eval_batch, jb=args.jvp_batch, h71_pool=take,
                   hb=args.hvp_batch)
    peak_measure = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float("nan")
    sec = rec["seconds"]
    h71_key = next(k for k in sec if k.startswith("H7.1 projections"))
    h71_full = sec[h71_key] * n_pool / take
    stage_s = sum(v for k, v in sec.items() if k != h71_key)
    n_stages = len(tr["stages"])
    name = gpu_name(dev)
    lo, hi, measured = ratio_range(name)
    out = {"gpu": name, "device": dev, "size": size, "cell": f"{noah}|{joseph}", "steps_timed": len(times),
           "micro_batch": mb, "batch_seqs": B, "seq_len": L, "step_seconds": step_s,
           "eval_batch": args.eval_batch, "jvp_batch": args.jvp_batch,
           "hvp_batch": int(args.hvp_batch or st["measure"]["hvp_batch"]),
           "step_seconds_p90": float(np.quantile(times, 0.9)), "tokens_per_second": B * L / step_s,
           "peak_train_gib": peak_train, "peak_measure_gib": peak_measure,
           "measure_seconds": sec, "h71_examples_timed": take, "h71_examples_full": n_pool,
           "h71_seconds_full_estimate": h71_full, "measure_seconds_per_stage_without_h71": stage_s,
           "stages": n_stages, "train_steps_per_run": n_steps, "train_seconds_per_run": n_steps * step_s,
           "run_hours_this_gpu_all_h71": (n_steps * step_s + n_stages * (stage_s + h71_full)) / 3600,
           "cpu_seconds_stream_build": t_build, "cpu_seconds_stream_check_estimate": t_check,
           "l40s_ratio": [lo, hi], "l40s_measured": measured, "price_per_hour": PRICE_PER_HOUR}
    out["l40s_component_hours"] = component_hours(out)
    out["designs"] = cost_table(out["l40s_component_hours"], n_stages)
    p1_tasks.atomic_json(run.out_dir / "timing.json", out)
    shutil.rmtree(c["dir"], ignore_errors=True)
    print_timing(out)
    return out


def print_timing(t: dict) -> None:
    lo, hi = t["l40s_ratio"]
    print(f"\nP6 timing on {t['gpu']} ({t['size']}, {t['cell']}, micro-batch {t['micro_batch']} of {t['batch_seqs']} "
          f"x {t['seq_len']} tokens):")
    print(f"  training: {t['step_seconds']:.3f} s/step (median of {t['steps_timed']}), "
          f"{t['tokens_per_second']:,.0f} tokens/s, peak {t['peak_train_gib']:.2f} GiB; "
          f"{t['train_steps_per_run']} steps: {t['train_seconds_per_run'] / 3600:.2f} h")
    print(f"  one measurement stage: {t['measure_seconds_per_stage_without_h71']:.0f} s + H7.1 JVPs "
          f"{t['h71_seconds_full_estimate']:.0f} s (scaled from {t['h71_examples_timed']} of "
          f"{t['h71_examples_full']} examples), peak {t['peak_measure_gib']:.2f} GiB")
    print(f"  one run with H7.1 at all {t['stages']} stages: {t['run_hours_this_gpu_all_h71']:.2f} h on this GPU")
    print("  this is the L40S: the hours below are measured" if t["l40s_measured"] else
          f"  L40S: {lo:g}-{hi:g}x faster than this GPU (prior; the queue's first job measures it)")
    for s, c in t["l40s_component_hours"].items():
        print(f"  L40S, {s}: training {c['train'][0]:.2f}-{c['train'][1]:.2f} h per run, measurements "
              f"{c['stage'][0] * 60:.1f}-{c['stage'][1] * 60:.1f} min per stage + H7.1 {c['h71'][0] * 60:.1f}-"
              f"{c['h71'][1] * 60:.1f} min per stage")
    for name, d in t["designs"].items():
        print(f"  {name}: {d['runs']} runs (H7.1 at stages {d['h71_stages']}), {d['gpu_hours'][0]:.1f}-"
              f"{d['gpu_hours'][1]:.1f} L40S h, ${d['cost_usd'][0]:.0f}-{d['cost_usd'][1]:.0f} at "
              f"${t['price_per_hour']}/h")


# prior from FLOPs and bytes moved (not a measurement; the timing runs replace it): L40S peaks and the fractions
# of them small kernels reach (OURS)
L40S_SPEC = {"bf16_tflops": 362.0, "fp32_tflops": 91.6, "gb_s": 864.0, "compute_eff": 0.35, "bandwidth_eff": 0.7,
             "step_overhead_s": 0.015, "jvp_call_s": 0.005}


def prior_component_hours(size: str, st: dict, spec: dict = L40S_SPEC) -> dict:
    """L40S hours per run component (training; one stage's measurements without H7.1; H7.1's JVPs per stage) from
    FLOPs and bytes per token: training 6 N_body + 8 N_head (the head recomputed) + 3 n_layer T d FLOPs and
    28 V + 6 n_layer (24 d + 8 ff) bytes per token (logits in bf16 and float32, forward and backward); JVPs
    4 (N_body + N_head) FLOPs and 24 V bytes per token and direction in float32. Time = max(compute, bytes) +
    0.3 x min(compute, bytes) + per-step or per-call overhead."""
    c = size_cfg(size)
    d, L, ff, V = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], st["stream"]["vocab"]
    T = st["stream"]["seq_len"]
    body, head = L * (4 * d * d + 2 * d * ff), d * V

    def t(n_tok, flops, nbytes, tflops):
        a = n_tok * flops / (spec["compute_eff"] * tflops * 1e12)
        b = n_tok * nbytes / (spec["bandwidth_eff"] * spec["gb_s"] * 1e9)
        return max(a, b) + 0.3 * min(a, b)

    tok = st["train"]["tokens"]
    train = t(tok, 6 * body + 8 * head + 3 * L * T * d, 28 * V + 6 * L * (24 * d + 8 * ff), spec["bf16_tflops"])
    train += steps_of(st) * spec["step_overhead_s"]
    m = st["measure"]
    ex = m["example_len"]
    n_h71 = m["h71"]["coarse_blocks"] * max(m["h71"]["batch_sizes"])
    dirs = m["n_random_dirs"] + m["n_hessian_dirs"]
    jvp = lambda n_ex, k: (t(n_ex * ex * k, 4 * (body + head), 24 * V, spec["fp32_tflops"])     # noqa: E731
                           + n_ex / 32 * k * spec["jvp_call_s"])
    h71 = jvp(n_h71, dirs)
    fwd = (m["eval_seqs"] * m["eval_len"] + m["surprisal_docs"] * T + 2 * m["probe_seqs"] * m["probe_len"])
    stage = jvp(m["coupling_blocks"] * m["coupling_B"], m["n_random_dirs"]) + t(fwd, 2 * (body + head), 16 * V,
                                                                               spec["fp32_tflops"]) + 20.0
    return {"train": train / 3600, "stage": stage / 3600, "h71": h71 / 3600}


def prior_peak_gib(size: str, st: dict, mb: int | None = None, chunk: int | None = None, eb: int = 4, jb: int = 32,
                   hb: int | None = None) -> dict:
    """GPU memory (GiB, tensors). Training (bfloat16, flash attention, micro-batch mb with the loss in chunks of
    `chunk` tokens): from the sizes of the weights, optimizer state, saved activations and one chunk of logits (a
    prior: the meta device cannot run the flash kernel; a CPU bfloat16 run of the same step gave 3.9 GiB at 14M and
    5.6 at 31M, 2026-11-18). Measurements: measured on the meta device (measure_peaks)."""
    c = size_cfg(size)
    d, L, ff, V = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], st["stream"]["vocab"]
    n_par = L * (4 * d * d + 2 * d * ff) + 2 * d * V
    static = n_par * (16 + 4 * st["train"]["n_hgrad_dirs"] + 4)       # weights, grads, AdamW, directions, flat grad
    act = int(mb or st["train"]["micro_batch"]) * st["stream"]["seq_len"] * L * (20 * d + 4 * ff)
    logits = int(chunk or st["train"]["loss_chunk"]) * V * 14
    meas = measure_peaks(size, st, eb, jb, hb)
    return {"train": (static + act + logits) / 2 ** 30, "measure": meas["measure"], "measure_steps": meas["steps"],
            "training_state": meas["training_state"]}


def measure_peaks(size: str, st: dict, eb: int = 4, jb: int = 32, hb: int | None = None) -> dict:
    """Peak GPU memory (GiB, tensors) of each step of a measurement stage, run on torch's meta device under
    memory_check.py's tracker (shapes only: no GPU, no data, seconds), plus the training state the GPU keeps while
    it measures (gradients, AdamW moments, the hgrad directions). The CUDA context and allocator rounding come on
    top, as for run_queue.py's peaks (1.05 x peak + 0.6 GiB)."""
    import memory_check as mc
    m, V, K = st["measure"], int(st["stream"]["vocab"]), int(st["stream"]["K"])
    hb = int(hb or m["hvp_batch"])
    with torch.device("meta"):
        model = build_model(size, 1, V)
    model.eval()
    names = lm.grad_param_names(model)
    params = {n: p for n, p in model.named_parameters() if n in names}
    T = int(m["example_len"])

    def ids(n, t):
        return torch.randint(0, V, (int(n), int(t)), device="meta")

    out = {}
    x = ids(eb, m["probe_len"])
    out[f"probe loss ({eb} x {m['probe_len']})"] = mc.peak_gib(lambda: mc.nograd_losses(model, x), mc.state(model, x))
    out[f"hidden states ({eb} x {m['probe_len']})"] = mc.peak_gib(lambda: mc.nograd_hidden(model, x),
                                                                  mc.state(model, x))
    xs = ids(eb, st["stream"]["seq_len"])

    def sur():
        with torch.no_grad():
            _surprisal_batch(model, xs, K)
    out[f"surprisal parts ({eb} x {st['stream']['seq_len']})"] = mc.peak_gib(sur, mc.state(model, xs))
    xe = ids(eb, m["eval_len"])
    out[f"eval loss ({eb} x {m['eval_len']})"] = mc.peak_gib(lambda: mc.nograd_losses(model, xe), mc.state(model, xe))
    k = int(m["n_random_dirs"]) + int(m["n_hessian_dirs"])
    dirs = [{n: torch.randn_like(p) for n, p in params.items()} for _ in range(k)]
    xj = ids(jb, T)
    out[f"per-example JVP ({jb} x {T}, {k} directions)"] = mc.peak_gib(
        lambda: gradients.per_example_projections(model, lm.per_example_lm_loss, xj, params, dirs),
        mc.state(model, dirs, xj))
    nh = int(m["hessian_batch"])
    xh = ids(nh, T)
    hvp, _ = gradients.hvp_fn(model, lm.per_example_lm_loss, xh, params, chunk=hb if hb < nh else None)
    v = {n: torch.randn_like(p) for n, p in params.items()}
    out[f"HVP, Hessian direction ({nh} x {T} in micro-batches of {min(hb, nh)})"] = mc.peak_gib(
        lambda: hvp(v), mc.state(model, v, xh))
    n_par = sum(p.numel() for p in model.parameters())
    state = n_par * (4 + 8 + 4 * int(st["train"]["n_hgrad_dirs"])) / 2 ** 30
    return {"steps": out, "training_state": state, "measure": state + max(out.values())}


def print_memory(args, st: dict) -> None:
    sizes = [x for x in (args.sizes or ",".join(SIZES)).split(",") if x]
    eb, jb = args.eval_batch, args.jvp_batch
    for size in sizes:
        r = prior_peak_gib(size, st, args.micro_batch, args.loss_chunk, eb, jb, args.hvp_batch)
        mb = int(args.micro_batch or st["train"]["micro_batch"])
        chunk = int(args.loss_chunk or st["train"]["loss_chunk"])
        print(f"{size}: GPU memory, tensors (GiB); the CUDA context and allocator rounding come on top")
        print(f"  training, micro-batch {mb} x {st['stream']['seq_len']}, loss chunk {chunk}: {r['train']:.2f} (prior)")
        for name, g in r["measure_steps"].items():
            print(f"  {name}: {g:.2f}")
        print(f"  training state kept during the measurements: {r['training_state']:.2f}")
        print(f"  measuring, largest step + state: {r['measure']:.2f}; job peak {max(r['train'], r['measure']):.2f}")


# ----------------------------------------------------------------------------- feasibility check (2026-11-18)
# Fixed 2026-11-18 before any P6 run; a feasibility check, not a test (one cell: no contrast between cells exists).
# In P1, Pythia's outlier channels are absent at step 1,000 (~2B tokens) and present at step 4,000 (~8B) at every
# size; P6 trains on 2^28 tokens (0.27B) in 8,192 steps. One full-length P6 run of one cell, measured only by P1's
# activation moments on the probe set at step 0 and the four stages, decides whether the token budget is raised.
FEASIBILITY = {
    "cell": ("zipf1.0", "H0.9"),     # both factors on: the cell most likely to grow outliers (Zipf 1.0 the heavier)
    "seed": 6,                       # outside the design's seeds 1-5: no run of the P6 sets is seen before P6
    "k_tol": 1.0,                    # K and K_max "at their initial level": at most 1 (excess kurtosis) above step 0
    "raise": {"tokens": 2 ** 33, "batch_seqs": 512},   # if they stay: x32 tokens (Pythia's step-4,000 count) by the
                                                       # batch, at the same 8,192 steps, stages and learning rate
}


def outlier_pass(model, ck: dict, sets: dict, dev: str, eb: int = 4) -> dict:
    """The feasibility check's measurement, and nothing else: gpu_pass's activation moments on the probe set
    (spectra.activation_moments on each layer's output, P1's definitions): K = median over layers of the excess
    kurtosis, K_max = its largest value, outlier channels = channels with |activation| > 6 anywhere on the probe set,
    summed over layers (p1_tasks.assemble)."""
    d = Path(ck["dir"])
    d.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    was_training = model.training
    model.eval()
    set_attention(model, "eager")
    hs = lm.hidden_states(model, sets["probe"], dev, batch=eb)
    moments = []
    for H in hs:
        Hd = H.to(dev)
        moments.append(spectra.activation_moments(Hd))
        del Hd
    del hs
    set_attention(model, "sdpa")
    model.train(was_training)
    layers = [{"layer": i, "K": m["K"], "n_outlier_channels": int(m["n_outlier_channels"])}
              for i, m in enumerate(moments)]
    row = {"stage": float(ck["stage"]), "step": int(ck["step"]),
           "K": float(np.median([m["K"] for m in moments])), "K_max": float(np.max([m["K"] for m in moments])),
           "n_outlier_channels": int(np.sum([m["n_outlier_channels"] for m in moments])),
           "gpu_pass_seconds": round(time.perf_counter() - t, 1)}
    rec = {"row": row, "layers": layers, "tasks": []}
    p1_tasks.atomic_json(d / "row.json", rec)
    return rec


def feasibility_verdict(rows: list[dict], diverged: bool = False, k_tol: float = FEASIBILITY["k_tol"]) -> dict:
    """The rule fixed on 2026-11-18. rows: step 0 and the stages. 'raise' if at the final stage there are no more
    outlier channels than at step 0 and neither K nor K_max is more than k_tol above its step-0 value (they stay at
    their initial level); otherwise 'keep'."""
    by = {float(r["stage"]): r for r in rows}
    if diverged:
        return {"verdict": "diverged", "reason": "the run diverged: the check gives no answer"}
    if 0.0 not in by or 1.0 not in by:
        return {"verdict": "incomplete", "reason": f"stages measured: {sorted(by)}"}
    a, b = by[0.0], by[1.0]
    up = {"outlier channels": b["n_outlier_channels"] > a["n_outlier_channels"],
          "K": b["K"] > a["K"] + k_tol, "K_max": b["K_max"] > a["K_max"] + k_tol}
    if any(up.values()):
        return {"verdict": "keep", "reason": "above the initial level at the final stage: "
                + ", ".join(k for k, v in up.items() if v)}
    return {"verdict": "raise", "reason": f"K, K_max and outlier channels at their initial level at the final stage "
            f"(within {k_tol:g} of step 0, no new outlier channel)"}


def feasibility(args, run, st: dict, prog) -> dict:
    noah, joseph = args.noah, args.joseph          # the defaults are FEASIBILITY["cell"]
    size = args.size or ("tiny" if st["smoke"] else "14m")
    seed = int(FEASIBILITY["seed"])
    key = run_key(size, noah, joseph, seed)
    c = {"key": key, "run": key, "size": size, "noah": noah, "joseph": joseph, "seed": seed,
         "dir": run.out_dir / "runs" / key}
    prog.say(f"feasibility check: one full-length run of {key} ({steps_of(st)} steps of "
             f"{st['train']['batch_seqs']} x {st['stream']['seq_len']} tokens), measured only by K, K_max and "
             f"outlier channels at step 0 and the stages {st['train']['stages']}")
    res = train_run(c, st, args, run, None, prog,
                    measure=lambda model, ck, sets, dev: outlier_pass(model, ck, sets, dev, eb=args.eval_batch))
    B, L = int(st["train"]["batch_seqs"]), int(st["stream"]["seq_len"])
    rows, layer_rows = [], []
    for f in sorted((Path(c["dir"]) / "checkpoints").glob("stage*/row.json")):
        rec = p6_stats.json_load(f)
        r = {**rec["row"], "tokens": rec["row"]["step"] * B * L}
        rows.append(r)
        layer_rows += [{"stage": r["stage"], "step": r["step"], **x} for x in rec["layers"]]
    rows.sort(key=lambda r: r["stage"])
    cols = ["stage", "step", "tokens", "K", "K_max", "n_outlier_channels"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(run.out_dir / "feasibility.csv", index=False)
    pd.DataFrame(layer_rows).sort_values(["stage", "layer"]).to_csv(run.out_dir / "feasibility_layers.csv",
                                                                     index=False)
    v = feasibility_verdict(rows, res["diverged"])
    out = {"cell": key, "rule": {k: FEASIBILITY[k] for k in ("k_tol", "raise")}, **v,
           "stages": df.to_dict(orient="records"), "train_hours": res["train_hours"]}
    p1_tasks.atomic_json(run.out_dir / "feasibility.json", out)
    print(f"\nP6 feasibility check, {key} ({res['train_hours']:.2f} h on {gpu_name(run.device)}):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"rule (fixed 2026-11-18): {v['verdict'].upper()}: {v['reason']}")
    if v["verdict"] == "raise":
        r = FEASIBILITY["raise"]
        print(f"  -> before P6 starts: {r['tokens']:,} tokens ({r['tokens'] / st['train']['tokens']:.0f}x), batch "
              f"{r['batch_seqs']} sequences at the same {steps_of(st)} steps (README, 'Feasibility check')")
    return out


MIN_SEEDS = 3


def budget_seeds(args, design: dict) -> list[int] | None:
    """--budget-hours with --timing-json (the queue's L40S timing jobs): the seeds the design may run, the first k of
    its seeds with k the largest number whose runs fit the budget at the measured hours, at least MIN_SEEDS (else
    the job stops and nothing starts). Fixed before any run: the number of seeds depends on the timing only.
    None: no budget given (every seed)."""
    if args.budget_hours is None:
        return None
    files = [Path(args.timing_json or "")] + [Path(x) for x in (args.timing_json_extra or "").split(",") if x]
    comp = {}
    for f in files:
        t = p6_stats.json_load(f) if f.exists() else None
        if not t:
            raise SystemExit(f"--budget-hours {args.budget_hours} needs the timing file {f} (missing)")
        comp.update({k: v for k, v in component_hours(t).items() if k == t["size"] or k not in comp})
    n_stages = len(TRAIN["stages"])
    per_seed = design_hours({**design, "seeds": [design["seeds"][0]]}, comp, n_stages)
    if per_seed is None:
        raise SystemExit(f"P6 budget: no timing for the sizes {design['sizes']} ({', '.join(map(str, files))})")
    h = per_seed[1]                                  # the upper end (= the measured hours, on the L40S)
    k = min(len(design["seeds"]), int(args.budget_hours // h)) if h > 0 else len(design["seeds"])
    if k < MIN_SEEDS:
        raise SystemExit(f"P6 budget: {MIN_SEEDS} seeds need {MIN_SEEDS * h:.1f} L40S h at the measured hours "
                         f"({h:.2f} h per seed) > --budget-hours {args.budget_hours}; nothing started")
    print(f"[p6] budget: {h:.2f} L40S h per seed ({len(plan({**design, 'seeds': [1]}))} runs); {k} of "
          f"{len(design['seeds'])} seeds fit {args.budget_hours} h ({k * h:.1f} h)", flush=True)
    return list(design["seeds"][:k])


# ----------------------------------------------------------------------------- analysis
def analyse(args, run, st: dict) -> None:
    src = Path(args.from_dir)
    rows = []
    for uj in sorted(src.glob("**/units/*.json")):
        if uj.name == "settings.json":
            continue
        u = json.loads(uj.read_text())
        rd = uj.parent.parent / "runs" / u["key"]
        if rd.exists():
            rows += assemble_run(rd, u)
    if not rows:
        raise SystemExit(f"no finished P6 runs below {src}")
    df = pd.DataFrame(rows)
    df.to_csv(run.out_dir / "checkpoints.csv", index=False)
    rng = np.random.default_rng(np.random.SeedSequence(units_mod.seed_for(config.SEED, "p6 analysis")))
    corners = st["stream"]["corners"]
    size0 = "tiny" if st["smoke"] else p6_stats.SIZE_PRIMARY
    prim = df[(df["size"] == size0) & np.isclose(df["stage"], p6_stats.STAGE_PRIMARY)
              & df["noah"].isin(corners["noah"]) & df["joseph"].isin(corners["joseph"]) & ~df["diverged"].astype(bool)]
    n_perm = 199 if st["smoke"] else p6_stats.N_PERM
    fam = p6_stats.family_tests(prim, rng, n_perm)
    fam.to_csv(run.out_dir / "family.csv", index=False)
    man = p6_stats.manipulation_checks(prim, rng, n_perm)
    man.to_csv(run.out_dir / "manipulation_checks.csv", index=False)
    sec = p6_stats.secondary_tests(df[~df["diverged"].astype(bool)], rng, 199 if st["smoke"] else 999)
    sec.to_csv(run.out_dir / "secondary.csv", index=False)
    header = (f"{len(prim)} runs in the confirmatory set ({size0}, corners, final stage); "
              f"{len(df)} checkpoint rows in all; diverged runs: {int(df.groupby('key')['diverged'].first().sum())}")
    (run.out_dir / "summary.md").write_text(p6_stats.to_markdown(fam, man, header))
    run.note(p6_stats.verdict(fam))


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = runtime.base_parser(__doc__.split("\n")[0])
    ap.add_argument("--design", choices=sorted(DESIGNS), default="small")
    ap.add_argument("--sizes", default=None, help="comma-separated sizes of the design to run (default all)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds of the design to run (default all)")
    ap.add_argument("--plan", action="store_true", help="list the runs and the cost model; run nothing")
    ap.add_argument("--timing", action="store_true", help="time --steps training steps and one measurement pass")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--size", default=None, help="timing: model size (default 14m)")
    ap.add_argument("--noah", default="zipf1.0", choices=sorted(p6_streams.P6["noah"]))
    ap.add_argument("--joseph", default="H0.9", choices=sorted(p6_streams.P6["joseph"]))
    ap.add_argument("--timing-pool", type=int, default=4096, help="timing: H7.1 examples timed (scaled to the pool)")
    ap.add_argument("--micro-batch", type=int, default=None, help="sequences per forward pass (default 16; the "
                                                                   "gradient is the same, accumulated)")
    ap.add_argument("--loss-chunk", type=int, default=None, help="tokens per chunk of the output projection and "
                                                                  "softmax (default 4,096; the same loss)")
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--jvp-batch", type=int, default=32)
    ap.add_argument("--hvp-batch", type=int, default=None, help="sequences per micro-batch of the Hessian-vector "
                                                                 "products of H7.1's Hessian direction (default 16; "
                                                                 "the same H v)")
    ap.add_argument("--feasibility", action="store_true", help="the feasibility check (2026-11-18): one full-length "
                                                                "run of Zipf 1.0 x H 0.9, seed 6, measured only by K, "
                                                                "K_max and outlier channels; prints the rule's verdict")
    ap.add_argument("--memory", action="store_true", help="GPU memory of each step on the meta device (no GPU); "
                                                           "with the batch flags given")
    ap.add_argument("--workers", type=int, default=2, help="CPU pool processes")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--timing-json", default=None, help="timing.json of the queue's L40S timing job")
    ap.add_argument("--timing-json-extra", default=None, help="comma-separated further timing.json (other sizes)")
    ap.add_argument("--keep-state", action="store_true", help="keep train_state.pt after a run")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--from", dest="from_dir", default=None, help="analyse: folder with the job folders")
    args = ap.parse_args(argv)
    design = SMOKE_DESIGN if args.smoke else DESIGNS[args.design]
    st = settings(args.smoke, design)
    if args.plan:
        comp = {s: {k: [v, v] for k, v in prior_component_hours(s, st).items()} for s in SIZES}
        for name, d in DESIGNS.items():
            h = design_hours(d, comp, len(st["train"]["stages"]))
            print(f"{name}: {len(plan(d))} runs = sizes {d['sizes']} x Noah {d['noah']} x Joseph {d['joseph']} x "
                  f"seeds {d['seeds']}; {steps_of(st)} steps of {st['train']['batch_seqs']} x "
                  f"{st['stream']['seq_len']} tokens per run; stages {st['train']['stages']}, H7.1 at "
                  f"{d['h71_stages']}; prior {h[0]:.1f} L40S h, ${h[0] * PRICE_PER_HOUR:.0f}")
        for s in SIZES:
            c, m = comp[s], prior_peak_gib(s, st)
            print(f"  prior per run, {s}: training {c['train'][0] * 60:.0f} min, measurements "
                  f"{c['stage'][0] * 60:.1f} min per stage + H7.1 {c['h71'][0] * 60:.1f} min per stage; GPU memory "
                  f"about {m['train']:.1f} GiB training, {m['measure']:.1f} GiB measuring (tensors; --memory)")
        print("  (hours: a prior from FLOPs and bytes moved, not a measurement: --timing measures)")
        return 0
    if args.memory:
        print_memory(args, st)
        return 0
    run = runtime.start_run(TEST, {**st, "design": args.design if not args.smoke else "smoke"}, args)
    prog = Progress(run.out_dir)
    if args.timing:
        timing(args, run, st, prog)
        runtime.finish_run(run)
        return 0
    if args.feasibility:
        out = feasibility(args, run, st, prog)
        runtime.finish_run(run, extra={"verdict": out["verdict"]})
        return 0
    if args.analyse:
        if not args.from_dir:
            raise SystemExit("--analyse needs --from <folder with the P6 job folders>")
        analyse(args, run, st)
        runtime.finish_run(run)
        return 0
    sizes = args.sizes.split(",") if args.sizes else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    allowed = budget_seeds(args, design)
    todo = plan(design, sizes, seeds)
    if allowed is not None:
        skipped = [c["key"] for c in todo if c["seed"] not in allowed]
        todo = [c for c in todo if c["seed"] in allowed]
        if skipped:
            p1_tasks.atomic_json(run.out_dir / "skipped_by_budget.json", {"runs": skipped, "seeds_run": allowed})
            run.note(f"budget: {len(skipped)} run(s) of seeds outside {allowed} skipped")
        if not todo:
            runtime.finish_run(run, extra={"runs": 0, "skipped_by_budget": skipped})
            return 0
    if not todo:
        raise SystemExit("no runs selected")
    stable.set_density("grid")
    grid_path = stable._GRID.path
    units = units_mod.Units(run.out_dir, {k: st[k] for k in ("stream", "train", "measure", "seed")}, TEST,
                            total=len(todo), log=run.note)
    units.plan([c["key"] for c in todo])
    for v in p1_tasks.THREAD_VARS:
        os.environ[v] = "1"
    start_cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    pool = P6Pool(lambda: start_cpus, max(1, args.workers), "grid", grid_path, prog, {})

    def on_signal(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        for c in todo:
            c.update(run=c["key"], dir=run.out_dir / "runs" / c["key"])
            n = resubmit_missing(Path(c["dir"]), pool)
            if n:
                prog.say(f"{c['key']}: {n} CPU task(s) of an earlier session queued again")
            if units.has(c["key"]):
                prog.say(f"{c['key']}: done in an earlier session")
                hg = p1_tasks.stats_path(Path(c["dir"]) / "hlink", "hlink_grad")
                u = units.get(c["key"])
                if not hg.exists() and not u.get("diverged") and (Path(c["dir"]) / "hgrad.npy").exists():
                    unit = {"source": "p4", "folder": c["key"], "key": f"{c['key']}/hgrad",
                            "path": str(Path(c["dir"]) / "hgrad.npy"), "stamp": "", "warm": u["warmup_steps"],
                            "tag": c["key"], "group": c["size"]}
                    pool.submit({"name": "hlink_grad", "kind": "hlink", "key": c["key"],
                                 "dir": str(Path(c["dir"]) / "hlink"), "params": {}, "unit": unit,
                                 "seed": p1_tasks.seed_for(st["seed"], c["key"], "hlink_grad"),
                                 "test": st["measure"]["hlink_test"]})
                continue
            units.start(c["key"])
            res = train_run(c, st, args, run, pool, prog)
            for rec in pool.collect():
                pass
            h = units.put(c["key"], res)
            prog.say(f"{c['key']}: run finished in {h:.2f} h (train {res['train_hours']:.2f} h incl. measurements); "
                     f"{pool.busy()} CPU tasks pending")
        prog.say(f"all runs trained; waiting for {pool.busy()} CPU tasks")
        pool.wait_all(lambda rec: None)
    finally:
        pool.shutdown()
    rows = []
    for c in todo:
        u = units.get(c["key"])
        if u:
            rows += assemble_run(Path(c["dir"]), u)
    if rows:
        pd.DataFrame(rows).to_csv(run.out_dir / "checkpoints.csv", index=False)
    runtime.finish_run(run, extra={"runs": len(todo)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
