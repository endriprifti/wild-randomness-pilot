"""Peak GPU memory of the pilot's largest steps, estimated without a GPU and without downloads.

Each model is built from its architecture on torch's meta device (shapes, no data) and each step runs under
a dispatch mode that adds up the bytes of the distinct tensor storages alive after every operation. The peak
counts tensors only: the CUDA context (about 0.5 GiB), cuBLAS workspaces and allocator rounding come on top,
so a step is marked "tight" above 85% of the limit. Calibration against the L40S (2026-11-15): P4's
transformer HVP on all 512 measurement sequences estimates 537 GiB (it failed with 44 GiB in use) and its
training step 21 GiB (it ran for an hour).

Checks, at the sizes in config.py (full runs, float32, eager attention as in common/hf.py):
  p1  every suite and size: no-grad forwards at eval_batch (probe loss and hidden states at probe_len, surprisal
      and eval loss at 2,048 tokens), per-example JVPs at jvp_batch x example_len, and the HVP at
      hessian_batch x example_len for the sizes that run H7.1
  p4  training step (AdamW / SGD state included) and every batch measure of measure() at its micro-batch
  p3  one training step of every arm, task and state size at matched parameters
  p2  each model in its config dtype (bfloat16): a perplexity window, PopQA generation (a batch of prompts of
      256 tokens, more than PopQA's few-shot prompts, then max_new_tokens steps with the cache), hidden states,
      and every scheme's restore + quantization of the weights; perplexity and PopQA again under the rotation and
      outlier schemes (their activations are rotated or copied in float32)
  p5  the escape grid's largest steps: a full-batch gradient of all restarts at once (injected noise, the descent
      of every exit check), the loss barrier on the segment, a minibatch gradient at the largest B, and the
      HVP and per-example projections of alpha_at

  python memory_check.py                    # every part against the L40S's 44.39 GiB
  python memory_check.py p1 --limit 23.5    # one part against another card
  python memory_check.py p4 --calibrate     # adds the unchunked P4 HVP that failed on the L40S
  python memory_check.py p2 --calibrate     # adds P2's quantization as before 2026-11-17 (every layer held)

run_queue.py admits jobs by these peaks (its PEAK_GIB table; tests/test_run_queue.py checks it against this script).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import torch
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import config
from common import gradients, lm, ssm

GiB = 2 ** 30
L40S_GIB = 44.39          # total capacity CUDA reports on the CloudRift L40S (p4_transformer.log)
TIGHT = 0.85

# public architecture configs (config.json of each repo); vocabularies are the padded embedding sizes
PYTHIA = {"14m": (128, 6, 4), "31m": (256, 6, 8), "70m": (512, 6, 8), "160m": (768, 12, 12),
          "410m": (1024, 24, 16), "1b": (2048, 16, 8), "1.4b": (2048, 24, 16)}     # hidden, layers, heads
OLMO2 = {"1b": (2048, 16, 16, 8192), "7b": (4096, 32, 32, 11008)}                  # + intermediate size


class PeakTracker(TorchDispatchMode):
    """Bytes of distinct live storages, swept after every op; `keep` are tensors alive throughout."""

    def __init__(self, keep=(), ignore=()):
        super().__init__()
        self.refs, self.peak = {}, 0
        # storages that live elsewhere (P2's CPU copies of the weights): an op that reads them (detach in deepcopy)
        # returns them, and they would otherwise be counted from then on
        self.ignore = set()
        for t in ignore:
            try:
                self.ignore.add(StorageWeakRef(t.untyped_storage()).cdata)
            except Exception:  # noqa: BLE001
                pass
        for t in keep:
            self._add(t)
        self._sweep()

    def _add(self, t):
        if not isinstance(t, torch.Tensor):
            return
        try:
            st = t.untyped_storage()
        except Exception:  # noqa: BLE001  functorch wrappers carry no storage of their own
            return
        ref = StorageWeakRef(st)
        if ref.cdata not in self.refs and ref.cdata not in self.ignore:
            self.refs[ref.cdata] = (ref, st.nbytes())

    def _sweep(self):
        for k in [k for k, (r, _) in self.refs.items() if r.expired()]:
            del self.refs[k]
        self.peak = max(self.peak, sum(nb for _, nb in self.refs.values()))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            self._add(t)
        self._sweep()
        return out


def peak_gib(fn, keep) -> float:
    tr = PeakTracker(keep)
    with tr:
        fn()
    return tr.peak / GiB


def peak_gib_model(fn, model, *extra, ignore=()) -> float:
    """As peak_gib, without holding the model's tensors: a step that replaces layers (P2's quantization) frees the
    old weights as the real run does. `ignore`: tensors that are not on the GPU in the real run."""
    tr = PeakTracker(state(model, *extra), ignore)    # the list dies here; the tracker keeps weak references only
    with tr:
        fn()
    return tr.peak / GiB


def state(model, *extra):
    return list(model.parameters()) + list(model.buffers()) + [t for e in extra for t in tree_leaves(e)]


# ----------------------------------------------------------------------------- models (on the meta device)
def causal_lm(family: str, size: str):
    if family == "pythia":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        h, n_layers, heads = PYTHIA[size]
        c = GPTNeoXConfig(vocab_size=50304, hidden_size=h, num_hidden_layers=n_layers, num_attention_heads=heads,
                          intermediate_size=4 * h, max_position_embeddings=2048, rotary_pct=0.25,
                          use_parallel_residual=True, tie_word_embeddings=False)
        cls = GPTNeoXForCausalLM
    else:
        from transformers import Olmo2Config, Olmo2ForCausalLM
        h, n_layers, heads, inter = OLMO2[size]
        c = Olmo2Config(vocab_size=100352, hidden_size=h, num_hidden_layers=n_layers, num_attention_heads=heads,
                        num_key_value_heads=heads, intermediate_size=inter, max_position_embeddings=4096,
                        tie_word_embeddings=False)
        cls = Olmo2ForCausalLM
    c._attn_implementation = "eager"
    return cls(c).eval()


def nograd_losses(model, x):
    """Mirrors lm.token_losses on one micro-batch (without the copy to the CPU)."""
    with torch.no_grad():
        logits = model(input_ids=x).logits.float()
        F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")


def nograd_hidden(model, x):
    """Mirrors lm.hidden_states on one micro-batch."""
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True)
        [h.float().reshape(-1, h.shape[-1]) for h in out.hidden_states[1:]]


def hvp_step(model, pel, batch, params):
    hvp, _ = gradients.hvp_fn(model, pel, batch, params)
    v = {n: torch.randn_like(p) for n, p in params.items()}
    return (lambda: hvp(v)), [v, batch]


def jvp_step(model, pel, batch, params):
    d = [{n: torch.randn_like(p) for n, p in params.items()}]
    return (lambda: gradients.per_example_projections(model, pel, batch, params, d)), [d, batch]


def backward_step(model, pel, batch):
    def f():
        pel(lambda *a, **kw: model(*a, **kw), batch).mean().backward()
        model.zero_grad(set_to_none=True)
    return f, [batch]


# ----------------------------------------------------------------------------- P1
def check_p1(rows, log):
    cfg = config.get("p1_checkpoints", False)
    h71 = cfg["h71"]["full_res_sizes"] + cfg["h71"]["coarse_sizes"]
    T, Te = cfg["example_len"], max(cfg["surprisal_len"], cfg["eval_len"])
    seen, listed = {}, set()
    for suite, sc in cfg["suites"].items():
        family = "olmo" if "olmo" in suite else "pythia"
        eb, jb = int(sc.get("eval_batch", 4)), int(sc.get("jvp_batch", 32))
        for size in sc["sizes"]:
            key = (family, size, eb, jb, suite == "pythia" and size in h71)
            job = f"p1_{suite}" if suite in ("pythia", "polypythia", "olmo", "olmo7b") else "p1_polypythia"
            if key in seen:                  # same architecture and batches: same numbers, listed once per job
                if (key, job) not in listed:
                    rows.extend({**r, "job": job} for r in seen[key])
                    listed.add((key, job))
                continue
            t0 = time.time()
            out = []
            with torch.device("meta"):
                model = causal_lm(family, size)
                V = model.config.vocab_size
                names = lm.grad_param_names(model)
                params = {n: p for n, p in model.named_parameters() if n in names}
                w = sum(p.numel() * p.element_size() for p in model.parameters()) / GiB
                out.append(("weights", "", w))
                x = torch.randint(0, V, (eb, cfg["probe_len"]))
                out.append(("probe loss", f"{eb} x {cfg['probe_len']}", peak_gib(lambda: nograd_losses(model, x), state(model, x))))
                out.append(("hidden states", f"{eb} x {cfg['probe_len']}", peak_gib(lambda: nograd_hidden(model, x), state(model, x))))
                x = torch.randint(0, V, (eb, Te))
                out.append(("surprisal / eval loss", f"{eb} x {Te}", peak_gib(lambda: nograd_losses(model, x), state(model, x))))
                f, extra = jvp_step(model, lm.per_example_lm_loss, torch.randint(0, V, (jb, T)), params)
                out.append(("per-example JVP", f"{jb} x {T}", peak_gib(f, state(model, *extra))))
                if key[-1]:
                    hb = cfg["hessian_batch"]
                    f, extra = hvp_step(model, lm.per_example_lm_loss, torch.randint(0, V, (hb, T)), params)
                    out.append(("HVP (Hessian direction)", f"{hb} x {T}", peak_gib(f, state(model, *extra))))
            out = [{"part": "p1", "job": job, "model": f"{family}-{size}", "step": s, "shape": sh, "peak_GiB": g}
                   for s, sh, g in out]
            seen[key] = out
            listed.add((key, job))
            rows.extend(out)
            log(f"p1 {family}-{size}: {max(r['peak_GiB'] for r in out):.1f} GiB ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- P4
def check_p4(rows, log, calibrate=False):
    import p4_clipping as p4
    from common import hf
    cfg = config.get("p4_clipping", False)
    M = cfg["measure"]
    for kind in cfg["models"]:
        t0 = time.time()
        mb = M.get("micro_batch", {}).get(kind)
        hb = M["hessian_batch"]
        out = []
        with torch.device("meta"):
            if kind == "transformer":
                c = cfg["transformer"]
                neox = dict(getattr(hf, c["neox"]))
                neox["max_position_embeddings"] = max(neox.get("max_position_embeddings", 2048), c["seq_len"])
                model = hf.neox_from_config(neox)
                V, L, Bt = model.config.vocab_size, c["seq_len"], c["batch"]
                mk = lambda n: torch.randint(0, V, (n, L))            # noqa: E731
                pel, opt = lm.per_example_lm_loss, torch.optim.AdamW(model.parameters(), lr=1e-3)
                unit = f" x {L}"
            else:
                c = cfg["resnet18"]
                model = p4.build_resnet(0)
                Bt = c["batch"]
                mk = lambda n: (torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))   # noqa: E731
                pel = p4.resnet_pel
                opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
                unit = " images"
            model.train()
            xb = mk(Bt)

            def train_step():
                pel(lambda *a, **kw: model(*a, **kw), xb).mean().backward()
                opt.step()
            out.append(("training step (+ optimizer state)", f"{Bt}{unit}", peak_gib(train_step, state(model, xb))))
            model.zero_grad(set_to_none=True)
            opt = None
            model.eval()
            params = gradients.select_params(model, None)
            ebs = min(64, mb) if mb else 64
            xe = mk(ebs)

            def evaluate():
                with torch.no_grad():
                    pel(lambda *a, **kw: model(*a, **kw), xe)
            out.append(("losses (no grad)", f"{ebs}{unit}", peak_gib(evaluate, state(model, xe))))
            hn = mb or hb
            f, extra = hvp_step(model, pel, mk(hn), params)
            out.append(("HVP: power iteration, trace", f"{hn}{unit}", peak_gib(f, state(model, *extra))))
            f, extra = backward_step(model, pel, mk(hn))
            out.append(("adaptive sharpness (gradient)", f"{hn}{unit}", peak_gib(f, state(model, *extra))))
            jn = min(32, mb) if mb else 32
            f, extra = jvp_step(model, pel, mk(jn), params)
            out.append(("per-example JVP", f"{jn}{unit}", peak_gib(f, state(model, *extra))))
            if calibrate and mb and hb > mb:
                f, extra = hvp_step(model, pel, mk(hb), params)
                out.append(("calibration: HVP without micro-batches", f"{hb}{unit}", peak_gib(f, state(model, *extra))))
        rows.extend({"part": "p4", "job": f"p4_{kind}", "model": kind, "step": s, "shape": sh, "peak_GiB": g}
                    for s, sh, g in out)
        log(f"p4 {kind}: {max(g for s, _, g in out if not s.startswith('calibration')):.1f} GiB ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- P3
def check_p3(rows, log):
    cfg = config.get("p3_fractional_memory", False)
    Bsz, done = cfg["batch"], set()
    for task, ct in cfg["tasks"].items():
        cls, L = task == "lra_image", ct["L"]
        if (cls, L) in done:                 # powerlaw and exponential have the same shapes
            continue
        done.add((cls, L))
        for n_s in cfg["n_states"]:
            t0 = time.time()
            with torch.device("meta"):
                ref = ssm.SSMModel(cfg["ref_arch"], 1, 10 if cls else 1, cfg["H_ref"], n_s, cfg["n_layers"], L, pool=cls)
                target = ssm.n_params(ref)
                for arch in cfg["archs"]:
                    H = ssm.width_for_budget(arch, 1, 10 if cls else 1, n_s, cfg["n_layers"], L, cls, target)[0]
                    m = ssm.SSMModel(arch, 1, 10 if cls else 1, H, n_s, cfg["n_layers"], L, pool=cls)
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                    x = torch.randn(Bsz, L, 1)
                    y = torch.randint(0, 10, (Bsz,)) if cls else torch.randn(Bsz, L, 1)

                    def step():
                        o = m(x)
                        (F.cross_entropy(o, y) if cls else F.mse_loss(o, y)).backward()
                        opt.step()
                    g = peak_gib(step, state(m, x, y))
                    rows.append({"part": "p3", "job": "p3_fractional", "model": f"{arch} (H={H})",
                                 "step": f"training step, {task}, n_s={n_s}", "shape": f"{Bsz} x {L}", "peak_GiB": g})
            log(f"p3 {task} n_s={n_s}: done ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- P2
# public configs (config.json) of P2's models: family, hidden, layers, heads, intermediate, padded vocabulary
P2_MODELS = {"EleutherAI/pythia-6.9b": ("pythia", 4096, 32, 32, 16384, 50432),
             "allenai/OLMo-7B-hf": ("olmo", 4096, 32, 32, 11008, 50304)}
POPQA_PROMPT = 256        # tokens; PopQA's 4-shot prompts are about 100


def p2_model(name: str):
    fam, h, n_layers, heads, inter, V = P2_MODELS[name]
    if fam == "pythia":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        c = GPTNeoXConfig(vocab_size=V, hidden_size=h, num_hidden_layers=n_layers, num_attention_heads=heads,
                          intermediate_size=inter, max_position_embeddings=2048, rotary_pct=0.25,
                          use_parallel_residual=True, tie_word_embeddings=False)
        cls = GPTNeoXForCausalLM
    else:
        from transformers import OlmoConfig, OlmoForCausalLM
        c = OlmoConfig(vocab_size=V, hidden_size=h, num_hidden_layers=n_layers, num_attention_heads=heads,
                       num_key_value_heads=heads, intermediate_size=inter, max_position_embeddings=2048,
                       tie_word_embeddings=False)
        cls = OlmoForCausalLM
    c._attn_implementation = "eager"
    return cls(c).eval()


def nograd_generate(model, x, new):
    """Mirrors generate() without sampling: the prompt with the cache, then one token at a time (full logits)."""
    with torch.no_grad():
        out = model(input_ids=x, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(new):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)


def check_p2(rows, log, calibrate=False):
    import copy
    import p2_w4a4 as p2
    from common import quant
    import torch.fx.experimental._config as fx_config
    fx_config.meta_nonzero_assume_all_nonzero = True      # boolean masks (outlier channels) select every channel:
    cfg = config.get("p2_w4a4", False)                     # a slight overestimate on the meta device
    dtype = getattr(torch, cfg["dtype"])
    pq = cfg["popqa"]
    for name in cfg["models"]:
        t0 = time.time()
        out = []
        with torch.device("meta"):
            model = p2_model(name).to(dtype)
            V = model.config.vocab_size
            out.append(("weights", cfg["dtype"], sum(p.numel() * p.element_size() for p in model.parameters()) / GiB))
            win = torch.randint(0, V, (1, cfg["ppl"]["ctx"]))
            prompt = torch.randint(0, V, (pq["batch"], POPQA_PROMPT))
            probe = torch.randint(0, V, (4, cfg["probe"]["len"]))

            def forwards(tag):
                out.append((f"perplexity window{tag}", f"1 x {cfg['ppl']['ctx']}",
                            peak_gib_model(lambda: nograd_losses(model, win), model, win)))
                out.append((f"PopQA generation{tag}", f"{pq['batch']} x {POPQA_PROMPT}+{pq['max_new_tokens']}",
                            peak_gib_model(lambda: nograd_generate(model, prompt, pq["max_new_tokens"]), model, prompt)))
            forwards("")
            out.append(("hidden states", f"4 x {cfg['probe']['len']}",
                        peak_gib_model(lambda: nograd_hidden(model, probe), model, probe)))
            lins = quant.linear_layers(model)
            absmax = {n: torch.rand(m.in_features) for n, m in lins.items()}
            bank = p2.WeightBank.__new__(p2.WeightBank)          # the real bank holds CPU copies: not counted
            bank.orig = {n: copy.deepcopy(m) for n, m in lins.items()}
            on_cpu = [t for m in bank.orig.values() for t in m.parameters()]
            del lins
            schemes = [(arm, quant.Scheme(**s)) for arm, ss in cfg["arms"].items() for s in ss]
            schemes.append(("control", quant.Scheme("isotropic", a_bits=None)))
            for arm, s in schemes:
                noise = {"sigma_w": 0.1, "sigma_a": 0.1, "seed": 0} if s.name == "isotropic" else None

                def step():
                    bank.restore(model, "meta")
                    quant.apply_scheme(model, s, absmax if s.name != "isotropic" else None, seed=0, noise=noise)
                out.append((f"restore + quantize: {arm} {s.name}", "weights",
                            peak_gib_model(step, model, absmax, ignore=on_cpu)))
                if arm == "w4a4" and s.name in ("rotation", "outlier"):
                    forwards(f" ({s.name} W4A4)")
                if calibrate and arm == "weight_only" and s.name == "nf4":
                    def old_step():
                        bank.restore(model, "meta")
                        held = list(quant.linear_layers(model).values())       # the old loop held every layer
                        quant.apply_scheme(model, s, absmax, seed=0)
                        del held
                    out.append(("calibration: quantize, every layer held (before 2026-11-17)", "weights",
                                peak_gib_model(old_step, model, absmax, ignore=on_cpu)))
        rows.extend({"part": "p2", "job": "rerun_p2_bf16", "model": name.split("/")[-1], "step": st, "shape": sh,
                     "peak_GiB": g} for st, sh, g in out)
        log(f"p2 {name}: {max(g for st, _, g in out if st != 'weights' and not st.startswith('calibration')):.1f} GiB "
            f"({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- P5
def check_p5(rows, log):
    import p5_escape as p5
    cfg = config.get("p5_escape", False)
    R, n, act = cfg["restarts"], cfg["n_train"], cfg["act"]
    t0 = time.time()
    out = []
    with torch.device("meta"):
        P = p5.init_params(R, 784, cfg["hidden"], 10, 0, "meta")
        X, Y = torch.randn(n, 784), torch.randint(0, 10, (n,))
        keep = lambda *e: [*P, X, Y, *e]                          # noqa: E731
        out.append(("full-batch gradient, all restarts", f"{R} x {n}", peak_gib(lambda: p5.grads(P, X, Y, act), keep())))
        out.append(("loss barrier on the segment", f"{R} x {n} x {cfg['segment_points']}",
                    peak_gib(lambda: p5.barrier(P, P, X, Y, act, cfg["segment_points"], with_end=True), keep())))
        idx = torch.randint(0, n, (R, max(cfg["batch_sizes"])))
        out.append(("minibatch gradient", f"{R} x {max(cfg['batch_sizes'])}",
                    peak_gib(lambda: p5.grads(P, X[idx], Y[idx], act), keep(idx))))

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i, t in enumerate(P):
                    self.register_parameter(f"p{i}", torch.nn.Parameter(t[0:1].clone()))

            def forward(self, x):
                return p5.forward([getattr(self, f"p{i}") for i in range(len(P))], x, act)[0]
        m = Wrap()
        pel = lambda run, b: F.cross_entropy(run(b[0]), b[1], reduction="none")      # noqa: E731
        params = gradients.select_params(m, None)
        b = (X[:1000], Y[:1000])
        f, extra = hvp_step(m, pel, b, params)
        out.append(("alpha at W_S: HVP", "1000", peak_gib(f, state(m, *extra))))
        f, extra = jvp_step(m, pel, b, params)
        out.append(("alpha at W_S: per-example projections", "1000", peak_gib(f, state(m, *extra))))
    rows.extend({"part": "p5", "job": "rerun_p5", "model": f"MLP {cfg['hidden']}x2", "step": st, "shape": sh,
                 "peak_GiB": g} for st, sh, g in out)
    log(f"p5: {max(g for _, _, g in out):.2f} GiB ({time.time() - t0:.0f} s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", default=["p1", "p4", "p3", "p2", "p5"], choices=["p1", "p4", "p3", "p2", "p5"])
    ap.add_argument("--limit", type=float, default=L40S_GIB, help="GPU memory in GiB (default: the L40S)")
    ap.add_argument("--calibrate", action="store_true", help="P4: add the unchunked HVP that failed on the L40S; "
                                                             "P2: add the quantization as before 2026-11-17")
    ap.add_argument("--csv", default=None, help="also write the rows to this CSV")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    log = lambda m: print(m, file=sys.stderr, flush=True)     # noqa: E731
    rows = []
    for part in args.parts:
        {"p1": lambda: check_p1(rows, log), "p4": lambda: check_p4(rows, log, args.calibrate),
         "p3": lambda: check_p3(rows, log), "p2": lambda: check_p2(rows, log, args.calibrate),
         "p5": lambda: check_p5(rows, log)}[part]()
    bad = 0
    print(f"{'job':16} {'model':22} {'step':40} {'shape':14} {'GiB':>8}  vs {args.limit:.1f} GiB")
    for r in rows:
        share = r["peak_GiB"] / args.limit
        status = "over" if share > 1 else ("tight" if share > TIGHT else "ok")
        if r["step"] == "weights":
            status = ""
        elif status != "ok" and not r["step"].startswith("calibration"):
            bad += 1
        r["status"] = status
        print(f"{r['job']:16} {r['model']:22} {r['step'][:40]:40} {r['shape']:14} {r['peak_GiB']:8.2f}  {status}")
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
    worst = max((r for r in rows if r["step"] != "weights" and not r["step"].startswith("calibration")),
                key=lambda r: r["peak_GiB"], default=None)
    if worst:
        print(f"largest: {worst['job']} {worst['model']} {worst['step']} at {worst['peak_GiB']:.1f} GiB; "
              f"{bad} step(s) over {TIGHT:.0%} of {args.limit:.1f} GiB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
