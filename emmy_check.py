"""Emmy check (OURS): Emmy-compiled against standard PyTorch inference on outlier-heavy Pythia checkpoints.

Emmy (PyPI emmy-ml, CloudRift/RiftStack) compiles PyTorch graphs into CUDA kernels through
torch.compile(model, backend="emmy", dynamic=False). Its README and recipes name no GPT-NeoX model, so
the script confirms support before it compares anything:

  1. Support probe. Compile the whole GPT-NeoX forward at one fixed shape, with a counting wrapper
     around Emmy's backend, along two routes:
       hf_model        torch.compile on GPTNeoXForCausalLM as it is (the README's one-liner)
       static_wrapper  Emmy's own static full-model wrapper adapted to GPT-NeoX: causal mask, position
                       ids and rotary cos/sin precomputed (NeoXStatic below)
     Each Dynamo graph either compiles under Emmy or runs eagerly, and its error is recorded. Verdicts:
       supported    every graph compiled and the logits are finite
       partial      some graphs ran eagerly (the comparison still runs; op_coverage says how much is Emmy)
       unsupported  no graph compiled, or the compiled forward raised
  2. Comparison, for supported or partial. On n_seqs held-out sequences, compare eager against compiled
     at the same dtype:
       logits         max |diff|, relative L2, mean and max KL(eager || compiled), top-1 agreement, NLL difference
       hidden states  per layer: max |diff|, relative L2, K, max |h|, outlier channels (|h| > 6, as in P1)
                      and the Jaccard overlap of the outlier-channel sets
     This runs for every dtype, with Emmy's fast math on (its default: EMMY_FAST_MATH unset) and off
     (EMMY_FAST_MATH=0). Each lane is its own subprocess, so a crash in a compiled kernel cannot take the run
     down. Emmy lets single precision knobs (EMMY_FAST_EXP, EMMY_FAST_DIV, ...) override the umbrella, so the
     worker's environment drops any such variable inherited from the shell and run.json lists what it dropped.

Checkpoints: the configured Pythia models, plus the top-k checkpoints by K in P1's checkpoints.csv
(the outlier-heavy ones), when a full P1 run exists.

Requirements: the separate Emmy environment (requirements-emmy.txt: Python >= 3.12, transformers 5.19.x),
an NVIDIA GPU and nvcc on PATH. Without them, a full run writes support.csv with status "blocked" and
stops. Under --smoke it instead runs the whole comparison with torch.compile(backend="aot_eager") as a
stand-in and says so in run.json (no Emmy result is produced).

--frontend-only runs step 1 on a CPU with emmy-ml installed and no GPU: each graph is traced into
Emmy's IR and the IR is evaluated with Emmy's torch reference against the graph (no kernels built).
Result of that probe here (2026-11-13, emmy-ml 0.3.25, transformers 5.19.0, random-init GPT-NeoX of
Pythia-14M shape and a smaller one):
  hf_model        unsupported: "no tracer mapping for aten.new_ones" (and aten.diff without an
                  attention mask), both from transformers' mask construction, not from GPT-NeoX layers
  static_wrapper  traced; Emmy's IR, run through its torch reference, equals the graph and the eager
                  logits (max |diff| 0), EXCEPT when seq_len equals hidden_size: Emmy's CPU evaluators
                  (GatherOp.forward, torch_ref) then read the embedding lookup as torch.gather, and the
                  fused-QKV view after it fails (docs/emmy-issue-seq-len-hidden-size.md has the minimal
                  reproduction). seq_len is kept off every model dimension for that reason.
Kernel lowering and the numerical comparison on a GPU are still untested (lowering the whole model on
a CPU did not finish here).

Outputs (results/emmy_check/[smoke/]): support.csv, compare.csv, layers.csv, environment.json, run.json

Commands:
  python emmy_check.py --frontend-only     # step 1 on CPU: Emmy's frontend only (no GPU, no nvcc)
  python emmy_check.py --probe-only        # step 1 on the GPU: fp32 and fp16, default fast math
  python emmy_check.py
  python emmy_check.py --models EleutherAI/pythia-410m@step143000
  python emmy_check.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from common import hf, runtime


# ----------------------------------------------------------------------------- environment

def emmy_environment(device: str) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "device": device,
            "cuda_available": torch.cuda.is_available(), "nvcc": shutil.which("nvcc")}
    for pkg in ("emmy-ml", "transformers"):
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = None
    reasons = []
    frontend_only = device == "frontend"
    if sys.version_info < (3, 12):
        reasons.append("Python < 3.12 (Emmy requires >= 3.12)")
    if info["emmy-ml"] is None:
        reasons.append("emmy-ml not installed (pip install -r requirements-emmy.txt in a separate venv)")
    else:
        try:
            from torch import _dynamo
            info["backend_registered"] = "emmy" in _dynamo.list_backends(exclude_tags=())
        except Exception as e:  # noqa: BLE001
            info["backend_registered"] = False
            info["backend_error"] = f"{type(e).__name__}: {e}"
        if not info["backend_registered"]:
            reasons.append("torch.compile backend 'emmy' is not registered")
    if not frontend_only and (not info["cuda_available"] or not device.startswith("cuda")):
        reasons.append("no CUDA device (Emmy runs compiled CUDA kernels)")
    if not frontend_only and not info["nvcc"]:
        reasons.append("nvcc not on PATH (Emmy builds its kernels with the CUDA toolkit)")
    info["ok"] = not reasons
    info["reason"] = "; ".join(reasons)
    return info


# ----------------------------------------------------------------------------- statistics

def kurtosis_from_sums(n, s1, s2, s3, s4):
    """Excess kurtosis K from raw power sums (float64), the same K as spectra.activation_stats."""
    m, e2, e3, e4 = s1 / n, s2 / n, s3 / n, s4 / n
    var = e2 - m * m
    m4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4
    return m4 / var ** 2 - 3.0


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class LayerAcc:
    """Streaming per-layer statistics for the eager and the compiled hidden states."""

    def __init__(self, n_layers: int, hidden: int):
        z = lambda: np.zeros(n_layers)  # noqa: E731
        self.n = 0
        self.sums = {k: [z(), z(), z(), z()] for k in ("e", "c")}
        self.chan_max = {k: torch.zeros(n_layers, hidden, dtype=torch.float64) for k in ("e", "c")}
        self.dmax, self.d2, self.h2 = z(), z(), z()

    def add(self, hs_e, hs_c):
        for l, (he, hc) in enumerate(zip(hs_e, hs_c)):
            he = he.reshape(-1, he.shape[-1]).double()
            hc = hc.reshape(-1, hc.shape[-1]).double()
            for key, h in (("e", he), ("c", hc)):
                for p in range(4):
                    self.sums[key][p][l] += float((h ** (p + 1)).sum())
                self.chan_max[key][l] = torch.maximum(self.chan_max[key][l], h.abs().amax(0).cpu())
            d = hc - he
            self.dmax[l] = max(self.dmax[l], float(d.abs().max()))
            self.d2[l] += float((d ** 2).sum())
            self.h2[l] += float((he ** 2).sum())
        self.n += hs_e[0].numel()

    def rows(self, threshold: float):
        out = []
        for l in range(len(self.dmax)):
            K = {k: kurtosis_from_sums(self.n, *(s[l] for s in self.sums[k])) for k in ("e", "c")}
            sets = {k: set(torch.nonzero(self.chan_max[k][l] > threshold).flatten().tolist()) for k in ("e", "c")}
            out.append({"layer": l + 1, "K_eager": K["e"], "K_compiled": K["c"],
                        "max_abs_eager": float(self.chan_max["e"][l].max()),
                        "max_abs_compiled": float(self.chan_max["c"][l].max()),
                        "max_abs_diff": self.dmax[l], "rel_l2": float(np.sqrt(self.d2[l] / max(self.h2[l], 1e-300))),
                        "n_outlier_eager": len(sets["e"]), "n_outlier_compiled": len(sets["c"]),
                        "outlier_jaccard": jaccard(sets["e"], sets["c"])})
        return out


def logit_metrics(le: torch.Tensor, lc: torch.Tensor, ids: torch.Tensor) -> dict:
    """le, lc: [T, V] float32 logits of one sequence; ids: [T]."""
    d = lc - le
    lpe, lpc = F.log_softmax(le, -1), F.log_softmax(lc, -1)
    kl = (lpe.exp() * (lpe - lpc)).sum(-1)
    nll_e = F.cross_entropy(le[:-1], ids[1:], reduction="sum")
    nll_c = F.cross_entropy(lc[:-1], ids[1:], reduction="sum")
    return {"max_abs": float(d.abs().max()), "d2": float((d ** 2).sum()), "l2": float((le ** 2).sum()),
            "kl_sum": float(kl.sum()), "kl_max": float(kl.max()), "n_tok": le.shape[0],
            "top1": float((le.argmax(-1) == lc.argmax(-1)).sum()),
            "nll_e": float(nll_e), "nll_c": float(nll_c), "n_pred": le.shape[0] - 1,
            "finite": bool(torch.isfinite(lc).all())}


# ----------------------------------------------------------------------------- worker

class _WorkerRun:
    """Stand-in for runtime.Run inside a worker subprocess (collects fallbacks and notes)."""

    def __init__(self, offline_ok: bool):
        self.offline_ok = offline_ok
        self.require_pins = False
        self.pins: dict[str, str] = {}            # hf.PINS.resolve records the commits it loads here
        self.fallbacks: list[str] = []
        self.notes: list[str] = []

    def fallback(self, what):
        print(f"[emmy_check worker] FALLBACK: {what}", flush=True)
        self.fallbacks.append(what)

    def note(self, msg):
        print(f"[emmy_check worker] {msg}", flush=True)
        self.notes.append(msg)


def emmy_frontend_backend(gm, example_inputs):
    """CPU-only check of Emmy's frontend: trace the Dynamo graph into Emmy's IR (the step that rejects
    unmapped aten ops), evaluate the IR with Emmy's torch reference and compare it with the graph.
    No CUDA kernels are built; the graph itself then runs eagerly."""
    from emmy.compiler.backend.torch_ref import build_callable, is_runnable
    from emmy.compiler.trace.torch import trace_module_functional
    g, _ = trace_module_functional(gm, tuple(example_inputs))
    if not is_runnable(g):
        raise NotImplementedError("Emmy IR holds ops without a torch reference (frontend traced, not checkable)")
    fn, ins = build_callable(g, {n: t.contiguous() for n, t in zip(g.inputs, example_inputs)})
    with torch.no_grad():
        got, ref = fn(*ins), gm(*example_inputs)
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    d = [float((a.float() - b.float()).abs().max()) for a, b in zip(got, ref)
         if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape]
    if d and max(d) > 1e-3:
        raise ArithmeticError(f"Emmy IR differs from the graph by {max(d):.3g}")
    return gm.forward


def counting_backend(name: str, stats: dict):
    if name == "emmy":
        from emmy.dynamo import emmy_backend as inner
    elif name == "emmy_frontend":
        inner = emmy_frontend_backend
    else:
        from torch._dynamo.backends.registry import lookup_backend
        inner = lookup_backend(name)

    def backend(gm, example_inputs):
        n_ops = sum(1 for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module"))
        stats["graphs"] += 1
        stats["fx_ops"] += n_ops
        try:
            fn = inner(gm, example_inputs)
            stats["compiled_graphs"] += 1
            stats["compiled_ops"] += n_ops
            return fn
        except Exception as e:  # noqa: BLE001  the graph runs eagerly; the error is the finding
            msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
            stats["errors"].append(f"{type(e).__name__}: {msg}")
            return gm.forward

    return backend


def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


class NeoXStatic(torch.nn.Module):
    """Emmy's static full-model wrapper (emmy.compiler.trace.huggingface.build_full_model_wrapper),
    adapted to GPT-NeoX, whose decoder sits at .gpt_neox rather than .model.

    The causal mask and the position ids are buffers and the rotary cos/sin are precomputed, so the
    traced graph holds none of transformers' mask-construction ops (aten.diff, aten.new_ones), which
    Emmy's frontend does not map. forward(input_ids) returns (logits, h_1, ..., h_L).
    Numerically identical to the unwrapped model (checked: wrapper_vs_model_max_abs).
    """

    def __init__(self, model, seq_len: int):
        super().__init__()
        self.model = model
        dtype = next(model.parameters()).dtype
        dev = next(model.parameters()).device
        mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=dev)
        mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), 1), torch.finfo(dtype).min)
        self.register_buffer("causal_mask", mask[None, None])
        self.register_buffer("position_ids", torch.arange(seq_len, device=dev)[None])
        dec = model.gpt_neox
        with torch.no_grad():
            cos, sin = dec.rotary_emb(torch.zeros(1, seq_len, model.config.hidden_size, dtype=dtype, device=dev),
                                      self.position_ids)
        dec.rotary_emb = _PassThroughRotary(cos, sin)

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, attention_mask=self.causal_mask, position_ids=self.position_ids,
                         use_cache=False, output_hidden_states=True, return_dict=False)
        return (out[0], *out[1][1:])


class _PassThroughRotary(torch.nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, *_, **__):
        return self.cos, self.sin


def _split_out(o):
    """(logits [T, V], [h_1 .. h_L]) from either a ModelOutput or the wrapper's tuple."""
    if isinstance(o, tuple):
        return o[0], list(o[1:])
    return o.logits, list(o.hidden_states[1:])


def _probe(route: str, module, call, x0, backend: str, dev: str, key: dict) -> tuple[dict, object]:
    """Compile `module` with the counting backend, run call(compiled, x0) once, and grade support."""
    from torch import _dynamo
    _dynamo.reset()
    explain = {}
    try:
        with torch.no_grad():
            ex = _dynamo.explain(lambda x: call(module, x))(x0)
        # informational: explain can count wrapper frames that the compiled call never produces;
        # the counting backend's "graphs" column is authoritative
        explain = {"explain_graph_count": ex.graph_count, "explain_graph_breaks": ex.graph_break_count,
                   "break_reasons": " | ".join(" ".join(str(getattr(r, "reason", r)).split())[:160] for r in ex.break_reasons[:3])}
    except Exception as e:  # noqa: BLE001
        explain = {"explain_graph_count": None, "break_reasons": f"explain failed: {type(e).__name__}: {str(e)[:160]}"}
    _dynamo.reset()
    stats = {"graphs": 0, "fx_ops": 0, "compiled_graphs": 0, "compiled_ops": 0, "errors": []}
    compiled = torch.compile(module, backend=counting_backend(backend, stats), dynamic=False)
    row = {**key, "route": route, **explain}
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            logits, _ = _split_out(call(compiled, x0))
        _sync(dev)
        finite = bool(torch.isfinite(logits).all())
    except Exception as e:  # noqa: BLE001
        row.update(status="unsupported", compile_s=time.perf_counter() - t0, finite=None,
                   errors=f"{type(e).__name__}: {str(e).strip()[:400]}",
                   **{k: v for k, v in stats.items() if k != "errors"})
        return row, None
    row.update(compile_s=time.perf_counter() - t0, finite=finite, **{k: v for k, v in stats.items() if k != "errors"})
    row["op_coverage"] = stats["compiled_ops"] / max(stats["fx_ops"], 1)
    row["errors"] = " | ".join(stats["errors"][:3])
    if stats["compiled_graphs"] == 0:
        row["status"] = "unsupported"
    elif stats["compiled_graphs"] == stats["graphs"] and finite:
        row["status"] = "supported"
    else:
        row["status"] = "partial"
    if backend == "emmy_frontend":  # only Emmy's frontend ran: no kernels were built
        row["status"] = {"supported": "frontend-ok", "partial": "frontend-partial"}.get(row["status"], "frontend-fails")
    return row, compiled


def _work(spec: dict, wrun: _WorkerRun) -> dict:
    dev, dtype = spec["device"], getattr(torch, spec["dtype"])
    key = {"model": spec["repo"], "revision": spec["revision"], "dtype": spec["dtype"],
           "fast_math": spec["fast_math"], "EMMY_FAST_MATH": os.environ.get("EMMY_FAST_MATH", "unset"),
           "backend": spec["backend"]}
    if dtype == torch.float16 and not dev.startswith("cuda"):
        return {"support": [{**key, "status": "skipped", "errors": "float16 on CPU"}]}
    model, tok, src = hf.load_causal_lm(spec["repo"], spec["revision"], wrun, dtype=dtype, device=dev)
    model.eval()
    texts = json.loads(Path(spec["texts"]).read_text())
    c = model.config
    hd = c.hidden_size // c.num_attention_heads
    dims = {c.hidden_size, c.intermediate_size, c.num_attention_heads, hd, 3 * hd, 3 * c.hidden_size,
            int(hd * getattr(c, "rotary_pct", 1.0))}
    seq_len = spec["seq_len"]
    while seq_len in dims:  # emmy-ml 0.3.25 reads the embedding as a gather when seq_len == hidden_size
        seq_len -= 1
    if seq_len != spec["seq_len"]:
        wrun.note(f"seq_len {spec['seq_len']} equals a model dimension; using {seq_len}")
    ids = hf.tokenize_stream(texts, tok, seq_len, spec["n_seqs"]).long()
    key["source"] = src
    key["seq_len"] = seq_len
    x0 = ids[:1].to(dev)

    # route 1: the documented one-liner on the Hugging Face model as it is
    def call_hf(m, x):
        return m(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_hidden_states=True)
    with torch.no_grad():
        ref_logits = model(input_ids=x0, use_cache=False).logits.float()
    support = []
    row, _ = _probe("hf_model", model, call_hf, x0, spec["backend"], dev, key)
    support.append(row)

    # route 2: the static wrapper (Emmy's own approach for Hugging Face models)
    wrapper = NeoXStatic(model, seq_len).eval()
    with torch.no_grad():
        w_logits, _ = _split_out(wrapper(x0))
    gap = float((w_logits.float() - ref_logits).abs().max())
    row, compiled = _probe("static_wrapper", wrapper, lambda m, x: m(x), x0, spec["backend"], dev, key)
    row["wrapper_vs_model_max_abs"] = gap
    support.append(row)
    if spec["probe_only"] or compiled is None or row["status"] in ("unsupported", "frontend-fails"):
        return {"support": support}

    n_layers = model.config.num_hidden_layers
    acc = LayerAcc(n_layers, model.config.hidden_size)
    lm, t_e, t_c = [], [], []
    with torch.no_grad():
        for i in range(len(ids)):
            x = ids[i:i + 1].to(dev)
            _sync(dev); t = time.perf_counter()
            le, he = _split_out(wrapper(x))
            _sync(dev); t_e.append(time.perf_counter() - t); t = time.perf_counter()
            lc, hc = _split_out(compiled(x))
            _sync(dev); t_c.append(time.perf_counter() - t)
            lm.append(logit_metrics(le[0].float(), lc[0].float(), x[0]))
            acc.add(he, hc)
            del le, he, lc, hc
    S = lambda k: sum(m[k] for m in lm)  # noqa: E731
    key = {**key, "route": "static_wrapper"}
    layers = [{**key, **r} for r in acc.rows(spec["outlier_threshold"])]
    tol = spec["tolerances"]
    cmp = {**key, "status": row["status"], "op_coverage": row["op_coverage"], "n_seqs": len(ids),
           "seq_len": ids.shape[1], "logit_max_abs_diff": max(m["max_abs"] for m in lm),
           "logit_rel_l2": float(np.sqrt(S("d2") / S("l2"))), "kl_mean": S("kl_sum") / S("n_tok"),
           "kl_max": max(m["kl_max"] for m in lm), "top1_agree": S("top1") / S("n_tok"),
           "nll_eager": S("nll_e") / S("n_pred"), "nll_compiled": S("nll_c") / S("n_pred"),
           "finite": all(m["finite"] for m in lm),
           "K_max_eager": max(r["K_eager"] for r in layers), "K_max_compiled": max(r["K_compiled"] for r in layers),
           "max_abs_eager": max(r["max_abs_eager"] for r in layers),
           "outlier_jaccard_min": min(r["outlier_jaccard"] for r in layers),
           "t_eager_ms": 1e3 * float(np.median(t_e)), "t_compiled_ms": 1e3 * float(np.median(t_c[1:] or t_c))}
    cmp["d_nll"] = cmp["nll_compiled"] - cmp["nll_eager"]
    cmp["speedup"] = cmp["t_eager_ms"] / cmp["t_compiled_ms"]
    cmp["agrees"] = bool(cmp["finite"] and cmp["top1_agree"] >= tol["top1_min"]
                         and abs(cmp["d_nll"]) <= tol["abs_d_nll_max"]
                         and cmp["outlier_jaccard_min"] >= tol["outlier_jaccard_min"])
    return {"support": support, "compare": cmp, "layers": layers}


def worker(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    runtime.set_all_seeds(spec["seed"])
    wrun = _WorkerRun(spec["offline_ok"])
    try:
        res = _work(spec, wrun)
    except Exception as e:  # noqa: BLE001
        res = {"support": [{"model": spec["repo"], "revision": spec["revision"], "dtype": spec["dtype"],
                            "fast_math": spec["fast_math"], "backend": spec["backend"], "status": "error",
                            "errors": f"{type(e).__name__}: {str(e)[:400]}"}], "traceback": traceback.format_exc()}
    res["fallbacks"], res["notes"] = wrun.fallbacks, wrun.notes
    Path(spec["out"]).write_text(json.dumps(res, default=str))


# ----------------------------------------------------------------------------- driver

def model_list(cfg: dict, args, run) -> list[tuple[str, str]]:
    if args.models:
        return [tuple(m.split("@", 1)) if "@" in m else (m, "main") for m in args.models.split(",")]
    models = [tuple(m) for m in cfg["models"]]
    k = cfg.get("from_p1_top_k", 0)
    # P1 results: results/p1_checkpoints/ (one P1 run) or results/p1_<suite>/ (run_all.sh, one folder per suite)
    res = runtime.PILOT_ROOT / "results"
    p1 = [f for f in sorted(res.glob("p1_*/checkpoints.csv")) if f.stat().st_size]
    if k and p1:
        df = pd.concat([pd.read_csv(f) for f in p1], ignore_index=True)
        if "K" not in df:
            df["K"] = np.nan
        df = df[(df.source == "hub") & df.repo.str.contains("pythia", case=False) & df.K.notna()]
        added = 0
        for _, r in df.sort_values("K", ascending=False).iterrows():
            m = (r.repo, r.revision)
            if m not in models:
                models.append(m)
                added += 1
                run.note(f"outlier-heavy checkpoint from P1: {m[0]}@{m[1]} (K = {r.K:.1f})")
            if added >= k:
                break
    elif k:
        run.note("no full P1 checkpoints.csv yet: only the configured models are checked")
    return models


# Emmy's precision-trading knobs (emmy.compiler.pipeline.search.space.PRECISION_KNOBS); an individual pin
# wins over the FAST_MATH umbrella, so none may leak into either lane from the user's shell
PRECISION_VARS = ("EMMY_FAST_MATH", "EMMY_FAST_EXP", "EMMY_F16_MMA_F32_ACC", "EMMY_FP8_MMA", "EMMY_FAST_DIV",
                  "EMMY_FAST_SQRT", "EMMY_FLUSH_TO_ZERO", "EMMY_CONTRACT_FMA")


def fast_math_env(base, fm: int, run=None) -> dict:
    """Worker environment for one lane: fm=1 leaves EMMY_FAST_MATH unset (Emmy's default, fast math on);
    fm=0 sets EMMY_FAST_MATH=0. Inherited precision pins are dropped and noted."""
    env = {k: v for k, v in base.items() if k not in PRECISION_VARS}
    dropped = sorted(k for k in base if k in PRECISION_VARS)
    if dropped and run is not None and not getattr(run, "_emmy_dropped_noted", False):
        run.note(f"dropped inherited Emmy precision pins from the worker environment: {dropped}")
        run._emmy_dropped_noted = True
    if not fm:
        env["EMMY_FAST_MATH"] = "0"
    return env


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--frontend-only", action="store_true",
                    help="CPU-only support probe of Emmy's frontend (trace + IR reference check); needs emmy-ml, no GPU")
    ap.add_argument("--models", default=None, help="comma list of repo@revision")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker)
    cfg = config.get("emmy_check", args.smoke)
    run = runtime.start_run("emmy_check", cfg, args)
    if args.frontend_only:
        args.probe_only = True
    env = emmy_environment("frontend" if args.frontend_only else run.device)
    (run.out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    run.note(f"Emmy environment: {'ok' if env['ok'] else env['reason']}")
    backend = "emmy_frontend" if args.frontend_only else "emmy"
    models = model_list(cfg, args, run)
    if not env["ok"]:
        if not run.offline_ok:
            runtime.write_csv(run.out_dir / "support.csv",
                              [{"model": m, "revision": r, "status": "blocked", "errors": env["reason"]} for m, r in models])
            runtime.finish_run(run, status="blocked", extra={"environment": env})
            return
        backend = "aot_eager"
        args.frontend_only = False
        run.fallback(f"Emmy unavailable ({env['reason']}): the comparison ran with torch.compile(backend='aot_eager') "
                     "as a stand-in, so these rows test the pipeline, not Emmy")

    ds = cfg["dataset"]
    n_docs = max(50, cfg["seq_len"] * cfg["n_seqs"] // 40)
    texts, text_src = hf.load_texts(ds, n_docs, run, seed=run.seed)
    work = run.out_dir / "_work"
    work.mkdir(exist_ok=True)
    (work / "texts.json").write_text(json.dumps(texts))

    dtypes = ["float32"] if args.frontend_only else cfg["dtypes"]
    fms = [1] if args.probe_only else cfg["fast_math"]
    support, compare, layers = [], [], []
    for repo, rev in models:
        for dt in dtypes:
            for fm in fms:
                tag = f"{repo.replace('/', '_')}_{rev}_{dt}_fm{fm}"
                spec = {"repo": repo, "revision": rev, "dtype": dt, "fast_math": fm, "backend": backend,
                        "device": "cpu" if args.frontend_only else run.device, "seed": run.seed, "offline_ok": run.offline_ok,
                        "seq_len": cfg["seq_len"], "n_seqs": cfg["n_seqs"], "texts": str(work / "texts.json"),
                        "outlier_threshold": cfg["outlier_threshold"], "tolerances": cfg["tolerances"],
                        "probe_only": bool(args.probe_only), "out": str(work / f"{tag}.result.json")}
                sp = work / f"{tag}.spec.json"
                sp.write_text(json.dumps(spec))
                Path(spec["out"]).unlink(missing_ok=True)
                env_vars = fast_math_env(os.environ, fm, run)
                env_vars["PYTHONHASHSEED"] = str(run.seed)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(sp)]
                run.note(f"{repo}@{rev} {dt} fast_math={fm} backend={backend}")
                try:
                    proc = subprocess.run(cmd, env=env_vars, timeout=cfg["worker_timeout_s"], cwd=runtime.PILOT_ROOT)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                key = {"model": repo, "revision": rev, "dtype": dt, "fast_math": fm, "backend": backend}
                if not Path(spec["out"]).exists():
                    support.append({**key, "status": "worker-crashed", "errors": f"return code {rc}"})
                    continue
                res = json.loads(Path(spec["out"]).read_text())
                for f in res.get("fallbacks", []):
                    if f not in run.fallbacks:
                        run.fallback(f)
                support.extend(res["support"])
                if res.get("compare"):
                    compare.append(res["compare"])
                    layers.extend(res["layers"])
                    c = res["compare"]
                    run.note(f"  {res['compare']['status']}: top1 {c['top1_agree']:.4f}, dNLL {c['d_nll']:+.2e}, "
                             f"KL {c['kl_mean']:.2e}, outlier Jaccard min {c['outlier_jaccard_min']:.2f}, "
                             f"speedup {c['speedup']:.2f}")
                for r in res["support"]:
                    run.note(f"  route {r.get('route', '-')}: {r.get('status')} {r.get('errors', '')}")
    runtime.write_csv(run.out_dir / "support.csv", support)
    runtime.write_csv(run.out_dir / "compare.csv", compare)
    runtime.write_csv(run.out_dir / "layers.csv", layers)
    shutil.rmtree(work, ignore_errors=True)
    verdicts = {f"{s['model']}@{s['revision']} {s['dtype']} fm{s['fast_math']} {s.get('route', '-')}": s["status"]
                for s in support}
    runtime.finish_run(run, extra={"environment": env, "backend": backend, "text_source": text_src,
                                   "support": verdicts,
                                   "agrees": {f"{c['model']}@{c['revision']} {c['dtype']} fm{c['fast_math']}": c["agrees"]
                                              for c in compare}})


if __name__ == "__main__":
    main()
