"""Minimal reproduction for emmy-ml 0.3.25: an embedding lookup is evaluated as torch.gather when the
index tensor's last dimension equals the embedding width (for a language model: seq_len == hidden_size).

CPU only, no CUDA kernels are built. Usage:  python emmy_seq_eq_hidden_repro.py
"""
import importlib.metadata as md

import numpy as np
import torch
from torch import _dynamo, nn

from emmy.compiler.backend.torch_ref import build_callable
from emmy.compiler.ir.tensor.ir import GatherOp
from emmy.compiler.trace.torch import trace_module_functional

V, H = 100, 64


def ir_op_forward(seq_len):
    """1. GatherOp.forward on NumPy arrays, the way an embedding lookup is traced (axis 0)."""
    rng = np.random.default_rng(0)
    weight, ids = rng.standard_normal((V, H)), rng.integers(0, V, (1, seq_len))
    got, want = GatherOp(axis=0).forward(weight, ids), weight[ids]
    return f"shape {got.shape}, expected {want.shape}"


def traced_embedding(seq_len):
    """2. nn.Embedding traced with Emmy's frontend and run with Emmy's torch reference."""
    torch.manual_seed(0)
    emb = nn.Embedding(V, H).eval()
    ids = torch.randint(0, V, (1, seq_len))
    out = {}

    def backend(gm, example_inputs):
        g, _ = trace_module_functional(gm, tuple(example_inputs))
        fn, ins = build_callable(g, {n: t.contiguous() for n, t in zip(g.inputs, example_inputs)})
        with torch.no_grad():
            got, ref = fn(*ins), gm(*example_inputs)
        got = got[0] if isinstance(got, (tuple, list)) else got
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        out["r"] = (f"shape {tuple(got.shape)}, expected {tuple(ref.shape)}" if got.shape != ref.shape
                    else f"max |IR - eager| = {float((got - ref).abs().max()):.3g}")
        return gm.forward

    _dynamo.reset()
    with torch.no_grad():
        torch.compile(lambda x: emb(x) * 2.0, backend=backend, dynamic=False)(ids)  # *2: keep the lookup in a graph
    return out["r"]


if __name__ == "__main__":
    print(f"emmy-ml {md.version('emmy-ml')}, torch {torch.__version__}, embedding ({V}, {H})")
    for s in (H - 1, H, H + 1):
        print(f"seq_len={s}: GatherOp.forward -> {ir_op_forward(s)}; traced nn.Embedding -> {traced_embedding(s)}")
