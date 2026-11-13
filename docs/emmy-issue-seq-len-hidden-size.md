# Draft issue for https://github.com/cloudrift-ai/emmy/issues

**Title:** Embedding lookup evaluated as `torch.gather` when the index's last dimension equals the embedding width (seq_len == hidden_size)

---

**What happens**

`GatherOp` carries three torch semantics (`gather`, `embedding`, `index_select`). The two CPU evaluators pick
between them by comparing shapes:

```python
# emmy/compiler/ir/tensor/ir.py, GatherOp.forward (and the same test in
# emmy/compiler/backend/torch_ref.py, _eval, GatherOp branch)
same_rank = data.ndim == indices.ndim
if same_rank and all(indices.shape[k] == data.shape[k] for k in range(data.ndim) if k != axis):
    return np.take_along_axis(data, indices, axis=axis)   # torch.gather semantics
return np.take(data, indices, axis=axis)                  # embedding / index_select
```

For an embedding, `data` is the weight `(V, H)` and `indices` is `(B, S)`, so both have rank 2. When `S == H`,
the only non-axis dimension matches and the lookup is evaluated as `torch.gather`. The result is `(B, S)`
instead of `(B, S, H)`. Downstream, either the shapes break (a following `view` fails) or, after
broadcasting, the values are wrong with no error.

In a language model this happens whenever the sequence length equals `hidden_size`. For example, Pythia-14M
(hidden 128) at seq_len 128, or a 64-wide model at seq_len 64.

**Minimal reproduction** (CPU, no CUDA needed): `emmy_seq_eq_hidden_repro.py`, attached below.

```
emmy-ml 0.3.25, torch 2.15.0+cu132, embedding (100, 64)
seq_len=63: GatherOp.forward -> shape (1, 63, 64), expected (1, 63, 64); traced nn.Embedding -> max |IR - eager| = 0
seq_len=64: GatherOp.forward -> shape (1, 64), expected (1, 64, 64); traced nn.Embedding -> max |IR - eager| = 11
seq_len=65: GatherOp.forward -> shape (1, 65, 64), expected (1, 65, 64); traced nn.Embedding -> max |IR - eager| = 0
```

A GPT-NeoX model, traced with a static full-model wrapper of the kind `build_full_model_wrapper` produces,
then fails at the fused-QKV `view` in the first attention block:

```
RuntimeError: shape '[1, 64, 4, 48]' is invalid for input of size 192    # hidden 64, seq_len 64
RuntimeError: shape '[1, 128, 4, 96]' is invalid for input of size 384   # hidden 128, seq_len 128
```

With batch > 1 (ids `(3, 64)`, H = 64) the broadcast fails instead:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (64) at non-singleton dimension 1
```

**Expected**

The output is `weight[ids]` with shape `(B, S, H)` for every S.

**Suggested fix**

Decide the semantics from the output rank, as the loop lowering already does.
`passes/loop/lowering/040_lower_gather.py` uses `out_rank == idx_rank + data_rank - 1` for
embedding/index_select. The alternative is for the tracer to record the source op on `GatherOp` (for example
`kind="gather" | "take"`), so that no evaluator has to guess from shapes.

**Scope**

- Confirmed: `GatherOp.forward` and `torch_ref` (CPU reference evaluation, as used by
  `torch_ref.build_callable`).
- Not tested: the CUDA kernels (we had no GPU in this environment). `040_lower_gather.py` decides by output
  rank, so the generated kernel is probably correct. A golden-reference comparison built on `torch_ref`
  would then flag a correct kernel as wrong at these shapes. We did not trace which other passes call
  `GatherOp.forward` (for example, constant folding of a gather with constant indices).

**Environment**

emmy-ml 0.3.25, torch 2.15.0+cu132, transformers 5.19.0, numpy 2.5.4, Python 3.13.18, Linux x86_64, CPU only.

**Workaround we use**

Keep seq_len away from the model's hidden size (we use 500 for Pythia-410M to 1.4B, whose hidden sizes are
1024 to 2048).

---

<details><summary>emmy_seq_eq_hidden_repro.py</summary>

```python
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
```

</details>
