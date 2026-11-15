"""Per-example gradient projections, Hessian-block eigenvectors and flatness measures.

Per-example gradients enter Section 7 only through projections on fixed directions u
(random, or leading eigenvectors of a layer's Hessian block, sec. 7.5). The projection of
example i's gradient on u is the directional derivative of its loss, so one forward-mode
JVP of the vector of per-example losses gives all projections in a batch at the cost of
about two forward passes, with no per-example backward pass.

Models with batch statistics (BatchNorm) must be measured in eval mode, otherwise
"per-example" gradients are coupled through the batch.

Micro-batches (chunk): every batch measure here is a mean over examples, so it can be computed
on micro-batches of at most `chunk` examples and combined with weights n_c / n: the Hessian-vector
product of the mean loss is sum_c (n_c / n) H_c v, with H_c the Hessian of micro-batch c's mean
loss (H v is linear in the loss); the same holds for the gradient and the loss in the sharpness
ascent, and per-example projections are simply stacked. The result equals the full-batch one up
to floating-point summation order, at the memory of one micro-batch. chunk=None keeps one batch.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch.func import functional_call, jvp, grad


def select_params(model: torch.nn.Module, patterns: Iterable[str] | None) -> dict[str, torch.Tensor]:
    """Parameters whose names contain any of the patterns (all trainable ones if None)."""
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if patterns is None or any(pt in name for pt in patterns):
            out[name] = p
    if not out:
        raise ValueError(f"no parameters match {patterns}")
    return out


def random_directions(params: dict[str, torch.Tensor], k: int, generator: torch.Generator) -> list[dict]:
    """k Gaussian directions over the selected parameters, each of unit total norm."""
    dirs = []
    for _ in range(k):
        d = {n: torch.randn(p.shape, generator=generator, device="cpu").to(p.device, p.dtype)
             for n, p in params.items()}
        norm = math.sqrt(sum(float((v.float() ** 2).sum()) for v in d.values()))
        dirs.append({n: v / norm for n, v in d.items()})
    return dirs


def batch_len(batch) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, (tuple, list)):
        return batch_len(batch[0])
    if isinstance(batch, dict):
        return batch_len(next(iter(batch.values())))
    raise TypeError(f"cannot split a batch of type {type(batch).__name__}")


def batch_slice(batch, s: int, e: int):
    if isinstance(batch, torch.Tensor):
        return batch[s:e]
    if isinstance(batch, (tuple, list)):
        return type(batch)(batch_slice(b, s, e) for b in batch)
    if isinstance(batch, dict):
        return {k: batch_slice(v, s, e) for k, v in batch.items()}
    raise TypeError(f"cannot split a batch of type {type(batch).__name__}")


def micro_batches(batch, chunk: int | None) -> list[tuple[object, float]]:
    """[(micro-batch, n_c / n)]; the whole batch with weight 1 if chunk is None or covers it."""
    n = batch_len(batch)
    if not chunk or chunk >= n:
        return [(batch, 1.0)]
    return [(batch_slice(batch, s, min(s + chunk, n)), (min(s + chunk, n) - s) / n) for s in range(0, n, chunk)]


def _split(model, sel):
    frozen = {n: p.detach() for n, p in model.named_parameters() if n not in sel}
    buffers = dict(model.named_buffers())
    return frozen, buffers


def per_example_projections(model: torch.nn.Module, per_example_loss: Callable, batch,
                            params: dict[str, torch.Tensor], directions: list[dict],
                            chunk: int | None = None) -> torch.Tensor:
    """Matrix [batch, n_directions] of grad L_i . u for each direction u (forward-mode JVP).

    per_example_loss(outputs_fn, batch) must return a vector of per-example losses, where
    outputs_fn(**inputs) runs the model with the substituted parameters. With chunk, the rows
    are computed on micro-batches of at most chunk examples and stacked.
    """
    parts = micro_batches(batch, chunk)
    if len(parts) > 1:
        return torch.cat([per_example_projections(model, per_example_loss, b, params, directions) for b, _ in parts])
    sel = {n: p.detach() for n, p in params.items()}
    frozen, buffers = _split(model, sel)

    def f(p):
        merged = {**frozen, **p, **buffers}
        run = lambda *a, **kw: functional_call(model, merged, a, kw)
        return per_example_loss(run, batch)

    cols = []
    for u in directions:
        _, t = jvp(f, (sel,), ({n: u[n] for n in sel},))
        cols.append(t.detach().float())
    return torch.stack(cols, dim=1)


def mean_loss_fn(model, per_example_loss, batch, params):
    sel_names = list(params)
    frozen, buffers = _split(model, params)

    def f(p):
        merged = {**frozen, **p, **buffers}
        run = lambda *a, **kw: functional_call(model, merged, a, kw)
        return per_example_loss(run, batch).mean()
    return f, {n: params[n].detach() for n in sel_names}


def hvp_fn(model, per_example_loss, batch, params, chunk: int | None = None):
    """v -> H v for the Hessian block of the selected parameters (forward-over-reverse).

    With chunk, H v = sum_c (n_c / n) H_c v over micro-batches of at most chunk examples (exact:
    the loss is a mean over examples), so memory is that of one micro-batch."""
    parts = micro_batches(batch, chunk)
    grads = []
    for b, w in parts:
        f, p0 = mean_loss_fn(model, per_example_loss, b, params)
        grads.append((grad(f), w))
    if len(grads) == 1:
        g = grads[0][0]

        def hvp(v):
            return jvp(g, (p0,), (v,))[1]
        return hvp, p0

    def hvp(v):
        out = None
        for g, w in grads:
            hv = jvp(g, (p0,), (v,))[1]
            if out is None:
                out = {n: t * w for n, t in hv.items()}
            else:
                for n, t in hv.items():
                    out[n].add_(t, alpha=w)
            del hv
        return out
    return hvp, p0


def _dot(a, b):
    return sum(float((a[n].double() * b[n].double()).sum()) for n in a)


def _axpy(alpha, x, y):
    return {n: y[n] + alpha * x[n] for n in y}


def _scale(alpha, x):
    return {n: alpha * v for n, v in x.items()}


def top_eigvecs(hvp, p0, k: int = 1, iters: int = 30, generator: torch.Generator | None = None,
                tol: float = 1e-3) -> tuple[list[float], list[dict]]:
    """Leading k eigenpairs of a Hessian block by power iteration with deflation."""
    gen = generator or torch.Generator().manual_seed(0)
    vals, vecs = [], []
    for _ in range(k):
        v = {n: torch.randn(p.shape, generator=gen).to(p.device, p.dtype) for n, p in p0.items()}
        v = _scale(1 / math.sqrt(_dot(v, v)), v)
        lam = 0.0
        for _ in range(iters):
            Hv = hvp(v)
            for lv, vv in zip(vals, vecs):  # deflate
                Hv = _axpy(-lv * _dot(vv, v), vv, Hv)
            new_lam = _dot(v, Hv)
            nrm = math.sqrt(max(_dot(Hv, Hv), 1e-300))
            v = _scale(1 / nrm, Hv)
            if abs(new_lam - lam) <= tol * max(abs(new_lam), 1e-12):
                lam = new_lam
                break
            lam = new_lam
        vals.append(lam)
        vecs.append({n: t.detach() for n, t in v.items()})
    return vals, vecs


def hutchinson_trace(hvp, p0, n_samples: int = 20, generator: torch.Generator | None = None) -> float:
    gen = generator or torch.Generator().manual_seed(0)
    est = []
    for _ in range(n_samples):
        v = {n: (torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.device, p.dtype)
             for n, p in p0.items()}
        est.append(_dot(v, hvp(v)))
    return float(sum(est) / len(est))


@torch.no_grad()
def _eval_loss(model, per_example_loss, batch, chunk: int | None = None):
    run = lambda *a, **kw: model(*a, **kw)
    return float(sum(w * float(per_example_loss(run, b).mean()) for b, w in micro_batches(batch, chunk)))


def adaptive_sharpness(model, per_example_loss, batch, rho: float = 0.05, steps: int = 5,
                       lr: float | None = None, chunk: int | None = None) -> float:
    """Adaptive (rescaling-invariant) sharpness: max over ||delta / |w| ||_2 <= rho of L(w + delta) - L(w).

    Approximated by a few steps of projected gradient ascent in the |w|-scaled coordinates
    (Kwon et al. 2021; Andriushchenko et al. 2023). Restores the weights afterwards. With chunk,
    the loss and its gradient are accumulated over micro-batches with weights n_c / n.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    w0 = [p.detach().clone() for p in params]
    parts = micro_batches(batch, chunk)
    base = _eval_loss(model, per_example_loss, batch, chunk)
    lr = lr if lr is not None else rho / max(steps, 1)
    deltas = [torch.zeros_like(p) for p in params]
    try:
        for _ in range(steps):
            for p, w, d in zip(params, w0, deltas):
                p.data.copy_(w + d)
            model.zero_grad(set_to_none=True)
            run = lambda *a, **kw: model(*a, **kw)
            for b, wc in parts:
                (per_example_loss(run, b).mean() * wc).backward()
            # gradient in scaled coordinates z = delta / |w|: dL/dz = |w| dL/ddelta
            zs = [d / (w.abs() + 1e-12) for d, w in zip(deltas, w0)]
            gz = [(p.grad if p.grad is not None else torch.zeros_like(p)) * w.abs() for p, w in zip(params, w0)]
            gn = math.sqrt(sum(float((g ** 2).sum()) for g in gz)) + 1e-12
            zs = [z + lr * g / gn for z, g in zip(zs, gz)]
            zn = math.sqrt(sum(float((z ** 2).sum()) for z in zs))
            if zn > rho:
                zs = [z * rho / zn for z in zs]
            deltas = [z * w.abs() for z, w in zip(zs, w0)]
        for p, w, d in zip(params, w0, deltas):
            p.data.copy_(w + d)
        worst = _eval_loss(model, per_example_loss, batch, chunk)
    finally:
        for p, w in zip(params, w0):
            p.data.copy_(w)
        model.zero_grad(set_to_none=True)
    return float(worst - base)
