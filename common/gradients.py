"""Per-example gradient projections, Hessian-block eigenvectors and flatness measures.

Per-example gradients enter Section 7 only through projections on fixed directions u
(random, or leading eigenvectors of a layer's Hessian block, sec. 7.5). The projection of
example i's gradient on u is the directional derivative of its loss, so one forward-mode
JVP of the vector of per-example losses gives all projections in a batch at the cost of
about two forward passes, with no per-example backward pass.

Models with batch statistics (BatchNorm) must be measured in eval mode, otherwise
"per-example" gradients are coupled through the batch.
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


def _split(model, sel):
    frozen = {n: p.detach() for n, p in model.named_parameters() if n not in sel}
    buffers = dict(model.named_buffers())
    return frozen, buffers


def per_example_projections(model: torch.nn.Module, per_example_loss: Callable, batch,
                            params: dict[str, torch.Tensor], directions: list[dict]) -> torch.Tensor:
    """Matrix [batch, n_directions] of grad L_i . u for each direction u (forward-mode JVP).

    per_example_loss(outputs_fn, batch) must return a vector of per-example losses, where
    outputs_fn(**inputs) runs the model with the substituted parameters.
    """
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


def hvp_fn(model, per_example_loss, batch, params):
    """v -> H v for the Hessian block of the selected parameters (forward-over-reverse)."""
    f, p0 = mean_loss_fn(model, per_example_loss, batch, params)
    g = grad(f)

    def hvp(v):
        return jvp(g, (p0,), (v,))[1]
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
def _eval_loss(model, per_example_loss, batch):
    run = lambda *a, **kw: model(*a, **kw)
    return float(per_example_loss(run, batch).mean())


def adaptive_sharpness(model, per_example_loss, batch, rho: float = 0.05, steps: int = 5,
                       lr: float | None = None) -> float:
    """Adaptive (rescaling-invariant) sharpness: max over ||delta / |w| ||_2 <= rho of L(w + delta) - L(w).

    Approximated by a few steps of projected gradient ascent in the |w|-scaled coordinates
    (Kwon et al. 2021; Andriushchenko et al. 2023). Restores the weights afterwards.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    w0 = [p.detach().clone() for p in params]
    base = _eval_loss(model, per_example_loss, batch)
    lr = lr if lr is not None else rho / max(steps, 1)
    deltas = [torch.zeros_like(p) for p in params]
    try:
        for _ in range(steps):
            for p, w, d in zip(params, w0, deltas):
                p.data.copy_(w + d)
            model.zero_grad(set_to_none=True)
            run = lambda *a, **kw: model(*a, **kw)
            loss = per_example_loss(run, batch).mean()
            loss.backward()
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
        worst = _eval_loss(model, per_example_loss, batch)
    finally:
        for p, w in zip(params, w0):
            p.data.copy_(w)
        model.zero_grad(set_to_none=True)
    return float(worst - base)
