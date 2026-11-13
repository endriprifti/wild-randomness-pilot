"""P5: escape tests (H3.3, escape clause of H6.1) and the injected-noise arm of H3.1.

A two-hidden-layer perceptron on 10^4 Fashion-MNIST images is driven to a converged minimum
W_S by full-batch descent. From W_S, n_rst restarts (batched on the GPU) run
  sgd       minibatch SGD at each (B, eta) on the grid, unclipped;
  clipped   minibatch SGD at the middle eta with norm clipping at each g_max;
  injected  full-batch descent plus SaS noise of known alpha with scale gamma_B = B^(1/alpha-1) gamma_1,
            gamma_1 set so that the noise has the same interquartile range for every alpha
            (positive control: the slope of ln E k_exit on ln B should be alpha - 1).
Every check_every steps, full-batch descent is run from each iterate and exit is declared when
the loss on the segment from that endpoint back to W_S rises above both ends by more than
barrier_tol (linear mode connectivity criterion). Restarts that do not exit by k_cens are
right-censored; E k_exit comes from censored maximum likelihood, and eq:s7-escape is compared
with the Gaussian rival (ln E k linear in B/eta) by censored likelihood. alpha is fitted to the
minibatch noise at W_S (per-example gradient projections, random and Hessian directions).

--part h31 trains from initialization with full-batch descent plus injected SaS noise at
matched interquartile range (alpha in {1.2, 1.5, 1.8, 2}, five seeds) and records Lambda_max,
the Hessian trace, adaptive sharpness and the generalization gap.

Outputs (results/p5_escape/[smoke/]): restarts.csv (one row per restart), grid.csv (per grid
point), fits.csv (model comparison and slopes), alpha_at_WS.csv, run.json; --part h31 writes
results/p5_escape_h31/[smoke/]: h31.csv, run.json

Commands:
  python p5_escape.py                 # escape grid (80 grid points x 50 restarts)
  python p5_escape.py --part h31      # injected-noise arm of H3.1
  python p5_escape.py --smoke
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from common import escape, gradients, runtime, stable


# ----------------------------------------------------------------------------- data
def fmnist(cfg, run, rng):
    n = cfg["n_train"]
    try:
        import torchvision
        root = str(runtime.PILOT_ROOT / "data")
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        g = torch.Generator().manual_seed(run.seed)
        idx = torch.randperm(len(tr.data), generator=g)[:n]
        X = tr.data[idx].float().reshape(n, -1) / 255.0
        Y = tr.targets[idx]
        Xte = te.data.float().reshape(len(te.data), -1) / 255.0
        Yte = te.targets
        src = "torchvision"
    except Exception as e:
        if not run.offline_ok:
            raise
        run.fallback(f"Fashion-MNIST: random teacher data ({type(e).__name__})")
        X = torch.rand(n, 784)
        W = torch.randn(784, 10)
        Y = (X @ W).argmax(1)
        Xte = torch.rand(1000, 784)
        Yte = (Xte @ W).argmax(1)
        src = "fallback"
    mu, sd = X.mean(0), X.std(0) + 1e-3
    return (X - mu) / sd, Y, (Xte - mu) / sd, Yte, src


# ----------------------------------------------------------------------------- batched MLP
def init_params(R, d_in, h, d_out, seed, device):
    g = torch.Generator().manual_seed(seed)
    def lin(a, b):
        return (torch.randn(a, b, generator=g) / math.sqrt(a)).to(device), torch.zeros(b, device=device)
    W1, b1 = lin(d_in, h); W2, b2 = lin(h, h); W3, b3 = lin(h, d_out)
    return [t.unsqueeze(0).repeat(R, *([1] * t.ndim)).contiguous() for t in (W1, b1, W2, b2, W3, b3)]


def forward(P, X, act):
    """P: list of [R, ...] tensors; X: [R, n, d] or [n, d] (broadcast)."""
    W1, b1, W2, b2, W3, b3 = P
    f = torch.tanh if act == "tanh" else F.relu
    if X.ndim == 2:
        X = X.unsqueeze(0).expand(W1.shape[0], -1, -1)
    h = f(torch.baddbmm(b1.unsqueeze(1), X, W1))
    h = f(torch.baddbmm(b2.unsqueeze(1), h, W2))
    return torch.baddbmm(b3.unsqueeze(1), h, W3)


def losses(P, X, Y, act):
    out = forward(P, X, act)
    R, n, k = out.shape
    Yb = Y.unsqueeze(0).expand(R, -1) if Y.ndim == 1 else Y
    return F.cross_entropy(out.reshape(-1, k), Yb.reshape(-1), reduction="none").reshape(R, n).mean(1)


def grads(P, X, Y, act):
    P = [p.detach().requires_grad_(True) for p in P]
    L = losses(P, X, Y, act)
    gs = torch.autograd.grad(L.sum(), P)
    return [g.detach() for g in gs], L.detach()


def full_descent(P, X, Y, act, steps, lr):
    P = [p.clone() for p in P]
    for _ in range(steps):
        g, _ = grads(P, X, Y, act)
        P = [p - lr * gg for p, gg in zip(P, g)]
    return P


@torch.no_grad()
def barrier(P_end, P_S, X, Y, act, m):
    ts = torch.linspace(0, 1, m, device=X.device)
    vals = []
    for t in ts:
        Pt = [(1 - t) * a + t * b for a, b in zip(P_end, P_S)]
        vals.append(losses(Pt, X, Y, act))
    V = torch.stack(vals, 1)  # [R, m]
    ends = torch.maximum(V[:, 0], V[:, -1])
    return V.max(1).values - ends


def gnorm(g):
    return torch.sqrt(sum((t.reshape(t.shape[0], -1) ** 2).sum(1) for t in g))


# ----------------------------------------------------------------------------- escape grid
def run_point(mode, P_S, X, Y, cfg, B, eta, g_max, alpha_inj, gamma1, seed, device, act):
    R = cfg["restarts"]
    P = [p.clone() for p in P_S]
    exited = torch.zeros(R, dtype=torch.bool, device=device)
    k_exit = torch.full((R,), float(cfg["k_cens"]), device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    gn = torch.Generator(device=device).manual_seed(seed)   # injected noise, drawn on the device
    n = len(X)
    clipped = 0
    for k in range(1, cfg["k_cens"] + 1):
        if mode == "injected":
            gr, _ = grads(P, X, Y, act)
            gamma_B = gamma1 * B ** (1 / alpha_inj - 1)
            noise = [stable.rvs_stable_torch(alpha_inj, 0.0, gamma_B, 0.0, t.shape, gn, device, t.dtype) for t in gr]
            gr = [a + b for a, b in zip(gr, noise)]
        else:
            idx = torch.randint(0, n, (R, B), generator=g).to(device)
            gr, _ = grads(P, X[idx], Y[idx], act)
            if mode == "clipped":
                nr = gnorm(gr)
                fac = torch.clamp(g_max / nr.clamp(min=1e-12), max=1.0)
                clipped += int((fac < 1).sum())
                gr = [t * fac.view(-1, *([1] * (t.ndim - 1))) for t in gr]
        live = (~exited).float()
        P = [p - eta * gg * live.view(-1, *([1] * (gg.ndim - 1))) for p, gg in zip(P, gr)]
        if k % cfg["check_every"] == 0:
            P_end = full_descent(P, X, Y, act, cfg["descent_steps"], cfg["descent_lr"])
            bar = barrier(P_end, P_S, X, Y, act, cfg["segment_points"])
            new = (bar > cfg["barrier_tol"]) & ~exited
            k_exit[new] = float(k)
            exited |= new
            if bool(exited.all()):
                break
    return exited.cpu().numpy(), k_exit.cpu().numpy(), clipped / max(1, k * cfg["restarts"])


def alpha_at(P_S, X, Y, cfg, device, act, rng, B=8):
    """Stable fit of minibatch noise at W_S along a random and a Hessian direction (restart 0)."""
    P0 = [p[0:1].clone() for p in P_S]
    names = [f"p{i}" for i in range(len(P0))]

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for nm, t in zip(names, P0):
                self.register_parameter(nm, torch.nn.Parameter(t.clone()))

        def forward(self, x):
            return forward([getattr(self, nm) for nm in names], x, act)[0]

    m = Wrap().to(device)
    pel = lambda run, b: F.cross_entropy(run(b[0]), b[1], reduction="none")
    params = gradients.select_params(m, None)
    gen = torch.Generator().manual_seed(0)
    dirs = gradients.random_directions(params, 1, gen)
    hvp, p0 = gradients.hvp_fn(m, pel, (X[:1000], Y[:1000]), params)
    _, vecs = gradients.top_eigvecs(hvp, p0, 1, 30, gen)
    P = []
    for s in range(0, len(X), 1000):
        P.append(gradients.per_example_projections(m, pel, (X[s:s + 1000], Y[s:s + 1000]), params, dirs + vecs).cpu())
    P = torch.cat(P).numpy()
    rows = []
    for j, nm in enumerate(["random0", "hessian0"]):
        noise = stable.block_noise(P[:, j], B)
        f = stable.fit_stable(noise)
        rows.append({"direction": nm, "B": B, "n_blocks": len(noise), "alpha": f.alpha, "beta": f.beta, "gamma": f.gamma})
    return rows


def converge(X, Y, cfg, device, act, seed):
    P = init_params(1, X.shape[1], cfg["hidden"], 10, seed, device)
    P = full_descent(P, X, Y, act, cfg["converge_steps"], cfg["converge_lr"])
    g, L = grads(P, X, Y, act)
    return P, float(L[0]), float(gnorm(g)[0])


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--part", default="escape", help="escape | h31")
    args = ap.parse_args()
    cfg = config.get("p5_escape", args.smoke)
    run = runtime.start_run("p5_escape" if args.part == "escape" else "p5_escape_h31", cfg, args)
    rng = runtime.rng(run.seed)
    dev = run.device
    act = cfg["act"]
    X, Y, Xte, Yte, src = fmnist(cfg, run, rng)
    X, Y, Xte, Yte = X.to(dev), Y.to(dev), Xte.to(dev), Yte.to(dev)

    if args.part == "h31":
        h = cfg["h31"]
        rows = []
        for a in h["alphas"]:
            iqr_std = stable.stable_iqr(a) if a < 2 else 2 * math.sqrt(2) * 0.6744897501960817
            gamma = h["iqr"] / iqr_std
            for seed in h["seeds"]:
                P = init_params(1, X.shape[1], cfg["hidden"], 10, seed, dev)
                gn = torch.Generator(device=dev).manual_seed(seed)   # noise drawn on the device
                for _ in range(h["steps"]):
                    gr, _ = grads(P, X, Y, act)
                    gr = [t + stable.rvs_stable_torch(a, 0, gamma, 0, t.shape, gn, dev, t.dtype) for t in gr]
                    P = [p - h["lr"] * g for p, g in zip(P, gr)]
                P = full_descent(P, X, Y, act, 200, h["lr"])  # settle without noise before measuring
                with torch.no_grad():
                    tr, te = float(losses(P, X, Y, act)[0]), float(losses(P, Xte, Yte, act)[0])
                m_rows = flatness(P, X, Y, act, dev)
                rows.append({"alpha_injected": a, "seed": seed, "gamma": gamma, "train_loss": tr, "test_loss": te,
                             "gap": te - tr, **m_rows})
                run.note(f"h31 alpha={a} seed={seed}: gap={te - tr:.4f}")
        runtime.write_csv(run.out_dir / "h31.csv", rows)
        runtime.finish_run(run, extra={"n": len(rows)})
        return

    P_S, L_S, gn_S = converge(X, Y, cfg, dev, act, run.seed)
    run.note(f"W_S: loss {L_S:.4f}, |grad| {gn_S:.2e}")
    a_rows = alpha_at(P_S, X, Y, cfg, dev, act, rng)
    alpha_hat = a_rows[0]["alpha"]
    runtime.write_csv(run.out_dir / "alpha_at_WS.csv", a_rows)
    P_S = [p.repeat(cfg["restarts"], *([1] * (p.ndim - 1))) for p in P_S]
    # g_max from quantiles of minibatch gradient norms at W_S (B = 64)
    idx = torch.randint(0, len(X), (cfg["restarts"], 64), generator=torch.Generator().manual_seed(1)).to(dev)
    gr, _ = grads(P_S, X[idx], Y[idx], act)
    norms = gnorm(gr).cpu().numpy()
    g_maxes = [float(np.quantile(norms, q)) for q in cfg["g_max_quantiles"]]
    etas = cfg["etas"]
    eta_mid = etas[len(etas) // 2]
    grid = [("sgd", B, e, None, None) for e in etas for B in cfg["batch_sizes"]]
    grid += [("clipped", B, eta_mid, gm, None) for gm in g_maxes for B in cfg["batch_sizes"]]
    grid += [("injected", B, eta_mid, None, a) for a in cfg["inject_alphas"] for B in cfg["batch_sizes"]]
    rest_rows, grid_rows = [], []
    for gi, (mode, B, eta, gm, a) in enumerate(grid):
        gamma1 = None
        if mode == "injected":
            iqr_std = stable.stable_iqr(a) if a < 2 else 2 * math.sqrt(2) * 0.6744897501960817
            gamma1 = cfg["inject_iqr"] / iqr_std
        t0 = time.perf_counter()
        ex, kk, pc = run_point(mode, P_S, X, Y, cfg, B, eta, gm, a, gamma1, run.seed + gi, dev, act)
        rate = escape.censored_rate(kk, ex)
        grid_rows.append({"mode": mode, "B": B, "eta": eta, "g_max": gm, "alpha_injected": a, "p_clip": pc,
                          **rate, "wall_s": time.perf_counter() - t0})
        rest_rows += [{"mode": mode, "B": B, "eta": eta, "g_max": gm, "alpha_injected": a, "restart": r,
                       "k": float(kk[r]), "exited": bool(ex[r])} for r in range(len(ex))]
        run.note(f"{mode} B={B} eta={eta} g_max={gm} alpha={a}: exits {rate['exits']}/{rate['n']}")
    rest = pd.DataFrame(rest_rows)
    fits = []
    f = escape.fit_grid(rest[rest["mode"] == "sgd"], alpha_hat)
    fits.append({"analysis": "H3.3 sgd grid, eq:s7-escape vs Gaussian rival", **f})
    for gm in g_maxes:
        sub = rest[(rest["mode"] == "clipped") & (rest.g_max == gm)]
        sl = escape.slope_in_lnB(sub)
        fits.append({"analysis": f"H6.1 clipped g_max={gm:.4g}", **sl, "slope_over_alpha_minus_1": sl["slope"] / (alpha_hat - 1) if alpha_hat > 1 else np.nan})
    for a in cfg["inject_alphas"]:
        sub = rest[(rest["mode"] == "injected") & (rest.alpha_injected == a)]
        sl = escape.slope_in_lnB(sub)
        fits.append({"analysis": f"positive control alpha={a}", **sl, "predicted": a - 1})
    runtime.write_csv(run.out_dir / "restarts.csv", rest_rows)
    runtime.write_csv(run.out_dir / "grid.csv", grid_rows)
    runtime.write_csv(run.out_dir / "fits.csv", fits)
    runtime.finish_run(run, extra={"alpha_hat": alpha_hat, "W_S_loss": L_S, "grid_points": len(grid)})


def flatness(P, X, Y, act, device):
    names = [f"p{i}" for i in range(len(P))]

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for nm, t in zip(names, P):
                self.register_parameter(nm, torch.nn.Parameter(t.clone()))

        def forward(self, x):
            return forward([getattr(self, nm) for nm in names], x, act)[0]
    m = Wrap().to(device)
    pel = lambda run, b: F.cross_entropy(run(b[0]), b[1], reduction="none")
    params = gradients.select_params(m, None)
    hvp, p0 = gradients.hvp_fn(m, pel, (X[:2000], Y[:2000]), params)
    gen = torch.Generator().manual_seed(0)
    vals, _ = gradients.top_eigvecs(hvp, p0, 1, 30, gen)
    return {"lambda_max": vals[0], "hessian_trace": gradients.hutchinson_trace(hvp, p0, 20, gen),
            "adaptive_sharpness": gradients.adaptive_sharpness(m, pel, (X[:2000], Y[:2000]))}


if __name__ == "__main__":
    main()
