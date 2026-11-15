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

Exit times are interval-censored: an exit seen at check k happened in (k_lo, k], k_lo being the
previous check, and every fit uses that likelihood (common/escape.py). A second reading runs
next to the barrier rule: a restart exits by distance or loss when the descended endpoint lies
farther than dist_tol (relative to |W_S|) from W_S or its loss exceeds L(W_S) by loss_tol. Each
restart records the loss and distance at its barrier exit and at its last step.

Outputs (results/p5_escape/[smoke/]): restarts.csv (one row per restart), grid.csv (per grid
point), fits.csv (model comparison and slopes, reading = barrier or distance_or_loss),
alpha_at_WS.csv, checks.csv (--log-checks: every check of every restart), run.json; --part h31
writes results/p5_escape_h31/[smoke/]: h31.csv (with diverged / minimum / converged flags;
sharpness only for converged runs, raw values in *_raw), h31_summary.csv, run.json

Commands:
  python p5_escape.py                 # escape grid (80 grid points x 50 restarts)
  python p5_escape.py --part h31      # injected-noise arm of H3.1
  python p5_escape.py --smoke
Reruns of selected cells keep each cell's seed, so the minibatches and injected noise are those of
the full grid (exit checks draw no random numbers):
  python p5_escape.py --floor-from results/p5_escape --fine-check 10:250 --out results/p5_escape_floor10
  python p5_escape.py --modes injected --alphas 1.2 --second-reading --log-checks --out results/p5_escape_a12
  python p5_escape.py --part refit --base results/p5_escape \
      --replace results/p5_escape_floor10,results/p5_escape_a12 --out results/p5_escape_refit
  python p5_escape.py --part h31-flags --base results/p5_escape_h31 --out results/p5_escape_h31_flags
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

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
def barrier(P_end, P_S, X, Y, act, m, with_end=False):
    """Loss barrier on the segment from P_end to P_S above both ends; with_end also returns L(P_end)."""
    ts = torch.linspace(0, 1, m, device=X.device)
    vals = []
    for t in ts:
        Pt = [(1 - t) * a + t * b for a, b in zip(P_end, P_S)]
        vals.append(losses(Pt, X, Y, act))
    V = torch.stack(vals, 1)  # [R, m]
    ends = torch.maximum(V[:, 0], V[:, -1])
    bar = V.max(1).values - ends
    return (bar, V[:, 0]) if with_end else bar


def gnorm(g):
    return torch.sqrt(sum((t.reshape(t.shape[0], -1) ** 2).sum(1) for t in g))


def rel_dist(P, P_S):
    """|P - W_S| / |W_S| per restart (all layers together)."""
    num = gnorm([a - b for a, b in zip(P, P_S)])
    return num / gnorm(P_S).clamp(min=1e-12)


def check_schedule(cfg):
    """k -> whether exit is checked at step k: every check_every steps, plus every fine_every steps up to
    fine_until when fine_check = (fine_every, fine_until) is set (finer timing of fast exits)."""
    every = int(cfg["check_every"])
    fine = cfg.get("fine_check")
    if not fine:
        return lambda k: k % every == 0
    f_every, f_until = int(fine[0]), int(fine[1])
    return lambda k: k % every == 0 or (k <= f_until and k % f_every == 0)


# ----------------------------------------------------------------------------- escape grid
def run_point(mode, P_S, X, Y, cfg, B, eta, g_max, alpha_inj, gamma1, seed, device, act, L_S=None,
              log_checks=False):
    """Run the restarts of one grid point. Returns a dict of per-restart arrays (barrier exit k, the check
    before it k_lo, the distance-or-loss reading, losses and distances) plus the clipped fraction and,
    with log_checks, one row per check and restart. With cfg second_reading.continue, a restart keeps
    stepping after its barrier exit until the second reading has fired too (or k_cens)."""
    R = cfg["restarts"]
    K = cfg["k_cens"]
    sr = cfg.get("second_reading", {})
    both = bool(sr.get("continue", False))
    d_tol, l_tol = float(sr.get("dist_tol", np.inf)), float(sr.get("loss_tol", np.inf))
    is_check = check_schedule(cfg)
    P = [p.clone() for p in P_S]
    f = lambda v: torch.full((R,), float(v), device=device)            # noqa: E731
    exited, exited_dl = torch.zeros(R, dtype=torch.bool, device=device), torch.zeros(R, dtype=torch.bool, device=device)
    k_exit, k_lo, k_dl, k_lo_dl, k_stop = f(K), f(K), f(K), f(K), f(K)
    loss_exit, dist_exit, dist_max = f(np.nan), f(np.nan), f(0.0)
    alive = torch.ones(R, dtype=torch.bool, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    gn = torch.Generator(device=device).manual_seed(seed)   # injected noise, drawn on the device
    n = len(X)
    clipped, prev, k = 0, 0, 0
    L_S = float(losses([p[:1] for p in P_S], X, Y, act)[0]) if L_S is None else L_S
    log = []
    for k in range(1, K + 1):
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
        live = alive.float()
        P = [p - eta * gg * live.view(-1, *([1] * (gg.ndim - 1))) for p, gg in zip(P, gr)]
        if is_check(k):
            active = alive.clone()                                    # moved during this interval
            P_end = full_descent(P, X, Y, act, cfg["descent_steps"], cfg["descent_lr"])
            bar, l_end = barrier(P_end, P_S, X, Y, act, cfg["segment_points"], with_end=True)
            d_end = rel_dist(P_end, P_S)
            dist_max = torch.where(active, torch.maximum(dist_max, d_end), dist_max)
            new = (bar > cfg["barrier_tol"]) & ~exited & active
            k_exit[new], k_lo[new] = float(k), float(prev)
            loss_exit[new], dist_exit[new] = l_end[new], d_end[new]
            exited |= new
            new_dl = ((d_end > d_tol) | (l_end - L_S > l_tol)) & ~exited_dl & active
            k_dl[new_dl], k_lo_dl[new_dl] = float(k), float(prev)
            exited_dl |= new_dl
            if log_checks:
                a = active.cpu().numpy()
                for r, (b_, l_, d_) in enumerate(zip(bar.cpu().numpy(), l_end.cpu().numpy(), d_end.cpu().numpy())):
                    if a[r]:
                        log.append({"k": k, "restart": r, "barrier": float(b_), "loss_end": float(l_), "dist_end": float(d_)})
            prev = k
            still = ~(exited & exited_dl) if both else ~exited
            stop = alive & ~still
            k_stop[stop] = float(k)
            alive &= still
            if not bool(alive.any()):
                break
    k_stop[alive] = float(k)
    # a reading that did not fire is censored where the restart stopped
    k_dl = torch.where(exited_dl, k_dl, k_stop)
    k_lo_dl = torch.where(exited_dl, k_lo_dl, k_stop)
    with torch.no_grad():
        loss_final = losses(P, X, Y, act)
        dist_final = rel_dist(P, P_S)
    c = lambda t: t.detach().cpu().numpy()                             # noqa: E731
    return {"exited": c(exited), "k": c(k_exit), "k_lo": c(torch.where(exited, k_lo, k_exit)),
            "exited_dl": c(exited_dl), "k_dl": c(k_dl), "k_lo_dl": c(k_lo_dl), "k_stop": c(k_stop),
            "loss_exit": c(loss_exit), "dist_exit": c(dist_exit), "loss_final": c(loss_final),
            "dist_final": c(dist_final), "dist_max": c(dist_max), "L_S": L_S,
            "p_clip": clipped / max(1, k * R), "checks": log}


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


# ----------------------------------------------------------------------------- grid cells, fits, refits
KEY = ["mode", "B", "eta", "g_max_q", "alpha_injected"]
READINGS = {"barrier": ("k", "k_lo", "exited"), "distance_or_loss": ("k_dl", "k_lo_dl", "exited_dl")}


def full_grid(cfg, g_maxes):
    """Every grid point with its index gi (the seed is run.seed + gi, so a rerun of a cell repeats it)."""
    etas = cfg["etas"]
    eta_mid = etas[len(etas) // 2]
    qs = cfg["g_max_quantiles"]
    grid = [("sgd", B, e, None, None, None) for e in etas for B in cfg["batch_sizes"]]
    grid += [("clipped", B, eta_mid, gm, q, None) for gm, q in zip(g_maxes, qs) for B in cfg["batch_sizes"]]
    grid += [("injected", B, eta_mid, None, None, a) for a in cfg["inject_alphas"] for B in cfg["batch_sizes"]]
    return list(enumerate(grid))


def _norm_key(df: pd.DataFrame, quantiles=None) -> pd.DataFrame:
    """Add g_max_q (the quantile behind g_max) to restarts saved before it was recorded."""
    df = df.copy()
    if "g_max_q" not in df:
        df["g_max_q"] = np.nan
        cl = df["mode"] == "clipped"
        if cl.any() and quantiles is not None:
            vals = sorted(df.loc[cl, "g_max"].unique())
            df.loc[cl, "g_max_q"] = df.loc[cl, "g_max"].map(dict(zip(vals, sorted(quantiles))))
    for c in ("eta", "g_max_q", "alpha_injected"):
        df[c] = df[c].astype(float).round(10)
    df["B"] = df["B"].astype(int)
    return df


def _cell_tuple(mode, B, eta, q, a):
    r = lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 10)   # noqa: E731
    return (mode, int(B), r(eta), r(q), r(a))


def _same_cell(a, b):
    return all((x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))
               for x, y in zip(a, b))


def load_restarts(folder) -> tuple[pd.DataFrame, dict]:
    """restarts.csv of a saved run with g_max_q and k_lo filled in (k_lo = k - check_every for exits of runs
    saved before k_lo was recorded); also returns the run's config."""
    folder = Path(folder)
    cfg = json.loads((folder / "run.json").read_text())["config"]
    df = _norm_key(pd.read_csv(folder / "restarts.csv"), cfg.get("g_max_quantiles"))
    if "k_lo" not in df:
        df["k_lo"] = np.where(df.exited, (df.k - int(cfg["check_every"])).clip(lower=0), df.k)
    return df, cfg


def floor_cells(folder, share: float = 1.0) -> list[tuple]:
    """Cells of a saved run whose restarts all exited and at least `share` of them at the first check."""
    df, cfg = load_restarts(folder)
    first = int(cfg["check_every"])
    out = []
    for key, g in df.groupby(KEY, dropna=False):
        if g.exited.all() and (g.k <= first).mean() >= share:
            out.append(_cell_tuple(*key))
    return out


def grid_summary(rest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in rest.groupby(KEY, dropna=False, sort=False):
        r = dict(zip(KEY, key))
        r.update(escape.censored_rate(g.k, g.exited, g.k_lo))
        if "exited_dl" in g and g.exited_dl.notna().all():
            dl = escape.censored_rate(g.k_dl, g.exited_dl.astype(bool), g.k_lo_dl)
            r.update({f"dl_{k}": v for k, v in dl.items() if k in ("exits", "E_k_ic", "E_k_ic_lo95", "E_k_ic_hi95", "floor")})
        for c in ("loss_final", "dist_final", "dist_max"):
            if c in g:
                r[f"{c}_median"] = float(g[c].median())
        rows.append(r)
    return pd.DataFrame(rows)


def fit_all(rest: pd.DataFrame, alpha_hat: float) -> list[dict]:
    """H3.3 grid fit, H6.1 clipped slopes and the injected positive control, for each reading present."""
    fits = []
    for reading, (kc, loc, ec) in READINGS.items():
        if ec not in rest:
            continue
        have = rest[rest[ec].notna()]                 # a rerun may carry the second reading for some cells only
        if not len(have):
            continue
        r = have.assign(k=have[kc], k_lo=have[loc], exited=have[ec].astype(bool))
        sgd = r[r["mode"] == "sgd"]
        if len(sgd):
            fits.append({"reading": reading, "analysis": "H3.3 sgd grid, eq:s7-escape vs Gaussian rival",
                         **escape.fit_grid(sgd, alpha_hat)})
        for q, sub in r[r["mode"] == "clipped"].groupby("g_max_q"):
            sl = escape.slope_in_lnB(sub)
            fits.append({"reading": reading, "analysis": f"H6.1 clipped g_max=q{q:g} ({sub.g_max.iloc[0]:.4g})",
                         **sl, "slope_over_alpha_minus_1": sl["slope"] / (alpha_hat - 1) if alpha_hat > 1 else np.nan})
        for a, sub in r[r["mode"] == "injected"].groupby("alpha_injected"):
            sl = escape.slope_in_lnB(sub)
            fits.append({"reading": reading, "analysis": f"positive control alpha={a:g}", **sl, "predicted": a - 1})
    return fits


def refit(args, run):
    """Combine a saved grid with reruns of some of its cells (each rerun replaces its cells) and refit every
    reading with the interval-censored likelihood. --dist-tol/--loss-tol re-threshold the second reading from
    a rerun's checks.csv (a restart that stopped before the new rule fired is censored where it stopped)."""
    base, bcfg = load_restarts(args.base)
    parts, sources = [base.assign(source=str(args.base))], []
    for folder in [f for f in (args.replace or "").split(",") if f]:
        if not (Path(folder) / "restarts.csv").exists() or (Path(folder) / "restarts.csv").stat().st_size == 0:
            run.note(f"{folder}: no restarts (no cells were selected there); skipped")
            continue
        df, _ = load_restarts(folder)
        if (args.dist_tol is not None or args.loss_tol is not None) and (Path(folder) / "checks.csv").exists():
            df = rethreshold(df, pd.read_csv(Path(folder) / "checks.csv"), args.dist_tol, args.loss_tol)
        keys = {_cell_tuple(*k) for k in df[KEY].itertuples(index=False, name=None)}
        for i, p in enumerate(parts):
            own = p[KEY].apply(lambda r: any(_same_cell(_cell_tuple(*r), k) for k in keys), axis=1)
            parts[i] = p[~own]
        parts.append(df.assign(source=str(folder)))
        sources.append({"folder": folder, "cells": len(keys), "restarts": len(df)})
    rest = pd.concat(parts, ignore_index=True)
    a_rows = pd.read_csv(Path(args.base) / "alpha_at_WS.csv")
    alpha_hat = float(a_rows.alpha.iloc[0])
    runtime.write_csv(run.out_dir / "restarts.csv", rest.to_dict("records"))
    runtime.write_csv(run.out_dir / "grid.csv", grid_summary(rest).to_dict("records"))
    runtime.write_csv(run.out_dir / "fits.csv", fit_all(rest, alpha_hat))
    runtime.write_csv(run.out_dir / "sources.csv", [{"folder": str(args.base), "cells": "all others"}] + sources)
    runtime.finish_run(run, extra={"alpha_hat": alpha_hat, "base_check_every": bcfg["check_every"], "replaced": sources})


def rethreshold(rest, checks, dist_tol, loss_tol):
    """Second reading from logged checks with new thresholds (needs L_S per restart)."""
    rest = rest.copy()
    checks = _norm_key(checks)
    out = []
    for key, g in rest.groupby(KEY, dropna=False, sort=False):
        c = checks
        for col, v in zip(KEY, key):
            c = c[c[col].isna()] if isinstance(v, float) and np.isnan(v) else c[c[col] == v]
        for _, r in g.iterrows():
            cr = c[c.restart == r.restart].sort_values("k")
            dt = dist_tol if dist_tol is not None else np.inf
            lt = loss_tol if loss_tol is not None else np.inf
            hit = cr[(cr.dist_end > dt) | (cr.loss_end - r.L_S > lt)]
            if len(hit):
                kk = int(hit.k.iloc[0])
                prev = cr[cr.k < kk].k
                out.append((r.name, kk, int(prev.iloc[-1]) if len(prev) else 0, True))
            else:
                out.append((r.name, r.k_stop, r.k_stop, False))
    for idx, kk, lo, e in out:
        rest.loc[idx, ["k_dl", "k_lo_dl", "exited_dl"]] = [kk, lo, e]
    return rest


def h31_flags(df: pd.DataFrame, h: dict) -> pd.DataFrame:
    """diverged: train loss not finite or above diverge_loss (ln 10, the loss of a uniform guess over the ten
    classes); minimum: lambda_max > 0 and, when measured, lambda_min >= -min_tol * lambda_max (lambda_max is
    the eigenvalue of largest magnitude, so a negative one is a saddle or a maximum); converged: both.
    Sharpness measures are kept only for converged runs (raw values in *_raw)."""
    df = df.copy()
    tl = df.train_loss.astype(float)
    df["diverged"] = ~np.isfinite(tl) | (tl > h.get("diverge_loss", math.log(10)))
    lmax = df.lambda_max.astype(float)
    df["minimum"] = lmax > 0
    df["minimum_check"] = "lambda_max > 0"
    if "lambda_min" in df and df.lambda_min.notna().any():
        tol = h.get("min_tol", 0.01)
        df["minimum"] &= df.lambda_min.astype(float) >= -tol * lmax
        df["minimum_check"] = f"lambda_max > 0 and lambda_min >= -{tol} lambda_max"
    df["converged"] = ~df.diverged & df.minimum
    for c in ("lambda_max", "hessian_trace", "adaptive_sharpness"):
        if f"{c}_raw" not in df:
            df[f"{c}_raw"] = df[c]
        df[c] = df[f"{c}_raw"].where(df.converged)
    return df


def h31_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for a, g in df.groupby("alpha_injected"):
        c = g[g.converged]
        rows.append({"alpha_injected": a, "runs": len(g), "diverged": int(g.diverged.sum()),
                     "non_minimum": int((~g.minimum & ~g.diverged).sum()), "converged": len(c),
                     **{f"{m}_mean_converged": float(c[m].mean()) if len(c) else np.nan
                        for m in ("lambda_max", "hessian_trace", "adaptive_sharpness", "gap")}})
    return rows


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--part", default="escape", help="escape | h31 | refit | h31-flags")
    ap.add_argument("--modes", default=None, help="escape: only these modes (sgd,clipped,injected)")
    ap.add_argument("--alphas", default=None, help="escape: only these injected alphas, e.g. 1.2")
    ap.add_argument("--floor-from", default=None, help="escape: only the cells of this saved run whose restarts "
                                                       "all exited at its first check")
    ap.add_argument("--floor-share", type=float, default=1.0, help="with --floor-from: share of first-check exits")
    ap.add_argument("--check-every", type=int, default=None, help="override check_every")
    ap.add_argument("--fine-check", default=None, help="EVERY:UNTIL, extra checks every EVERY steps up to UNTIL")
    ap.add_argument("--second-reading", action="store_true", help="keep stepping after a barrier exit until the "
                                                                  "distance-or-loss reading has fired too")
    ap.add_argument("--log-checks", action="store_true", help="write checks.csv (every check of every restart)")
    ap.add_argument("--base", default=None, help="refit / h31-flags: the saved run")
    ap.add_argument("--replace", default=None, help="refit: comma-separated reruns whose cells replace the base's")
    ap.add_argument("--dist-tol", type=float, default=None, help="refit: re-threshold the second reading")
    ap.add_argument("--loss-tol", type=float, default=None, help="refit: re-threshold the second reading")
    args = ap.parse_args()
    cfg = config.get("p5_escape", args.smoke)
    if args.check_every:
        cfg["check_every"] = args.check_every
    if args.fine_check:
        cfg["fine_check"] = [int(v) for v in args.fine_check.split(":")]
    if args.second_reading:
        cfg["second_reading"]["continue"] = True
    name = {"escape": "p5_escape", "h31": "p5_escape_h31", "refit": "p5_escape_refit", "h31-flags": "p5_escape_h31_flags"}
    run = runtime.start_run(name[args.part], cfg, args)
    if args.part == "refit":
        return refit(args, run)
    if args.part == "h31-flags":
        df = h31_flags(pd.read_csv(Path(args.base) / "h31.csv"), cfg["h31"])
        runtime.write_csv(run.out_dir / "h31.csv", df.to_dict("records"))
        runtime.write_csv(run.out_dir / "h31_summary.csv", h31_summary(df))
        runtime.finish_run(run, extra={"n": len(df), "converged": int(df.converged.sum())})
        return
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
                g_end, _ = grads(P, X, Y, act)
                m_rows = flatness(P, X, Y, act, dev) if math.isfinite(tr) else \
                    {"lambda_max": np.nan, "lambda_min": np.nan, "hessian_trace": np.nan, "adaptive_sharpness": np.nan}
                rows.append({"alpha_injected": a, "seed": seed, "gamma": gamma, "train_loss": tr, "test_loss": te,
                             "gap": te - tr, "grad_norm": float(gnorm(g_end)[0]), **m_rows})
                run.note(f"h31 alpha={a} seed={seed}: train loss {tr:.4f}, gap={te - tr:.4f}")
        df = h31_flags(pd.DataFrame(rows), h)
        runtime.write_csv(run.out_dir / "h31.csv", df.to_dict("records"))
        runtime.write_csv(run.out_dir / "h31_summary.csv", h31_summary(df))
        runtime.finish_run(run, extra={"n": len(rows), "converged": int(df.converged.sum())})
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
    grid = full_grid(cfg, g_maxes)
    if args.modes:
        grid = [c for c in grid if c[1][0] in args.modes.split(",")]
    if args.alphas:
        want = [float(a) for a in args.alphas.split(",")]
        grid = [c for c in grid if c[1][0] != "injected" or any(abs(c[1][5] - a) < 1e-9 for a in want)]
    if args.floor_from:
        cells = floor_cells(args.floor_from, args.floor_share)
        def chosen(cell):
            m, B, e, _, q, a = cell
            return any(_same_cell(_cell_tuple(m, B, e, q, a), k) for k in cells)
        grid = [c for c in grid if chosen(c[1])]
        run.note(f"floor cells from {args.floor_from}: {len(cells)} found, {len(grid)} in this grid")
    if not grid:
        run.note("no grid cells selected; nothing to run")
        runtime.finish_run(run, status="empty", extra={"grid_points": 0})
        return
    rest_rows, grid_rows, check_rows = [], [], []
    for gi, (mode, B, eta, gm, q, a) in grid:
        gamma1 = None
        if mode == "injected":
            iqr_std = stable.stable_iqr(a) if a < 2 else 2 * math.sqrt(2) * 0.6744897501960817
            gamma1 = cfg["inject_iqr"] / iqr_std
        t0 = time.perf_counter()
        out = run_point(mode, P_S, X, Y, cfg, B, eta, gm, a, gamma1, run.seed + gi, dev, act, L_S, args.log_checks)
        cell = {"mode": mode, "B": B, "eta": eta, "g_max": gm, "g_max_q": q, "alpha_injected": a}
        rate = escape.censored_rate(out["k"], out["exited"], out["k_lo"])
        dl = escape.censored_rate(out["k_dl"], out["exited_dl"], out["k_lo_dl"])
        grid_rows.append({**cell, "gi": gi, "p_clip": out["p_clip"], **rate,
                          **{f"dl_{k}": v for k, v in dl.items() if k in ("exits", "E_k_ic", "E_k_ic_lo95", "E_k_ic_hi95", "floor")},
                          "wall_s": time.perf_counter() - t0})
        per = ("k", "k_lo", "exited", "k_dl", "k_lo_dl", "exited_dl", "k_stop", "loss_exit", "dist_exit",
               "loss_final", "dist_final", "dist_max")
        rest_rows += [{**cell, "restart": r, **{c: (bool(out[c][r]) if c.startswith("exited") else float(out[c][r]))
                                                for c in per}, "L_S": out["L_S"]} for r in range(cfg["restarts"])]
        check_rows += [{**cell, **row} for row in out["checks"]]
        run.note(f"{mode} B={B} eta={eta} g_max={gm} alpha={a}: exits {rate['exits']}/{rate['n']} "
                 f"(first check {rate['first_check_exits']}), distance-or-loss exits {dl['exits']}")
    rest = _norm_key(pd.DataFrame(rest_rows))
    fits = fit_all(rest, alpha_hat)
    runtime.write_csv(run.out_dir / "restarts.csv", rest.to_dict("records"))
    runtime.write_csv(run.out_dir / "grid.csv", grid_rows)
    runtime.write_csv(run.out_dir / "fits.csv", fits)
    if args.log_checks:
        runtime.write_csv(run.out_dir / "checks.csv", check_rows)
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
    lam = vals[0]
    # most negative curvature: the top eigenvalue of H - lambda_max I is lambda_min - lambda_max
    if lam > 0:
        shifted = lambda v: {n: hv - lam * v[n] for n, hv in hvp(v).items()}   # noqa: E731
        lam_min = lam + gradients.top_eigvecs(shifted, p0, 1, 30, gen)[0][0]
    else:
        lam_min = np.nan
    return {"lambda_max": lam, "lambda_min": lam_min, "hessian_trace": gradients.hutchinson_trace(hvp, p0, 20, gen),
            "adaptive_sharpness": gradients.adaptive_sharpness(m, pel, (X[:2000], Y[:2000]))}


if __name__ == "__main__":
    main()
