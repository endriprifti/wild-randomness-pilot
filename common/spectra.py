"""Weight spectra (alpha_ESD) and activation statistics (K, outliers) of a trained model.

alpha_ESD is the density exponent of the eigenvalues x of W^T W / N for one layer W (N x M),
rho_ESD(x) ~ x^(-alpha_ESD) above x_min chosen by KS distance; with survival exponent zeta-hat
from the CSN fit, alpha_ESD = zeta-hat + 1. It is compared with exponential and exponentially
truncated fits (Yang et al. 2023) and never mapped to alpha or zeta. p_KS for eigenvalues is
indicative only (eigenvalues are not independent draws). The model value is the median over
layers.

K is the excess kurtosis of activations on a probe set of fixed size, reported with zeta-hat
of |activation| (K is finite only for zeta > 4). Kurt0 is the non-excess kurtosis about zero
used in Section 6. Outlier channels: hidden dimensions whose max |activation| exceeds a
threshold (6.0 by default, the LLM.int8() magnitude).
"""

from __future__ import annotations

import numpy as np
import torch

from . import tails


def layer_matrices(model: torch.nn.Module, min_dim: int = 32) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= min_dim and "embed" not in name and "wte" not in name \
                and "lm_head" not in name and "embed_out" not in name:
            out[name] = p.detach()
    return out


def esd(W: torch.Tensor) -> np.ndarray:
    W = W.float()
    N, M = W.shape
    if N < M:
        W = W.T
        N, M = M, N
    s = torch.linalg.svdvals(W.cpu() if W.device.type == "cpu" else W)
    return (s.double() ** 2 / N).cpu().numpy()


def alpha_esd_values(x: np.ndarray, min_tail: int = 10, n_boot: int = 0, rng=None) -> dict:
    """alpha_ESD of one layer from its ESD eigenvalues (the CPU part of alpha_esd_layer)."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    try:
        r = tails.read_tail(x, n_boot=n_boot, min_tail=min(min_tail, max(5, len(x) // 10)),
                            max_candidates=len(x), hill_db=False, rng=rng)
    except ValueError as e:
        return {"alpha_esd": np.nan, "error": str(e)}
    f = r.fit
    return {"alpha_esd": f.zeta + 1.0, "esd_xmin": f.xmin, "esd_n_tail": f.n_tail, "esd_n": f.n,
            "esd_ks": f.ks, "esd_decades": f.decades_x, "esd_p_ks_indicative": r.p_ks,
            "esd_vs_exp_z": r.tests["exp"]["z"], "esd_vs_exp_p": r.tests["exp"]["p"],
            "esd_vs_trunc_lr": r.tests["truncated"]["lr"], "esd_vs_trunc_p": r.tests["truncated"]["p"],
            "esd_label": r.label}


def alpha_esd_layer(W: torch.Tensor, min_tail: int = 10, n_boot: int = 0, rng=None) -> dict:
    return alpha_esd_values(esd(W), min_tail=min_tail, n_boot=n_boot, rng=rng)


def layer_esds(model, min_dim: int = 32) -> list[tuple[str, list, np.ndarray]]:
    """(name, shape, ESD eigenvalues) of every layer matrix: the GPU part of model_alpha_esd."""
    return [(name, list(W.shape), esd(W)) for name, W in layer_matrices(model, min_dim).items()]


def alpha_esd_rows(esds, rng=None) -> tuple[float, list[dict]]:
    """model_alpha_esd from layer_esds' output: median alpha_ESD over layers and one row per layer."""
    rows = []
    for name, shape, x in esds:
        d = alpha_esd_values(x, rng=rng)
        d["layer"] = name
        d["shape"] = list(shape)
        rows.append(d)
    vals = [r["alpha_esd"] for r in rows if np.isfinite(r.get("alpha_esd", np.nan))]
    return (float(np.median(vals)) if vals else np.nan), rows


def model_alpha_esd(model, min_dim: int = 32, rng=None) -> tuple[float, list[dict]]:
    return alpha_esd_rows(layer_esds(model, min_dim), rng=rng)


def activation_moments(H: torch.Tensor, outlier_threshold: float = 6.0) -> dict:
    """Moment statistics of one layer's activations [tokens, hidden], on H's device (the GPU in P1)."""
    x = H.float().reshape(-1)
    m = x.mean()
    c = x - m
    var = (c ** 2).mean()
    K = float((c ** 4).mean() / var ** 2 - 3.0)
    kurt0 = float((x ** 4).mean() / (x ** 2).mean() ** 2)
    chan_max = H.float().abs().amax(dim=0)
    n_out = int((chan_max > outlier_threshold).sum())
    return {"K": K, "Kurt0": kurt0, "n_outlier_channels": n_out, "max_abs": float(chan_max.max())}


ACT_TAIL_SAMPLE = 200_000


def activation_tail_sample(H: torch.Tensor) -> np.ndarray | None:
    """The values whose tail gives act_zeta: up to ACT_TAIL_SAMPLE nonzero |activations|, drawn without replacement
    with a fixed seed (None if there are 200 or fewer). The filtering runs on H's device; only the draw leaves it."""
    a = H.float().reshape(-1).abs()
    a = a[a > 0]
    if len(a) <= 200:
        return None
    idx = np.random.default_rng(0).choice(len(a), min(len(a), ACT_TAIL_SAMPLE), replace=False)
    return a[torch.as_tensor(idx, device=a.device)].cpu().numpy()


def activation_tail_fit(sub: np.ndarray | None) -> dict:
    """act_zeta: CSN Pareto fit (no bootstrap) of activation_tail_sample's values."""
    if sub is None:
        return {}
    try:
        f = tails.fit_power_law(sub, min_tail=100)
    except ValueError:
        return {}
    return {"act_zeta": f.zeta, "act_xmin": f.xmin, "act_n_tail": f.n_tail, "act_decades": f.decades_x}


def activation_stats(H: torch.Tensor, outlier_threshold: float = 6.0, tail_n_boot: int = 0, rng=None) -> dict:
    """H: [tokens, hidden] activations of one layer on the probe set."""
    out = activation_moments(H, outlier_threshold)
    out.update(activation_tail_fit(activation_tail_sample(H)))
    return out
