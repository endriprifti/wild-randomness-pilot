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


def alpha_esd_layer(W: torch.Tensor, min_tail: int = 10, n_boot: int = 0, rng=None) -> dict:
    x = esd(W)
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


def model_alpha_esd(model, min_dim: int = 32, rng=None) -> tuple[float, list[dict]]:
    rows = []
    for name, W in layer_matrices(model, min_dim).items():
        d = alpha_esd_layer(W, rng=rng)
        d["layer"] = name
        d["shape"] = list(W.shape)
        rows.append(d)
    vals = [r["alpha_esd"] for r in rows if np.isfinite(r.get("alpha_esd", np.nan))]
    return (float(np.median(vals)) if vals else np.nan), rows


def activation_stats(H: torch.Tensor, outlier_threshold: float = 6.0, tail_n_boot: int = 0, rng=None) -> dict:
    """H: [tokens, hidden] activations of one layer on the probe set."""
    x = H.float().reshape(-1)
    m = x.mean()
    c = x - m
    var = (c ** 2).mean()
    K = float((c ** 4).mean() / var ** 2 - 3.0)
    kurt0 = float((x ** 4).mean() / (x ** 2).mean() ** 2)
    chan_max = H.float().abs().amax(dim=0)
    n_out = int((chan_max > outlier_threshold).sum())
    out = {"K": K, "Kurt0": kurt0, "n_outlier_channels": n_out, "max_abs": float(chan_max.max())}
    a = x.abs().cpu().numpy()
    a = a[a > 0]
    if len(a) > 200:
        sub = np.random.default_rng(0).choice(a, min(len(a), 200_000), replace=False)
        try:
            f = tails.fit_power_law(sub, min_tail=100)
            out.update({"act_zeta": f.zeta, "act_xmin": f.xmin, "act_n_tail": f.n_tail, "act_decades": f.decades_x})
        except ValueError:
            pass
    return out
