"""Language-model helpers shared by P1, P2 and P4: per-example losses, hidden states, losses."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def per_example_lm_loss(run, batch):
    """Mean next-token loss of each sequence; `run` is model(**inputs) with substituted params."""
    ids = batch.long()
    out = run(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    lp = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                         reduction="none").reshape(ids.shape[0], -1)
    return lp.mean(dim=1)


@torch.no_grad()
def token_losses(model, ids: torch.Tensor, device, batch: int = 4) -> torch.Tensor:
    """Per-token NLL [n_seqs, len-1]."""
    outs = []
    for s in range(0, len(ids), batch):
        x = ids[s:s + batch].to(device).long()
        logits = model(input_ids=x).logits.float()
        outs.append(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                                    reduction="none").reshape(x.shape[0], -1).cpu())
    return torch.cat(outs)


@torch.no_grad()
def hidden_states(model, ids: torch.Tensor, device, batch: int = 4) -> list[torch.Tensor]:
    """Per layer output [tokens, hidden] on the probe set (embedding output excluded)."""
    acc = None
    for s in range(0, len(ids), batch):
        out = model(input_ids=ids[s:s + batch].to(device).long(), output_hidden_states=True)
        hs = [h.float().reshape(-1, h.shape[-1]).cpu() for h in out.hidden_states[1:]]
        acc = hs if acc is None else [torch.cat([a, h]) for a, h in zip(acc, hs)]
    return acc


def non_embedding_params(model) -> int:
    n = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ("embed_in", "embed_out", "embed_tokens", "lm_head", "wte", "wpe")):
            continue
        n += p.numel()
    return n


def grad_param_names(model) -> list[str]:
    """Last block's attention input projection and MLP output projection."""
    names = [n for n, _ in model.named_parameters()]
    mt = getattr(model.config, "model_type", "")
    if mt == "gpt_neox":
        L = model.config.num_hidden_layers - 1
        pats = [f"layers.{L}.attention.query_key_value.weight", f"layers.{L}.mlp.dense_4h_to_h.weight"]
    else:
        L = model.config.num_hidden_layers - 1
        pats = [f"layers.{L}.self_attn.q_proj.weight", f"layers.{L}.mlp.down_proj.weight"]
    sel = [n for n in names if any(n.endswith(p) for p in pats)]
    if not sel:
        raise ValueError(f"no gradient layers found for model_type={mt}")
    return sel


def fit_scaling(N: np.ndarray, L: np.ndarray) -> dict:
    """L = E + A N^(-a_N) with E free (nonnegative)."""
    from scipy.optimize import curve_fit
    N = np.asarray(N, float)
    L = np.asarray(L, float)
    if len(N) < 3:
        return {"a_N": np.nan, "E": np.nan, "A": np.nan, "status": "fewer than 3 sizes"}

    def f(logN, E, logA, a):
        return E + np.exp(logA - a * logN)
    try:
        p, _ = curve_fit(f, np.log(N), L, p0=[max(L.min() - 0.5, 0.0), math.log(max(L.max(), 1e-3)) + 0.3 * math.log(N.min()), 0.3],
                         bounds=([0, -50, 0], [L.min(), 50, 3]), maxfev=20000)
        return {"a_N": float(p[2]), "E": float(p[0]), "A": float(np.exp(p[1])), "status": "ok",
                "decades_N": float(np.log10(N.max() / N.min()))}
    except Exception as e:  # noqa: BLE001
        return {"a_N": np.nan, "E": np.nan, "A": np.nan, "status": f"fit failed: {e}"}
