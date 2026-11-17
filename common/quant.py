"""Simulated (fake) quantization for H6.2 (Section 6.2; register row H6.2).

All schemes are simulated in plain PyTorch: weights are quantized and dequantized once,
activations are quantized and dequantized on the fly. No custom CUDA kernel is used, so the
numerics are the same on Turing, Ampere, Ada and Blackwell GPUs and on the CPU.

Schemes (round-to-nearest for every arm, so that only the treatment of outliers differs):
  uniform   INT4 symmetric, per-group scales along the input dimension; activations INT4
            symmetric per token (W4A4) or full precision (weight-only control).
  outlier   as uniform, but the input channels with the largest calibrated activation
            magnitude stay in 16 bits in both weights and activations (LLM.int8()/SpQR style).
            The kept channels are left out of the per-token activation scale and of the weight
            group scales, so they do not set the step size of the other channels (fixed
            2026-11-15: before, the scales were taken over all channels and the kept ones were
            only restored afterwards).
  rotation  QuaRot-style: x -> x Q and W -> W Q with Q a randomized Hadamard matrix
            (Haar-random orthogonal when the dimension is not a power of two), then uniform.
            Simulated online at every linear input; numerically this is the quantization error
            QuaRot's fused rotations produce, without their kernels. GPTQ is not applied.
  nf4       weight-only control arm: blockwise absmax NF4 code book (Dettmers et al. 2023).

Embeddings and the LM head stay in 16 bits in every arm. Bits per weight include scales and
16-bit outlier columns and must match across arms within 0.05 (sec. 7.3).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NF4_CODE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


@dataclass
class Scheme:
    name: str                 # uniform | outlier | rotation | nf4 | none
    w_bits: int = 4
    a_bits: int | None = 4    # None = activations in 16 bits (weight-only arm)
    group: int = 128          # weight group size along the input dimension (0 = per output channel)
    outlier_frac: float = 0.0
    rotation_nonpow2: str = "haar"

    def bits_per_weight(self, d_in: int) -> float:
        if self.name == "none":
            return 16.0
        g = self.group if self.group > 0 else d_in
        base = self.w_bits + 16.0 / g
        if self.name == "outlier":
            k = max(1, int(round(self.outlier_frac * d_in)))
            f = k / d_in
            return (1 - f) * base + f * 16.0
        return base


def check_bits_matched(schemes: list[Scheme], d_in: int, tol: float = 0.05) -> dict:
    bits = {s.name: s.bits_per_weight(d_in) for s in schemes if s.name != "none"}
    spread = max(bits.values()) - min(bits.values())
    return {"bits": bits, "spread": spread, "matched": spread <= tol + 1e-12}


# ----------------------------------------------------------------------------- primitives
def quant_sym(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Symmetric round-to-nearest fake quantization with absmax scale along `dim`."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale


def quant_weight_groups(W: torch.Tensor, bits: int, group: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Per-group INT fake quantization along the input dimension. Columns in `keep` (16-bit outlier
    channels) are left out of the group scales and returned unchanged."""
    out_f, in_f = W.shape
    Wz = W
    if keep is not None and len(keep):
        Wz = W.clone()
        Wz[:, keep] = 0          # a zero never sets an absmax, so the kept columns do not set any scale
    if group <= 0 or group >= in_f:
        Wq = quant_sym(Wz, bits, dim=1)
    else:
        pad = (-in_f) % group
        Wp = F.pad(Wz, (0, pad)) if pad else Wz
        Wg = Wp.reshape(out_f, -1, group)
        Wq = quant_sym(Wg, bits, dim=2).reshape(out_f, -1)[:, :in_f]
    if keep is not None and len(keep):
        Wq[:, keep] = W[:, keep]
    return Wq


def quant_act(x: torch.Tensor, bits: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Per-token symmetric fake quantization of activations (computed in at least float32). Channels in
    `keep` stay in the input precision and are left out of the per-token absmax, so an outlier channel
    does not set the scale of the others."""
    work = torch.promote_types(x.dtype, torch.float32)
    if keep is None or not len(keep):
        return quant_sym(x.to(work), bits, dim=-1).to(x.dtype)
    rest = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    rest[keep] = False
    xq = x.clone()
    xq[..., rest] = quant_sym(x[..., rest].to(work), bits, dim=-1).to(x.dtype)
    return xq


def quant_nf4(W: torch.Tensor, block: int = 64, chunk_blocks: int = 1 << 16) -> torch.Tensor:
    """Blockwise absmax NF4. The nearest code is found chunk_blocks blocks at a time: each element's code depends
    on its own value and block only, so the result is the same as in one pass, without the [blocks, block, 16]
    distance tensor of the whole matrix (4.3 GiB in float32 for a 4096 x 16384 weight)."""
    flat = W.reshape(-1)
    pad = (-flat.numel()) % block
    fp = F.pad(flat, (0, pad)) if pad else flat
    blocks = fp.reshape(-1, block)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    code = NF4_CODE.to(W.device, W.dtype)
    deq = torch.empty_like(blocks)
    for s in range(0, blocks.shape[0], chunk_blocks):
        normed = blocks[s:s + chunk_blocks] / absmax[s:s + chunk_blocks]
        idx = (normed.unsqueeze(-1) - code).abs().argmin(dim=-1)
        deq[s:s + chunk_blocks] = code[idx] * absmax[s:s + chunk_blocks]
    return deq.reshape(-1)[: flat.numel()].reshape(W.shape)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last dimension (power of two)."""
    d = x.shape[-1]
    h = 1
    y = x.reshape(-1, d)
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return (y.reshape(x.shape) / math.sqrt(d))


class Rotation:
    """Randomized Hadamard (power of two) or Haar orthogonal matrix, applied as x -> x Q."""

    def __init__(self, d: int, seed: int, device, dtype, nonpow2: str = "haar"):
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.pow2 = (d & (d - 1)) == 0
        if self.pow2:
            self.signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(device, dtype)
            self.Q = None
        else:
            if nonpow2 != "haar":
                raise ValueError("only Haar rotation is implemented for non-power-of-two dimensions")
            A = torch.randn(d, d, generator=g, dtype=torch.float64)
            Qm, R = torch.linalg.qr(A)
            Qm = Qm * torch.sign(torch.diagonal(R))
            self.Q = Qm.to(device, dtype)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.pow2:
            return fwht(x * self.signs)  # x diag(s) H
        return x @ self.Q


# ----------------------------------------------------------------------------- module
class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, scheme: Scheme, act_absmax: torch.Tensor | None,
                 rotation: Rotation | None, noise: dict | None = None):
        super().__init__()
        self.scheme = scheme
        self.bias = lin.bias
        self.rotation = rotation
        self.noise = noise or {}
        W = lin.weight.data
        dtype = W.dtype
        self.keep = None
        with torch.no_grad():
            if scheme.name == "rotation":
                # (x Q)(W Q)^T = x W^T, computed in float32 so that a bfloat16 model's 8-bit mantissa does not
                # add rounding error to this arm alone
                W = rotation.apply(W.to(torch.promote_types(dtype, torch.float32)))
            if scheme.name == "outlier":
                k = max(1, int(round(scheme.outlier_frac * W.shape[1])))
                self.keep = torch.topk(act_absmax.to(W.device).float(), k).indices
            if scheme.name in ("uniform", "rotation", "outlier"):
                work = torch.promote_types(W.dtype, torch.float32)
                Wq = quant_weight_groups(W.to(work), scheme.w_bits, scheme.group, self.keep).to(dtype)
            elif scheme.name == "nf4":
                Wq = quant_nf4(W.float(), scheme.group if scheme.group > 0 else 64).to(W.dtype)
            elif scheme.name == "isotropic":
                g = torch.Generator(device="cpu").manual_seed(int(self.noise.get("seed", 0)))
                rms = W.float().pow(2).mean().sqrt()
                Wq = (W.float() + self.noise.get("sigma_w", 0.0) * rms *
                      torch.randn(W.shape, generator=g).to(W.device)).to(W.dtype)
            else:
                Wq = W
        self.weight = nn.Parameter(Wq, requires_grad=False)

    def forward(self, x):
        s = self.scheme
        dtype = x.dtype
        if s.name == "rotation":
            x = self.rotation.apply(x.to(torch.promote_types(dtype, torch.float32)))
        if s.a_bits is not None and s.name in ("uniform", "rotation", "outlier"):
            x = quant_act(x, s.a_bits, self.keep)
        elif s.name == "isotropic" and self.noise.get("sigma_a", 0.0) > 0:
            rms = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            x = (x.float() + self.noise["sigma_a"] * rms * torch.randn_like(x, dtype=torch.float32)).to(x.dtype)
        return F.linear(x.to(dtype), self.weight, self.bias)


def linear_layers(model: nn.Module, skip=("embed", "lm_head", "embed_out", "wte")) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(sk in n for sk in skip)}


@torch.no_grad()
def calibrate_absmax(model: nn.Module, batches, device) -> dict[str, torch.Tensor]:
    """Max |activation| per input channel of every quantized linear layer."""
    stats: dict[str, torch.Tensor] = {}
    hooks = []
    for name, lin in linear_layers(model).items():
        def hook(mod, inp, out, name=name):
            a = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(dim=0).float()
            stats[name] = torch.maximum(stats[name], a) if name in stats else a
        hooks.append(lin.register_forward_hook(hook))
    try:
        for b in batches:
            model(input_ids=b.to(device))
    finally:
        for h in hooks:
            h.remove()
    return stats


def apply_scheme(model: nn.Module, scheme: Scheme, absmax: dict | None = None, seed: int = 0,
                 noise: dict | None = None) -> nn.Module:
    """Replace every linear layer (except embeddings and LM head) in place. Each layer is looked up when it is
    replaced and no reference to it is kept, so the original weights are freed one layer at a time: the peak is
    the model plus one layer (python memory_check.py p2), not the model twice (holding every original layer
    until the end, as before 2026-11-17, gave about 25 GiB for a 7B model in bfloat16)."""
    rotations: dict[int, Rotation] = {}
    for name in list(linear_layers(model)):
        lin = model.get_submodule(name)
        rot = None
        if scheme.name == "rotation":
            d = lin.in_features
            if d not in rotations:
                rdt = torch.promote_types(lin.weight.dtype, torch.float32)       # rotations in at least float32
                rotations[d] = Rotation(d, seed + d, lin.weight.device, rdt, scheme.rotation_nonpow2)
            rot = rotations[d]
        am = absmax.get(name) if absmax else None
        nz = None
        if noise is not None:
            nz = dict(noise)
            nz["seed"] = noise.get("seed", 0) + zlib.crc32(name.encode()) % 100003
        q = QuantLinear(lin, scheme, am, rot, nz)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, q)
        del lin, q
    return model
