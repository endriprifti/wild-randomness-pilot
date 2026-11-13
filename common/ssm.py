"""State-space blocks for H4.1 (Section 4.1; register row H4.1), all in plain PyTorch.

Every block has n_s state dimensions per channel, the quantity H4.1 holds equal:

  frac     Fractional block D_t^nu y = A y + b g with its n_s modes tied to (nu, lambda).
           The impulse response G(t) = t^(nu-1) E_{nu,nu}(-lambda t^nu) is completely monotone,
           G(t) = int_0^inf e^(-r t) h(r) dr with h(r) = r^nu sin(nu pi) / (pi (r^(2nu) +
           2 lambda r^nu cos(nu pi) + lambda^2)) (Titchmarsh inversion of 1/(s^nu + lambda)).
           The n_s modes sit at fixed log-spaced rates r_m with quadrature weights
           h(r_m) dr_m; only (nu, lambda), a step size, an output scale and a skip are
           learned per channel. This is our construction for the test, in the spirit of
           FRAC/FRACTAL (prior art, sec. 4.1).
  s4_legs  S4 with HiPPO-LegS initialization (Gu et al. 2020, 2022): A and B learned, shared
           across channels, per-channel step size, bilinear discretization. Implemented with
           a dense A (n_s <= 64), which contains the NPLR parametrization of S4 as a special
           case; the official S4 Cauchy kernel is not used.
  legs     HiPPO-LegS with A and B frozen (optional arm).
  s4d      S4D-Lin (Gu et al. 2022b): n_s/2 complex diagonal modes (n_s real state dims), ZOH.
  mamba    Mamba (Gu and Dao 2023) selective SSM with d_state = n_s, reference scan in PyTorch
           (chunked); the mamba-ssm CUDA kernels are not required.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_conv(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Causal convolution along the last dim: u [B, H, L], K [H, L]."""
    L = u.shape[-1]
    n = 2 * L
    y = torch.fft.irfft(torch.fft.rfft(u.float(), n=n) * torch.fft.rfft(K.float(), n=n), n=n)[..., :L]
    return y.to(u.dtype)


# ----------------------------------------------------------------------------- kernels
class FracKernel(nn.Module):
    def __init__(self, H: int, n_s: int, L_max: int, r_min: float | None = None, r_max: float = 2.0,
                 nu_init: float = 0.5, lam_init: float = 1.0):
        super().__init__()
        r_min = r_min if r_min is not None else 1.0 / (4 * L_max)
        lr = torch.linspace(math.log(r_min), math.log(r_max), n_s)
        self.register_buffer("r", torch.exp(lr))
        dlog = (lr[1] - lr[0]) if n_s > 1 else torch.tensor(math.log(r_max / r_min))
        self.register_buffer("dr", torch.exp(lr) * dlog)  # trapezoid in log r
        self.nu_raw = nn.Parameter(torch.full((H,), math.log(nu_init / (1 - nu_init))))
        self.lam_raw = nn.Parameter(torch.full((H,), math.log(math.expm1(lam_init))))
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001) + math.log(10))
        self.scale = nn.Parameter(torch.randn(H) * 0.5 + 1.0)

    def weights(self):
        nu = torch.sigmoid(self.nu_raw).unsqueeze(-1)            # [H, 1]
        lam = F.softplus(self.lam_raw).unsqueeze(-1)
        r = self.r.unsqueeze(0)                                   # [1, n]
        rn = r ** nu
        h = rn * torch.sin(nu * math.pi) / (math.pi * (rn ** 2 + 2 * lam * rn * torch.cos(nu * math.pi) + lam ** 2))
        return h * self.dr                                        # [H, n]

    def forward(self, L: int):
        w = self.weights()
        dt = torch.exp(self.log_dt).unsqueeze(-1)                 # [H, 1]
        rdt = self.r.unsqueeze(0) * dt                            # [H, n]
        zoh = -torch.expm1(-rdt) / self.r.unsqueeze(0)            # (1 - e^{-r dt}) / r
        k = torch.arange(L, device=w.device, dtype=w.dtype)
        decay = torch.exp(-rdt.unsqueeze(-1) * k)                 # [H, n, L]
        K = torch.einsum("hn,hnl->hl", w * zoh, decay)
        return self.scale.unsqueeze(-1) * K


def legs_matrices(n: int):
    q = torch.arange(n, dtype=torch.float64)
    r = torch.sqrt(2 * q + 1)
    A = -(r[:, None] * r[None, :])
    A = torch.tril(A, -1) - torch.diag(q + 1)
    B = r.clone()
    return A, B


class DenseKernel(nn.Module):
    """C (I - dt/2 A)^-1 ... bilinear discretization of a dense (A, B), kernel by repeated doubling."""

    def __init__(self, H: int, n_s: int, learn_AB: bool = True, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        A, B = legs_matrices(n_s)
        if learn_AB:
            self.A = nn.Parameter(A.float())
            self.B = nn.Parameter(B.float())
        else:
            self.register_buffer("A", A.float())
            self.register_buffer("B", B.float())
        self.C = nn.Parameter(torch.randn(H, n_s) / math.sqrt(n_s))
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))

    def forward(self, L: int):
        n = self.A.shape[0]
        dt = torch.exp(self.log_dt)[:, None, None]                  # [H,1,1]
        I = torch.eye(n, device=self.A.device, dtype=self.A.dtype)
        M = I - dt / 2 * self.A
        Ab = torch.linalg.solve(M, I + dt / 2 * self.A)             # [H,n,n]
        Bb = torch.linalg.solve(M, (dt[..., 0] * self.B).unsqueeze(-1))  # [H,n,1]
        V = Bb
        P = Ab
        while V.shape[-1] < L:
            V = torch.cat([V, P @ V], dim=-1)
            P = P @ P
        V = V[..., :L]                                               # [H,n,L]
        return torch.einsum("hn,hnl->hl", self.C, V)


class S4DKernel(nn.Module):
    def __init__(self, H: int, n_s: int, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        m = max(1, n_s // 2)  # complex modes; n_s real state dimensions
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, m)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(m).float().repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, m, 2) * (0.5 ** 0.5))
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))

    def forward(self, L: int):
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        C = torch.view_as_complex(self.C)
        dA = A * dt
        Cz = C * (torch.exp(dA) - 1.0) / A
        k = torch.arange(L, device=A.device)
        V = torch.exp(dA.unsqueeze(-1) * k)
        return 2 * torch.einsum("hn,hnl->hl", Cz, V).real


class ConvSSM(nn.Module):
    """LTI SSM layer: y = K * u + D u, channel-wise."""

    def __init__(self, kind: str, H: int, n_s: int, L_max: int):
        super().__init__()
        if kind == "frac":
            self.kernel = FracKernel(H, n_s, L_max)
        elif kind == "s4_legs":
            self.kernel = DenseKernel(H, n_s, learn_AB=True)
        elif kind == "legs":
            self.kernel = DenseKernel(H, n_s, learn_AB=False)
        elif kind == "s4d":
            self.kernel = S4DKernel(H, n_s)
        else:
            raise ValueError(kind)
        self.D = nn.Parameter(torch.randn(H))

    def forward(self, u):  # u [B, L, H]
        L = u.shape[1]
        K = self.kernel(L)
        y = fft_conv(u.transpose(1, 2), K).transpose(1, 2)
        return y + u * self.D


# ----------------------------------------------------------------------------- Mamba
def selective_scan(u, delta, A, Bm, Cm, D, chunk: int = 32):
    """Reference selective scan. u, delta [B, L, Dn]; A [Dn, N]; Bm, Cm [B, L, N]; D [Dn].

    h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t,  y_t = C_t . h_t + D u_t.
    Within chunks of length T the recurrence is unrolled with the segment sums
    exp(S_t - S_s) (s <= t), which are bounded by one; chunks are chained sequentially.
    """
    Bsz, L, Dn = u.shape
    N = A.shape[1]
    h = u.new_zeros(Bsz, Dn, N, dtype=torch.float32)
    ys = []
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        T = e - s
        dl = (delta[:, s:e, :, None].float() * A[None, None].float())          # [B,T,Dn,N] log decay
        S = torch.cumsum(dl, dim=1)                                             # [B,T,Dn,N]
        b = delta[:, s:e, :, None].float() * Bm[:, s:e, None, :].float() * u[:, s:e, :, None].float()
        diff = S[:, :, None] - S[:, None, :]                                    # [B,t,s,Dn,N]
        mask = torch.tril(torch.ones(T, T, device=u.device, dtype=torch.bool))
        Lmat = torch.where(mask[None, :, :, None, None], torch.exp(diff.clamp(max=0)), torch.zeros_like(diff))
        hs = torch.einsum("btsdn,bsdn->btdn", Lmat, b) + torch.exp(S) * h[:, None]
        ys.append(torch.einsum("btdn,btn->btd", hs, Cm[:, s:e].float()))
        h = hs[:, -1]
    y = torch.cat(ys, dim=1)
    return (y + u.float() * D[None, None].float()).to(u.dtype)


def selective_scan_naive(u, delta, A, Bm, Cm, D):
    Bsz, L, Dn = u.shape
    h = u.new_zeros(Bsz, Dn, A.shape[1], dtype=torch.float32)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t, :, None] * A[None])
        h = dA * h + delta[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
        ys.append((h * Cm[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + u * D


class MambaMixer(nn.Module):
    def __init__(self, H: int, n_s: int, expand: int = 2, d_conv: int = 4, dt_rank: int | None = None,
                 chunk: int = 32):
        super().__init__()
        Dn = expand * H
        self.Dn, self.N, self.chunk = Dn, n_s, chunk
        self.dt_rank = dt_rank or max(1, math.ceil(H / 16))
        self.in_proj = nn.Linear(H, 2 * Dn, bias=False)
        self.conv = nn.Conv1d(Dn, Dn, d_conv, groups=Dn, padding=d_conv - 1)
        self.x_proj = nn.Linear(Dn, self.dt_rank + 2 * n_s, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, Dn)
        dt = torch.exp(torch.rand(Dn) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_s + 1).float()).repeat(Dn, 1))
        self.D = nn.Parameter(torch.ones(Dn))
        self.out_proj = nn.Linear(Dn, H, bias=False)

    def forward(self, x):  # [B, L, H]
        L = x.shape[1]
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2))
        dbc = self.x_proj(xi)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.N, self.N], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = selective_scan(xi, delta, A, Bm, Cm, self.D, self.chunk)
        return self.out_proj(y * F.silu(z))


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, kind, H, n_s, L_max, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        if kind == "mamba":
            self.mixer = MambaMixer(H, n_s)
            self.out = nn.Identity()
        else:
            self.mixer = ConvSSM(kind, H, n_s, L_max)
            self.out = nn.Sequential(nn.GELU(), nn.Linear(H, H))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.out(self.mixer(self.norm(x))))


class SSMModel(nn.Module):
    def __init__(self, kind: str, d_in: int, d_out: int, H: int, n_s: int, n_layers: int, L_max: int,
                 pool: bool, dropout: float = 0.0):
        super().__init__()
        self.enc = nn.Linear(d_in, H)
        self.blocks = nn.ModuleList([Block(kind, H, n_s, L_max, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(H)
        self.dec = nn.Linear(H, d_out)
        self.pool = pool

    def forward(self, x):  # x [B, L, d_in]
        h = self.enc(x)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        if self.pool:
            h = h.mean(dim=1)
        return self.dec(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def width_for_budget(kind, d_in, d_out, n_s, n_layers, L_max, pool, target: int,
                     H_grid=range(2, 1025, 2)) -> tuple[int, int]:
    """Channel width H whose parameter count is closest to `target` (matched parameters)."""
    best = None
    for H in H_grid:
        m = SSMModel(kind, d_in, d_out, H, n_s, n_layers, L_max, pool)
        p = n_params(m)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (H, p)
        if p > target and best[1] <= p:
            break
    return best
