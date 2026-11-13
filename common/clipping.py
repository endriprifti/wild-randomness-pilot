"""Gradient clipping rules for H6.1 (Section 6.1, eq:s6-clip).

  fixed     norm clipping min{1, g_max/||g||} g with g_max fixed after a calibration window
            as the (1 - p_clip) quantile of unclipped gradient norms; p_clip = 0 never clips,
            p_clip = 1 sets g_max below every calibration norm (clips every step).
  autoclip  g_max = the p-th percentile of all gradient norms seen so far (Seetharaman et al. 2020).
  zclip     z-score rule on an EMA of the gradient norm and its variance (Kumar et al. 2025):
            when z = (||g|| - mu)/sigma exceeds z_thresh, clip to mu + z_thresh * sigma.
            [VERIFY] ZClip's default variant may clip to mu + (z_thresh^2 / z) sigma instead.
  agc       unit-wise adaptive gradient clipping (Brock et al. 2021): each row's gradient is
            capped at lambda * max(||W_row||, eps) ||.

The realized clipped fraction p_clip and the mean update length are recorded by every rule;
`ClipStats` exposes them for the matched-update-length reruns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


def grad_norm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(s)


def scale_grads(params, factor: float):
    for p in params:
        if p.grad is not None:
            p.grad.mul_(factor)


@dataclass
class ClipStats:
    steps: int = 0
    clipped: int = 0
    norms: list = field(default_factory=list)

    @property
    def p_clip(self) -> float:
        return self.clipped / max(self.steps, 1)


class Clipper:
    def __init__(self, rule: str = "none", p_clip: float = 0.0, calib_steps: int = 200,
                 autoclip_pct: float = 10.0, zclip_z: float = 2.5, zclip_beta: float = 0.97,
                 agc_lambda: float = 0.01, agc_eps: float = 1e-3):
        self.rule = rule
        self.p_clip_target = p_clip
        self.calib_steps = calib_steps
        self.g_max = None
        self.autoclip_pct = autoclip_pct
        self.z, self.beta = zclip_z, zclip_beta
        self.mu = self.var = None
        self.agc_lambda, self.agc_eps = agc_lambda, agc_eps
        self.stats = ClipStats()
        self._hist: list[float] = []

    def __call__(self, model: torch.nn.Module) -> float:
        """Clip gradients in place; returns the unclipped global norm."""
        params = [p for p in model.parameters() if p.grad is not None]
        g = grad_norm(params)
        self.stats.steps += 1
        self.stats.norms.append(g)
        clipped = False
        if self.rule == "fixed":
            if self.g_max is None:
                self._hist.append(g)
                if len(self._hist) >= self.calib_steps:
                    if self.p_clip_target >= 1.0:
                        self.g_max = 0.5 * min(self._hist)
                    elif self.p_clip_target <= 0.0:
                        self.g_max = float("inf")
                    else:
                        self.g_max = float(np.quantile(self._hist, 1 - self.p_clip_target))
            elif g > self.g_max:
                scale_grads(params, self.g_max / g)
                clipped = True
        elif self.rule == "autoclip":
            self._hist.append(g)
            thr = float(np.percentile(self._hist, self.autoclip_pct))
            if g > thr:
                scale_grads(params, thr / g)
                clipped = True
        elif self.rule == "zclip":
            if self.mu is None:
                self._hist.append(g)
                if len(self._hist) >= self.calib_steps:
                    self.mu = float(np.mean(self._hist))
                    self.var = float(np.var(self._hist))
            else:
                sd = math.sqrt(max(self.var, 1e-24))
                zval = (g - self.mu) / sd
                g_eff = g
                if zval > self.z:
                    g_eff = self.mu + self.z * sd
                    scale_grads(params, g_eff / g)
                    clipped = True
                self.mu = self.beta * self.mu + (1 - self.beta) * g_eff
                self.var = self.beta * self.var + (1 - self.beta) * (g_eff - self.mu) ** 2
        elif self.rule == "agc":
            for name, p in model.named_parameters():
                if p.grad is None or p.ndim < 2:
                    continue
                w = p.detach().float().reshape(p.shape[0], -1).norm(dim=1).clamp(min=self.agc_eps)
                gr = p.grad.detach().float().reshape(p.shape[0], -1)
                gn = gr.norm(dim=1)
                maxn = self.agc_lambda * w
                fac = torch.where(gn > maxn, maxn / gn.clamp(min=1e-12), torch.ones_like(gn))
                if bool((fac < 1).any()):
                    clipped = True
                p.grad.mul_(fac.reshape(-1, *([1] * (p.ndim - 1))).to(p.grad.dtype))
        if clipped:
            self.stats.clipped += 1
        return g
