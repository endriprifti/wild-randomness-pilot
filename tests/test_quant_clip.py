"""Simulated quantization primitives and the clipping rules."""
import math

import numpy as np
import pytest
import torch
from scipy import stats

from common import clipping, quant


def test_nf4_code_book_is_the_qlora_normal_map():
    offset = 0.9677083
    v1 = stats.norm.ppf(np.linspace(offset, 0.5, 9)[:-1]).tolist()
    v3 = (-stats.norm.ppf(np.linspace(offset, 0.5, 8)[:-1])).tolist()
    v = np.sort(np.array(v1 + [0.0] + v3))
    v = v / np.abs(v).max()
    assert np.allclose(quant.NF4_CODE.numpy(), v, atol=1e-6)


def test_fwht_is_orthonormal():
    from scipy.linalg import hadamard
    x = torch.randn(3, 64, dtype=torch.float64)
    assert torch.allclose(quant.fwht(x), x @ torch.tensor(hadamard(64), dtype=torch.float64) / 8)
    assert torch.allclose(quant.fwht(quant.fwht(x)), x)


@pytest.mark.parametrize("d", [64, 96])
def test_rotation_leaves_linear_layer_unchanged(d):
    rot = quant.Rotation(d, seed=3, device="cpu", dtype=torch.float64)
    x, W = torch.randn(5, d, dtype=torch.float64), torch.randn(7, d, dtype=torch.float64)
    assert torch.allclose(rot.apply(x) @ rot.apply(W).T, x @ W.T)


def test_quant_sym_error_bound():
    x = torch.randn(10, 128)
    q = quant.quant_sym(x, 4)
    scale = x.abs().amax(-1, keepdim=True) / 7
    assert torch.all((q - x).abs() <= scale / 2 + 1e-6)
    assert torch.unique(torch.round(q / scale)).numel() <= 16


def test_bits_are_matched_across_schemes():
    s = [quant.Scheme("uniform", group=128), quant.Scheme("rotation", group=128),
         quant.Scheme("outlier", group=256, outlier_frac=0.005), quant.Scheme("nf4", group=128)]
    assert quant.Scheme("uniform", group=128).bits_per_weight(4096) == 4.125
    assert quant.check_bits_matched(s, 4096)["matched"]


def test_fixed_clipper_hits_target_rate():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    clip = clipping.Clipper("fixed", p_clip=0.1, calib_steps=1000)
    rng = np.random.default_rng(0)
    after = []
    for _ in range(4000):
        for p in model.parameters():
            p.grad = torch.randn_like(p) * float(rng.lognormal(0, 0.5))
        clip(model)
        after.append(clipping.grad_norm(list(model.parameters())))
    clipped_rate = clip.stats.clipped / (clip.stats.steps - 1000)
    assert abs(clipped_rate - 0.1) < 0.03          # quantile from 1,000 calibration steps: SE about 0.01
    assert max(after[1000:]) <= clip.g_max * (1 + 1e-6)
