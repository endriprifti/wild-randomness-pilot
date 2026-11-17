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


def test_nf4_in_chunks_equals_one_pass():
    W = torch.randn(300, 257) * 3                       # padding in the last block, many chunks
    flat = torch.nn.functional.pad(W.reshape(-1), (0, (-W.numel()) % 64)).reshape(-1, 64)
    am = flat.abs().amax(1, keepdim=True).clamp(min=1e-8)
    one = (quant.NF4_CODE[((flat / am).unsqueeze(-1) - quant.NF4_CODE).abs().argmin(-1)] * am).reshape(-1)
    one = one[:W.numel()].reshape(W.shape)
    assert torch.equal(quant.quant_nf4(W, 64), one) and torch.equal(quant.quant_nf4(W, 64, chunk_blocks=7), one)


def test_apply_scheme_frees_each_layer_as_it_goes():
    import gc
    import weakref
    m = torch.nn.Sequential(*[torch.nn.Linear(16, 16) for _ in range(3)])
    refs = [weakref.ref(l) for l in m]
    seen = []
    orig = quant.QuantLinear.__init__

    def spy(self, lin, *a, **k):
        gc.collect()
        seen.append(sum(r() is not None for r in refs))
        orig(self, lin, *a, **k)
    quant.QuantLinear.__init__ = spy
    try:
        quant.apply_scheme(m, quant.Scheme("uniform", a_bits=None, group=8))
    finally:
        quant.QuantLinear.__init__ = orig
    assert seen == [3, 2, 1]                            # the layers already replaced are gone


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


# ----------------------------------------------------------------------------- outlier channels
def _outlier_layer(W, keep_ch, frac):
    lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False).double()
    lin.weight.data = W.clone()
    absmax = torch.ones(W.shape[1], dtype=torch.float64)
    absmax[keep_ch] = 100.0                                   # calibration marks the outlier channel
    return quant.QuantLinear(lin, quant.Scheme("outlier", a_bits=4, group=64, outlier_frac=frac), absmax, None)


def test_kept_channel_1000x_larger_leaves_the_other_channels_error_unchanged():
    g = torch.Generator().manual_seed(0)
    d, j = 128, 37
    x = torch.randn(6, d, generator=g, dtype=torch.float64)
    big = x.clone()
    big[:, j] *= 1000.0
    keep = torch.tensor([j])
    rest = torch.arange(d) != j
    # activations: the kept channel is exact and the others' error does not depend on its size
    e_small, e_big = quant.quant_act(x, 4, keep) - x, quant.quant_act(big, 4, keep) - big
    assert torch.equal(e_small[:, rest], e_big[:, rest]) and torch.all(e_big[:, j] == 0)
    # the bug fixed on 2026-11-15: a scale over all channels lets the kept channel crush the others
    e_old = quant.quant_sym(big, 4) - big                            # most other entries round to zero
    assert e_old[:, rest].abs().mean() > 5 * e_big[:, rest].abs().mean()
    # weights: a 1000x column in a group does not change the other columns' quantized values
    W = torch.randn(8, d, generator=g, dtype=torch.float64)
    Wb = W.clone()
    Wb[:, j] *= 1000.0
    q, qb = quant.quant_weight_groups(W, 4, 64, keep), quant.quant_weight_groups(Wb, 4, 64, keep)
    assert torch.equal(q[:, rest], qb[:, rest]) and torch.equal(qb[:, j], Wb[:, j])
    # the whole layer: output error with a 1000x input channel equals the error without it
    layer = _outlier_layer(W, j, 1 / d)
    assert torch.equal(layer.keep, keep)
    err_small = layer(x) - x @ W.T
    err_big = layer(big) - big @ W.T
    assert torch.allclose(err_small, err_big, rtol=0, atol=1e-9)


def test_outlier_scheme_never_does_worse_than_uniform_when_one_channel_dominates():
    g = torch.Generator().manual_seed(1)
    d, j = 256, 5
    x = torch.randn(32, d, generator=g, dtype=torch.float64)
    x[:, j] *= 300.0
    W = torch.randn(16, d, generator=g, dtype=torch.float64) / d ** 0.5
    lin = torch.nn.Linear(d, 16, bias=False).double()
    lin.weight.data = W.clone()
    absmax = x.abs().amax(0)
    uni = quant.QuantLinear(lin, quant.Scheme("uniform", a_bits=4, group=128), None, None)
    out = quant.QuantLinear(lin, quant.Scheme("outlier", a_bits=4, group=256, outlier_frac=1 / d), absmax, None)
    ref = x @ W.T
    assert (out(x) - ref).norm() < 0.2 * (uni(x) - ref).norm()


def test_rotation_arm_in_bfloat16_adds_no_rounding_beyond_bfloat16_storage():
    g = torch.Generator().manual_seed(2)
    d = 1024
    x = torch.randn(8, d, generator=g)
    W = torch.randn(32, d, generator=g) / d ** 0.5
    lin = torch.nn.Linear(d, 32, bias=False).to(torch.bfloat16)
    lin.weight.data = W.to(torch.bfloat16)
    rot = quant.Rotation(d, 0, "cpu", torch.float32)
    q = quant.QuantLinear(lin, quant.Scheme("rotation", w_bits=16, a_bits=16, group=128), None, rot)
    ref = x.to(torch.bfloat16).float() @ W.to(torch.bfloat16).float().T
    out = q(x.to(torch.bfloat16)).float()
    assert q.weight.dtype == torch.bfloat16 and out.dtype == torch.float32
    assert (out - ref).norm() / ref.norm() < 1e-2          # bfloat16 storage of the rotated weights and output only
