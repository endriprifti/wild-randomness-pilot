"""Statistics used by the Emmy check (they run without Emmy)."""
import numpy as np
import torch

import emmy_check
from common import spectra


def test_streaming_kurtosis_matches_direct():
    torch.manual_seed(0)
    H = torch.randn(4000, 32) * torch.linspace(0.1, 8, 32)
    H[:, 3] += 30 * (torch.rand(4000) < 0.01)
    x = H.double().reshape(-1)
    K = emmy_check.kurtosis_from_sums(x.numel(), *(float((x ** p).sum()) for p in range(1, 5)))
    assert abs(K - spectra.activation_stats(H)["K"]) < 1e-3 * max(1, abs(K))


def test_layer_accumulator_and_logit_metrics():
    torch.manual_seed(0)
    hs = [torch.randn(1, 16, 8) * 4 for _ in range(3)]
    acc = emmy_check.LayerAcc(3, 8)
    acc.add(hs, [h.clone() for h in hs])
    rows = acc.rows(6.0)
    assert all(r["max_abs_diff"] == 0 and r["outlier_jaccard"] == 1.0 for r in rows)
    le = torch.randn(16, 50)
    ids = torch.randint(0, 50, (16,))
    same = emmy_check.logit_metrics(le, le.clone(), ids)
    assert same["kl_sum"] == 0 and same["top1"] == 16 and same["nll_e"] == same["nll_c"]
    other = emmy_check.logit_metrics(le, le + 0.5 * torch.randn(16, 50), ids)
    assert other["kl_sum"] > 0 and other["max_abs"] > 0


def test_fast_math_lanes_environment():
    base = {"PATH": "/bin", "EMMY_FAST_MATH": "1", "EMMY_FAST_EXP": "1", "EMMY_OTHER": "x"}
    on = emmy_check.fast_math_env(base, 1)
    off = emmy_check.fast_math_env(base, 0)
    assert "EMMY_FAST_MATH" not in on and "EMMY_FAST_EXP" not in on          # default lane: nothing pinned
    assert off["EMMY_FAST_MATH"] == "0" and "EMMY_FAST_EXP" not in off
    assert on["EMMY_OTHER"] == off["EMMY_OTHER"] == "x" and on["PATH"] == "/bin"
