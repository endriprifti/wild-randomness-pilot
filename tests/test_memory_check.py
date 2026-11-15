"""memory_check's tracker: live storage bytes on the meta device, freed tensors leave the count."""
import torch

import memory_check as mc


def test_peak_counts_live_storages_and_frees():
    with torch.device("meta"):
        a = torch.empty(1024, 256)                       # 1 MiB of float32, alive throughout

        def step():
            b = a @ torch.empty(256, 512)                 # 0.5 MiB operand + 2 MiB result
            del b
            torch.empty(1024, 1024)                       # 4 MiB, freed at once
        peak = mc.peak_gib(step, [a]) * 2 ** 30
    assert peak == 4 * 2 ** 20 + 2 ** 20                  # the 4 MiB tensor plus a; the matmul peak (3.5 MiB) is lower


def test_transformer_hvp_grows_with_the_micro_batch():
    with torch.device("meta"):
        from common import hf, lm, gradients
        model = hf.neox_from_config(dict(hf.TINY_NEOX)).eval()
        params = gradients.select_params(model, None)
        peaks = []
        for n in (2, 8):
            f, extra = mc.hvp_step(model, lm.per_example_lm_loss, torch.randint(0, 512, (n, 64)), params)
            peaks.append(mc.peak_gib(f, mc.state(model, *extra)))
    assert 0 < peaks[0] < peaks[1]
