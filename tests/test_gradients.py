"""Per-example JVP projections against autograd; Hessian-block eigenvalues and trace on a quadratic."""
import numpy as np
import torch

from common import gradients


def _pel(run, batch):
    X, y = batch
    return 0.5 * (run(X).squeeze(-1) - y) ** 2


def test_jvp_projections_match_autograd():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    X, y = torch.randn(6, 5), torch.randn(6)
    params = gradients.select_params(model, None)
    dirs = gradients.random_directions(params, 3, torch.Generator().manual_seed(1))
    P = gradients.per_example_projections(model, _pel, (X, y), params, dirs)
    ref = torch.zeros(6, 3)
    for i in range(6):
        model.zero_grad()
        (0.5 * (model(X[i:i + 1]).squeeze() - y[i]) ** 2).backward()
        for j, u in enumerate(dirs):
            ref[i, j] = sum((p.grad * u[n]).sum() for n, p in model.named_parameters())
    assert torch.allclose(P, ref, atol=1e-5)


def test_hessian_eigs_and_trace_on_linear_regression():
    torch.manual_seed(0)
    model = torch.nn.Linear(6, 1, bias=False)
    X = torch.randn(200, 6) * torch.tensor([3.0, 2.0, 1.0, 1.0, 0.5, 0.2])
    y = torch.randn(200)
    params = gradients.select_params(model, None)
    hvp, p0 = gradients.hvp_fn(model, _pel, (X, y), params)
    Hm = (X.T @ X / 200).double().numpy()                     # exact Hessian of the mean loss
    ev = np.sort(np.linalg.eigvalsh(Hm))[::-1]
    vals, _ = gradients.top_eigvecs(hvp, p0, k=2, iters=300, tol=1e-9)
    assert abs(vals[0] / ev[0] - 1) < 1e-3 and abs(vals[1] / ev[1] - 1) < 1e-2
    tr = gradients.hutchinson_trace(hvp, p0, n_samples=400)
    assert abs(tr / np.trace(Hm) - 1) < 0.1


# ----------------------------------------------------------------------------- micro-batches
def _tiny_lm():
    # float64 weights; lm.per_example_lm_loss casts the logits to float32, so chunked and full results
    # agree to ~1e-7 relative (a wrong chunk weight is off by 10% or more on these uneven chunks)
    from common import hf
    torch.manual_seed(0)
    cfg = dict(hf.TINY_NEOX, vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
               intermediate_size=64, max_position_embeddings=64)
    return hf.neox_from_config(cfg, seed=0).double().eval()


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def test_micro_batches_weights_cover_the_batch():
    X, y = torch.randn(10, 3), torch.randn(10)
    parts = gradients.micro_batches((X, y), 4)
    assert [len(b[0]) for b, _ in parts] == [4, 4, 2]
    assert sum(w for _, w in parts) == 1.0 and parts[-1][1] == 0.2
    assert torch.equal(torch.cat([b[0] for b, _ in parts]), X)
    assert gradients.micro_batches((X, y), None) == [((X, y), 1.0)]
    assert len(gradients.micro_batches(X, 10)) == 1


def test_chunked_hvp_matches_full_batch_on_a_small_transformer():
    from common import lm
    model = _tiny_lm()
    ids = torch.randint(0, 64, (10, 12), generator=torch.Generator().manual_seed(1))
    params = gradients.select_params(model, None)
    v = {n: torch.randn(p.shape, generator=torch.Generator().manual_seed(2), dtype=p.dtype) for n, p in params.items()}
    full, _ = gradients.hvp_fn(model, lm.per_example_lm_loss, ids, params)
    Hv = full(v)
    for chunk in (1, 3, 4, 10):                       # uneven last micro-batch included (10 = 3 + 3 + 3 + 1)
        hvp, _ = gradients.hvp_fn(model, lm.per_example_lm_loss, ids, params, chunk=chunk)
        Hc = hvp(v)
        for n in Hv:
            assert _rel(Hc[n], Hv[n]) < 1e-6, (chunk, n)


def test_chunked_trace_eigs_sharpness_and_projections_match():
    from common import lm
    model = _tiny_lm()
    ids = torch.randint(0, 64, (8, 10), generator=torch.Generator().manual_seed(3))
    params = gradients.select_params(model, ["layers.1"])
    full, p0 = gradients.hvp_fn(model, lm.per_example_lm_loss, ids, params)
    chunked, _ = gradients.hvp_fn(model, lm.per_example_lm_loss, ids, params, chunk=3)
    t_full = gradients.hutchinson_trace(full, p0, 5, torch.Generator().manual_seed(4))
    t_chunk = gradients.hutchinson_trace(chunked, p0, 5, torch.Generator().manual_seed(4))
    assert abs(t_chunk - t_full) <= 1e-6 * abs(t_full)
    l_full = gradients.top_eigvecs(full, p0, 1, 10, torch.Generator().manual_seed(5))[0][0]
    l_chunk = gradients.top_eigvecs(chunked, p0, 1, 10, torch.Generator().manual_seed(5))[0][0]
    assert abs(l_chunk - l_full) <= 1e-6 * abs(l_full)
    s_full = gradients.adaptive_sharpness(model, lm.per_example_lm_loss, ids, 0.05, 3)
    s_chunk = gradients.adaptive_sharpness(model, lm.per_example_lm_loss, ids, 0.05, 3, chunk=3)
    assert abs(s_chunk - s_full) <= 1e-6 and s_full > 0           # a difference of two float32 losses near 4
    dirs = [{n: torch.randn(p.shape, generator=torch.Generator().manual_seed(6), dtype=p.dtype) for n, p in params.items()}]
    P = gradients.per_example_projections(model, lm.per_example_lm_loss, ids, params, dirs)
    Pc = gradients.per_example_projections(model, lm.per_example_lm_loss, ids, params, dirs, chunk=3)
    assert Pc.shape == P.shape and _rel(Pc, P) < 1e-6


def test_chunked_hvp_with_tuple_batches():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()
    X, y = torch.randn(7, 5, dtype=torch.float64), torch.randn(7, dtype=torch.float64)
    params = gradients.select_params(model, None)
    v = {n: torch.randn_like(p) for n, p in params.items()}
    Hv = gradients.hvp_fn(model, _pel, (X, y), params)[0](v)
    Hc = gradients.hvp_fn(model, _pel, (X, y), params, chunk=2)[0](v)
    assert all(torch.allclose(Hc[n], Hv[n], rtol=1e-10, atol=1e-12) for n in Hv)
