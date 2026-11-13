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
