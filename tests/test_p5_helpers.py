"""P5 helpers: check schedule, cell reruns that repeat the full grid's trajectories, refits, H3.1 flags."""
import json
import math

import numpy as np
import pandas as pd
import torch

import p5_escape as p5


def test_check_schedule_adds_fine_checks_only_up_to_the_limit():
    every = p5.check_schedule({"check_every": 250})
    fine = p5.check_schedule({"check_every": 250, "fine_check": [10, 250]})
    assert [k for k in range(1, 801) if every(k)] == [250, 500, 750]
    ks = [k for k in range(1, 801) if fine(k)]
    assert ks[:3] == [10, 20, 30] and ks[24:] == [250, 500, 750] and len(ks) == 27


def _cfg(**kw):
    c = {"restarts": 6, "k_cens": 120, "check_every": 40, "descent_steps": 5, "descent_lr": 0.1,
         "segment_points": 5, "barrier_tol": 0.002, "second_reading": {"dist_tol": 0.05, "loss_tol": 0.05}}
    c.update(kw)
    return c


def test_a_finer_schedule_only_refines_exits_inside_the_first_interval():
    torch.manual_seed(0)
    X, Y = torch.randn(64, 12), torch.randint(0, 10, (64,))
    P_S = p5.full_descent(p5.init_params(1, 12, 8, 10, 0, "cpu"), X, Y, "tanh", 300, 0.5)
    P_S = [p.repeat(6, *([1] * (p.ndim - 1))) for p in P_S]
    args = ("injected", P_S, X, Y)
    kw = dict(B=8, eta=0.2, g_max=None, alpha_inj=1.5, gamma1=0.3, seed=11, device="cpu", act="tanh")
    a = p5.run_point(*args, _cfg(), **kw)
    b = p5.run_point(*args, _cfg(fine_check=[5, 40]), **kw)
    # the same trajectories, checked at a superset of the steps: an exit is never seen later, and a restart
    # with no exit inside the fine window exits exactly where the coarse schedule saw it
    assert np.all(b["k"] <= a["k"]) and np.all(b["exited"] >= a["exited"])
    after = b["k"] > 40
    assert np.array_equal(a["k"][after], b["k"][after])
    early = a["exited"] & (a["k"] <= 40)
    assert np.all(b["k"][early] <= 40) and np.all(b["k_lo"][early] == b["k"][early] - 5)
    assert np.all(a["k_lo"][a["exited"]] == a["k"][a["exited"]] - 40)
    assert np.all(a["k_lo"][~a["exited"]] == 120)
    assert a["exited"].any() and np.isfinite(a["loss_final"]).all() and (a["dist_final"] >= 0).all()
    c = p5.run_point(*args, _cfg(second_reading={"dist_tol": 0.05, "loss_tol": 0.05, "continue": True}), **kw)
    assert np.array_equal(c["k"], a["k"]) and np.all(c["k_stop"] >= a["k_stop"])   # barrier reading unchanged


def test_saved_runs_get_k_lo_and_quantiles_and_floor_cells(tmp_path):
    rows = []
    for r in range(4):
        rows.append({"mode": "sgd", "B": 8, "eta": 0.1, "g_max": np.nan, "alpha_injected": np.nan, "restart": r,
                     "k": 250.0, "exited": True})                           # a floor cell
        rows.append({"mode": "clipped", "B": 8, "eta": 0.1, "g_max": 0.7 if r % 2 else 0.3, "alpha_injected": np.nan,
                     "restart": r, "k": 750.0 if r else 50_000.0, "exited": bool(r)})
    pd.DataFrame(rows).to_csv(tmp_path / "restarts.csv", index=False)
    (tmp_path / "run.json").write_text(json.dumps({"config": {"check_every": 250, "g_max_quantiles": [0.5, 0.9]}}))
    df, _ = p5.load_restarts(tmp_path)
    assert set(df[df["mode"] == "clipped"].g_max_q) == {0.5, 0.9}
    assert (df[df.exited].k_lo == df[df.exited].k - 250).all() and (df[~df.exited].k_lo == df[~df.exited].k).all()
    cells = p5.floor_cells(tmp_path)
    assert len(cells) == 1 and cells[0][0] == "sgd"
    g = p5.grid_summary(df)
    assert bool(g[g["mode"] == "sgd"].floor.iloc[0]) and g[g["mode"] == "sgd"].E_k.iloc[0] == 250


def test_h31_flags_from_a_first_run_csv():
    df = pd.DataFrame({"alpha_injected": [1.2, 1.2, 1.5, 2.0, 2.0], "seed": range(5),
                       "train_loss": [2930.0, np.nan, 4.0, 0.05, 0.04], "gap": [0.1] * 5,
                       "lambda_max": [3.0, np.nan, 2.0, -0.5, 8.0], "hessian_trace": [1.0] * 5,
                       "adaptive_sharpness": [0.1] * 5})
    out = p5.h31_flags(df, {"diverge_loss": math.log(10)})
    assert out.diverged.tolist() == [True, True, True, False, False]
    assert out.minimum.tolist() == [True, False, True, False, True] and out.minimum_check.iloc[0] == "lambda_max > 0"
    assert out.converged.tolist() == [False, False, False, False, True]
    assert out.lambda_max.isna().tolist() == [True, True, True, True, False] and out.lambda_max_raw.iloc[3] == -0.5
    s = pd.DataFrame(p5.h31_summary(out)).set_index("alpha_injected")
    assert s.loc[2.0, "non_minimum"] == 1 and s.loc[2.0, "converged"] == 1 and s.loc[1.2, "diverged"] == 2
    with_min = p5.h31_flags(df.assign(lambda_min=[0.0, np.nan, 0.0, -1.0, -0.5]), {"min_tol": 0.01})
    assert not with_min.minimum.iloc[4]                                     # lambda_min = -0.5 < -0.08
