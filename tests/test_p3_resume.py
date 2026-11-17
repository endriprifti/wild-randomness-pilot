"""P3 saves every run as it finishes and goes on from the saved runs; the LRA designs; P3's GPU lock."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import p3_fractional_memory as p3

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_lra_designs_and_run_counts():
    cfg = config.get("p3_fractional_memory")
    full, red = p3.task_settings(cfg, "lra_image", "full"), p3.task_settings(cfg, "lra_image", "reduced")
    syn = p3.task_settings(cfg, "powerlaw", "reduced")
    assert full["steps"] == 20 * (50_000 // 32) == 31_240 and syn["steps"] == 3_000     # the first plan's count
    assert red["steps"] == 3 * 1_562 and red["n_states"] == [2, 8, 32] and red["seeds"] == [1, 2, 3]
    archs = cfg["archs"]
    counts = {k: tuple(map(len, p3.plan_task("lra_image", s, archs))) for k, s in (("full", full), ("reduced", red))}
    assert counts == {"full": (12, 120), "reduced": (12, 36)} and p3.plan_task("powerlaw", syn, archs)[1][0]["seed"] == 0
    # the reduced design tunes on a seed it does not measure, so no main run repeats a tuning run
    tune, main = p3.plan_task("lra_image", red, archs)
    assert not {p3.run_key({**m, "lr": t["lr"], "phase": "tune"}) for m in main for t in tune} & {p3.run_key(t) for t in tune}
    # main runs go seed by seed
    assert [m["seed"] for m in main] == sorted(m["seed"] for m in main)


def test_synthetic_data_do_not_depend_on_the_tasks_run():
    cfg = config.get("p3_fractional_memory", smoke=True)
    both = p3.synthetic_data(cfg, 7, ["powerlaw", "exponential"])
    alone = p3.synthetic_data(cfg, 7, ["exponential"])
    assert set(alone) == {"exponential"} and all(torch.equal(a, b) for a, b in zip(both["exponential"], alone["exponential"]))
    rng = np.random.default_rng(7)                                    # the old code's draws, in config order
    for task in ("powerlaw", "exponential"):
        ct = cfg["tasks"][task]
        old = p3.memory_task(task, ct["L"], ct["n_train"], ct, rng) + p3.memory_task(task, ct["L"], ct["n_test"], ct, rng)
        assert all(torch.equal(a, b) for a, b in zip(old, both[task]))


def _p3(out, *extra, env=None):
    r = subprocess.run([PY, str(ROOT / "p3_fractional_memory.py"), "--smoke", "--tasks", "powerlaw", "--archs",
                        "frac,s4d", "--device", "cpu", "--out", str(out), *extra], capture_output=True, text=True,
                       timeout=600, env=env, cwd=ROOT)
    return r


def test_saved_runs_are_reused_and_lost_ones_run_again(tmp_path):
    out = tmp_path / "p3"
    r = _p3(out)
    assert r.returncode == 0, r.stderr[-2000:]
    first = pd.read_csv(out / "runs.csv")
    files = sorted((out / "runs" / "powerlaw").glob("*_steps*.json"))
    assert len(first) == len(files) == 2 + 2 * 2                      # 2 arms x 1 LR tuning, 2 arms x 2 n_s x 1 seed
    assert (out / "curves.csv").exists() and json.loads((out / "runs" / "powerlaw" / "settings.json").read_text())["seed"]
    for f in files[:3]:
        f.unlink()                                                     # as if P3 had stopped before these
    r = _p3(out)
    assert r.returncode == 0 and r.stdout.count("acc ") == 3, r.stdout[-2000:]
    again = pd.read_csv(out / "runs.csv")
    cols = ["task", "phase", "arch", "n_s", "seed", "lr", "H", "acc", "params"]
    pd.testing.assert_frame_equal(first[cols], again[cols])
    r = _p3(out, "--plan")
    assert r.returncode == 0 and "left: 0 runs" in r.stdout
    # other settings in the same folder: refused
    s = out / "runs" / "powerlaw" / "settings.json"
    d = json.loads(s.read_text())
    d["weight_decay"] = 0.5
    s.write_text(json.dumps(d))
    r = _p3(out)
    assert r.returncode != 0 and "weight_decay" in r.stderr


def test_p3_gpu_lock(tmp_path, monkeypatch):
    import argparse
    from common import gpu_locks, runtime
    monkeypatch.setattr(runtime, "PILOT_ROOT", tmp_path)
    a = argparse.Namespace(smoke=False, plan=False)
    f, waited = p3.gpu_lock(a, "cuda")
    assert f is not None and gpu_locks.is_held(tmp_path / "logs" / ".p3_gpu.lock")
    probe = subprocess.run([PY, "-c", "import fcntl, sys; f = open(sys.argv[1], 'a'); "
                            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)", str(tmp_path / "logs" / ".p3_gpu.lock")],
                           capture_output=True, text=True)
    assert probe.returncode != 0 and "BlockingIOError" in probe.stderr   # another process cannot take it
    f.close()
    assert p3.gpu_lock(argparse.Namespace(smoke=True, plan=False), "cuda") == (None, 0.0)
    assert p3.gpu_lock(a, "cpu") == (None, 0.0)
