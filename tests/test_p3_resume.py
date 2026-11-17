"""P3 saves every run as it finishes and goes on from the saved runs; the LRA designs; runs recovered from a process
of the code before 2026-11-17 (p3_rescue.py) are adopted only with the same settings."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_adopt_takes_runs_only_with_the_same_settings(tmp_path):
    src = tmp_path / "src"
    assert _p3(src).returncode == 0
    rows = pd.read_csv(src / "runs.csv")
    old_cols = ["task", "arch", "n_s", "H", "lr", "seed", "phase", "target_params", "data_source", "acc", "params",
                "steps", "wall_s", "peak_mem_mb", "diverged", "tokens_seen", "flops_proxy"]   # the old code's row
    cfg = config.get("p3_fractional_memory", smoke=True)
    seed = json.loads((src / "runs" / "powerlaw" / "settings.json").read_text())["seed"]
    rescued = {"rows": rows[old_cols].to_dict("records"), "cfg": cfg, "seed": seed, "smoke": True, "pid": 1,
               "utc": "2026-11-17T00:00:00Z"}
    rescued["rows"].append({**rescued["rows"][0], "steps": 999})        # not in this plan
    (tmp_path / "rescued.json").write_text(json.dumps(rescued))
    r = _p3(tmp_path / "new", "--adopt", str(tmp_path / "rescued.json"), "--plan")
    assert r.returncode == 0 and "adopted 6, already saved 0, not in this plan 1" in r.stdout, r.stdout + r.stderr
    assert "left: 0 runs" in r.stdout
    new = pd.read_csv(tmp_path / "new" / "runs.csv")
    assert new.origin.str.startswith("rescued").all() and np.allclose(new.acc, rows.acc)
    bad = {**rescued, "cfg": {**cfg, "batch": 64}}                        # the process trained with another batch
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    r = _p3(tmp_path / "new2", "--adopt", str(tmp_path / "bad.json"), "--plan")
    assert r.returncode == 0 and "other settings 7" in r.stdout


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


OLD_P3 = '''
import time
def main():
    cfg = {"steps": 3000, "tasks": {"powerlaw": {"L": 8}}}
    class R: seed = 5; smoke = False; out_dir = "x"
    run = R()
    rows = []
    for i in range(10_000):
        rows.append({"task": "powerlaw", "arch": "frac", "n_s": 2, "H": 4, "lr": 0.001, "seed": i, "phase": "main",
                     "acc": float("nan") if i == 1 else i / 10, "steps": 3000})
        if i == 2:
            print("ready", flush=True)
        if i >= 2:
            time.sleep(0.05)
main()
'''


@pytest.mark.skipif(shutil.which("gdb") is None or not sys.platform.startswith("linux"), reason="needs gdb on Linux")
def test_rescue_reads_the_runs_of_a_running_process(tmp_path):
    import p3_rescue
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "p3_fractional_memory.py").write_text(OLD_P3)
    proc = subprocess.Popen([PY, str(tmp_path / "old" / "p3_fractional_memory.py")], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        try:
            data = p3_rescue.rescue(proc.pid, tmp_path / "out" / "rescued.json", timeout=120)
        except SystemExit as e:
            if "ptrace" in str(e) or "Operation not permitted" in str(e):
                pytest.skip("ptrace not permitted here")
            raise
        assert proc.poll() is None                                     # still running
        n = len(data["rows"])
        assert n >= 3 and [r["seed"] for r in data["rows"]] == list(range(n)) and np.isnan(data["rows"][1]["acc"])
        assert data["seed"] == 5 and data["cfg"]["steps"] == 3000 and data["how"] == "main() frame"
        assert "recovered" in p3_rescue.summary(data)
        time.sleep(0.3)
        assert proc.poll() is None
    finally:
        proc.kill()
