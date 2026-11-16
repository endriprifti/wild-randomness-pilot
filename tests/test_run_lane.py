"""run_lane.sh: the same jobs, commands and bookkeeping as run_all.sh, and its safety checks, with a stand-in
Python that only writes a results folder."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("flock") is None or shutil.which("bash") is None,
                                reason="needs bash and flock (util-linux)")


def _jobs(text):
    return re.search(r'^(?:ALL_)?JOBS="([^"]+)"', text, re.M).group(1).split()


def test_lane_uses_run_all_order_and_commands():
    run_all, lane = (ROOT / "run_all.sh").read_text(), (ROOT / "run_lane.sh").read_text()
    assert _jobs(lane) == _jobs(run_all)
    gpu = dict(re.findall(r"^gpu_job (p1_\w+)\s+\$PY (.+)$", run_all, re.M))
    mine = dict(re.findall(r"^\s+(p1_\w+)\)\s+echo \"(.+)\" ;;$", lane, re.M))
    assert set(gpu) == set(mine) == {"p1_pythia", "p1_polypythia", "p1_olmo", "p1_olmo7b"}
    assert all(gpu[j].strip() == mine[j] for j in gpu)


@pytest.fixture
def pilot(tmp_path):
    d = tmp_path / "pilot"
    d.mkdir()
    shutil.copy(ROOT / "run_lane.sh", d)
    (d / "pins.json").write_text("{}\n")
    (d / "logs").mkdir()
    fake = tmp_path / "fakepy"
    fake.write_text("#!/usr/bin/env bash\n"            # stand-in for python: writes --out, fails for --suite olmo
                    'out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out="$2"; shift; done\n'
                    'mkdir -p "$out"; echo "{}" > "$out/run.json"; echo "fake job $out"\n'
                    '[[ "$out" == *p1_olmo ]] && exit 1; exit 0\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), PATH="/usr/bin:/bin")   # no nvidia-smi: memory check skipped
    return d, env


def _run(d, env, *jobs):
    return subprocess.run(["bash", "run_lane.sh", *jobs], cwd=d, env=env, capture_output=True, text=True, timeout=60)


def test_lane_bookkeeping_next_to_a_running_run_all(pilot):
    d, env = pilot
    (d / "logs" / "progress.txt").write_text(
        "2026-11-15T00:00:00Z  END    p1_pythia  exit 1  elapsed 0.33 h  total 50.00 h\n"
        "2026-11-16T00:56:00Z  START  p3_fractional  total 51.00 h\n")
    holder = subprocess.Popen(["flock", str(d / "logs" / ".run_all.lock"), "sleep", "30"])   # run_all.sh is running
    try:
        import time
        time.sleep(0.3)
        r = _run(d, env, "p1_olmo7b")                       # run_all.sh (at p3_fractional) will still run it
        assert r.returncode == 1 and "will still run it" in r.stderr
        r = _run(d, env, "p4_transformer")
        assert r.returncode == 1 and "not a P1 job" in r.stderr
        r = _run(d, env, "p1_pythia", "p1_polypythia", "p1_olmo")
    finally:
        holder.kill()
    assert r.returncode == 0, r.stderr
    prog = (d / "logs" / "progress.txt").read_text().splitlines()
    lane = [x for x in prog if x.endswith("(lane)")]
    assert [x.split()[1:3] for x in lane] == [["START", "p1_pythia"], ["END", "p1_pythia"],
                                              ["START", "p1_polypythia"], ["END", "p1_polypythia"],
                                              ["START", "p1_olmo"], ["END", "p1_olmo"]]
    ends = {x.split()[2]: x.split()[4] for x in lane if x.split()[1] == "END"}
    assert ends == {"p1_pythia": "0", "p1_polypythia": "0", "p1_olmo": "1"}
    assert (d / "logs" / "p1_pythia.done").exists() and not (d / "logs" / "p1_olmo.done").exists()
    for j in ("p1_pythia", "p1_polypythia", "p1_olmo"):
        assert (d / "results_archive" / f"{j}.tar.gz").exists()
        assert "--out results/" + j in (d / "logs" / f"{j}.log").read_text()
    assert (d / "results_archive" / "pins.json").exists() and not list((d / "results_archive").glob("*.tmp"))
    r = _run(d, env, "p1_pythia")                           # run_all.sh not running now; .done: skipped
    assert r.returncode == 0 and "SKIP   p1_pythia" in r.stdout


def test_a_second_lane_does_not_start(pilot):
    d, env = pilot
    holder = subprocess.Popen(["flock", str(d / "logs" / ".run_lane.lock"), "sleep", "30"])
    try:
        import time
        time.sleep(0.3)
        r = _run(d, env, "p1_pythia")
    finally:
        holder.kill()
    assert r.returncode == 1 and "already running" in r.stderr


def test_p1_gpu_lock_makes_a_second_p1_job_wait(tmp_path, monkeypatch):
    import argparse
    import sys
    import time
    import p1_checkpoints as p1
    from common import runtime
    monkeypatch.setattr(runtime, "PILOT_ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(1.5)",
                               str(tmp_path / "logs" / ".p1_gpu.lock")], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    args = argparse.Namespace(smoke=False, pin_only=False, device="cuda")
    t0 = time.perf_counter()
    f, waited = p1.gpu_lock(args)
    assert f is not None and waited > 0.8 and time.perf_counter() - t0 > 0.8
    holder.wait()
    assert p1.gpu_lock(argparse.Namespace(smoke=True, pin_only=False, device=None)) == (None, 0.0)


def test_pins_saved_by_two_jobs_keep_both_jobs_entries(tmp_path):
    import json
    from common import hf
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"a@main": {"commit": "1" * 40}}) + "\n")
    lane, other = hf.Pins(path), hf.Pins(path)              # both read the file at start (as at import)
    lane.data["b@step1"] = {"commit": "2" * 40}
    lane.data["a@main"]["deduped_check"] = {"verdict": "differs"}
    lane.save()
    other.data["c@step2"] = {"commit": "3" * 40}             # other's copy of a@main is the old one
    other.save()
    disk = json.loads(path.read_text())
    assert set(disk) == {"a@main", "b@step1", "c@step2"}
    assert disk["a@main"]["deduped_check"] == {"verdict": "differs"}       # not undone by the older copy
    assert not list(tmp_path.glob("*.tmp"))
