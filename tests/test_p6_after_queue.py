"""p6_after_queue.py with stand-in processes: the P6 queue starts only after the running queue has ended and its jobs
are gone, p1_olmo7b's lane wait (common/gpu_locks.wait_for_lanes, the code olmo7b runs) never sees the lanes free
between the two queues, and the refusals, the read-only check and the stop."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPT = str(ROOT / "p6_after_queue.py")


def hold(path, *extra):
    """A process holding an flock on path until it is killed or the file `path`.release appears."""
    p = subprocess.Popen([PY, "-c", "import fcntl, os, sys, time; f = open(sys.argv[1], 'a'); "
                          "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True)\n"
                          "while not os.path.exists(sys.argv[1] + '.release'): time.sleep(0.02)", str(path), *extra],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def olmo7b_standin(logs, out):
    """p1_olmo7b's first step, with its real code: wait until no logs/.run_lane*.lock is held (every 10 ms here,
    every 60 s on the server), then record the time."""
    code = ("import sys, time; sys.path.insert(0, sys.argv[3]); from common import gpu_locks; "
            "gpu_locks.wait_for_lanes(__import__('pathlib').Path(sys.argv[1]), poll=0.01, log=lambda m: None); "
            "open(sys.argv[2], 'w').write(repr(time.time()))")
    env = {k: v for k, v in os.environ.items() if k != "PILOT_LANE_LOCK"}
    return subprocess.Popen([PY, "-c", code, str(logs), str(out), str(ROOT)], env=env)


def setup(tmp_path, job_seconds=1.0, name="p6_standin"):
    logs = tmp_path / "logs"
    logs.mkdir()
    end = tmp_path / "job_end"
    jobs = [{"name": name, "args": ["-c", f"import sys, time; time.sleep({job_seconds}); "
                                          f"open({str(end)!r}, 'w').write(repr(time.time()))"],
             "out": f"results/{name}", "gpu": 0.0}]
    jf = tmp_path / "p6_queue_test.json"
    jf.write_text(json.dumps(jobs))
    return logs, jf, end


def waiter(tmp_path, jf, *extra):
    return subprocess.Popen([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf), "--poll", "0.2", *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for(cond, timeout=30.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.02)


def read_until(p, text, timeout=30.0):
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        ln = p.stdout.readline()
        if not ln:
            break
        lines.append(ln)
        if text in ln:
            return "".join(lines)
    raise AssertionError(f"{text!r} not in {''.join(lines)!r}")


def test_p6_starts_after_the_queue_and_olmo7b_waits_through_the_handover(tmp_path):
    logs, jf, end = setup(tmp_path)
    old = hold(logs / ".run_lane.queue.lock")
    passed = tmp_path / "olmo7b_passed"
    olmo = olmo7b_standin(logs, passed)
    w = waiter(tmp_path, jf)
    read_until(w, "waiting for the running run_queue.py to end")
    time.sleep(0.5)
    assert not end.exists() and not passed.exists()          # nothing starts while the old queue runs
    t_release = time.time()
    (logs / ".run_lane.queue.lock.release").write_text("")   # the old queue ends
    old.wait(10)
    out, _ = w.communicate(timeout=60)
    assert w.returncode == 0, out
    olmo.wait(30)
    t_job_end, t_olmo = float(end.read_text()), float(passed.read_text())
    assert t_release < t_job_end < t_olmo                     # P6 ran after the queue; olmo7b only after P6
    prog = (logs / "progress.txt").read_text()
    assert "START  p6_standin" in prog and "END    p6_standin  exit 0" in prog
    assert "holds logs/.run_lane.queue.lock; logs/.run_lane.p6.lock released" in out
    assert (logs / "p6_standin.done").exists()


def test_without_the_handover_lock_olmo7b_slips_into_the_gap(tmp_path):
    """Negative control: releasing the queue lock and then starting run_queue.py (what a shell 'wait, then start'
    does) leaves the lanes free for a moment, and olmo7b's wait sees it."""
    logs, jf, end = setup(tmp_path)
    old = hold(logs / ".run_lane.queue.lock")
    passed = tmp_path / "olmo7b_passed"
    olmo = olmo7b_standin(logs, passed)
    time.sleep(0.2)
    (logs / ".run_lane.queue.lock.release").write_text("")
    old.wait(10)
    q = subprocess.Popen([PY, str(ROOT / "run_queue.py"), "--root", str(tmp_path), "--jobs-file", str(jf),
                          "--poll", "0.2"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    q.wait(60)
    olmo.wait(30)
    assert float(passed.read_text()) < float(end.read_text())  # olmo7b passed before the P6 job ended


def test_p6_waits_for_jobs_the_old_queue_left_running(tmp_path):
    logs, jf, end = setup(tmp_path)
    old = hold(logs / ".run_lane.queue.lock")
    # a p4_transformer left running by a queue killed with -9: its pid file and its folder in the command line
    orphan = subprocess.Popen([PY, "-c", "import time; time.sleep(60)", "--out", "results/p4_transformer"])
    (logs / "p4_transformer.queue.pid").write_text(str(orphan.pid))
    passed = tmp_path / "olmo7b_passed"
    olmo = olmo7b_standin(logs, passed)
    w = waiter(tmp_path, jf)
    read_until(w, "waiting for the running run_queue.py to end")
    (logs / ".run_lane.queue.lock.release").write_text("")
    old.wait(10)
    read_until(w, "old queue jobs still running: p4_transformer")
    time.sleep(1.0)
    assert not end.exists() and not passed.exists()          # P6 and olmo7b both wait for the orphan
    t_kill = time.time()
    orphan.kill()
    orphan.wait()
    out, _ = w.communicate(timeout=60)
    assert w.returncode == 0, out
    olmo.wait(30)
    assert t_kill < float(end.read_text()) < float(passed.read_text())


def test_refusals(tmp_path):
    logs, jf, end = setup(tmp_path)
    # no queue running
    r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "no run_queue.py runs" in r.stdout and not end.exists()
    # missing jobs file
    old = hold(logs / ".run_lane.queue.lock")
    r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(tmp_path / "none.json")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "is missing" in r.stdout
    # a second waiter
    w = waiter(tmp_path, jf)
    read_until(w, "waiting for the running run_queue.py to end")
    r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "another p6_after_queue.py runs" in r.stdout
    # stopped while waiting: nothing starts, the old queue goes on
    w.send_signal(signal.SIGTERM)
    w.wait(10)
    assert old.poll() is None
    (logs / ".run_lane.queue.lock.release").write_text("")
    old.wait(10)
    time.sleep(0.5)
    assert not end.exists() and not (logs / "progress.txt").exists()
    # a P6 queue already running
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "run_queue.py").write_text("import time; time.sleep(60)\n")
    q = subprocess.Popen([PY, str(fake / "run_queue.py"), "--jobs-file", "p6_queue_small.json"])
    (logs / ".run_lane.queue.lock.release").unlink()
    old = hold(logs / ".run_lane.queue.lock")
    try:
        time.sleep(0.3)
        r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "a P6 queue already runs" in r.stdout
    finally:
        q.kill()
        old.kill()


def test_check_changes_nothing(tmp_path):
    logs, jf, end = setup(tmp_path)
    (logs / "progress.txt").write_text("2026-11-17T10:40:00Z  START  p1_olmo7b  total 1.00 h\n")
    (logs / "p1_olmo7b.log").write_text("### 2026-11-17T10:40:00Z  run_all.sh: python p1_checkpoints.py --suite "
                                        "olmo7b\n### git 3463b1c; free disk: 100G\n[p1_checkpoints] waiting for "
                                        "run_lane.sh to finish (.run_lane.lock, .run_lane.queue.lock held)\n")
    old = hold(logs / ".run_lane.queue.lock")
    try:
        before = sorted(p.name for p in logs.iterdir())
        r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert sorted(p.name for p in logs.iterdir()) == before
        assert "logs/.run_lane.queue.lock: HELD" in r.stdout
        assert "p1_olmo7b (run_all.sh): 2026-11-17T10:40:00Z" in r.stdout and "git 3463b1c" in r.stdout
        assert "would wait for the running queue, then start P6" in r.stdout and "WARNING" not in r.stdout
        # an olmo7b started from older code is flagged
        (logs / "p1_olmo7b.log").write_text("### x\n### git b13cadc; free disk: 1G\n")
        r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert "WARNING: p1_olmo7b loaded b13cadc, older than 3463b1c" in r.stdout
    finally:
        old.kill()


def test_stop_after_the_start_stops_the_p6_queue(tmp_path):
    logs, jf, end = setup(tmp_path, job_seconds=60)
    old = hold(logs / ".run_lane.queue.lock")
    w = waiter(tmp_path, jf)
    read_until(w, "waiting for the running run_queue.py to end")
    (logs / ".run_lane.queue.lock.release").write_text("")
    old.wait(10)
    read_until(w, "logs/.run_lane.p6.lock released")
    wait_for(lambda: "START  p6_standin" in (logs / "progress.txt").read_text())
    w.send_signal(signal.SIGTERM)
    out, _ = w.communicate(timeout=60)
    assert w.returncode == 130, out
    prog = (logs / "progress.txt").read_text()
    assert "END    p6_standin  exit 130 (interrupted)" in prog and not end.exists()


@pytest.mark.parametrize("name", ["p6_queue_small.json", "p6_queue_full.json"])
def test_the_real_queue_files_are_accepted(name):
    r = subprocess.run([PY, str(ROOT / "run_queue.py"), "--jobs-file", str(ROOT / name), "--dry-run"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0 and "p6_timing_14m" in r.stdout
