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


# ----------------------------------------------------------------------------- --take-over (2026-11-18)
OLD_COMMIT = "6c5bf9a"      # the p6_after_queue.py that the server's waiter runs (unchanged until --take-over)


def old_waiter_script(tmp_path):
    """The waiter as started on the server: p6_after_queue.py at 6c5bf9a, next to links to this run_queue.py and
    common/ (its HERE); None if git cannot give it (the Library zip has no history)."""
    r = subprocess.run(["git", "show", f"{OLD_COMMIT}:p6_after_queue.py"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    d = tmp_path / "old_code"
    d.mkdir()
    (d / "p6_after_queue.py").write_text(r.stdout)
    for name in ("run_queue.py", "common", "config.py"):
        (d / name).symlink_to(ROOT / name)
    return d / "p6_after_queue.py"


def two_queues(tmp_path, old_seconds=1.0, new_seconds=1.0):
    """The design's queue file (a stand-in job) and the raised one (another), each writing its end time."""
    logs, jf, end = setup(tmp_path, job_seconds=old_seconds, name="p6_standin")
    end_new = tmp_path / "job_end_raised"
    jobs = [{"name": "p6r_standin", "args": ["-c", f"import time; time.sleep({new_seconds}); "
                                                   f"open({str(end_new)!r}, 'w').write(repr(time.time()))"],
             "out": "results/p6r_standin", "gpu": 0.0}]
    jf_new = tmp_path / "p6_queue_raised_test.json"
    jf_new.write_text(json.dumps(jobs))
    return logs, jf, end, jf_new, end_new


def take_over(tmp_path, jf_new, *extra):
    return subprocess.Popen([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf_new), "--poll", "0.2",
                             "--take-over", *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def started_waiter(tmp_path, jf, old: bool):
    script = old_waiter_script(tmp_path) if old else Path(SCRIPT)
    if script is None:
        pytest.skip(f"git cannot show {OLD_COMMIT}")
    return subprocess.Popen([PY, str(script), "--root", str(tmp_path), "--jobs-file", str(jf), "--poll", "0.2"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def kill_left(tmp_path):
    """Kill any process of this test still running (its arguments name tmp_path), so a failure leaves no queue
    behind for the next test's refusal checks."""
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != os.getpid():
            try:
                if str(tmp_path).encode() in (d / "cmdline").read_bytes():
                    os.kill(int(d.name), signal.SIGKILL)
            except OSError:
                pass


@pytest.mark.parametrize("old", [True, False])
def test_take_over_while_the_waiter_waits(tmp_path, old):
    """The verdict comes while the old queue still runs: the waiter (the server's code, 6c5bf9a) is replaced, the
    design's queue never starts, the raised one starts when the old queue ends, and olmo7b waits through it all."""
    try:
        logs, jf, end, jf_new, end_new = two_queues(tmp_path)
        old_q = hold(logs / ".run_lane.queue.lock")
        passed = tmp_path / "olmo7b_passed"
        olmo = olmo7b_standin(logs, passed)
        w = started_waiter(tmp_path, jf, old)
        read_until(w, "waiting for the running run_queue.py to end")
        t = take_over(tmp_path, jf_new)
        read_until(t, "released (the queue lock keeps p1_olmo7b waiting)")
        w.wait(10)                                                   # the old waiter is gone, having started nothing
        time.sleep(0.5)
        assert not end.exists() and not end_new.exists() and not passed.exists() and old_q.poll() is None
        t_release = time.time()
        (logs / ".run_lane.queue.lock.release").write_text("")      # the old queue ends
        old_q.wait(10)
        out, _ = t.communicate(timeout=60)
        assert t.returncode == 0, out
        olmo.wait(30)
        assert t_release < float(end_new.read_text()) < float(passed.read_text())
        assert not end.exists()                                      # the design's 2^28 queue never ran
        prog = (logs / "progress.txt").read_text()
        assert "START  p6r_standin" in prog and "p6_standin" not in prog
    finally:
        kill_left(tmp_path)


def test_take_over_after_the_design_queue_started(tmp_path):
    """The verdict comes after P6 started: its queue and job are stopped (exit 130), the raised queue starts, and
    olmo7b never sees the lanes free in between."""
    try:
        logs, jf, end, jf_new, end_new = two_queues(tmp_path, old_seconds=60)
        old_q = hold(logs / ".run_lane.queue.lock")
        passed = tmp_path / "olmo7b_passed"
        olmo = olmo7b_standin(logs, passed)
        w = started_waiter(tmp_path, jf, old=True)
        read_until(w, "waiting for the running run_queue.py to end")
        (logs / ".run_lane.queue.lock.release").write_text("")
        old_q.wait(10)
        read_until(w, "logs/.run_lane.p6.lock released")
        prog_f = logs / "progress.txt"
        wait_for(lambda: prog_f.exists() and "START  p6_standin" in prog_f.read_text())
        t = take_over(tmp_path, jf_new)
        out, _ = t.communicate(timeout=120)
        assert t.returncode == 0, out
        w.wait(30)
        olmo.wait(30)
        prog = (logs / "progress.txt").read_text()
        assert "END    p6_standin  exit 130 (interrupted)" in prog and not end.exists()
        assert "START  p6r_standin" in prog and "END    p6r_standin  exit 0" in prog
        assert float(end_new.read_text()) < float(passed.read_text())
        assert "stopping the P6 queue" in out or "stopping p6_after_queue.py" in out
    finally:
        kill_left(tmp_path)


def test_take_over_with_nothing_running_starts_the_queue(tmp_path):
    try:
        logs, jf, end, jf_new, end_new = two_queues(tmp_path)
        passed = tmp_path / "olmo7b_passed"
        t = take_over(tmp_path, jf_new)
        out, _ = t.communicate(timeout=60)
        assert t.returncode == 0 and "no queue runs" in out, out
        assert end_new.exists() and not end.exists() and not passed.exists()
    finally:
        kill_left(tmp_path)


def test_take_over_refusals_and_check(tmp_path):
    logs, jf, end, jf_new, end_new = two_queues(tmp_path)
    r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(tmp_path / "none.json"),
                        "--take-over"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "is missing" in r.stdout
    sw = hold(logs / ".run_lane.p6switch.lock")
    r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf_new), "--take-over"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "another --take-over runs" in r.stdout
    sw.kill()
    old_q = hold(logs / ".run_lane.queue.lock")
    w = waiter(tmp_path, jf)
    try:
        read_until(w, "waiting for the running run_queue.py to end")
        before = sorted(p.name for p in logs.iterdir())
        r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf_new), "--take-over",
                            "--check"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and f"would stop p6_after_queue.py pid {w.pid}" in r.stdout, r.stdout
        assert "wait for the running queue" in r.stdout and sorted(p.name for p in logs.iterdir()) == before
        assert w.poll() is None                                  # --check stopped nothing
    finally:
        w.kill()
        old_q.kill()
    # the raised queue already runs
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "run_queue.py").write_text("import time; time.sleep(60)\n")
    q = subprocess.Popen([PY, str(fake / "run_queue.py"), "--jobs-file", jf_new.name])
    try:
        time.sleep(0.3)
        r = subprocess.run([PY, SCRIPT, "--root", str(tmp_path), "--jobs-file", str(jf_new), "--take-over"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "runs already" in r.stdout
    finally:
        q.kill()


def test_the_raised_queue_file_is_accepted():
    r = subprocess.run([PY, str(ROOT / "run_queue.py"), "--jobs-file", str(ROOT / "p6_queue_small_raised.json"),
                        "--dry-run"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0 and "p6r_timing_14m" in r.stdout, r.stdout + r.stderr
