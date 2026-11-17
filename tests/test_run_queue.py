"""run_queue.py: the admission rule (lanes keep room for their next job, olmo7b alone), the records it writes next
to run_all.sh and the lanes, dependencies, retries, stop and restart, and the peaks it uses."""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import run_queue as rq
from common import gpu_locks

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def lane(name, peak, nxt):
    return rq.Lane(name, peak, nxt)


def hold(path):
    """A process holding an flock on path (itself, so that killing it frees the lock at once)."""
    p = subprocess.Popen([PY, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
                          "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(60)", str(path)],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_admission_keeps_room_for_the_lanes():
    P = rq.PEAK_GIB
    p1_next, p3 = lane("P1", P["p1_olmo"], P["p1_olmo"]), lane("P3", P["p3_fractional"], None)
    # now: the P1 lane (olmo still to start) and P3 run; only the P5 reruns fit next to them
    assert rq.admit(P["rerun_p5_floor10"], [p1_next, p3], [])[0]
    assert rq.admit(P["rerun_p5_a12"], [p1_next, p3], [("floor10", P["rerun_p5_floor10"])])[0]
    assert not rq.admit(P["rerun_p2_bf16"], [p1_next, p3], [])[0]
    ok, why = rq.admit(P["p4_transformer"], [p1_next, p3], [])
    assert not ok
    # after the P1 lane: P3 next to P4 or P2, not both
    assert rq.admit(P["p4_transformer"], [p3], [])[0] and rq.admit(P["rerun_p2_bf16"], [p3], [])[0]
    assert not rq.admit(P["rerun_p2_bf16"], [p3], [("p4", P["p4_transformer"])])[0]
    # a lane with a job still to start keeps room for it plus run_lane.sh's 4 GiB
    assert rq.admit(P["p4_transformer"], [p1_next], [])[0]
    ok, why = rq.admit(21.6, [p1_next], [])                  # fits the GPU now, not with the lane's next job
    assert not ok and "next job" in why
    assert rq.admit(21.6, [lane("P1", P["p1_olmo"], None)], [])[0]
    # with no lane, P4 and P2 fit together, with one P5 rerun
    assert rq.admit(P["p4_transformer"], [], [("p2", P["rerun_p2_bf16"]), ("floor10", P["rerun_p5_floor10"])])[0]
    assert not rq.admit(P["rerun_p5_a12"], [], [("p2", P["rerun_p2_bf16"]), ("floor10", P["rerun_p5_floor10"]),
                                                ("p4", P["p4_transformer"])])[0]
    # nvidia-smi has the last word
    ok, why = rq.admit(P["rerun_p5_a12"], [], [], free=1.0)
    assert not ok and "nvidia-smi" in why


def test_every_admitted_set_fits_with_room_for_each_lane():
    """Everything at its peak: the sets --combos lists stay within the GPU, and a lane with a job to start finds
    run_lane.sh's free memory for it."""
    args = type("A", (), {"gpu_total": rq.GPU_TOTAL_GIB})()
    by = {j.name: j.gpu for j in rq.JOBS}
    for c in rq.combos(args):
        for s in c["largest sets that fit"]:
            names = [] if s == "none" else s.strip("{}").split(", ")
            lanes_gib = c["lanes_GiB"]
            jobs_gib = sum(rq.use_gib(by[n]) for n in names)
            assert lanes_gib + jobs_gib + rq.MARGIN_GIB <= rq.GPU_TOTAL_GIB + 1e-9, (c, s)
            if "next job" in c["lanes"]:
                assert rq.GPU_TOTAL_GIB - (lanes_gib - rq.use_gib(rq.PEAK_GIB["p1_olmo"])) - jobs_gib >= \
                    rq.PEAK_GIB["p1_olmo"] + rq.LANE_HEADROOM_GIB


def test_peaks_agree_with_run_lane_and_memory_check():
    lane_sh = (ROOT / "run_lane.sh").read_text()
    for job, gib in re.findall(r"(p1_\w+|p3_fractional)\) echo ([\d.]+) ;;", lane_sh):
        assert rq.PEAK_GIB[job] == float(gib), job
    import memory_check as mc
    rows = []
    mc.check_p5(rows, lambda m: None)
    p5 = max(r["peak_GiB"] for r in rows)
    assert rq.PEAK_GIB["rerun_p5_floor10"] == rq.PEAK_GIB["rerun_p5_a12"] >= p5 > 0.5


def test_olmo7b_waits_for_the_queue(tmp_path):
    """p1_checkpoints.py's olmo7b waits while any logs/.run_lane*.lock is held: the queue's lock is one of them."""
    logs = tmp_path / "logs"
    logs.mkdir()
    holder = hold(logs / rq.QUEUE_LOCK)
    try:
        assert [p.name for p in gpu_locks.lanes_running(logs)] == [rq.QUEUE_LOCK]
    finally:
        holder.kill()
        holder.wait()
    assert gpu_locks.lanes_running(logs) == []


# ----------------------------------------------------------------------------- the queue itself, with stand-in jobs
def _code(body):
    return ["-c", "import sys, time, pathlib\nout = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                  "out.mkdir(parents=True, exist_ok=True)\n" + body]


@pytest.fixture
def qroot(tmp_path):
    d = tmp_path / "pilot"
    (d / "logs").mkdir(parents=True)
    (d / "pins.json").write_text("{}\n")
    (d / "logs" / "progress.txt").write_text("2026-11-17T00:00:00Z  END    p3_fractional  exit 143  elapsed 1.00 h  "
                                             "total 10.00 h\n")
    return d


def _queue(d, jobs, *extra, env=None, wait=True):
    (d / "jobs.json").write_text(json.dumps(jobs))
    cmd = [PY, str(ROOT / "run_queue.py"), "--root", str(d), "--jobs-file", str(d / "jobs.json"), "--poll", "0.2",
           *extra]
    e = dict(os.environ, RUN_QUEUE_FREE_GIB="", **(env or {}))
    if not wait:
        return subprocess.Popen(cmd, cwd=d, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, cwd=d, env=e, capture_output=True, text=True, timeout=120)


def _progress(d):
    return (d / "logs" / "progress.txt").read_text()


def test_queue_runs_jobs_with_run_all_records(qroot):
    d = qroot
    jobs = [{"name": "gpu_a", "args": _code("(out / 'a.txt').write_text('a')"), "out": "results/gpu_a", "gpu": 1.0},
            {"name": "fails", "args": _code("sys.exit(3)"), "out": "results/fails", "gpu": 1.0},
            {"name": "cpu_b", "args": _code("(out / 'b.txt').write_text('b')"), "out": "results/cpu_b",
             "after": ["gpu_a"]},
            {"name": "after_fail", "args": _code("pass"), "out": "results/after_fail", "after": ["fails"]},
            {"name": "no_input", "args": _code("pass"), "out": "results/no_input", "needs": ["results/x.csv"]}]
    r = _queue(d, jobs)
    assert r.returncode == 1, r.stdout + r.stderr
    prog = _progress(d)
    assert re.search(r"START  gpu_a  total 1\.00 h  \(queue\)", prog)            # run_all.sh's running total
    assert re.search(r"END    gpu_a  exit 0  elapsed 0\.00 h  total 1\.00 h  \(queue\)", prog)
    assert prog.count("START  fails") == 2 and prog.count("END    fails  exit 3") == 2       # retried once
    assert "after_fail" not in prog and "after_fail not run: fails did not end with 0" in r.stdout
    assert "no_input not run: missing input results/x.csv" in r.stdout
    assert prog.index("END    gpu_a") < prog.index("START  cpu_b")
    for j in ("gpu_a", "cpu_b"):
        assert (d / "logs" / f"{j}.done").exists() and (d / "results_archive" / f"{j}.tar.gz").exists()
        assert (d / "logs" / f"{j}.log").read_text().startswith("### ")
        assert not (d / "logs" / f"{j}.queue.pid").exists()
    assert not (d / "logs" / "fails.done").exists() and (d / "results_archive" / "pins.json").exists()
    # a second queue skips what is done and runs the rest again
    r = _queue(d, jobs[:3], "--retries", "0")
    assert "SKIP   gpu_a" in _progress(d) and _progress(d).count("START  fails") == 3


def test_queue_waits_for_lane_room_and_for_olmo7b(qroot):
    d = qroot
    logs = d / "logs"
    # a P1 lane runs (pythia done, polypythia running, olmo still to start): no room for a 25 GiB job
    (logs / "p1_pythia.done").write_text("x\n")
    with open(logs / "progress.txt", "a") as f:
        f.write("2026-11-17T01:00:00Z  START  p1_polypythia  total 11.00 h  (lane)\n")
    lane_lock = hold(logs / ".run_lane.lock")
    olmo = None
    try:
        jobs = [{"name": "big", "args": _code("pass"), "out": "results/big", "gpu": 25.0},
                {"name": "small", "args": _code("pass"), "out": "results/small", "gpu": 1.0}]
        q = _queue(d, jobs, wait=False)
        time.sleep(2.5)
        prog = _progress(d)
        assert "START  small" in prog and "START  big" not in prog
        # the lane ends; olmo7b is on the GPU (P3's lock held without a P3 lane): big still waits
        olmo = hold(logs / ".p3_gpu.lock")
        lane_lock.kill()
        lane_lock.wait()
        time.sleep(1.5)
        assert "START  big" not in _progress(d)
        olmo.kill()
        olmo.wait()
        out, _ = q.communicate(timeout=60)
        assert q.returncode == 0, out
        assert "big waits: " in out and "p1_olmo7b is on the GPU" in out and "START  big" in _progress(d)
    finally:
        for p in (lane_lock, olmo):
            if p is not None and p.poll() is None:
                p.kill()


def test_queue_stops_its_jobs_and_goes_on_after_a_restart(qroot):
    d = qroot
    jobs = [{"name": "long", "args": _code("(out / 'started').write_text('1'); time.sleep(60)"),
             "out": "results/long", "gpu": 1.0}]
    q = _queue(d, jobs, wait=False)
    for _ in range(100):
        if (d / "results" / "long" / "started").exists():
            break
        time.sleep(0.1)
    pid = int((d / "logs" / "long.queue.pid").read_text())
    # a second queue does not start while the first holds its lock
    r = _queue(d, jobs)
    assert r.returncode == 1 and "already running" in r.stderr
    q.send_signal(signal.SIGTERM)
    out, _ = q.communicate(timeout=60)
    assert q.returncode == 130, out
    assert re.search(r"END    long  exit 130 \(interrupted\)", _progress(d))
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(pid, 0)
            time.sleep(0.1)
    # restarted, the job runs again
    jobs[0]["args"] = _code("pass")
    r = _queue(d, jobs)
    assert r.returncode == 0 and (d / "logs" / "long.done").exists()


def test_queue_refuses_while_a_job_of_an_earlier_queue_runs(qroot):
    d = qroot
    out = "results/orphan"
    orphan = subprocess.Popen([PY, "-c", "import time; time.sleep(30)", "--out", out])
    try:
        (d / "logs" / "orphan.queue.pid").write_text(str(orphan.pid))
        jobs = [{"name": "orphan", "args": _code("pass"), "out": out, "gpu": 1.0}]
        r = _queue(d, jobs)
        assert r.returncode == 1 and "still running from an earlier run_queue.py: orphan" in r.stderr
    finally:
        orphan.kill()
        orphan.wait()
    with open(d / "logs" / "progress.txt", "a") as f:
        f.write("2026-11-17T02:00:00Z  START  orphan  total 11.00 h  (queue)\n")
    r = _queue(d, jobs)
    assert r.returncode == 0 and "END    orphan  exit 255 (lost" in _progress(d)


def test_real_jobs_commands():
    by = {j.name: j for j in rq.JOBS}
    assert by["p4_transformer"].args == ["p4_clipping.py", "--model", "transformer"]     # run_all.sh's command
    assert by["p4_transformer"].out == "results/p4_transformer"
    assert "gpu_job p4_transformer   $PY p4_clipping.py --model transformer" in (ROOT / "run_all.sh").read_text()
    assert set(by["rerun_p5_refit"].after) == {"rerun_p5_floor10", "rerun_p5_a12"}
    assert all(j.gpu == 0 for j in rq.JOBS if j.name in ("rerun_p5_refit", "rerun_p5_h31_flags", "rerun_p2_popqa"))
    # every job writes a new folder; none writes over a first-run folder other than p4_transformer's (it failed)
    assert len({j.out for j in rq.JOBS}) == len(rq.JOBS)
    assert all(j.out not in ("results/p5_escape", "results/p2_w4a4", "results/p5_escape_h31") for j in rq.JOBS)
