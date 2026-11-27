"""Start P7 (p7_queue.json) once P6c (p6c_queue_small.json) has ended (added 2026-11-27, before any P7 run; new
file: p4_clipping.py's P4 paths, p6_*.py, p6c_after_olmo.py, common/, run_queue.py, run_all.sh and the jobs files in
use are unchanged). P7 never competes with P6c for the GPU: it starts only after P6c's queue and its waiter are both
gone.

  python p7_after_p6c.py --check            # what it waits for and what it would do; changes nothing
  setsid nohup nice -n 10 taskset -c 0-10 python p7_after_p6c.py > logs/p7_after_p6c.out 2>&1 < /dev/null &
  kill -TERM $(cat logs/p7_after_p6c.pid)   # while waiting: stop waiting; after the start: stop P7 (saved runs stay)

What it does, in order:
  1. Refuses to start (exit 2) if p7_queue.json or P6c's jobs file is missing, a P7 queue or another p7_after_p6c.py
     runs, every P7 job is already done, or P6c is neither waiting (p6c_after_olmo.py holds its lock), running (a
     run_queue.py with p6c_queue_small.json) nor ended (a progress line for one of its jobs): it would wait for ever.
  2. Waits, reading only (processes in /proc, logs/progress.txt, /proc/locks; it takes no lock the others look at),
     until all of these hold:
       - no p6c_after_olmo.py holds logs/.p6c_after_olmo.lock and no run_queue.py runs p6c_queue_small.json;
       - logs/progress.txt has a line for at least one P6c job, and the last line of every P6c job that has one is
         END or SKIP (whatever its exit code: a failed or stopped P6c has ended too);
       - nobody holds logs/.run_lane.queue.lock (the queue that ran P6c has ended).
     A P6c START line with no process (a queue killed with -9) is reported every hour and waited out.
  3. If results/p6c_small/STOP exists then (P6c stopped for lack of credit), it starts nothing and exits 0: P7 would
     spend credit that is not there; start the queue by hand if wanted (README "P7").
  4. Otherwise starts run_queue.py --jobs-file p7_queue.json (its output in logs/run_queue_p7.out): Phase 0 and
     Phase 1 in one GPU job, then h_link.py on its series, the width decomposition and the analysis on the CPU. Waits
     for it and exits with its code.

To stop P7 after the seed that is running (lack of credit): touch results/p7_transformer_sgd/STOP. The job then
starts nothing more, writes stopped.json and exits 0, and h_link, the decomposition and the analysis run on the
complete seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6_after_queue as aq  # noqa: E402  (its read-only helpers; nothing runs on import)
import run_queue  # noqa: E402

JOBS_FILE = "p7_queue.json"
AFTER_JOBS_FILE = "p6c_queue_small.json"         # P6c's (the P6 thread's file; read only)
AFTER_LOCK = ".p6c_after_olmo.lock"              # held by P6c's waiter while it waits and runs
AFTER_STOP = "results/p6c_small/STOP"
OWN_LOCK = ".p7_after_p6c.lock"                  # not a .run_lane* name: invisible to the lanes and the queues
PID_FILE = "p7_after_p6c.pid"
OUT_FILE = "run_queue_p7.out"
LOST_REPORT_S = 3600.0


def say(msg: str) -> None:
    print(f"{aq.stamp()} p7_after_p6c.py: {msg}", flush=True)


# ----------------------------------------------------------------------------- reading the machine (read-only)
def last_events(logs: Path, jobs: list[str]) -> dict[str, tuple[str, str]]:
    """The kind (START, END or SKIP) and text of the last line of logs/progress.txt for each job that has one, and
    under "" the last line for any of them."""
    out: dict[str, tuple[str, str]] = {}
    try:
        lines = (logs / "progress.txt").read_text(errors="replace").splitlines()
    except OSError:
        return out
    want = set(jobs)
    for ln in lines:
        w = ln.split()
        if len(w) >= 3 and w[1] in ("START", "END", "SKIP") and w[2] in want:
            out[w[2]] = out[""] = (w[1], ln.strip())
    return out


def job_names(jobs_file: Path) -> list[str]:
    return [j["name"] for j in json.loads(jobs_file.read_text())]


def queues_of(procs: dict[int, list[str]], me: set[int], name: str) -> list[int]:
    return [p for p, a in aq.running_queues(procs, me) if aq.queue_jobs_file(a)
            and Path(aq.queue_jobs_file(a)).name == name]


def state(root: Path, procs: dict[int, list[str]], me: set[int], after_jobs: list[str]) -> dict:
    logs = root / "logs"
    ev = last_events(logs, after_jobs)
    after_queue = queues_of(procs, me, AFTER_JOBS_FILE)
    after_waiter = [p for p in aq.lock_holders(logs / AFTER_LOCK) if p not in me]
    started = [j for j in after_jobs if j in ev and j]
    lost = [j for j in started if ev[j][0] == "START"] if not after_queue else []
    ended = bool(started) and not after_queue and not after_waiter and not lost
    queue_holders = [p for p in aq.lock_holders(logs / run_queue.QUEUE_LOCK) if p not in me]
    last = ev.get("", ("", ""))[1]
    return {"after_queue": after_queue, "after_waiter": after_waiter, "started": started, "lost": lost,
            "ended": ended, "known": bool(after_queue or after_waiter or started), "last_line": last,
            "n_after": len(after_jobs), "n_ended": sum(ev[j][0] in ("END", "SKIP") for j in started),
            "queue_holders": queue_holders, "ready": ended and not queue_holders}


def jobs_left(root: Path, jobs_file: Path) -> list[str]:
    return [n for n in job_names(jobs_file) if not (root / "logs" / f"{n}.done").exists()]


def refusals(root: Path, jobs_file: Path, after_file: Path, procs, me, st: dict | None) -> list[str]:
    if not jobs_file.exists():
        return [f"{jobs_file} is missing (git pull first)"]
    if not after_file.exists():
        return [f"{after_file} (P6c's jobs file) is missing (git pull first)"]
    out = []
    if queues_of(procs, me, JOBS_FILE):
        out.append(f"a P7 queue already runs (pid {queues_of(procs, me, JOBS_FILE)})")
    if aq.lock_holders(root / "logs" / OWN_LOCK):
        out.append(f"another p7_after_p6c.py runs (logs/{OWN_LOCK})")
    if not jobs_left(root, jobs_file):
        out.append("every P7 job is done already (logs/<job>.done)")
    if st is not None and not st["known"]:
        out.append("P6c is neither waiting (p6c_after_olmo.py), running nor ended (no progress line for its jobs): "
                   "start P6c first, or start P7 by hand")
    return out


def describe(st: dict) -> list[str]:
    def pids(x):
        return ", ".join(f"pid {p}, started {aq.started_utc(p)}" for p in x)
    if st["ended"]:
        p6c = "ended"
    elif st["after_queue"]:
        p6c = f"running ({pids(st['after_queue'])})"
    elif st["after_waiter"]:
        p6c = f"waiting to start (p6c_after_olmo.py: {pids(st['after_waiter'])})"
    elif st["lost"]:
        p6c = f"START line with no process for {', '.join(st['lost'])} (queue killed?)"
    else:
        p6c = "not started"
    held = f"held by pid {', '.join(map(str, st['queue_holders']))}" if st["queue_holders"] else "free"
    return [f"P6c: {p6c}; {st['n_ended']} of {st['n_after']} jobs ended; last progress line: "
            f"{st['last_line'] or 'none'}",
            f"logs/{run_queue.QUEUE_LOCK}: {held}"]


def check(root: Path, jobs_file: Path, after_file: Path) -> int:
    procs, me = aq.processes(), {os.getpid()}
    st = state(root, procs, me, job_names(after_file)) if after_file.exists() else None
    for ln in describe(st) if st else [f"{after_file} is missing"]:
        print(ln)
    waiters = aq.lock_holders(root / "logs" / OWN_LOCK)
    print(f"p7_after_p6c.py waiting: {'pid ' + ', '.join(map(str, waiters)) if waiters else 'none'}; P7 queue: "
          + (", ".join(f"pid {p}" for p in queues_of(procs, me, JOBS_FILE)) or "not running"))
    if jobs_file.exists():
        left = jobs_left(root, jobs_file)
        print(f"P7 jobs not done: {len(left)} of {len(job_names(jobs_file))}" + (f" (next: {left[0]})" if left else ""))
    problems = refusals(root, jobs_file, after_file, procs, me, st) if not waiters else []
    if waiters:
        print("-> a waiter runs; it starts P7 " + ("now" if st and st["ready"] else "once the lines above say "
                                                                                    "ended/free"))
    elif problems:
        print("-> a new waiter would not start: " + "; ".join(problems))
    else:
        print("-> a new waiter would " + ("start P7 at once" if st["ready"] else "wait, then start P7"))
    if (root / AFTER_STOP).exists():
        print(f"{AFTER_STOP} exists: once P6c has ended the waiter starts nothing (lack of credit)")
    if (root / "results" / "p7_transformer_sgd" / "STOP").exists():
        print("results/p7_transformer_sgd/STOP exists: the P7 job will start no run")
    return 0


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the state and what it would do; change nothing")
    ap.add_argument("--jobs-file", default=JOBS_FILE, help=argparse.SUPPRESS)        # tests
    ap.add_argument("--after-jobs-file", default=AFTER_JOBS_FILE, help=argparse.SUPPRESS)   # tests
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks, here and in the P7 queue")
    ap.add_argument("--root", default=str(HERE), help=argparse.SUPPRESS)            # tests
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    logs = root / "logs"
    jobs_file, after_file = (Path(f) if Path(f).is_absolute() else root / f
                             for f in (args.jobs_file, args.after_jobs_file))
    if args.check:
        return check(root, jobs_file, after_file)
    procs, me = aq.processes(), {os.getpid()}
    st = state(root, procs, me, job_names(after_file)) if after_file.exists() else None
    problems = refusals(root, jobs_file, after_file, procs, me, st)
    if problems:
        say("not started: " + "; ".join(problems))
        return 2
    after_jobs = job_names(after_file)
    own = aq.flock(logs / OWN_LOCK, block=False)
    if own is None:
        say(f"not started: another p7_after_p6c.py runs (logs/{OWN_LOCK})")
        return 2
    (logs / PID_FILE).write_text(f"{os.getpid()}\n")
    child = None
    stop = {"sig": None}

    def on_signal(signum, frame):
        stop["sig"] = signum
        if child is None:
            say("stopped while waiting; nothing started")
            raise SystemExit(130)
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    say(f"pid {os.getpid()}; waiting until P6c has ended, checking every {args.poll:g} s. Now: "
        + " | ".join(describe(st)))

    # 2. wait, reading only
    told, lost_told = None, 0.0
    while not st["ready"]:
        key = (st["ended"], bool(st["after_queue"]), bool(st["after_waiter"]), bool(st["queue_holders"]),
               st["n_ended"])
        if key != told:
            say("waiting: " + " | ".join(describe(st)))
            told = key
        if st["lost"] and time.time() - lost_told > LOST_REPORT_S:
            say(f"P6c has a START line and no process ({', '.join(st['lost'])}); still waiting for its END")
            lost_told = time.time()
        time.sleep(args.poll)
        st = state(root, aq.processes(), me, after_jobs)
    say("P6c has ended: " + " | ".join(describe(st)))
    if stop["sig"]:
        return 130

    # 3. lack of credit
    if (root / AFTER_STOP).exists():
        say(f"{AFTER_STOP} exists (P6c stopped for lack of credit): P7 not started. To start it anyway: "
            f"python run_queue.py --jobs-file {JOBS_FILE}")
        own.close()
        return 0

    # 4. the P7 queue
    out_file = logs / OUT_FILE
    cmd = [sys.executable, str(HERE / "run_queue.py"), "--jobs-file", str(jobs_file), "--poll", f"{args.poll:g}"]
    if root != HERE:
        cmd += ["--root", str(root)]
    for attempt in range(3):
        offset = out_file.stat().st_size if out_file.exists() else 0
        with open(out_file, "a") as log:
            log.write(f"### {aq.stamp()}  p7_after_p6c.py: {' '.join(cmd)}\n")
            log.flush()
            child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        say(f"started the P7 queue (pid {child.pid}): {' '.join(cmd)}; its output in {out_file}")
        while child.poll() is None and not aq.child_holds_queue_lock(child, logs / run_queue.QUEUE_LOCK, out_file,
                                                                     offset):
            time.sleep(0.05)
        if child.poll() is None:
            break
        text = out_file.read_bytes()[offset:].decode(errors="replace")
        if "is already running" in text and attempt < 2 and not stop["sig"]:
            say(f"the P7 queue found logs/{run_queue.QUEUE_LOCK} taken for a moment; trying again in 5 s")
            time.sleep(5)
            continue
        break
    rc = child.wait()
    say(f"the P7 queue ended with exit {rc}")
    own.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
