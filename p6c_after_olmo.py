"""Start P6c (p6c_queue_small.json) once run_all.sh's p1_olmo7b has ended (added 2026-11-27, before any P6c run; new
file: p4_clipping.py, p1_checkpoints.py, config.py, common/, run_queue.py, run_all.sh and the jobs files in use are
unchanged). P6c never competes with OLMo-7B for the GPU: it starts only after p1_olmo7b has ended and no queue runs.

  python p6c_after_olmo.py --check            # what it waits for and what it would do; changes nothing
  setsid nohup nice -n 10 taskset -c 0-10 python p6c_after_olmo.py > logs/p6c_after_olmo.out 2>&1 < /dev/null &
  kill -TERM $(cat logs/p6c_after_olmo.pid)   # while waiting: stop waiting; after the start: stop P6c (saved runs stay)

What it does, in order (p6b_after_olmo.py's steps without the P4 condition, which P6b already met):
  1. Refuses to start (exit 2) if the jobs file is missing, a P6c queue or another p6c_after_olmo.py runs, every P6c
     job is already done, or p1_olmo7b is neither running (or waiting) nor ended (it would wait for ever).
  2. Waits, reading only (processes in /proc, logs/progress.txt, /proc/locks; it takes no lock the others look at,
     so nothing running or waiting notices it), until both of these hold:
       - no python process runs p1_checkpoints.py for olmo7b, and logs/progress.txt's last line for p1_olmo7b is END
         or SKIP (whatever its exit code: an interrupted or failed OLMo-7B has ended too);
       - nobody holds logs/.run_lane.queue.lock (no queue runs).
     A p1_olmo7b START line with no process (run_all.sh killed with -9) is reported every hour and waited out.
  3. Starts run_queue.py --jobs-file p6c_queue_small.json (its output in logs/run_queue_p6c.out): the 20 seeds one
     after another, each job all four corners of its seed, then the analysis. Waits for it and exits with its code.

To stop P6c after the seed that is running (lack of credit): touch results/p6c_small/STOP. Every later seed job then
starts nothing and exits 0, and the analysis runs on the complete seeds.
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
from p6b_after_olmo import last_event  # noqa: E402  (read-only; nothing runs on import)

JOBS_FILE = "p6c_queue_small.json"
OWN_LOCK = ".p6c_after_olmo.lock"                 # not a .run_lane* name: invisible to p1_olmo7b and the queues
PID_FILE = "p6c_after_olmo.pid"
OUT_FILE = "run_queue_p6c.out"
OLMO_JOB = "p1_olmo7b"
LOST_REPORT_S = 3600.0


def say(msg: str) -> None:
    print(f"{aq.stamp()} p6c_after_olmo.py: {msg}", flush=True)


# ----------------------------------------------------------------------------- reading the machine (read-only)
def p6c_queues(procs: dict[int, list[str]], me: set[int]) -> list[int]:
    return [p for p, a in aq.running_queues(procs, me) if aq.queue_jobs_file(a)
            and Path(aq.queue_jobs_file(a)).name == JOBS_FILE]


def state(root: Path, procs: dict[int, list[str]], me: set[int]) -> dict:
    logs = root / "logs"
    olmo_kind, olmo_line = last_event(logs, OLMO_JOB)
    olmo_pids = aq.olmo7b_processes(procs)
    olmo_ended = not olmo_pids and olmo_kind in ("END", "SKIP")
    queue_holders = [p for p in aq.lock_holders(logs / run_queue.QUEUE_LOCK) if p not in me]
    return {"olmo_pids": olmo_pids, "olmo_line": olmo_line, "olmo_ended": olmo_ended,
            "olmo_known": bool(olmo_pids) or olmo_ended,
            "olmo_lost": not olmo_pids and olmo_kind == "START",
            "queue_holders": queue_holders,
            "ready": olmo_ended and not queue_holders}


def jobs_left(root: Path, jobs_file: Path) -> list[str]:
    jobs = json.loads(jobs_file.read_text())
    return [j["name"] for j in jobs if not (root / "logs" / f"{j['name']}.done").exists()]


def refusals(root: Path, jobs_file: Path, procs, me, st: dict) -> list[str]:
    out = []
    if not jobs_file.exists():
        return [f"{jobs_file} is missing (git pull first)"]
    if p6c_queues(procs, me):
        out.append(f"a P6c queue already runs (pid {p6c_queues(procs, me)})")
    if aq.lock_holders(root / "logs" / OWN_LOCK):
        out.append(f"another p6c_after_olmo.py runs (logs/{OWN_LOCK})")
    if not jobs_left(root, jobs_file):
        out.append("every P6c job is done already (logs/<job>.done)")
    if not st["olmo_known"] and not st["olmo_lost"]:
        out.append(f"{OLMO_JOB} is neither running or waiting nor ended (no process, last progress line: "
                   f"{st['olmo_line'] or 'none'}): start run_all.sh first, or start P6c by hand")
    return out


def describe(st: dict) -> list[str]:
    def pids(x):
        return ", ".join(f"pid {p}, started {aq.started_utc(p)}" for p in x)
    if st["olmo_ended"]:
        olmo = "ended"
    elif st["olmo_pids"]:
        olmo = f"running or waiting ({pids(st['olmo_pids'])})"
    elif st["olmo_lost"]:
        olmo = "START line with no process (run_all.sh killed?)"
    else:
        olmo = "not running, not ended"
    held = f"held by pid {', '.join(map(str, st['queue_holders']))}" if st["queue_holders"] else "free"
    return [f"p1_olmo7b: {olmo}; last progress line: {st['olmo_line'] or 'none'}",
            f"logs/{run_queue.QUEUE_LOCK}: {held}"]


def check(root: Path, jobs_file: Path) -> int:
    procs, me = aq.processes(), {os.getpid()}
    st = state(root, procs, me)
    for ln in describe(st):
        print(ln)
    waiters = [p for p in aq.lock_holders(root / "logs" / OWN_LOCK)]
    print(f"p6c_after_olmo.py waiting: {'pid ' + ', '.join(map(str, waiters)) if waiters else 'none'}; P6c queue: "
          + (", ".join(f"pid {p}" for p in p6c_queues(procs, me)) or "not running"))
    if jobs_file.exists():
        left = jobs_left(root, jobs_file)
        print(f"P6c jobs not done: {len(left)} of {len(json.loads(jobs_file.read_text()))}"
              + (f" (next: {left[0]})" if left else ""))
    problems = refusals(root, jobs_file, procs, me, st) if not waiters else []
    if waiters:
        print("-> a waiter runs; it starts P6c " + ("now" if st["ready"] else "once the lines above say ended/free"))
    elif problems:
        print("-> a new waiter would not start: " + "; ".join(problems))
    else:
        print("-> a new waiter would " + ("start P6c at once" if st["ready"] else "wait, then start P6c"))
    stop = root / "results" / "p6c_small" / "STOP"
    if stop.exists():
        print(f"{stop} exists: P6c seed jobs will start no run")
    return 0


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the state and what it would do; change nothing")
    ap.add_argument("--jobs-file", default=JOBS_FILE, help=argparse.SUPPRESS)       # tests
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks, here and in the P6c queue")
    ap.add_argument("--root", default=str(HERE), help=argparse.SUPPRESS)           # tests
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    logs = root / "logs"
    jobs_file = Path(args.jobs_file)
    jobs_file = jobs_file if jobs_file.is_absolute() else root / jobs_file
    if args.check:
        return check(root, jobs_file)
    procs, me = aq.processes(), {os.getpid()}
    st = state(root, procs, me)
    problems = refusals(root, jobs_file, procs, me, st)
    if problems:
        say("not started: " + "; ".join(problems))
        return 2
    own = aq.flock(logs / OWN_LOCK, block=False)
    if own is None:
        say(f"not started: another p6c_after_olmo.py runs (logs/{OWN_LOCK})")
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
    say(f"pid {os.getpid()}; waiting until p1_olmo7b has ended and no queue runs, checking every {args.poll:g} s. "
        "Now: " + " | ".join(describe(st)))

    # 2. wait, reading only
    told, lost_told = None, 0.0
    while not st["ready"]:
        now = " | ".join(describe(st))
        key = (st["olmo_ended"], bool(st["olmo_pids"]), bool(st["queue_holders"]))
        if key != told:
            say("waiting: " + now)
            told = key
        if st["olmo_lost"] and time.time() - lost_told > LOST_REPORT_S:
            say("p1_olmo7b has a START line and no process (run_all.sh killed with -9?); still waiting for its END")
            lost_told = time.time()
        time.sleep(args.poll)
        st = state(root, aq.processes(), me)
    say("OLMo-7B has ended and no queue runs: " + " | ".join(describe(st)))
    if stop["sig"]:
        return 130

    # 3. the P6c queue
    out_file = logs / OUT_FILE
    cmd = [sys.executable, str(HERE / "run_queue.py"), "--jobs-file", str(jobs_file), "--poll", f"{args.poll:g}"]
    if root != HERE:
        cmd += ["--root", str(root)]
    for attempt in range(3):
        offset = out_file.stat().st_size if out_file.exists() else 0
        with open(out_file, "a") as log:
            log.write(f"### {aq.stamp()}  p6c_after_olmo.py: {' '.join(cmd)}\n")
            log.flush()
            child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        say(f"started the P6c queue (pid {child.pid}): {' '.join(cmd)}; its output in {out_file}")
        while child.poll() is None and not aq.child_holds_queue_lock(child, logs / run_queue.QUEUE_LOCK, out_file,
                                                                     offset):
            time.sleep(0.05)
        if child.poll() is None:
            break
        text = out_file.read_bytes()[offset:].decode(errors="replace")
        if "is already running" in text and attempt < 2 and not stop["sig"]:
            say(f"the P6c queue found logs/{run_queue.QUEUE_LOCK} taken for a moment; trying again in 5 s")
            time.sleep(5)
            continue
        break
    rc = child.wait()
    say(f"the P6c queue ended with exit {rc}")
    own.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
