"""Start the small P6 design on the L40S as soon as the running run_queue.py has ended, before run_all.sh's
p1_olmo7b (added 2026-11-18; new file: run_all.sh, run_lane.sh, run_queue.py, p1_checkpoints.py and common/ are
unchanged).

  setsid nohup nice -n 10 taskset -c 0-10 python p6_after_queue.py > logs/p6_after_queue.out 2>&1 < /dev/null &
  python p6_after_queue.py --check            # what it would wait for, the locks and the running jobs; changes nothing
  kill -TERM $(cat logs/p6_after_queue.pid)   # while waiting: stop waiting; after the start: stop P6 (saved runs stay)

What it does, in order:
  1. Refuses to start (exit 2) unless a run_queue.py runs now (logs/.run_lane.queue.lock held), it is not already a
     P6 queue, no other p6_after_queue.py runs, p1_olmo7b is not on the GPU and the jobs file is there.
  2. Waits for logs/.run_lane.queue.lock with a blocking flock. run_queue.py holds that lock for its whole run and
     ends only after every job it started has ended (rerun_p2_bf16 and p4_transformer included); the kernel hands the
     lock to this process when the queue's process ends. Waiting changes nothing for the running queue, the lanes or
     p1_olmo7b: a blocked flock request is invisible to their non-blocking checks.
  3. Holding the queue lock (so p1_olmo7b keeps waiting), waits until no job of the old queue is alive: neither a
     logs/<job>.queue.pid of run_queue.py's jobs with its process alive, nor any python process running p2_w4a4.py,
     p4_clipping.py, p5_escape.py or common.popqa (a queue killed with -9 leaves its jobs running).
  4. Hand-over: takes logs/.run_lane.p6.lock, releases the queue lock, starts
     run_queue.py --jobs-file p6_queue_small.json (its output in logs/run_queue_p6.out), and releases
     .run_lane.p6.lock once that queue holds logs/.run_lane.queue.lock. p1_olmo7b waits while any logs/.run_lane*.lock
     is held (common/gpu_locks.wait_for_lanes, a check every 60 s), and one of the two is held at every moment.
  5. Waits for the P6 queue (timing job, then the runs, whose seed count the budget guard sets from the measured L40S
     timing, then the analysis) and exits with its exit code. The queue lock is then free, and p1_olmo7b starts at
     its next check once the P1 and P3 lanes have ended.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_queue  # noqa: E402  (stdlib and common.gpu_locks only; nothing runs on import)
from common import gpu_locks  # noqa: E402

QUEUE_LOCK = run_queue.QUEUE_LOCK                 # .run_lane.queue.lock
HANDOVER_LOCK = ".run_lane.p6.lock"               # matches p1_olmo7b's .run_lane*.lock; held only during the hand-over
OWN_LOCK = ".p6_after_queue.lock"                 # not a .run_lane* name: invisible to p1_olmo7b and the queues
OLD_SCRIPTS = ("p2_w4a4.py", "p4_clipping.py", "p5_escape.py")
OLD_MODULES = ("common.popqa",)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(msg: str) -> None:
    print(f"{stamp()} p6_after_queue.py: {msg}", flush=True)


# ----------------------------------------------------------------------------- reading the machine (read-only)
def processes() -> dict[int, list[str]]:
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        argv = [a for a in argv if a]
        if argv:
            out[int(d.name)] = argv
    return out


def is_python(argv: list[str]) -> bool:
    return Path(argv[0]).name.startswith("python")


def runs_script(argv: list[str], script: str) -> bool:
    return is_python(argv) and any(Path(a).name == script for a in argv[1:3] if not a.startswith("-"))


def started_utc(pid: int) -> str:
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(ln.split()[1]) for ln in Path("/proc/stat").read_text().splitlines()
                     if ln.startswith("btime"))
        return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(btime + ticks / os.sysconf("SC_CLK_TCK")))
    except (OSError, ValueError, IndexError, StopIteration):
        return "?"


def lock_holders(path: Path) -> list[int]:
    """Pids holding an flock on `path`, from /proc/locks (read-only; [] if it cannot tell)."""
    try:
        st = path.stat()
        lines = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        w = ln.split()
        if len(w) < 6 or w[1] == "->" or w[1] != "FLOCK":
            continue
        try:
            maj, mnr, ino = w[5].split(":")
            if int(ino) == st.st_ino and int(maj, 16) == os.major(st.st_dev) and int(mnr, 16) == os.minor(st.st_dev):
                out.append(int(w[4]))
        except ValueError:
            continue
    return out


def queue_jobs_file(argv: list[str]) -> str | None:
    return argv[argv.index("--jobs-file") + 1] if "--jobs-file" in argv[:-1] else None


def running_queues(procs: dict[int, list[str]], me: set[int]) -> list[tuple[int, list[str]]]:
    report = {"--status", "--dry-run", "--combos"}
    return [(p, a) for p, a in procs.items() if p not in me and runs_script(a, "run_queue.py")
            and not report & set(a)]


def old_jobs_alive(root: Path, procs: dict[int, list[str]], me: set[int]) -> list[str]:
    """Jobs of the old queue still running: run_queue.py's pid files (its rule in lost_jobs) and any python process
    running one of their scripts."""
    alive = []
    for job in run_queue.JOBS:
        f = root / "logs" / f"{job.name}.queue.pid"
        try:
            pid = int(f.read_text().strip() or 0)
        except (OSError, ValueError):
            continue
        argv = procs.get(pid)
        if pid and argv and job.out in " ".join(argv):
            alive.append(f"{job.name} (pid {pid}, {f.name})")
    for pid, argv in procs.items():
        if pid in me or not is_python(argv):
            continue
        hit = [s for s in OLD_SCRIPTS if runs_script(argv, s)]
        hit += [m for m in OLD_MODULES if "-m" in argv and m in argv]
        if hit and not any(f"pid {pid}," in a for a in alive):
            alive.append(f"{hit[0]} (pid {pid}: {' '.join(argv)[:160]})")
    return alive


def olmo7b_processes(procs: dict[int, list[str]]) -> list[int]:
    return [p for p, a in procs.items() if runs_script(a, "p1_checkpoints.py") and "olmo7b" in " ".join(a)]


OLMO7B_WAITS_FOR_LANES = "3463b1c"     # first commit whose p1_checkpoints.gpu_lock waits for every .run_lane*.lock


def git(root: Path, *a) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None


def log_header(log: Path) -> tuple[str, str]:
    """(start line, git commit) of the last '### ' header that run_all.sh, run_lane.sh and run_queue.py write into
    logs/<job>.log when they start a job: the code the job loaded."""
    start, commit = "?", "?"
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return start, commit
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("### git "):
            commit = lines[i].split()[2].rstrip(";")
            start = lines[i - 1][4:].strip()[:120] if i > 0 and lines[i - 1].startswith("### ") else "?"
            break
    return start, commit


def check(root: Path, jobs_file: Path) -> int:
    """Print what this script would wait for now, the lock holders, the running jobs with the commit each loaded,
    and the old queue's jobs. Changes nothing."""
    logs = root / "logs"
    procs = processes()
    me = {os.getpid()}
    print(f"{stamp()} p6_after_queue.py --check (read-only)")
    print("locks:")
    for name in (QUEUE_LOCK, ".run_lane.lock", ".run_lane.p3.lock", HANDOVER_LOCK, ".p1_gpu.lock", ".p3_gpu.lock",
                 ".run_all.lock", OWN_LOCK):
        p = logs / name
        held = gpu_locks.is_held(p)
        who = ", ".join(f"pid {h} ({' '.join(procs.get(h, ['?']))[:110]})" for h in lock_holders(p)) if held else ""
        print(f"  logs/{name}: {'HELD' if held else 'free'}{' by ' + who if who else ''}")
    st = run_queue.State(root, run_queue.GPU_TOTAL_GIB)
    warnings = []
    print("running jobs (START without END in logs/progress.txt) and the commit each loaded (logs/<job>.log):")
    for tag, who in ((None, "run_all.sh"), ("(lane)", "run_lane.sh"), ("(queue)", "run_queue.py")):
        for job in st.running_from_progress(tag):
            start, commit = log_header(logs / f"{job}.log")
            print(f"  {job} ({who}): {start}; git {commit}")
            if job == "p1_olmo7b":
                r = git(HERE, "merge-base", "--is-ancestor", OLMO7B_WAITS_FOR_LANES, commit)
                if r is None or r.returncode not in (0, 1):
                    warnings.append(f"cannot tell whether p1_olmo7b's commit {commit} includes "
                                    f"{OLMO7B_WAITS_FOR_LANES} (git); check it by hand")
                elif r.returncode == 1:
                    warnings.append(f"p1_olmo7b loaded {commit}, older than {OLMO7B_WAITS_FOR_LANES}: it does not "
                                    "wait for the lanes and the queue, so it could start next to P6; do not use this")
    for pid, argv in running_queues(procs, me):
        jf = queue_jobs_file(argv)
        print(f"  run_queue.py pid {pid}, started {started_utc(pid)}, jobs {'(built-in)' if jf is None else jf}")
    for pid in olmo7b_processes(procs):
        print(f"  p1_olmo7b python pid {pid}, started {started_utc(pid)}")
    olog = logs / "p1_olmo7b.log"
    if olog.exists():
        waits = [ln for ln in olog.read_text(errors="replace").splitlines() if "waiting for" in ln]
        if waits:
            print(f"  logs/p1_olmo7b.log, last wait line: {waits[-1].strip()[:220]}")
    r = git(HERE, "log", "--format=%h", "--", "run_queue.py")
    if r is not None and r.returncode == 0:
        print(f"  run_queue.py versions in git: {' '.join(r.stdout.split()) or '?'} (one version: the running queue "
              "runs this file's code)")
    print("old queue (run_queue.py's built-in jobs):")
    alive = old_jobs_alive(root, procs, me)
    for job in run_queue.JOBS:
        state = "done" if st.done(job.name) else ("running" if job.name in st.running_from_progress("(queue)")
                                                  else "not done")
        print(f"  {job.name}: {state}")
    print(f"  alive now: {', '.join(alive) or 'none'}")
    print(f"p1_olmo7b on the GPU: {'YES' if st.olmo7b_on_gpu() else 'no'}")
    r = git(HERE, "log", "-1", "--format=%h %cI")
    print(f"git HEAD {r.stdout.strip() if r is not None else '?'}; {jobs_file.name} "
          f"{'present' if jobs_file.exists() else 'MISSING'}")
    problems = refusals(root, jobs_file, procs, me, st)
    for w in warnings:
        print(f"WARNING: {w}")
    print("verdict: " + ("would wait for the running queue, then start P6" if not problems else
                         "would refuse: " + "; ".join(problems)))
    return 0


def refusals(root: Path, jobs_file: Path, procs, me, st) -> list[str]:
    logs = root / "logs"
    out = []
    if not jobs_file.exists():
        out.append(f"{jobs_file} is missing (git pull first)")
    if not gpu_locks.is_held(logs / QUEUE_LOCK):
        out.append(f"no run_queue.py runs (logs/{QUEUE_LOCK} is free): start the P6 queue directly")
    for pid, argv in running_queues(procs, me):
        jf = queue_jobs_file(argv)
        if jf and "p6_" in Path(jf).name:
            out.append(f"a P6 queue already runs (pid {pid}, {jf})")
    if st.olmo7b_on_gpu():
        out.append("p1_olmo7b is on the GPU already")
    if gpu_locks.is_held(logs / OWN_LOCK):
        out.append(f"another p6_after_queue.py runs (logs/{OWN_LOCK})")
    return out


# ----------------------------------------------------------------------------- the hand-over
def flock(path: Path, block: bool):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
    except BlockingIOError:
        f.close()
        return None
    return f


def child_holds_queue_lock(child: subprocess.Popen, lock: Path, out_file: Path, offset: int) -> bool:
    if child.pid in lock_holders(lock):
        return True
    try:
        with open(out_file, "rb") as f:
            f.seek(offset)
            return f"started (pid {child.pid}); holding logs/{QUEUE_LOCK}".encode() in f.read()
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the state and what it would do; change nothing")
    ap.add_argument("--jobs-file", default="p6_queue_small.json", help="P6 queue file (default the small design)")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks, here and in the P6 queue")
    ap.add_argument("--queue-args", default="", help="further arguments for the P6 run_queue.py (space-separated)")
    ap.add_argument("--root", default=str(HERE), help=argparse.SUPPRESS)          # tests
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    logs = root / "logs"
    jobs_file = Path(args.jobs_file)
    jobs_file = jobs_file if jobs_file.is_absolute() else root / jobs_file
    if args.check:
        return check(root, jobs_file)
    procs, me = processes(), {os.getpid()}
    st = run_queue.State(root, run_queue.GPU_TOTAL_GIB)
    problems = refusals(root, jobs_file, procs, me, st)
    if problems:
        say("not started: " + "; ".join(problems))
        return 2
    own = flock(logs / OWN_LOCK, block=False)
    if own is None:
        say(f"not started: another p6_after_queue.py runs (logs/{OWN_LOCK})")
        return 2
    (logs / "p6_after_queue.pid").write_text(f"{os.getpid()}\n")
    queues = running_queues(procs, me)
    say(f"pid {os.getpid()}; waiting for the running run_queue.py to end ("
        + (", ".join(f"pid {p}, started {started_utc(p)}" for p, _ in queues) or "its process not found")
        + f"; logs/{QUEUE_LOCK} held). Then: {jobs_file.name}")

    # 2. the queue lock, handed over by the kernel when the queue's process ends
    qlock = flock(logs / QUEUE_LOCK, block=True)
    say(f"the queue has ended; holding logs/{QUEUE_LOCK} (p1_olmo7b keeps waiting)")
    # 3. no job of the old queue may still run
    told = None
    while True:
        alive = old_jobs_alive(root, processes(), me)
        if not alive:
            break
        if alive != told:
            say("old queue jobs still running: " + ", ".join(alive) + f"; checking every {args.poll:g} s")
            told = alive
        time.sleep(args.poll)
    if st.olmo7b_on_gpu():
        say("p1_olmo7b took the GPU as the queue ended; the P6 queue starts now and waits for it (run_queue.py's rule)")

    # 4. hand-over: one of the two .run_lane* locks is held at every moment
    hand = flock(logs / HANDOVER_LOCK, block=False)
    if hand is None:
        say(f"logs/{HANDOVER_LOCK} is held by another process; going on with the queue lock released only after the "
            "P6 queue starts")
    out_file = logs / "run_queue_p6.out"
    cmd = [sys.executable, str(HERE / "run_queue.py"), "--jobs-file", str(jobs_file), "--poll", f"{args.poll:g}",
           *args.queue_args.split()]
    if root != HERE:
        cmd += ["--root", str(root)]
    child = None
    stop = {"sig": None}

    def forward(signum, frame):
        stop["sig"] = signum
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    if stop["sig"]:
        return 130
    for attempt in range(3):
        offset = out_file.stat().st_size if out_file.exists() else 0
        log = open(out_file, "a")
        log.write(f"### {stamp()}  p6_after_queue.py: {' '.join(cmd)}\n")
        log.flush()
        if qlock is not None:
            qlock.close()                  # released; HANDOVER_LOCK stays held until the P6 queue holds it
            qlock = None
        child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        say(f"started the P6 queue (pid {child.pid}): {' '.join(cmd)}; its output in {out_file}")
        while child.poll() is None and not child_holds_queue_lock(child, logs / QUEUE_LOCK, out_file, offset):
            time.sleep(0.05)
        log.close()
        if child.poll() is None:
            break
        text = out_file.read_bytes()[offset:].decode(errors="replace")
        if "is already running" in text and attempt < 2 and not stop["sig"]:
            say(f"the P6 queue found logs/{QUEUE_LOCK} taken for a moment (another process tested it); "
                "trying again in 1 s")
            time.sleep(1)
            continue
        break
    if hand is not None:
        hand.close()
    if child.poll() is None:
        say(f"the P6 queue holds logs/{QUEUE_LOCK}; logs/{HANDOVER_LOCK} released. p1_olmo7b waits for this queue")
    rc = child.wait()
    say(f"the P6 queue ended with exit {rc}" + (f" (signal {stop['sig']} forwarded)" if stop["sig"] else "")
        + f"; logs/{QUEUE_LOCK} is free, so p1_olmo7b starts at its next check once the lanes have ended")
    own.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
