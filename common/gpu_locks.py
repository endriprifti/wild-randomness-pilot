"""Locks between the jobs that share one GPU (run_all.sh and the run_lane.sh lanes next to it).

  logs/.p1_gpu.lock       one P1 job on the GPU at a time (p1_checkpoints.py)
  logs/.p3_gpu.lock       P3 (p3_fractional_memory.py) and P1's olmo7b, never together: olmo7b needs 31.5 GiB and
                          P3 10-13 GiB (python memory_check.py), more than the L40S's 44 GiB with CUDA contexts
  logs/.run_lane*.lock    held by each run_lane.sh for its whole run (logs/.run_lane.lock for P1 jobs,
                          logs/.run_lane.p3.lock for P3); olmo7b waits until no lane runs, so that it stays the
                          last job, as planned, instead of taking the P1 lock in the gap between two suites of a lane

Every lock is an exclusive flock held by an open file: the kernel releases it when the process ends, however it
ends. A job takes its locks before CUDA starts, so a waiting job holds no GPU memory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def take(path: Path, waiting_note: str, log=print):
    """Exclusive flock on `path`, waiting for it if needed. Returns (open file, seconds waited); (None, 0.0) where
    flock does not exist."""
    try:
        import fcntl
    except ImportError:
        return None, 0.0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    t0 = time.perf_counter()
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(waiting_note)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f, time.perf_counter() - t0


def is_held(path: Path) -> bool:
    """True if another process holds an flock on `path` (tested without keeping it)."""
    try:
        import fcntl
    except ImportError:
        return False
    if not Path(path).exists():
        return False
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def lanes_running(logs_dir: Path, own: str | None = None) -> list[Path]:
    """The run_lane.sh locks held now, apart from `own` (the lane this process runs in, PILOT_LANE_LOCK)."""
    own_path = Path(own).resolve() if own else None
    return [p for p in sorted(Path(logs_dir).glob(".run_lane*.lock"))
            if p.resolve() != own_path and is_held(p)]


def wait_for_lanes(logs_dir: Path, own: str | None = None, poll: float = 60.0, log=print) -> float:
    """Wait until no run_lane.sh runs (apart from our own lane); returns the seconds waited."""
    t0 = time.perf_counter()
    told = None
    while True:
        busy = lanes_running(logs_dir, own if own is not None else os.environ.get("PILOT_LANE_LOCK"))
        if not busy:
            return time.perf_counter() - t0
        names = ", ".join(p.name for p in busy)
        if names != told:
            log(f"waiting for run_lane.sh to finish ({names} held); checking every {poll:.0f} s")
            told = names
        time.sleep(poll)
