"""A third lane next to run_all.sh and the run_lane.sh lanes: the remaining reruns, each started as soon as the GPU
has room for it, with run_all.sh's bookkeeping.

Jobs, in this order (python run_queue.py --dry-run lists them with what each waits for):
  rerun_p5_floor10    P5 floor cells with checks every 10 steps up to 250      GPU  1.1 GiB
  rerun_p5_a12        P5 injected alpha = 1.2 cells, second reading, checks    GPU  1.1 GiB
  rerun_p5_refit      P5 refit of the grid with both reruns (after both)       CPU
  rerun_p5_h31_flags  H3.1 flags from the first run's h31.csv                  CPU
  rerun_p2_popqa      PopQA trends of the first P2 run                         CPU
  rerun_p2_bf16       P2 in bfloat16 (both models)                             GPU 16.5 GiB
  p4_transformer      run_all.sh's p4_transformer (same command and folder)    GPU 21.4 GiB
GPU peaks are python memory_check.py's (tensors only). Every job is resumable: P2, P4 and P5 save each finished
unit (a scheme, a training run, a grid cell) and a restart runs only the others; the CPU jobs take minutes and run
again whole. Each resumable job writes progress.json in its folder with the measured hours of every unit and the
hours left at the measured mean; python run_queue.py --status shows them.

What it writes, as run_lane.sh does: logs/progress.txt START/END lines tagged "(queue)"; logs/<job>.log (the job's
output, appended, with a header); logs/<job>.done when the job exits with 0 (a restarted queue, and run_all.sh for
p4_transformer, skip the job); results_archive/<job>.tar.gz and results_archive/pins.json after each job. A failed
job is retried once (--retries) when it fits again; a job whose `after` job failed does not run.

When a GPU job starts. Every process on the GPU is counted at 1.05 x its peak + 0.6 GiB (allocator rounding and
the CUDA context): the queue's running jobs, run_all.sh's running job, and each running lane at the largest peak of
the jobs it has left (P1 lane: p1_pythia, p1_polypythia, p1_olmo not done; P3 lane: p3_fractional). A job starts if
  (1) the sum with the new job, plus 1.5 GiB, is at most the GPU's 44.39 GiB;
  (2) for every lane with a job still to start: the others, plus that job's peak, plus the 4 GiB that run_lane.sh
      waits for, are at most 44.39 GiB, so a lane never waits for the queue;
  (3) nvidia-smi shows at least 1.05 x its peak + 0.6 GiB free.
Otherwise it waits and checks again every minute (--poll). CPU jobs start at once, one at a time. Jobs start in
list order whenever they fit (a later small job may start while an earlier large one waits).
python run_queue.py --combos lists which jobs fit together next to which lanes.

p1_olmo7b stays last and alone. It waits until no run_lane*.lock is held, and this queue holds
logs/.run_lane.queue.lock for its whole run (also while it waits), so olmo7b starts only after the queue ends.
If olmo7b is already on the GPU when the queue starts (logs/.p3_gpu.lock held with no P3 lane), the queue starts
no GPU job until olmo7b ends.

Usage, in the pilot folder inside the venv (CPUs shared with the P3 lane; P3 and these jobs each keep about one
CPU busy):
  nohup taskset -c 0-6 python run_queue.py > logs/run_queue.out 2>&1 &
  python run_queue.py --dry-run          # what would start now and why the others wait; runs nothing
  python run_queue.py --status           # jobs, units done, measured hours per unit, hours left
  python run_queue.py --combos           # which jobs fit together next to the lanes
  kill -TERM <pid of run_queue.py>       # stops the queue and its jobs (saved units stay); start it again to go on
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import gpu_locks  # noqa: E402

GPU_TOTAL_GIB = 44.39     # capacity CUDA reports on the CloudRift L40S (memory_check.L40S_GIB)
FRAG = 1.05               # allocator rounding on top of the tensor peak
CONTEXT_GIB = 0.6         # CUDA context and workspaces per process
MARGIN_GIB = 1.5          # left free when everything is at its peak
LANE_HEADROOM_GIB = 4.0   # free memory run_lane.sh waits for above a job's peak (LANE_HEADROOM_GIB there)

# peak GPU memory in GiB, tensors only (python memory_check.py, 2026-11-15 to 2026-11-17)
PEAK_GIB = {
    "p1_pythia": 16.3, "p1_polypythia": 7.7, "p1_olmo": 17.3, "p1_olmo7b": 31.5, "p3_fractional": 12.9,
    "p4_transformer": 21.4, "p4_resnet18": 2.5, "p2_w4a4": 16.5, "p5_escape": 1.1, "p5_escape_h31": 1.1,
    "rerun_p2_bf16": 16.5, "rerun_p5_floor10": 1.1, "rerun_p5_a12": 1.1, "p0_calibration": 0.0,
}
LANES = {".run_lane.lock": ("P1 lane", ["p1_pythia", "p1_polypythia", "p1_olmo"]),
         ".run_lane.p3.lock": ("P3 lane", ["p3_fractional"])}
QUEUE_LOCK = ".run_lane.queue.lock"


@dataclasses.dataclass
class Job:
    name: str
    args: list            # after the interpreter; --out <out> is added
    out: str
    gpu: float = 0.0      # peak GiB; 0 = CPU only
    after: tuple = ()     # jobs that must have ended with 0 first
    needs: tuple = ()     # input files that must exist

    @property
    def use(self) -> float:
        return use_gib(self.gpu)


JOBS = [
    Job("rerun_p5_floor10", ["p5_escape.py", "--floor-from", "results/p5_escape", "--alphas", "1.5,1.8,2.0",
                             "--fine-check", "10:250"], "results/p5_escape_floor10", PEAK_GIB["rerun_p5_floor10"],
        needs=("results/p5_escape/restarts.csv",)),
    Job("rerun_p5_a12", ["p5_escape.py", "--modes", "injected", "--alphas", "1.2", "--fine-check", "10:250",
                         "--second-reading", "--log-checks"], "results/p5_escape_a12", PEAK_GIB["rerun_p5_a12"]),
    Job("rerun_p5_refit", ["p5_escape.py", "--part", "refit", "--base", "results/p5_escape", "--replace",
                           "results/p5_escape_floor10,results/p5_escape_a12", "--device", "cpu"],
        "results/p5_escape_refit", after=("rerun_p5_floor10", "rerun_p5_a12"), needs=("results/p5_escape/run.json",)),
    Job("rerun_p5_h31_flags", ["p5_escape.py", "--part", "h31-flags", "--base", "results/p5_escape_h31",
                               "--device", "cpu"], "results/p5_escape_h31_flags",
        needs=("results/p5_escape_h31/h31.csv",)),
    Job("rerun_p2_popqa", ["-m", "common.popqa", "results/p2_w4a4"], "results/p2_w4a4_popqa",
        needs=("results/p2_w4a4/popqa_items.csv",)),
    Job("rerun_p2_bf16", ["p2_w4a4.py"], "results/p2_w4a4_bf16", PEAK_GIB["rerun_p2_bf16"]),
    Job("p4_transformer", ["p4_clipping.py", "--model", "transformer"], "results/p4_transformer",
        PEAK_GIB["p4_transformer"]),
]


def use_gib(peak: float) -> float:
    return FRAG * peak + CONTEXT_GIB if peak > 0 else 0.0


# ----------------------------------------------------------------------------- the admission rule
@dataclasses.dataclass
class Lane:
    name: str
    peak: float                    # largest peak of the jobs it has left (running or to start)
    pending_peak: float | None     # largest peak of the jobs it has still to start (None: none)


def admit(peak: float, lanes: list[Lane], others: list[tuple[str, float]], total: float = GPU_TOTAL_GIB,
          free: float | None = None) -> tuple[bool, str]:
    """Whether a GPU job of `peak` GiB may start next to `lanes` and `others` (name, peak) now running."""
    procs = [(ln.name, ln.peak) for ln in lanes] + list(others) + [("this job", peak)]
    used = sum(use_gib(p) for _, p in procs)
    if used + MARGIN_GIB > total:
        return False, (f"{used:.1f} + {MARGIN_GIB} GiB with it > {total} GiB ("
                       + ", ".join(f"{n} {p:g}" for n, p in procs if p > 0) + ")")
    for ln in lanes:
        if ln.pending_peak is not None:
            rest = used - use_gib(ln.peak)
            if rest + ln.pending_peak + LANE_HEADROOM_GIB > total:
                return False, (f"{ln.name}'s next job ({ln.pending_peak:g} GiB + {LANE_HEADROOM_GIB:g}) would not "
                               f"fit next to it ({rest:.1f} GiB)")
    if free is not None and free < use_gib(peak):
        return False, f"nvidia-smi shows {free:.1f} GiB free, it needs {use_gib(peak):.1f}"
    return True, f"fits: {used:.1f} + {MARGIN_GIB} of {total} GiB"


# ----------------------------------------------------------------------------- the machine's state
class State:
    def __init__(self, root: Path, total: float):
        self.root, self.logs, self.total = root, root / "logs", total
        self.progress = self.logs / "progress.txt"

    def done(self, job: str) -> bool:
        return (self.logs / f"{job}.done").exists()

    def progress_lines(self) -> list[list[str]]:
        if not self.progress.exists():
            return []
        return [ln.split() for ln in self.progress.read_text().splitlines() if ln.strip()]

    def running_from_progress(self, tag: str | None) -> list[str]:
        """Jobs with a START line and no END line after it; tag '(lane)', '(queue)' or None (run_all.sh's)."""
        open_: dict[str, bool] = {}
        for w in self.progress_lines():
            if len(w) < 3 or w[1] not in ("START", "END"):
                continue
            t = w[-1] if w[-1] in ("(lane)", "(queue)") else None
            if t != tag:
                continue
            if w[1] == "START":
                open_[w[2]] = True
            else:
                open_.pop(w[2], None)
        return list(open_)

    def run_all_active(self) -> bool:
        return gpu_locks.is_held(self.logs / ".run_all.lock")

    def olmo7b_on_gpu(self) -> bool:
        """olmo7b holds logs/.p3_gpu.lock while it runs; a P3 run holds it from inside the P3 lane."""
        return (gpu_locks.is_held(self.logs / ".p3_gpu.lock")
                and not gpu_locks.is_held(self.logs / ".run_lane.p3.lock"))

    def lanes(self) -> list[Lane]:
        out = []
        lane_running = set(self.running_from_progress("(lane)"))
        for p in sorted(self.logs.glob(".run_lane*.lock")):
            if p.name == QUEUE_LOCK or not gpu_locks.is_held(p):
                continue
            name, jobs = LANES.get(p.name, (f"lane {p.name}", LANES[".run_lane.lock"][1]))
            left = [j for j in jobs if not self.done(j)]
            pending = [j for j in left if j not in lane_running]
            out.append(Lane(name, max((PEAK_GIB[j] for j in left), default=0.0),
                            max((PEAK_GIB[j] for j in pending), default=None) if pending else None))
        return out

    def run_all_jobs(self) -> list[tuple[str, float]]:
        if not self.run_all_active():
            return []
        out = []
        for j in self.running_from_progress(None):
            if j == "p1_olmo7b" and not self.olmo7b_on_gpu():
                continue               # waiting for the lanes and this queue: no GPU memory yet
            out.append((f"run_all.sh {j}", PEAK_GIB.get(j, max(PEAK_GIB.values()))))
        return out


def gpu_free_gib() -> float | None:
    fake = os.environ.get("RUN_QUEUE_FREE_GIB")          # tests
    if fake is not None:
        return float(fake) if fake else None
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.split()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------------------- the queue
def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def uptime() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except OSError:
        return time.monotonic()


class Queue:
    def __init__(self, root: Path, jobs: list[Job], args):
        self.root, self.jobs, self.args = root, jobs, args
        self.state = State(root, args.gpu_total)
        self.logs = root / "logs"
        self.running: dict[str, tuple[subprocess.Popen, float, object]] = {}
        self.failed: dict[str, int] = {}
        self.tries: dict[str, int] = {}
        self.final: dict[str, int] = {}
        self.told: dict[str, str] = {}
        self.stopping = False

    # records -----------------------------------------------------------------
    def total_hours(self) -> float:
        s = 0.0
        for w in self.state.progress_lines():
            if len(w) > 1 and w[1] == "END" and "elapsed" in w:
                try:
                    s += float(w[w.index("elapsed") + 1])
                except (ValueError, IndexError):
                    pass
        return s

    def record(self, line: str) -> None:
        print(line, flush=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.state.progress, "a") as f:     # one write per line with O_APPEND
            f.write(line + "\n")

    def say(self, msg: str) -> None:
        print(f"{stamp()} run_queue.py: {msg}", flush=True)

    # selection ------------------------------------------------------------------
    def selected(self) -> list[Job]:
        only = set(filter(None, (self.args.only or "").split(",")))
        skip = set(filter(None, (self.args.skip or "").split(",")))
        return [j for j in self.jobs if (not only or j.name in only) and j.name not in skip]

    def waiting_reason(self, job: Job, cpu_busy: bool, extra=()) -> str | None:
        """None if the job may start now, else why it waits. extra: (name, peak) of jobs --dry-run would start
        first."""
        for d in job.after:
            if not self.state.done(d):
                return f"waits for {d}"
        missing = [n for n in job.needs if not (self.root / n).exists()]
        if missing:
            return "missing input " + ", ".join(missing)
        if job.gpu <= 0:
            return "another CPU job runs" if cpu_busy else None
        if self.state.olmo7b_on_gpu():
            return "p1_olmo7b is on the GPU (it runs alone)"
        others = (self.state.run_all_jobs() + [(n, self.job(n).gpu) for n in self.running if self.job(n).gpu > 0]
                  + list(extra))
        ok, why = admit(job.gpu, self.state.lanes(), others, self.args.gpu_total, gpu_free_gib())
        return None if ok else why

    def job(self, name: str) -> Job:
        return next(j for j in self.jobs if j.name == name)

    # running ------------------------------------------------------------------
    def env(self) -> dict:
        t = str(self.args.threads)
        return {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t,
                "OPENBLAS_NUM_THREADS": t, "NUMEXPR_NUM_THREADS": t, "LOKY_MAX_CPU_COUNT": t}

    def start(self, job: Job) -> None:
        cmd = [self.args.python, *job.args, "--out", job.out]
        log = open(self.logs / f"{job.name}.log", "a")
        try:
            git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=self.root, capture_output=True,
                                 text=True, timeout=10).stdout.strip() or "?"
        except (OSError, subprocess.SubprocessError):
            git = "?"
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "?"
        log.write(f"### {stamp()}  run_queue.py: {' '.join(cmd)}  (CPUs: {cpus}; GPU peak {job.gpu:g} GiB)\n"
                  f"### git {git}; free disk: {shutil.disk_usage(self.root).free / 2 ** 30:.0f} GiB\n")
        log.flush()
        self.record(f"{stamp()}  START  {job.name}  total {self.total_hours():.2f} h  (queue)")
        p = subprocess.Popen(cmd, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, env=self.env())
        (self.logs / f"{job.name}.queue.pid").write_text(str(p.pid))
        self.running[job.name] = (p, uptime(), log)
        self.tries[job.name] = self.tries.get(job.name, 0) + 1
        self.say(f"started {job.name} (pid {p.pid}, try {self.tries[job.name]}): {' '.join(cmd)}")

    def finish(self, name: str, rc: int, note: str = "") -> None:
        p, t0, log = self.running.pop(name)
        log.close()
        el = (uptime() - t0) / 3600
        self.record(f"{stamp()}  END    {name}  exit {rc}{note}  elapsed {el:.2f} h  "
                    f"total {self.total_hours() + el:.2f} h  (queue)")
        (self.logs / f"{name}.queue.pid").unlink(missing_ok=True)
        if rc == 0:
            (self.logs / f"{name}.done").write_text(stamp() + "\n")
        job = self.job(name)
        out = self.root / job.out
        arch = self.root / "results_archive"
        arch.mkdir(exist_ok=True)
        if out.exists():
            tmp = arch / f"{name}.tar.gz.queue.tmp"
            r = subprocess.run(["tar", "-czf", str(tmp), job.out, f"logs/{name}.log"], cwd=self.root,
                               capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(tmp, arch / f"{name}.tar.gz")
            else:
                self.say(f"archive of {name} failed: {r.stderr.strip()[-300:]}")
        pins = self.root / "pins.json"
        if pins.exists():
            shutil.copyfile(pins, arch / "pins.json.queue.tmp")
            os.replace(arch / "pins.json.queue.tmp", arch / "pins.json")
        if rc == 0:
            self.final[name] = 0
        elif self.stopping:
            pass
        elif self.tries.get(name, 0) <= self.args.retries:
            self.say(f"{name} exited with {rc}; it will run again (it goes on from its saved units) when it fits")
        else:
            self.final[name] = rc
            self.say(f"{name} exited with {rc}; not retried again (see logs/{name}.log)")

    def lost_jobs(self) -> int:
        """Jobs an earlier run_queue.py left: refuse to start while one still runs; close the records of the dead."""
        alive = []
        for job in self.jobs:
            pid_file = self.logs / f"{job.name}.queue.pid"
            if not pid_file.exists():
                continue
            pid = int(pid_file.read_text().strip() or 0)
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            except OSError:
                cmdline = ""
            if pid and job.out in cmdline:
                alive.append(f"{job.name} (pid {pid})")
                continue
            if job.name in self.state.running_from_progress("(queue)"):
                self.record(f"{stamp()}  END    {job.name}  exit 255 (lost: an earlier run_queue.py stopped)  "
                            f"elapsed 0.00 h  total {self.total_hours():.2f} h  (queue)")
            pid_file.unlink(missing_ok=True)
        if alive:
            print(f"run_queue.py: still running from an earlier run_queue.py: {', '.join(alive)}; stop them "
                  f"(kill -TERM <pid>) or wait for them, then start run_queue.py again", file=sys.stderr)
            return 1
        return 0

    def stop(self, signum, frame) -> None:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)            # a second signal does not interrupt the cleanup
        self.stopping = True
        self.say(f"signal {signum}: stopping {', '.join(self.running) or 'nothing'}")
        for name, (p, _, _) in list(self.running.items()):
            try:
                p.terminate()
            except OSError:
                pass
        for name, (p, _, _) in list(self.running.items()):
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            self.finish(name, 130, " (interrupted)")
        sys.exit(130)

    def run(self) -> int:
        todo = [j for j in self.selected()]
        for j in todo:
            if self.state.done(j.name):
                self.record(f"{stamp()}  SKIP   {j.name}  (finished OK: {(self.logs / f'{j.name}.done').read_text().strip()}; "
                            f"delete logs/{j.name}.done to rerun)  (queue)")
        pending = [j for j in todo if not self.state.done(j.name)]
        while pending or self.running:
            for name, (p, _, _) in list(self.running.items()):
                rc = p.poll()
                if rc is not None:
                    self.finish(name, rc)
            for j in list(pending):
                if j.name in self.running:
                    continue
                if j.name in self.final:
                    pending.remove(j)
                    continue
                dead = [d for d in j.after if self.final.get(d, 0) != 0 or
                        (d not in {x.name for x in todo} and not self.state.done(d))]
                if dead:
                    self.say(f"{j.name} not run: {', '.join(dead)} did not end with 0")
                    self.final[j.name] = -1
                    pending.remove(j)
                    continue
                cpu_busy = any(self.job(n).gpu <= 0 for n in self.running)
                why = self.waiting_reason(j, cpu_busy)
                if why is None:
                    self.start(j)
                elif why.startswith("missing input"):            # (after its dependencies ended with 0)
                    self.say(f"{j.name} not run: {why}")
                    self.final[j.name] = -2
                    pending.remove(j)
                elif self.told.get(j.name) != why:
                    self.say(f"{j.name} waits: {why}")
                    self.told[j.name] = why
            if not pending and not self.running:
                break
            time.sleep(self.args.poll)
            pending = [j for j in pending if j.name not in self.final]
        bad = {n: rc for n, rc in self.final.items() if rc != 0}
        self.say("done. " + ("All jobs ended with 0." if not bad else
                             "Not finished: " + ", ".join(f"{n} ({'not run' if rc < 0 else f'exit {rc}'})"
                                                         for n, rc in bad.items())))
        return 1 if bad else 0


# ----------------------------------------------------------------------------- reports
def status(root: Path, jobs: list[Job], args) -> int:
    st = State(root, args.gpu_total)
    q = Queue(root, jobs, args)
    running = set(st.running_from_progress("(queue)"))
    free = gpu_free_gib()
    lanes = st.lanes()
    print(f"GPU: {args.gpu_total} GiB; nvidia-smi free: {'?' if free is None else f'{free:.1f}'} GiB; "
          f"lanes: {', '.join(f'{ln.name} (peak {ln.peak:g}, next {ln.pending_peak})' for ln in lanes) or 'none'}; "
          f"run_all.sh on the GPU: {', '.join(n for n, _ in st.run_all_jobs()) or 'nothing'}"
          + ("; p1_olmo7b ON THE GPU" if st.olmo7b_on_gpu() else ""))
    print(f"{'job':20} {'state':10} {'GPU':>5} {'units':>9} {'h/unit':>7} {'h left':>7}  note")
    sim, sim_cpu = [], False            # jobs a queue started now would start, in order
    for j in q.selected():
        state = "done" if st.done(j.name) else ("running" if j.name in running else "pending")
        units = hu = left = ""
        note = ""
        prog = root / j.out / "progress.json"
        if prog.exists():
            try:
                d = json.loads(prog.read_text())
                units = f"{d.get('units_done')}/{d.get('units_total')}"
                if d.get("mean_unit_hours") is not None:
                    hu, left = f"{d['mean_unit_hours']:.2f}", f"{d['hours_left']:.1f}"
                    note = f"measured on {d.get('measured_units')} unit(s)"
                else:
                    note = "no unit measured yet"
            except (ValueError, OSError):
                note = "progress.json unreadable"
        if state == "pending":
            why = q.waiting_reason(j, sim_cpu, sim)
            if why is None:
                if j.gpu > 0:
                    sim.append((j.name, j.gpu))
                else:
                    sim_cpu = True
            note = (note + "; " if note else "") + (why or "would start now")
        print(f"{j.name:20} {state:10} {j.gpu:5.1f} {units:>9} {hu:>7} {left:>7}  {note}")
    return 0


def combos(args) -> list[dict]:
    """Which queue GPU jobs fit together next to each set of lanes (the admission rule, everything at its peak)."""
    gpu_jobs = [j for j in JOBS if j.gpu > 0]
    p1_next = Lane("P1 lane, next job to start", PEAK_GIB["p1_olmo"], PEAK_GIB["p1_olmo"])
    p1_last = Lane("P1 lane, last job running", PEAK_GIB["p1_olmo"], None)
    p3 = Lane("P3 lane", PEAK_GIB["p3_fractional"], None)
    scenarios = [[p1_next, p3], [p1_last, p3], [p3], [p1_next], [p1_last], []]
    out = []
    for lanes in scenarios:
        fits = []
        for k in range(len(gpu_jobs), 0, -1):
            for sub in itertools.combinations(gpu_jobs, k):
                ok = all(admit(j.gpu, lanes, [(o.name, o.gpu) for o in sub if o is not j], args.gpu_total)[0]
                         for j in sub)
                if ok and not any({j.name for j in sub} <= {j.name for j in f} for f in fits):
                    fits.append(sub)
        alone_ok = [j.name for j in gpu_jobs if admit(j.gpu, lanes, [], args.gpu_total)[0]]
        used = sum(use_gib(ln.peak) for ln in lanes)
        out.append({"lanes": " + ".join(ln.name for ln in lanes) or "no lane", "lanes_GiB": used,
                    "largest sets that fit": ["{" + ", ".join(j.name for j in s) + "}" for s in fits] or ["none"],
                    "GiB": [used + sum(j.use for j in s) for s in fits], "fits alone": alone_ok})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and why each waits; run nothing")
    ap.add_argument("--status", action="store_true", help="jobs, units done, measured hours per unit, hours left")
    ap.add_argument("--combos", action="store_true", help="which jobs fit together next to which lanes")
    ap.add_argument("--only", default=None, help="comma-separated jobs to run (default: all)")
    ap.add_argument("--skip", default=None, help="comma-separated jobs to leave out")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks (default 60)")
    ap.add_argument("--retries", type=int, default=1, help="times a failed job runs again (default 1)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("QUEUE_THREADS", 2)),
                    help="threads per job's thread pools (default 2)")
    ap.add_argument("--gpu-total", type=float, default=GPU_TOTAL_GIB, help="GPU memory in GiB (default: the L40S)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the jobs (default: this one)")
    ap.add_argument("--root", default=str(HERE), help=argparse.SUPPRESS)          # tests
    ap.add_argument("--jobs-file", default=None, help=argparse.SUPPRESS)          # tests: JSON list of jobs
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    jobs = JOBS
    if args.jobs_file:
        jobs = [Job(**{**d, "after": tuple(d.get("after", ())), "needs": tuple(d.get("needs", ()))})
                for d in json.loads(Path(args.jobs_file).read_text())]
    names = {j.name for j in jobs}
    for n in filter(None, f"{args.only or ''},{args.skip or ''}".split(",")):
        if n not in names:
            print(f"run_queue.py: unknown job {n!r}; jobs: {' '.join(sorted(names))}", file=sys.stderr)
            return 2
    if args.combos:
        print(f"Each process counted at {FRAG} x peak + {CONTEXT_GIB} GiB; {MARGIN_GIB} GiB kept free; a lane with a "
              f"job to start keeps that job's peak + {LANE_HEADROOM_GIB:g} GiB; GPU {args.gpu_total} GiB. Largest "
              f"sets of queue jobs that fit, with the GiB counted (lanes included):")
        for c in combos(args):
            print(f"{c['lanes']} ({c['lanes_GiB']:.1f} GiB):")
            for name, g in zip(c["largest sets that fit"], c["GiB"]):
                print(f"  {name}  {g:.1f} GiB")
        return 0
    if args.status or args.dry_run:
        return status(root, jobs, args)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    lock = open(root / "logs" / QUEUE_LOCK, "a")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"run_queue.py is already running (logs/{QUEUE_LOCK})", file=sys.stderr)
        return 1
    except ImportError:
        pass
    q = Queue(root, jobs, args)
    if q.lost_jobs():
        return 1
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, q.stop)
    q.say(f"started (pid {os.getpid()}); holding logs/{QUEUE_LOCK}, so p1_olmo7b waits for this queue")
    return q.run()


if __name__ == "__main__":
    sys.exit(main())
