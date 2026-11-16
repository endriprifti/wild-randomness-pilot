"""P1: measurements on public checkpoints, no training (Pythia, PolyPythias, OLMo).

Per checkpoint (log-spaced steps):
  alpha, beta, gamma  stable fit to minibatch gradient noise at B = coupling_B (per-example
                      gradients of the last block, projected on fixed random directions by JVP)
  zeta (both tails)   CSN fit of the same noise, with the sec. 7.1 reading label
  alpha_ESD           median over layers of the ESD density exponent (+ per-layer CSV)
  K, Kurt0, outliers  activations of every layer on a fixed probe set, with zeta of |activation|
  Delta h_rep         width of the final-layer representations of the probe set (dimension clause)
  H_tok               EXPLORATORY: Hurst exponent of the per-token loss series along long
                      documents, with its shuffled baseline. This is NOT H_grad: checkpoints carry
                      no per-step gradient series, so H_grad comes only from P4 (appendix, P1).
At the final step: eval loss for a_N (one fit per suite and seed = one family).
H7.1 transformer arm: at 1, 10, 50, 100% of training, pools of n_blocks * max(B) examples for
Pythia-14M (1,600 blocks) and coarse sizes (200 blocks), random and Hessian-block directions.
Finally an exploratory coupling matrix across seeds within each (suite, size) cell.

Suites (config.py): pythia (seed 0, 14M-1.4B); polypythia = pythia-{size}-seed1..9 (initialization and
batch composition vary jointly; run-to-run variance); polypythia_data / polypythia_weight = the 160M
data-seed1-3 and weight-seed1-3 runs (data order alone, initialization alone); olmo = OLMo 2 1B with its
stage-1 checkpoints; olmo7b = OLMo 2 7B, OPTIONAL and never run by default (L40S, last job).
Every model is loaded at a commit pinned in pins.json; checkpoints.csv records commit and commit date. For
pythia-14m/31m (re-created with the standard-Pile models on 24 February 2026; the deduplicated ones moved to
pythia-14m-deduped / pythia-31m-deduped) two checks: dedup_check flags a commit whose model.safetensors has the
same LFS sha256 as the same revision of the -deduped repository (Hub API metadata, no download), and
name_switch_check flags a commit dated before 24 February 2026. --pin-only runs both for every pinned name and
writes name_switch_check.csv.
Loss-spike check (PolyPythias 410M seeds 3 and 4 had loss spikes), two passes:
  (a) probe loss only, at every 1,000-step checkpoint, for seeds 3 and 4 and one stable seed; a jump is a
      loss increase, net of the stable seed's change, with robust z > 3.5 over the run's intervals;
  (b) alpha, alpha_ESD and K for all nine 410M seeds at every 1,000-step checkpoint within 5,000 steps
      either side of each jump; a flagged seed's change counts as a jump if it exceeds every other seed's
      and is more than 3.5 robust z above them. spike_check.csv holds both passes and the jump steps.
Each checkpoint is deleted from the Hugging Face cache after it is read (--keep-cache keeps them).

GPU pass and CPU statistics (2026-11-16): the main process runs each checkpoint's GPU measurements, saves what the
statistics need under checkpoints/<pass>/<repo>@<revision>/ and goes on to the next checkpoint; the statistics
(stable and CSN fits, alpha_ESD, act_zeta, Delta h_rep, H_tok surrogates, H7.1 fits and bootstrap) run in a pool of
processes (common/p1_tasks.py), one per CPU the job may use (--workers, --cpus-after, logs/p1_cpus). A checkpoint
whose statistics are saved gets result.json; checkpoints.csv and h71_summary.csv are rewritten after each one. The
same command resumes: finished checkpoints are read back, the rest measured. Progress lines per measurement in the
log, seconds in timings.csv. H7.1 runs at one checkpoint per stage (the nearest to 1, 10, 50, 100% of training).

Outputs (results/p1_checkpoints/[smoke/]): checkpoints.csv, esd_layers.csv, activations_layers.csv,
h71_per_B.csv, h71_summary.csv, scaling.csv, seed_variance.csv, spike_check.csv, coupling_pairs.csv,
coupling.json, timings.csv, checkpoints/ (per checkpoint: samples/, stats/, row.json, result.json), run.json;
with --pin-only: name_switch_check.csv and run.json

Commands:
  python p1_checkpoints.py --pin-only            # resolve and record every commit first (CPU is enough);
                                                 # run.json then gives the download volume of each pass
  python p1_checkpoints.py --suite pythia
  python p1_checkpoints.py --suite polypythia,polypythia_data,polypythia_weight   # includes both spike passes
  python p1_checkpoints.py --suite polypythia --spike-only                      # only the spike passes
  python p1_checkpoints.py --suite olmo           # OLMo 2 1B, fits the RTX 5090
  python p1_checkpoints.py --suite olmo7b         # OPTIONAL, last: OLMo 2 7B on the L40S
  python p1_checkpoints.py --smoke
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import contextlib
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from common import coupling, gradients, hf, lm, p1_tasks, runtime, spectra, stable


def suite_models(cfg, suite):
    s = cfg["suites"][suite]
    for size in s["sizes"]:
        for seed in s["seeds"]:
            repo = s["repo"].format(size=size, seed=seed)
            yield size, seed, repo, s.get("variant", "base")


def default_suites(cfg) -> list[str]:
    return [k for k, v in cfg["suites"].items() if not v.get("optional")]


def wanted_steps(cfg, suite: str, size: str, steps_avail: dict) -> list[int]:
    """Steps of the main grid for one model (the loss-spike passes choose their own)."""
    s = cfg["suites"][suite]
    final = max(steps_avail)
    if "step_fracs" in s:   # OLMo: fractions of the last pretraining checkpoint
        wanted = [max(1, int(round(f * final))) for f in s["step_fracs"]]
    else:
        wanted = list(cfg["steps"])
    h71 = cfg["h71"]
    if size in h71["full_res_sizes"] + h71["coarse_sizes"] and suite == "pythia":
        wanted += [int(round(st * final)) for st in h71.get("stages", cfg["stages"])]
    return sorted(set(wanted))


def scan_grid(sc: dict) -> list[int]:
    lo, hi = sc["scan_range"]
    return list(range(lo, hi + 1, sc["scan_every"]))


def scan_steps(avail: dict, sc: dict) -> list[int]:
    """Pass (a) steps: every released checkpoint on the scan grid (multiples of scan_every in scan_range)."""
    lo, hi = sc["scan_range"]
    return [s for s in sorted(avail) if lo <= s <= hi and s % sc["scan_every"] == 0]


def loss_jumps(df: pd.DataFrame, sc: dict) -> list[dict]:
    """Pass (a). For each flagged seed, the change in probe loss between consecutive scan checkpoints minus
    the reference seeds' mean change over the same interval (this removes the common downward trend);
    robust z = (excess - median) / (1.4826 MAD) over all intervals of that seed. A jump is a loss
    increase with z > sc['robust_z']."""
    if not len(df):
        return []
    d = df[(df.suite == sc["suite"]) & (df["size"] == sc["size"])]
    piv = d.pivot_table(index="step", columns="seed", values="probe_loss", aggfunc="first").sort_index()
    steps = piv.index.to_list()
    diffs = piv.diff()
    refs = [r for r in sc["reference_seeds"] if r in diffs.columns]
    ref = diffs[refs].mean(axis=1) if refs else pd.Series(0.0, index=diffs.index)
    out = []
    for s in sc["flagged_seeds"]:
        if s not in diffs.columns:
            continue
        excess = (diffs[s] - ref).iloc[1:]
        ok = excess.dropna().to_numpy(float)
        med = float(np.median(ok)) if len(ok) else np.nan
        mad = float(1.4826 * np.median(np.abs(ok - med))) if len(ok) else np.nan
        for i in range(1, len(steps)):
            e = excess.iloc[i - 1]
            z = (e - med) / mad if np.isfinite(e) and np.isfinite(mad) and mad > 0 else np.nan
            out.append({"pass": "a", "quantity": "probe_loss", "seed": s, "step_from": steps[i - 1], "step_to": steps[i],
                        "value_from": float(piv.iloc[i - 1][s]), "value_to": float(piv.iloc[i][s]),
                        "dq": float(diffs[s].iloc[i]), "reference_dq": float(ref.iloc[i]), "excess_dq": float(e),
                        "robust_z": float(z), "jump": bool(np.isfinite(z) and z > sc["robust_z"])})
    return out


def merge_windows(jumps: list[tuple[int, int]], half: int) -> list[tuple[int, int, list]]:
    """[step_from - half, step_to + half] around each jump; overlapping windows merge."""
    out: list = []
    for a, b in sorted(jumps):
        lo, hi = a - half, b + half
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi), out[-1][2] + [(a, b)])
        else:
            out.append((lo, hi, [(a, b)]))
    return out


def name_switch_flag(cfg, repo: str, commit_date: str | None) -> str:
    ns = cfg.get("name_switch") or {}
    if repo not in ns.get("repos", []):
        return ""
    if not commit_date:
        return "unknown: no commit date"
    if commit_date[:10] < ns["date"]:
        return "CHECK: commit predates the name switch (may be the deduplicated model)"
    return "after switch"


def dedup_content_flag(cfg, repo: str, rev: str, commit: str | None, lookup) -> dict:
    """Content check of the name switch, independent of dates: the LFS sha256 of the weights at the pinned commit
    against the same revision (step branch) of repo + '-deduped'. lookup(repo, revision) -> (commit,
    {filename: sha256}) reads Hub metadata only. Flagged ("CHECK") only if the files are identical; files that
    differ are not byte-identical, which is all a hash can say. {} for repos without a name switch."""
    ns = cfg.get("name_switch") or {}
    if repo not in ns.get("repos", []):
        return {}
    other = repo + ns.get("deduped_suffix", "-deduped")
    out = {"dedup_repo": other, "dedup_revision": rev}
    try:
        _, mine = lookup(repo, commit or rev)
        out["dedup_commit"], theirs = lookup(other, rev)
    except Exception as e:  # noqa: BLE001  a missing branch or no network: report, do not stop the run
        return {**out, "dedup_check": f"unknown: Hub lookup failed ({type(e).__name__}: {str(e)[:100]})"}
    fname, same = hf.compare_weights(mine, theirs, ns.get("content_files", hf.WEIGHTS_FILES))
    if fname is None:
        return {**out, "dedup_check": f"unknown: no weights file in common (pinned: {sorted(mine) or 'none'}, "
                                      f"{other}: {sorted(theirs) or 'none'})"}
    out.update(weights_file=fname, weights_sha256=mine[fname], dedup_sha256=theirs[fname])
    out["dedup_check"] = (f"CHECK: {fname} identical to {other}@{rev} (the deduplicated model)" if same
                          else f"differs from {other}@{rev}")
    return out


def content_check(cfg, repo: str, rev: str, pins, lookup) -> dict:
    """dedup_content_flag with its result cached in the pin entry (pins.json: deduped_check), so a committed
    pins.json carries the evidence and later runs reuse it. Lookup failures are not cached."""
    entry = pins.entry(repo, rev)
    if entry and "deduped_check" in entry:
        return dict(entry["deduped_check"])
    res = dedup_content_flag(cfg, repo, rev, (entry or {}).get("commit"), lookup)
    if res and entry and not res["dedup_check"].startswith("unknown"):
        res["dedup_checked_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        entry["deduped_check"] = dict(res)
        pins.save()
    return res


def spike_check(df: pd.DataFrame, sc: dict) -> list[dict]:
    """Pass (b). Per quantity, |change| between consecutive steps of each flagged seed against the other
    seeds of the same size over the same interval: robust z = (|dq| - median) / (1.4826 MAD) of the
    other seeds' |dq|; a jump needs |dq| above every other seed and z > sc['robust_z']."""
    d = df[(df.suite == sc["suite"]) & (df["size"] == sc["size"])]
    flagged = list(sc["flagged_seeds"])
    out = []
    for q in sc["quantities"]:
        if q not in d or d[q].notna().sum() == 0:
            continue
        piv = d.pivot_table(index="step", columns="seed", values=q, aggfunc="first").sort_index()
        steps = piv.index.to_list()
        diffs = piv.diff()
        for i in range(1, len(steps)):
            r = diffs.iloc[i]
            others = np.abs(r.drop(labels=flagged, errors="ignore").dropna().to_numpy(float))
            med = float(np.median(others)) if len(others) else np.nan
            mad = float(1.4826 * np.median(np.abs(others - med))) if len(others) else np.nan
            for s in flagged:
                if s not in r.index or not np.isfinite(r[s]):
                    continue
                a = abs(float(r[s]))
                z = (a - med) / mad if np.isfinite(mad) and mad > 0 else np.nan
                jump = bool(len(others) >= 3 and a > others.max() and np.isfinite(z) and z > sc["robust_z"])
                out.append({"quantity": q, "seed": s, "step_from": steps[i - 1], "step_to": steps[i],
                            "value_from": float(piv.iloc[i - 1][s]), "value_to": float(piv.iloc[i][s]),
                            "dq": float(r[s]), "others_n": int(len(others)), "others_median_abs_dq": med,
                            "others_max_abs_dq": float(others.max()) if len(others) else np.nan,
                            "robust_z": z, "jump": jump})
    return out


SEED_VAR_QS = ["probe_loss", "alpha", "zeta_right", "zeta_left", "alpha_esd", "K", "Kurt0", "H_dep", "dh_rep", "eval_loss"]


def seed_variance(df: pd.DataFrame) -> list[dict]:
    """Spread across seeds per (variant, size, step): joint (init and batch composition), data order
    alone, initialization alone. Only (size, step) pairs that more than one variant reached."""
    d = df[df.variant.isin(["joint", "data", "weight"])]
    out = []
    for (size, step), g in d.groupby(["size", "step"]):
        if g.variant.nunique() < 2:
            continue
        for q in [q for q in SEED_VAR_QS if q in g]:
            sds = {}
            for var, h in g.groupby("variant"):
                v = h[q].dropna().to_numpy(float)
                sds[var] = (len(v), float(np.mean(v)) if len(v) else np.nan,
                            float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan)
            ref = sds.get("joint", (0, np.nan, np.nan))[2]
            for var, (n, m, sd) in sds.items():
                out.append({"size": size, "step": step, "quantity": q, "variant": var, "n_seeds": n,
                            "mean": m, "sd": sd, "sd_ratio_to_joint": sd / ref if ref and np.isfinite(ref) else np.nan})
    return out


def jvp_projections(model, ids, params, dirs, device, batch):
    outs = []
    for s in range(0, len(ids), batch):
        x = ids[s:s + batch].to(device)
        outs.append(gradients.per_example_projections(model, lm.per_example_lm_loss, x, params, dirs).cpu())
    return torch.cat(outs).numpy()


def directions(model, ids_hess, device, n_random, n_hessian, seed):
    names = lm.grad_param_names(model)
    params = {n: p for n, p in model.named_parameters() if n in names}
    gen = torch.Generator().manual_seed(seed)
    dirs = [("random", i, d) for i, d in enumerate(gradients.random_directions(params, n_random, gen))]
    if n_hessian > 0:
        hvp, p0 = gradients.hvp_fn(model, lm.per_example_lm_loss, ids_hess.to(device), params)
        vals, vecs = gradients.top_eigvecs(hvp, p0, k=n_hessian, iters=20, generator=gen)
        dirs += [("hessian", i, v) for i, v in enumerate(vecs)]
    return params, dirs


noise_quantities = p1_tasks.noise_quantities   # moved to common/p1_tasks.py (runs in the pool)


def gpu_lock(args):
    """One P1 job on the GPU at a time (an exclusive lock on logs/.p1_gpu.lock, held until the process ends).
    run_lane.sh runs P1 suites next to run_all.sh; the lock keeps them from overlapping each other or run_all.sh's
    p1_olmo7b (31.5 GiB by memory_check.py, which with a 17.3 GiB suite would exceed the L40S's 44 GiB): the later
    job waits. Not taken under --smoke, --pin-only or --device cpu. Returns (open lock file, seconds waited)."""
    if args.smoke or args.pin_only or (args.device or "").startswith("cpu"):
        return None, 0.0
    try:
        import fcntl
    except ImportError:
        return None, 0.0
    path = runtime.PILOT_ROOT / "logs" / ".p1_gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    t0 = time.perf_counter()
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[p1_checkpoints] another P1 job holds {path}; waiting for it to finish", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f, time.perf_counter() - t0




# ----------------------------------------------------------------------------- H7.1 stages
def h71_stages(cfg, suite: str, size: str, steps_avail: dict) -> dict[int, float]:
    """step -> H7.1 stage for the checkpoints that carry the H7.1 test: for each stage, the available step nearest to
    round(stage * final) (the step wanted_steps adds); {} for models without the test. Until 2026-11-16 a checkpoint
    counted as a stage when |step / final - stage| < 0.02, which made every Pythia step up to 4,000 (of 143,000) a
    1% stage and 16,000 a 10% stage: ten H7.1 tests per size and direction instead of four."""
    h71 = cfg["h71"]
    if suite != "pythia" or size not in h71["full_res_sizes"] + h71["coarse_sizes"]:
        return {}
    final = max(steps_avail)
    out: dict[int, float] = {}
    for st in h71.get("stages", cfg["stages"]):
        step, _ = hf.nearest_steps(steps_avail, [int(round(st * final))])[0]
        out.setdefault(step, st)
    return out


# ----------------------------------------------------------------------------- CPUs for the statistics pool
def parse_cpus(text: str) -> set[int]:
    """'0-6,8' -> {0, ..., 6, 8}; 'all' -> every CPU of the machine."""
    text = text.strip()
    if text == "all":
        return set(range(os.cpu_count() or 1))
    out: set[int] = set()
    for part in filter(None, text.replace(" ", "").split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def job_ended(progress_txt: Path, job: str) -> bool:
    """True if run_all.sh's progress.txt shows `job` ended: its last START or END line is an END."""
    if not progress_txt.exists():
        return False
    last = None
    for line in progress_txt.read_text().splitlines():
        f = line.split()
        if len(f) >= 3 and f[2] == job and f[1] in ("START", "END"):
            last = f[1]
    return last == "END"


def cpu_rule(cpus_after: str | None, start: set[int], root: Path = runtime.PILOT_ROOT):
    """The CPUs the statistics may use now: logs/p1_cpus if that file exists (e.g. '0-10', read every minute),
    else after `JOB=CPUS` (--cpus-after or P1_CPUS_AFTER, e.g. p3_fractional=all) once progress.txt shows JOB ended,
    else the CPUs this process started on (taskset)."""
    def now() -> set[int]:
        f = root / "logs" / "p1_cpus"
        if f.exists():
            try:
                c = parse_cpus(f.read_text())
                if c:
                    return c
            except ValueError:
                pass
        if cpus_after:
            job, _, spec = cpus_after.partition("=")
            if job_ended(root / "logs" / "progress.txt", job.strip()):
                return parse_cpus(spec or "all")
        return set(start)
    return now


def fmt_cpus(c) -> str:
    c = sorted(c)
    parts, i = [], 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[j] + 1:
            j += 1
        parts.append(f"{c[i]}-{c[j]}" if j > i else f"{c[i]}")
        i = j + 1
    return ",".join(parts)


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Progress:
    """Progress lines (stdout, i.e. logs/<job>.log) and timings.csv: one row per measurement of the GPU pass and
    per statistic of the pool, with its seconds."""
    COLS = ["utc", "checkpoint", "measurement", "where", "seconds"]

    def __init__(self, out_dir: Path):
        self.path = Path(out_dir) / "timings.csv"
        if not self.path.exists():
            self.path.write_text(",".join(self.COLS) + "\n")
        self.lock = threading.Lock()

    def say(self, msg: str) -> None:
        print(f"{utc()} [p1_checkpoints] {msg}", flush=True)

    def record(self, key: str, what: str, where: str, seconds: float) -> None:
        line = f"{utc()},{key},{what},{where},{seconds:.3f}\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)


class StatPool:
    """p1_tasks.run_task in a pool of 'spawn' processes (no CUDA state is inherited), one task per allowed CPU at a
    time. A thread refills free slots as soon as a task ends, so the pool keeps working while the main thread runs
    a GPU pass; finished tasks are handed to the main thread by collect(). A pool process that dies (killed, e.g.
    out of memory) breaks the pool: it is started again and the interrupted tasks run again (twice at most, then
    recorded as failed)."""

    def __init__(self, cpus_fn, n_fixed: int, density: str, grid_path, prog: Progress, labels: dict):
        self.cpus_fn, self.n_fixed, self.density, self.grid_path = cpus_fn, n_fixed, density, grid_path
        self.prog, self.labels = prog, labels
        self.max_workers = max(os.cpu_count() or 1, n_fixed or 1)
        self.cv = threading.Condition()
        self.pending: collections.deque = collections.deque()
        self.running: dict = {}
        self.finished: list = []
        self.tries: collections.Counter = collections.Counter()
        self.error: BaseException | None = None
        self.ex = None
        self.cpus, self.cpus_t = None, -1e18
        self.n_done, self.cpu_seconds = 0, 0.0
        self.stop = False
        self.thread = threading.Thread(target=self._loop, name="p1-stat-pool", daemon=True)
        self.thread.start()

    # -- main thread
    def submit(self, spec: dict) -> None:
        with self.cv:
            self.pending.append(spec)
            self.cv.notify_all()

    def collect(self) -> list[dict]:
        with self.cv:
            if self.error is not None:
                raise self.error
            out, self.finished = self.finished, []
        return out

    def busy(self) -> int:
        with self.cv:
            return len(self.pending) + len(self.running)

    def wait_all(self, on_record, status_every: float = 600.0) -> None:
        t = time.monotonic()
        while True:
            for rec in self.collect():
                on_record(rec)
            with self.cv:
                if self.error is None and not self.pending and not self.running and not self.finished:
                    return
                self.cv.wait(timeout=5.0)
            if time.monotonic() - t > status_every:
                t = time.monotonic()
                self.prog.say(f"pool: waiting for {self.busy()} CPU tasks ({len(self.running)} running)")

    def shutdown(self) -> None:
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.thread.join(timeout=10)
        self._kill_executor()

    # -- pool thread
    def _allowed(self) -> list[int]:
        if time.monotonic() - self.cpus_t > 60:
            new = sorted(self.cpus_fn())
            if self.cpus is not None and new != self.cpus:
                self.prog.say(f"pool: CPUs {fmt_cpus(self.cpus)} -> {fmt_cpus(new)}")
            self.cpus, self.cpus_t = new, time.monotonic()
        return self.cpus

    def _executor(self):
        if self.ex is None:
            self.ex = cf.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp.get_context("spawn"),
                                             initializer=p1_tasks.init_worker,
                                             initargs=(self.density, str(self.grid_path) if self.grid_path else None))
        return self.ex

    def _kill_executor(self) -> None:
        ex, self.ex = self.ex, None
        if ex is None:
            return
        procs = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(self) -> None:   # under self.cv
        cpus = self._allowed()
        slots = self.n_fixed or len(cpus)
        while self.pending and len(self.running) < slots:
            spec = {**self.pending.popleft(), "cpus": cpus}
            self.running[self._executor().submit(p1_tasks.run_task, spec)] = (spec, time.monotonic())

    def _failed(self, spec: dict, why: str) -> dict:
        rec = {"key": spec.get("key"), "name": spec["name"], "kind": spec["kind"], "result": None, "error": why,
               "seconds": 0.0, "pid": None, "finished_utc": utc()}
        path = p1_tasks.stats_path(spec["dir"], spec["name"])
        path.parent.mkdir(parents=True, exist_ok=True)
        p1_tasks.atomic_json(path, rec)
        return rec

    def _loop(self) -> None:
        while True:
            with self.cv:
                if self.stop:
                    return
                try:
                    self._dispatch()
                except Exception as e:  # noqa: BLE001  e.g. no process could start: the main thread raises it
                    self.error = e
                    self.cv.notify_all()
                    return
                futs = list(self.running)
                if not futs:
                    self.cv.wait(timeout=1.0)
                    continue
            done, _ = cf.wait(futs, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            if not done:
                continue
            with self.cv:
                broken = []
                for fut in done:
                    spec, t0 = self.running.pop(fut)
                    try:
                        rec = fut.result()
                    except BrokenProcessPool:
                        broken.append(spec)
                        continue
                    except BaseException as e:  # noqa: BLE001  a bug in a statistic (p1_tasks.CODE_ERRORS): stop
                        self.error = RuntimeError(f"CPU task {spec['name']} of {spec.get('key')} failed: "
                                                  f"{type(e).__name__}: {e}")
                        self.error.__cause__ = e
                        self.cv.notify_all()
                        return
                    rec["key"] = spec.get("key")
                    self.finished.append(rec)
                    self.n_done += 1
                    self.cpu_seconds += rec.get("seconds") or 0.0
                    self.prog.record(spec.get("key", ""), spec["name"], "pool", rec.get("seconds") or 0.0)
                    err = f"  ERROR {rec['error']}" if rec.get("error") else ""
                    self.prog.say(f"pool: {self.labels.get(spec.get('key'), spec.get('key'))} {spec['name']} "
                                  f"{rec.get('seconds', 0):.1f} s{err}  ({len(self.pending)} queued, "
                                  f"{len(self.running)} running)")
                if broken:   # every running task fails with the pool: start a new pool, run them again
                    broken += [sp for sp, _ in self.running.values()]
                    self.running.clear()
                    self._kill_executor()
                    for spec in broken:
                        self.tries[(spec.get("key"), spec["name"])] += 1
                        if self.tries[(spec.get("key"), spec["name"])] > 2:
                            self.finished.append(self._failed(spec, "a pool process died three times running it "
                                                                    "(killed, e.g. out of memory)"))
                        else:
                            self.pending.appendleft({k: v for k, v in spec.items() if k != "cpus"})
                    self.prog.say(f"pool: a pool process died; restarted the pool, {len(broken)} task(s) again")
                self.cv.notify_all()


class Store:
    """Per-checkpoint results: result.json in each checkpoint's folder once its CPU tasks are done (the saved state
    a restart resumes from), and the CSVs rewritten from them (checkpoints.csv and h71_summary.csv after every
    checkpoint, the per-layer and per-B CSVs at most every 10 minutes and at the end)."""

    LAYER_EVERY = 600.0

    def __init__(self, out_dir: Path, prog: Progress, run, keep_samples: bool):
        self.out_dir, self.prog, self.run, self.keep = Path(out_dir), prog, run, keep_samples
        self.order: list[str] = []
        self.cks: dict = {}
        self.records: dict = {}
        self.left: dict = {}
        self.results: dict = {}
        self.layers_t = time.monotonic()
        self.n_complete_here = 0

    def register(self, c: dict) -> None:
        if c["key"] not in self.cks:
            self.order.append(c["key"])
        self.cks[c["key"]] = c

    def load_result(self, c: dict) -> dict:
        res = json.loads((c["dir"] / "result.json").read_text())
        self.results[c["key"]] = res
        return res

    def pending(self, c: dict, record: dict, pool) -> None:
        key, d = c["key"], c["dir"]
        self.records[key] = record
        left = [t for t in record["tasks"] if not p1_tasks.stats_path(d, t["name"]).exists()]
        self.left[key] = {t["name"] for t in left}
        for t in left:
            pool.submit({**t, "dir": str(d), "key": key})
        if not left:
            self.complete(key)

    def task_done(self, rec: dict) -> None:
        key = rec.get("key")
        if key not in self.left:
            return
        self.left[key].discard(rec["name"])
        if not self.left[key]:
            self.complete(key)

    def complete(self, key: str) -> None:
        c, record = self.cks[key], self.records.pop(key)
        self.left.pop(key, None)
        d = c["dir"]
        stats = {}
        for t in record["tasks"]:
            p = p1_tasks.stats_path(d, t["name"])
            stats[t["name"]] = json.loads(p.read_text()) if p.exists() else {"error": "no result"}
        res = p1_tasks.assemble(record, stats, self.h71_kw)
        for name, e in res["errors"].items():
            self.run.note(f"{c['label']}: {name} failed: {e}")
        p1_tasks.atomic_json(d / "result.json", res)
        self.results[key] = res
        self.n_complete_here += 1
        if not self.keep:
            for f in ("act.npz", "dh.npy"):          # the large samples; grad, h71, htok and esd ones stay
                (d / "samples" / f).unlink(missing_ok=True)
        self.write()
        done = sum(k in self.results for k in self.order)
        self.prog.say(f"complete: {c['label']} ({done} of {len(self.order)} checkpoints so far)")

    h71_kw: dict = {}

    def rows(self, part: str = "row") -> list[dict]:
        out = []
        for k in self.order:
            r = self.results.get(k)
            if r is not None:
                out += [r[part]] if part == "row" else r[part]
        return out

    def write(self, final: bool = False) -> None:
        def w(name, rows):
            tmp = self.out_dir / f".{name}.tmp"
            runtime.write_csv(tmp, rows)
            os.replace(tmp, self.out_dir / name)
        w("checkpoints.csv", self.rows())
        w("h71_summary.csv", self.rows("h71_summ"))
        if final or time.monotonic() - self.layers_t > self.LAYER_EVERY:
            self.layers_t = time.monotonic()
            w("esd_layers.csv", self.rows("esd_rows"))
            w("activations_layers.csv", self.rows("act_rows"))
            w("h71_per_B.csv", self.rows("h71_rows"))


def fingerprint(cfg, seed, skip, density, boot_chunk) -> dict:
    """What a resumed checkpoint must share with this run: the P1 config, seed, skipped parts and stable density."""
    blob = json.dumps(runtime.to_jsonable(cfg), sort_keys=True, default=str)
    return {"format": 1, "config_sha256": hashlib.sha256(blob.encode()).hexdigest(), "seed": int(seed),
            "skip": sorted(skip), "stable_density": density, "h71_boot_chunk": int(boot_chunk)}


def check_fingerprint(root: Path, fp: dict) -> None:
    path = root / "fingerprint.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = [k for k in fp if old.get(k) != fp[k]]
        if diff:
            raise SystemExit(f"{root} holds checkpoints measured with other settings ({', '.join(diff)} differ). "
                             f"Use another --out, or move {root} away to measure every checkpoint again.")
        return
    root.mkdir(parents=True, exist_ok=True)
    p1_tasks.atomic_json(path, fp)


def ck_key(tag: str, repo: str, rev: str) -> str:
    return f"{tag}/{repo.replace('/', '__')}@{rev}"


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--suite", default=None, help="comma-separated suites from config.py (default: every suite "
                                                  "not marked optional, so olmo7b runs only when named)")
    ap.add_argument("--pin-only", action="store_true", help="resolve and record the commit of every model and "
                                                            "revision in pins.json, then stop (no measurement)")
    ap.add_argument("--sizes", default=None, help="comma-separated subset of sizes")
    ap.add_argument("--spike-only", action="store_true", help="run only the two loss-spike passes")
    ap.add_argument("--keep-cache", action="store_true", help="keep every checkpoint in the Hugging Face cache "
                                                              "(default: delete each one after it is read; "
                                                              "--smoke always keeps them)")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: "
                                               "grad,esd,act,dh,htok,h71,scaling,spike,coupling")
    ap.add_argument("--jvp-batch", type=int, default=None,
                    help="sequences per JVP micro-batch (default: the suite's jvp_batch in config.py, else 32)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("P1_WORKERS", "0")),
                    help="processes for the CPU statistics (default: one per CPU this job may use, see --cpus-after)")
    ap.add_argument("--cpus-after", default=os.environ.get("P1_CPUS_AFTER") or None,
                    help="JOB=CPUS: once logs/progress.txt shows JOB ended, the statistics use CPUS ('all' or e.g. "
                         "'0-10'; default from P1_CPUS_AFTER). A file logs/p1_cpus (e.g. '0-10') overrides both")
    ap.add_argument("--stable-density", choices=["grid", "fft"], default="grid",
                    help="stable density for the fits: grid (precomputed, about 17x faster; common/stable.py) or "
                         "fft (a new FFT table per evaluation, as before 2026-11-16)")
    ap.add_argument("--h71-boot-chunk", type=int, default=25,
                    help="bootstrap replicates of one H7.1 test per pool task")
    ap.add_argument("--keep-samples", action="store_true",
                    help="keep every saved sample (default: delete dh.npy and act.npz once their statistic is saved)")
    args = ap.parse_args()
    cfg = config.get("p1_checkpoints", args.smoke)
    _held, waited = gpu_lock(args)               # before CUDA starts: a waiting job holds no GPU memory; the lock
                                                 # lasts as long as this process (and _held)
    run = runtime.start_run("p1_checkpoints", cfg, args)
    if waited > 1:
        run.note(f"waited {waited / 3600:.2f} h for the P1 GPU lock (logs/.p1_gpu.lock); progress.txt's elapsed "
                 f"hours for this job include the wait")
    dev = run.device
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    skip = set(filter(None, args.skip.split(",")))
    T = cfg["tails"]
    S = cfg["scale"]
    suites = args.suite.split(",") if args.suite else default_suites(cfg)
    unknown = [x for x in suites if x not in cfg["suites"]]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; config.py has {list(cfg['suites'])}")
    h71cfg = cfg["h71"]
    sc = cfg.get("spike_check") or {}
    h71_Bs = [int(b) for b in h71cfg.get("batch_sizes", cfg["batch_sizes_h71"])]
    n_boot_h71 = int(h71cfg.get("n_boot", config.H71["n_boot"]))
    data_cache: dict = {}
    counts = collections.Counter()

    def available(suite, repo):
        """step -> branch of one repo (offline stand-in: the configured steps plus the spike scan grid)."""
        fallback = list(cfg["steps"])
        if sc and suite == sc["suite"]:
            fallback += scan_grid(sc)
        return hf.list_steps(repo, run, fallback, branch_regex=cfg["suites"][suite].get("branch_regex", r"^step(\d+)$"))

    def pin_all(repo, revs, tag):
        for rev in revs:
            try:
                hf.PINS.resolve(repo, rev, run)
                pinned[tag].append(hf.PINS.key(repo, rev))
            except Exception as e:  # noqa: BLE001  report every missing name, then fail
                run.note(f"PIN FAILED {repo}@{rev}: {type(e).__name__}: {str(e)[:160]}")

    pinned: dict = {"grid": [], "spike_a": []}

    # ---- the main grid, planned first (so that progress lines can count)
    plan = []
    if not args.spike_only:
        for suite in suites:
            for size, seed, repo, variant in suite_models(cfg, suite):
                if args.sizes and size not in args.sizes.split(","):
                    continue
                steps_avail = available(suite, repo)
                final = max(steps_avail)
                chosen = hf.nearest_steps(steps_avail, wanted_steps(cfg, suite, size, steps_avail))
                if args.pin_only:
                    pin_all(repo, [rev for _, rev in chosen], "grid")
                    continue
                stages = h71_stages(cfg, suite, size, steps_avail)
                for step, rev in chosen:
                    plan.append(dict(suite=suite, size=size, seed=seed, repo=repo, variant=variant, step=step, rev=rev,
                                     final=final, skip=skip, tag="grid", stage=stages.get(step)))

    do_spike = bool(sc) and "spike" not in skip and sc["suite"] in suites and (
        not args.sizes or sc["size"] in args.sizes.split(","))
    if args.pin_only:
        if do_spike:
            repo_of = {seed: repo for size, seed, repo, _ in suite_models(cfg, sc["suite"]) if size == sc["size"]}
            for seed in list(sc["flagged_seeds"]) + list(sc["reference_seeds"]):
                avail = available(sc["suite"], repo_of[seed])
                pin_all(repo_of[seed], [avail[st] for st in scan_steps(avail, sc)], "spike_a")
        pin_only_report(cfg, run, pinned)
        return

    # ---- statistics pool, results store, resume
    density = args.stable_density
    grid_path = None
    if density == "grid":
        t0 = time.perf_counter()
        stable.set_density("grid")
        grid_path = stable._GRID.path
        if time.perf_counter() - t0 > 2:
            run.note(f"built the stable density grid in {time.perf_counter() - t0:.0f} s: {grid_path}")
    prog = Progress(run.out_dir)
    ck_root = run.out_dir / "checkpoints"
    boot_chunk = max(1, int(args.h71_boot_chunk))
    check_fingerprint(ck_root, fingerprint(cfg, run.seed, skip, density, boot_chunk))
    start_cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    cpus_now = cpu_rule(args.cpus_after, start_cpus)
    torch.set_num_threads(torch.get_num_threads())        # keep this process's thread pools as they are, then
    for v in p1_tasks.THREAD_VARS:                         # one thread per library in the pool processes
        os.environ[v] = "1"
    labels: dict = {}
    pool = StatPool(cpus_now, max(0, args.workers), density, grid_path, prog, labels)
    store = Store(run.out_dir, prog, run, args.keep_samples)
    store.h71_kw = {"level": config.H71["level"], "inv_tol": config.H71["inv_tol"],
                    "drift_tol": config.H71["drift_tol"]}
    run.note(f"CPU statistics: {density} stable density; pool of {args.workers or 'one per CPU'} process(es) on "
             f"CPUs {fmt_cpus(cpus_now())}" + (f", then {args.cpus_after} once that job has ended"
                                               if args.cpus_after else ""))

    def on_signal(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_signal)

    def task(c, name, kind, params, seed_parts=()):
        return {"name": name, "kind": kind, "params": params,
                "seed": p1_tasks.seed_for(run.seed, c["key"], name, *seed_parts)}

    def gpu_pass(c) -> dict:
        """Load one checkpoint, run every GPU measurement not in its skip set, save the samples the CPU statistics
        need and write row.json (last, so that a folder without it is a pass that did not finish)."""
        d, label, skip_here = c["dir"], c["label"], c["skip"]
        if d.exists():
            shutil.rmtree(d)                     # an earlier pass that did not finish: measure again
        sd = d / "samples"
        sd.mkdir(parents=True)
        t_ck = time.perf_counter()
        parts: dict = {}

        @contextlib.contextmanager
        def timed(what):
            t = time.perf_counter()
            yield
            sec = time.perf_counter() - t
            parts[what] = sec
            prog.record(c["key"], what, "gpu", sec)
            prog.say(f"{label}: {what} {sec:.1f} s ({time.perf_counter() - t_ck:.0f} s into the checkpoint)")

        suite, size, seed, repo, variant, step, rev, final, tag = (c[k] for k in (
            "suite", "size", "seed", "repo", "variant", "step", "rev", "final", "tag"))
        with timed("load"):
            model, tok, src = hf.load_causal_lm(repo, rev, run, dtype=dtype, device=dev)
        # micro-batches: sequences per no-grad forward (losses, hidden states) and per JVP; olmo7b sets smaller
        # ones so that 27 GiB of float32 weights plus activations stay well under the L40S's 44 GiB
        eb = int(cfg["suites"][suite].get("eval_batch", 4))
        jb = int(args.jvp_batch or cfg["suites"][suite].get("jvp_batch", 32))
        pin = (hf.PINS.entry(repo, rev) or {}) if src == "hub" else {}
        key = type(tok).__name__ + str(getattr(tok, "name_or_path", ""))
        if key not in data_cache:  # same examples for every checkpoint of a tokenizer
            with timed("data (first checkpoint of this tokenizer)"):
                need_pool = cfg["coupling_blocks"] * cfg["coupling_B"]
                n_h71 = h71cfg.get("n_blocks", config.H71["n_blocks"]) * max(h71_Bs)
                n_pool = max(need_pool, n_h71 if h71cfg["full_res_sizes"] else 0)
                pool_texts, _ = hf.load_texts(cfg["pool_dataset"], n=max(64, n_pool * cfg["example_len"] // 400 + 64),
                                              run=run, seed=run.seed)
                pool_ids = hf.tokenize_stream(pool_texts, tok, cfg["example_len"], n_pool)
                ev_texts, _ = hf.load_texts(cfg["eval_dataset"], n=max(16, cfg["eval_seqs"] * cfg["eval_len"] // 400 + 16),
                                            run=run, seed=run.seed + 1)
                probe = hf.tokenize_stream(ev_texts, tok, cfg["probe_len"], cfg["probe_seqs"])
                evals = hf.tokenize_stream(ev_texts, tok, cfg["eval_len"], cfg["eval_seqs"])
                sur = hf.tokenize_stream(ev_texts[::-1], tok, cfg["surprisal_len"], cfg["surprisal_docs"])
                data_cache[key] = (pool_ids, probe, evals, sur)
        pool_ids, probe, evals, sur = data_cache[key]
        vocab = model.get_output_embeddings().weight.shape[0]
        pool_v, probe_v, evals_v, sur_v = (t.clamp(max=vocab - 1) for t in (pool_ids, probe, evals, sur))
        ck_rng = p1_tasks.rng_for(p1_tasks.seed_for(run.seed, c["key"], "gpu pass"))
        row = {"suite": suite, "variant": variant, "size": size, "seed": seed, "repo": repo, "step": step,
               "revision": rev, "commit": pin.get("commit"), "commit_date": pin.get("commit_date"),
               "name_switch_check": name_switch_flag(cfg, repo, pin.get("commit_date")) if src == "hub" else "",
               "source": src, "pass": tag,
               "loss_spike_seed": bool(suite == sc.get("suite") and size == sc.get("size")
                                       and seed in sc.get("flagged_seeds", [])),
               "n_params": sum(p.numel() for p in model.parameters()),
               "n_nonembed": lm.non_embedding_params(model)}
        if src == "hub":
            row.update(content_check(cfg, repo, rev, hf.PINS, hf.hub_lfs))
        for col in ("dedup_check", "name_switch_check"):
            if str(row.get(col, "")).startswith("CHECK"):
                run.note(f"{repo}@{rev} ({row['commit']}): {col} {row[col]}")
        if c.get("window"):
            row["spike_window"] = c["window"]
        tasks: list = []
        record = {"row": row, "tasks": tasks}
        with timed("probe_loss"):
            row["probe_loss"] = float(lm.token_losses(model, probe_v, dev, batch=eb).mean())   # locates loss spikes

        if "esd" not in skip_here:
            with timed("esd (SVD of every layer)"):
                esds = spectra.layer_esds(model)
                arrays = {f"ev_{i}": ev for i, (_, _, ev) in enumerate(esds)}
                p1_tasks.atomic_npz(sd / "esd.npz", names=np.array([n for n, _, _ in esds]),
                                    shapes=np.array([s for _, s, _ in esds]), **arrays)
            tasks.append(task(c, "esd", "esd", {}))
        if "act" not in skip_here or "dh" not in skip_here:
            with timed("hidden states"):
                hs = lm.hidden_states(model, probe_v, dev, batch=eb)
            if "act" not in skip_here:
                with timed("activation moments"):
                    moments, subs = [], {}
                    for li, H in enumerate(hs):
                        Hd = H.to(dev)
                        moments.append(spectra.activation_moments(Hd))
                        sub = spectra.activation_tail_sample(Hd)
                        if sub is not None:
                            subs[f"sub_{li}"] = sub
                        del Hd
                    p1_tasks.atomic_npz(sd / "act.npz", **subs)
                record["act_moments"] = moments
                tasks.append(task(c, "act", "act", {"n_layers": len(hs)}))
            if "dh" not in skip_here:
                X = hs[-1].numpy()
                if len(X) > cfg["dh_points"]:
                    X = X[ck_rng.choice(len(X), cfg["dh_points"], replace=False)]
                p1_tasks.atomic_npy(sd / "dh.npy", np.ascontiguousarray(X, dtype=np.float32))
                tasks.append(task(c, "dh", "dh", {"q_lo": S["q_lo"], "q_hi": S["q_hi"]}))
            del hs
        if "htok" not in skip_here:
            with timed("token losses (H_tok series)"):
                series = lm.token_losses(model, sur_v, dev, batch=eb).reshape(-1).numpy()
                p1_tasks.atomic_npy(sd / "htok.npy", series)
            tasks.append(task(c, "htok", "htok", {"n_surr": S["n_surr"], "wavelet_j1": S["wavelet_j1"]}))
        if "grad" not in skip_here:
            with timed("gradient projections (JVP)"):
                B = cfg["coupling_B"]
                n = cfg["coupling_blocks"] * B
                params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev, cfg["n_random_dirs"], 0, run.seed)
                P = jvp_projections(model, pool_v[:n], params, [d_ for _, _, d_ in dirs], dev, jb)
                p1_tasks.atomic_npy(sd / "grad.npy", P)
            tasks.append(task(c, "grad", "grad", {"B": B, "n_blocks": cfg["coupling_blocks"], "T": T}))
        # H7.1 at the four training stages
        if "h71" not in skip_here and c.get("stage") is not None:
            nb = h71cfg.get("n_blocks", config.H71["n_blocks"]) if size in h71cfg["full_res_sizes"] else h71cfg["coarse_blocks"]
            n_pool = nb * max(h71_Bs)
            with timed(f"H7.1 projections ({n_pool} examples)"):
                params, dirs = directions(model, pool_v[:cfg["hessian_batch"]], dev,
                                          cfg["n_random_dirs"], cfg["n_hessian_dirs"], run.seed)
                P = jvp_projections(model, pool_v[:n_pool], params, [d_ for _, _, d_ in dirs], dev, jb)
                p1_tasks.atomic_npy(sd / "h71.npy", P)
            names = [f"{kind}{i}" for kind, i, _ in dirs]
            chunks = [min(boot_chunk, n_boot_h71 - s0) for s0 in range(0, n_boot_h71, boot_chunk)]
            record["h71"] = {"directions": names, "stage": c["stage"], "frac": round(step / final if final else 1.0, 4),
                             "nb": nb, "Bs": h71_Bs, "n_pool": int(len(P)), "n_boot": n_boot_h71,
                             "n_chunks": len(chunks)}
            for j, dname in enumerate(names):
                tasks.append(task(c, f"h71_fit.{dname}", "h71_fit", {"j": j, "Bs": h71_Bs, "nb": nb}))
                for ci, nbt in enumerate(chunks):
                    tasks.append(task(c, f"h71_boot.{dname}.{ci}", "h71_boot",
                                      {"j": j, "Bs": h71_Bs, "nb": nb, "n_boot": nbt}))
        if step == final and "scaling" not in skip_here:
            with timed("eval loss"):
                row["eval_loss"] = float(lm.token_losses(model, evals_v, dev, batch=eb).mean())
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        if src == "hub" and not (args.keep_cache or run.smoke):   # smoke re-reads small checkpoints: keep them
            hf.purge_revision(repo, pin.get("commit"), run)   # disk holds about one checkpoint at a time
        row["gpu_pass_seconds"] = round(time.perf_counter() - t_ck, 1)
        p1_tasks.atomic_json(d / "row.json", record)
        counts[f"read_{tag}"] += 1
        run.note(f"{label}: GPU pass {row['gpu_pass_seconds']:.0f} s ("
                 + ", ".join(f"{k} {v:.0f}" for k, v in parts.items()) + f"); {len(tasks)} CPU tasks queued")
        return record

    def process(c) -> None:
        c["key"] = ck_key(c["tag"], c["repo"], c["rev"])
        c["dir"] = ck_root / c["key"]
        c["label"] = f"{c['repo']}@{c['rev']} [{c['tag']}]" + (f" {c['i']}/{c['n']}" if "n" in c else "")
        labels[c["key"]] = f"{c['repo'].split('/')[-1]}@{c['rev']} [{c['tag']}]"
        store.register(c)
        for rec in pool.collect():
            store.task_done(rec)
        if (c["dir"] / "result.json").exists():
            store.load_result(c)
            counts["resumed_complete"] += 1
            prog.say(f"{c['label']}: complete in an earlier run (result.json)")
            return
        rj = c["dir"] / "row.json"
        if rj.exists():
            record = json.loads(rj.read_text())
            counts["resumed_gpu_pass"] += 1
            left = sum(not p1_tasks.stats_path(c["dir"], t["name"]).exists() for t in record["tasks"])
            prog.say(f"{c['label']}: GPU pass done in an earlier run; {left} of {len(record['tasks'])} CPU tasks to run")
        else:
            record = gpu_pass(c)
        store.pending(c, record, pool)

    spike_rows, windows = [], []
    try:
        for i, c in enumerate(plan, 1):
            c.update(i=i, n=len(plan))
            process(c)

        # ---- loss-spike check, pass (a): probe loss at every scan step, flagged seeds and the stable reference
        if do_spike:
            variant = cfg["suites"][sc["suite"]].get("variant", "base")
            repo_of = {seed: repo for size, seed, repo, _ in suite_models(cfg, sc["suite"]) if size == sc["size"]}
            skip_a = skip | {"esd", "act", "dh", "htok", "grad", "h71", "scaling"}
            plan_a = []
            for seed in list(sc["flagged_seeds"]) + list(sc["reference_seeds"]):
                repo = repo_of[seed]
                avail = available(sc["suite"], repo)
                plan_a += [dict(suite=sc["suite"], size=sc["size"], seed=seed, repo=repo, variant=variant, step=st,
                                rev=avail[st], final=max(avail), skip=skip_a, tag="spike_a")
                           for st in scan_steps(avail, sc)]
            for i, c in enumerate(plan_a, 1):
                c.update(i=i, n=len(plan_a))
                process(c)
            df_a = pd.DataFrame([r for r in store.rows() if r["pass"] == "spike_a"])
            a_out = loss_jumps(df_a, sc)
            spike_rows += a_out
            jumps = [(r["step_from"], r["step_to"]) for r in a_out if r["jump"]]
            if not jumps and run.smoke and sc.get("smoke_force_top") and a_out:
                top = max(a_out, key=lambda r: r["robust_z"] if np.isfinite(r["robust_z"]) else -np.inf)
                jumps = [(top["step_from"], top["step_to"])]
                run.note(f"smoke: no loss jump found; pass (b) runs around the largest z ({top['step_to']}) to exercise the code")
            windows = merge_windows(jumps, sc["window"])

            # ---- pass (b): alpha, alpha_ESD and K for every seed of that size around each jump
            skip_b = skip | {"dh", "htok", "h71", "scaling"}
            plan_b = []
            for seed, repo in repo_of.items():
                avail = available(sc["suite"], repo)
                for lo, hi, js in windows:
                    plan_b += [dict(suite=sc["suite"], size=sc["size"], seed=seed, repo=repo, variant=variant, step=st,
                                    rev=avail[st], final=max(avail), skip=skip_b, tag="spike_b", window=f"{lo}-{hi}")
                               for st in scan_steps(avail, sc) if lo <= st <= hi]
            for i, c in enumerate(plan_b, 1):
                c.update(i=i, n=len(plan_b))
                process(c)

        prog.say(f"GPU passes finished; waiting for {pool.busy()} CPU tasks")
        pool.wait_all(store.task_done)
    finally:
        pool.shutdown()
    store.write(final=True)
    rows = store.rows()

    extra = {"stable_density": density, "cpu_tasks_run": pool.n_done,
             "cpu_task_hours": round(pool.cpu_seconds / 3600, 3), **{k: v for k, v in counts.items()}}
    if do_spike:
        df_b = pd.DataFrame([r for r in rows if r["pass"] == "spike_b"])
        for lo, hi, js in windows:
            w = df_b[df_b.spike_window == f"{lo}-{hi}"] if len(df_b) else df_b
            if not len(w):
                continue
            for r in spike_check(w, sc):
                spike_rows.append({"pass": "b", "window": f"{lo}-{hi}",
                                   "loss_jump_steps": ";".join(f"{a}-{b}" for a, b in js), **r})
    analyse(cfg, run, rows, skip, do_spike, sc, spike_rows, windows, extra)


def pin_only_report(cfg, run, pinned) -> None:
    extra = {}
    failed = [m for m in run.notes if m.startswith("PIN FAILED")]
    # name switch: content check (cached in pins.json) and date check for every pinned pythia-14m/31m name
    ns_rows = []
    for k in dict.fromkeys(k for keys in pinned.values() for k in keys):
        repo, rev = k.rsplit("@", 1)
        if repo not in (cfg.get("name_switch") or {}).get("repos", []):
            continue
        entry = hf.PINS.data.get(k) or {}
        r = {"repo": repo, "revision": rev, "commit": entry.get("commit"), "commit_date": entry.get("commit_date"),
             "name_switch_check": name_switch_flag(cfg, repo, entry.get("commit_date")),
             **content_check(cfg, repo, rev, hf.PINS, hf.hub_lfs)}
        ns_rows.append(r)
        for col in ("dedup_check", "name_switch_check"):
            if str(r.get(col, "")).startswith("CHECK"):
                run.note(f"{k} ({r['commit']}): {col} {r[col]}")
    if ns_rows:
        runtime.write_csv(run.out_dir / "name_switch_check.csv", ns_rows)
        def verdict(v):  # CHECK / differs / unknown / after switch
            v = str(v)
            return "differs" if v.startswith("differs") else v.split(":")[0]
        extra["name_switch"] = {col: pd.Series([verdict(r.get(col, "")) for r in ns_rows]).value_counts().to_dict()
                                for col in ("dedup_check", "name_switch_check")}
        run.note(f"name switch, {len(ns_rows)} pinned names: content {extra['name_switch']['dedup_check']}, "
                 f"date {extra['name_switch']['name_switch_check']}")
    extra["pins_resolved"] = len(run.pins)
    extra["download"] = {}
    for tag, keys in pinned.items():
        sizes = [(hf.PINS.data.get(k) or {}).get("weights_bytes") for k in dict.fromkeys(keys)]
        known = [b for b in sizes if b]
        extra["download"][tag] = {"checkpoints": len(sizes), "with_size": len(known),
                                  "GB": round(sum(known) / 1e9, 1)}
        run.note(f"download for {tag}: {len(sizes)} checkpoints, {sum(known) / 1e9:.1f} GB "
                 f"({len(known)} with a known size)")
    runtime.finish_run(run, extra=extra)
    if failed:
        raise SystemExit(f"{len(failed)} name(s) could not be pinned; see run.json notes")


def analyse(cfg, run, rows, skip, do_spike, sc, spike_rows, windows, extra) -> None:
    """Scaling fits, seed variance, the spike check and the coupling matrix from the checkpoints' rows."""
    rng = p1_tasks.rng_for(p1_tasks.seed_for(run.seed, "analysis"))
    scale_rows = []
    df = pd.DataFrame(rows)
    if not len(df):
        extra["n_checkpoints"] = 0
        runtime.finish_run(run, extra=extra)
        return
    df["family"] = df["suite"] + ":" + df["seed"].astype(str)
    full = df[df["pass"] == "grid"]   # the spike passes enter only spike_check.csv

    # a_N per family (suite, seed) from final eval losses; a fit needs at least three sizes
    if "eval_loss" in full:
        fin = full.dropna(subset=["eval_loss"])
        for (suite, seed), g in fin.groupby(["suite", "seed"]):
            if g["size"].nunique() < 3:
                continue
            fit = lm.fit_scaling(g["n_nonembed"].to_numpy(), g["eval_loss"].to_numpy())
            scale_rows.append({"suite": suite, "seed": seed, "family": f"{suite}:{seed}", "n_sizes": len(g), **fit})
        runtime.write_csv(run.out_dir / "scaling.csv", scale_rows)

    # spread across seeds: joint vs data order alone vs initialization alone (160M)
    sv = seed_variance(full)
    runtime.write_csv(run.out_dir / "seed_variance.csv", sv)

    # loss-spike seeds: pass (a) loss jumps, pass (b) do alpha, alpha_ESD or K jump around them?
    if do_spike:
        runtime.write_csv(run.out_dir / "spike_check.csv", spike_rows)
        pick = ("pass", "quantity", "seed", "step_from", "step_to", "robust_z")
        extra["spike_check"] = {"suite": sc["suite"], "size": sc["size"], "flagged_seeds": sc["flagged_seeds"],
                                "reference_seeds": sc["reference_seeds"],
                                "loss_jumps": [{k: r[k] for k in pick} for r in spike_rows if r["pass"] == "a" and r["jump"]],
                                "windows": [f"{lo}-{hi}" for lo, hi, _ in windows],
                                "quantity_jumps": [{k: r[k] for k in pick} for r in spike_rows
                                                   if r["pass"] == "b" and r["jump"]]}

    # exploratory coupling across seeds within (suite, size) cells (final checkpoints); a_N per family
    if "coupling" not in skip:
        fin = full[full.step == full.groupby("repo").step.transform("max")].copy()
        sc_df = pd.DataFrame(scale_rows)
        if len(sc_df):
            fin = fin.merge(sc_df[["family", "a_N"]], on="family", how="left")
        fin = fin[fin.groupby(["suite", "size"]).repo.transform("size") >= 2]   # a one-run cell adds nothing
        qs = [q for q in ["alpha", "alpha_esd", "K", "H_dep", "dh_rep"] if q in fin and fin[q].notna().sum() >= 4]
        if len(qs) >= 2 and len(fin) and fin.groupby(["suite", "size"]).size().max() >= 3:
            res = coupling.coupling_test(fin, qs, ["suite", "size"], n_perm=200 if run.smoke else 5000,
                                         family_col="family", family_quantity="a_N" if "a_N" in fin else None, rng=rng)
            pair_rows = [{"q1": k[0], "q2": k[1], **v} for k, v in res["pairs"].items()]
            runtime.write_csv(run.out_dir / "coupling_pairs.csv", pair_rows)
            (run.out_dir / "coupling.json").write_text(json.dumps(runtime.to_jsonable(
                {k: v for k, v in res.items() if k != "pairs"}), indent=2))
            extra["coupling_primary"] = res["primary"]
        else:
            run.note("coupling matrix skipped: needs >= 3 seeds in one (suite, size) cell and >= 2 quantities")
    extra["n_checkpoints"] = len(rows)
    runtime.finish_run(run, extra=extra)


if __name__ == "__main__":
    main()
