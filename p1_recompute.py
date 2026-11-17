"""P1's statistics again, on the CPU, from the samples a P1 job saved: no GPU pass, no download, no GPU memory.

For a P1 folder measured before the batch order of 2026-11-17, also while its job is still running with the earlier
code. Each finished checkpoint of SRC (result.json in SRC/checkpoints/<pass>/<repo>@<revision>/) gets a folder of the
same name under DST/checkpoints/ with
  row.json     SRC's record, with order_seed added to the grad and H7.1 main-fit tasks (p1_tasks.upgrade_tasks)
  stats/       those tasks computed again from SRC's samples (grad.npy, h71.npy) with SRC's task seeds; every other
               statistic copied from SRC: alpha_ESD, act_zeta, Delta h_rep, H_tok, and the H7.1 bootstrap, which
               resamples the pool i.i.d. and so does not depend on its order (common/batch.py)
  result.json  merged by the code as it is now: Delta h_rep NaN when the dimension clause fails or the width is
               negative, the H7.1 verdict in the summary rows
and DST holds checkpoints.csv, h71_summary.csv, h71_per_B.csv, esd_layers.csv, activations_layers.csv and the analysis
of P1's end (scaling.csv, seed_variance.csv, spike_check.csv, coupling_pairs.csv, coupling.json) from those rows, with
run.json. They are what the current p1_checkpoints.py gives on the same checkpoints (tests/test_batch_order.py).

SRC is only read, so the running job is not touched. A checkpoint it finishes later is recomputed by the next run of
the same command (finished ones in DST are kept): run it again after the job has ended to complete DST. A P1 job
started with the current code on SRC itself upgrades SRC in the same way (p1_checkpoints.py), so this command is for
jobs that run on with the earlier code.

CPU: --workers processes (default 2) at nice 19 (--nice), on the CPUs the command is started on (taskset), so the
running jobs keep priority; CUDA_VISIBLE_DEVICES is emptied before torch loads. Before starting, the command prints
the CPU hours that the tasks to recompute took in SRC (their stats record their seconds on that machine).

Command (in the pilot folder inside the venv):
  nohup taskset -c 7-10 python p1_recompute.py results/p1_pythia > logs/p1_recompute_pythia.log 2>&1 &
writes results/p1_pythia_recomputed/ (--out sets another folder; --dry-run counts and estimates only).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch is imported (here and in the spawned workers)

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import pandas as pd

import config
import p1_checkpoints as p1
from common import p1_tasks, runtime, stable


def config_for(fp: dict) -> tuple[dict, bool]:
    """The P1 config (full or smoke) whose hash is in SRC's fingerprint."""
    for smoke in (False, True):
        cfg = config.get("p1_checkpoints", smoke)
        blob = json.dumps(runtime.to_jsonable(cfg), sort_keys=True, default=str)
        if hashlib.sha256(blob.encode()).hexdigest() == fp.get("config_sha256"):
            return cfg, smoke
    raise SystemExit("the P1 settings in config.py differ from those SRC was measured with (fingerprint.json's "
                     "config_sha256): recompute with the code of that run")


def finished_checkpoints(src: Path) -> list[tuple[str, Path]]:
    """(key, folder) of SRC's checkpoints with result.json, in the order of SRC's checkpoints.csv, then the rest."""
    root = src / "checkpoints"
    found = {f"{rj.parent.parent.name}/{rj.parent.name}": rj.parent for rj in root.glob("*/*/result.json")}
    order = []
    csv = src / "checkpoints.csv"
    if csv.exists() and csv.stat().st_size:
        df = pd.read_csv(csv, usecols=lambda c: c in ("pass", "repo", "revision"))
        if {"pass", "repo", "revision"} <= set(df.columns):
            order = [p1.ck_key(t, r, v) for t, r, v in df[["pass", "repo", "revision"]].itertuples(index=False)]
    keys = [k for k in dict.fromkeys(order) if k in found] + sorted(k for k in found if k not in set(order))
    return [(k, found[k]) for k in keys]


def copy_atomic(a: Path, b: Path) -> None:
    b.parent.mkdir(parents=True, exist_ok=True)
    tmp = b.with_name(f"{b.name}.{os.getpid()}.tmp")
    shutil.copyfile(a, tmp)
    os.replace(tmp, b)


def prepare(key: str, sd: Path, dd: Path, seed: int) -> tuple[dict, list[dict], float]:
    """DST's row.json and copied stats for one checkpoint; returns its record, the task specs still to run and the
    seconds those tasks took in SRC."""
    record = json.loads((sd / "row.json").read_text())
    changed = set(p1_tasks.upgrade_tasks(record, seed))
    record["samples_from"] = str(sd.resolve())
    dd.mkdir(parents=True, exist_ok=True)
    p1_tasks.atomic_json(dd / "row.json", record)
    todo, seconds = [], 0.0
    for t in record["tasks"]:
        src_stats, dst_stats = p1_tasks.stats_path(sd, t["name"]), p1_tasks.stats_path(dd, t["name"])
        if dst_stats.exists():
            continue
        if t["name"] in changed:
            todo.append({**t, "dir": str(dd), "samples_dir": str(sd), "key": key})
            if src_stats.exists():
                seconds += float(json.loads(src_stats.read_text()).get("seconds") or 0.0)
        elif src_stats.exists():
            copy_atomic(src_stats, dst_stats)
    return record, todo, seconds


def merge(dd: Path, record: dict) -> dict:
    stats = {}
    for t in record["tasks"]:
        p = p1_tasks.stats_path(dd, t["name"])
        stats[t["name"]] = json.loads(p.read_text()) if p.exists() else {"error": "no result"}
    res = p1_tasks.assemble(record, stats, p1.h71_kw())
    p1_tasks.atomic_json(dd / "result.json", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="a P1 job's folder (results/p1_pythia)")
    ap.add_argument("--out", default=None, help="folder for the recomputed results (default: SRC_recomputed)")
    ap.add_argument("--workers", type=int, default=2, help="processes for the statistics (0: in this process)")
    ap.add_argument("--nice", type=int, default=19, help="niceness of the worker processes")
    ap.add_argument("--dry-run", action="store_true", help="count the checkpoints and tasks, then stop")
    args = ap.parse_args(argv)
    src = Path(args.src)
    fp_path = src / "checkpoints" / "fingerprint.json"
    if not fp_path.exists():
        raise SystemExit(f"{fp_path} not found: SRC must be a P1 job's folder (results/p1_<suite>)")
    fp = json.loads(fp_path.read_text())
    cfg, smoke = config_for(fp)
    seed, density, skip = int(fp["seed"]), fp.get("stable_density", "fft"), set(fp.get("skip", []))
    dst = Path(args.out) if args.out else src.with_name(src.name + "_recomputed")
    if dst.resolve() == src.resolve():
        raise SystemExit("--out must be another folder than SRC")

    cks = finished_checkpoints(src)
    n_open = sum(1 for d in (src / "checkpoints").glob("*/*") if d.is_dir() and not (d / "result.json").exists())
    plan, done, todo, src_seconds = [], [], [], 0.0
    for key, sd in cks:
        dd = dst / "checkpoints" / key
        if (dd / "result.json").exists():
            done.append((key, sd, dd))
            continue
        if args.dry_run:
            rec = json.loads((sd / "row.json").read_text())
            n = sum(t["kind"] in p1_tasks.ORDERED_KINDS and t["params"].get("order_seed") is None for t in rec["tasks"])
            todo += [None] * n
            src_seconds += sum(float(json.loads(p1_tasks.stats_path(sd, t["name"]).read_text()).get("seconds") or 0)
                               for t in rec["tasks"] if t["kind"] in p1_tasks.ORDERED_KINDS
                               and t["params"].get("order_seed") is None and p1_tasks.stats_path(sd, t["name"]).exists())
            plan.append((key, sd, dd, None))
            continue
        record, specs, sec = prepare(key, sd, dd, seed)
        plan.append((key, sd, dd, record))
        todo += specs
        src_seconds += sec
    w = max(1, args.workers)
    print(f"[p1_recompute] {src}: {len(cks)} finished checkpoints ({len(done)} already in {dst}), {n_open} not finished "
          f"in SRC (left for a later run); {len(todo)} tasks to recompute, which took {src_seconds / 3600:.2f} CPU-hours "
          f"in SRC: about {src_seconds / 3600 / w:.2f} h on {w} process(es) with whole CPUs (longer at nice "
          f"{args.nice} while other jobs use them)", flush=True)
    if args.dry_run:
        return 0

    stable.set_density(density)
    grid_path = str(stable._GRID.path) if density == "grid" and stable._GRID is not None else None
    left = {key: {s["name"] for s in todo if s["key"] == key} for key, _, _, _ in plan}
    records = {key: rec for key, _, _, rec in plan}
    dirs = {key: dd for key, _, dd, _ in plan}
    results = {key: json.loads((dd / "result.json").read_text()) for key, _, dd in done}
    order = [k for k, _ in cks]
    run = runtime.start_run("p1_recompute", cfg, argparse.Namespace(seed=seed, device="cpu", out=str(dst),
                                                                     smoke=smoke, offline_ok=False))
    store = p1.Store(dst, None, run, keep_samples=True)
    store.order, store.results = order, results
    t0, n_done = time.perf_counter(), 0

    def finish(key):
        nonlocal n_done
        res = merge(dirs[key], records[key])
        for name, e in res["errors"].items():
            run.note(f"{key}: {name} failed: {e}")
        results[key] = res
        n_done += 1
        store.write()
        print(f"[p1_recompute] {n_done}/{len(plan)} recomputed: {key} ({(time.perf_counter() - t0) / 60:.1f} min)",
              flush=True)

    for key in [k for k in left if not left[k]]:
        finish(key)
    if todo:
        if args.workers <= 0:
            p1_tasks.init_worker(density, grid_path, nice=0)
            for spec in todo:
                rec = p1_tasks.run_task(spec)
                left[spec["key"]].discard(rec["name"])
                if not left[spec["key"]]:
                    finish(spec["key"])
        else:
            with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                        initializer=p1_tasks.init_worker,
                                        initargs=(density, grid_path, args.nice)) as ex:
                futs = {ex.submit(p1_tasks.run_task, spec): spec for spec in todo}
                for f in cf.as_completed(futs):
                    spec = futs[f]
                    rec = f.result()               # a bug in a statistic (p1_tasks.CODE_ERRORS) stops the command
                    left[spec["key"]].discard(rec["name"])
                    if not left[spec["key"]]:
                        finish(spec["key"])
    store.write(final=True)
    rows = store.rows()
    sc = cfg.get("spike_check") or {}
    do_spike = bool(sc) and "spike" not in skip and any(r.get("pass") == "spike_a" for r in rows)
    spike_rows, windows = [], []
    if do_spike:
        spike_rows, windows = p1.spike_windows(rows, sc, run)
        spike_rows = list(spike_rows) + p1.spike_b_rows(rows, windows, sc)
    extra = {"source": str(src.resolve()), "source_fingerprint": fp, "checkpoints": len(rows),
             "recomputed_here": n_done, "tasks_recomputed": len(todo), "source_not_finished": n_open,
             "source_task_cpu_hours": round(src_seconds / 3600, 3)}
    p1.analyse(cfg, run, rows, skip, do_spike, sc, spike_rows, windows, extra)
    print(f"[p1_recompute] done: {len(rows)} checkpoints in {dst} ({n_open} not finished in SRC yet)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
