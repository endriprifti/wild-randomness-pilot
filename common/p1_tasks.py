"""P1's CPU statistics as tasks for a process pool, so that the GPU pass of p1_checkpoints.py never waits for them.

The GPU pass of each checkpoint saves what the statistics need to <out>/checkpoints/<pass>/<repo>@<revision>/:
  samples/   grad.npy (JVP projections, examples x random directions), h71.npy (the H7.1 pool, examples x
             directions), htok.npy (per-token loss series), esd.npz (ESD eigenvalues per layer), act.npz (up to
             200,000 |activations| per layer for act_zeta), dh.npy (final-layer representation points)
  row.json   the checkpoint's row so far (losses, activation moments, commit) and the list of its CPU tasks
Each task (run_task, in a pool process) reads its samples, computes one statistic and writes stats/<task>.json
(atomically: a temporary file, then rename); p1_checkpoints.py merges the stats into result.json when all are
there. Every task has its own random generator, seeded from the run seed, the checkpoint and the task name, so the
results do not depend on the number of processes, the order the tasks finish in, or a restart.

Tasks: grad (stable fit and CSN tails with Vuong tests of the minibatch noise at coupling_B), esd (alpha_ESD per
layer), act (act_zeta per layer), dh (Delta h_rep, on the CPU), htok (H_tok with shuffled surrogates), and per H7.1
direction h71_fit (alpha-hat at each B) and h71_boot (a chunk of the bootstrap replicates).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from . import batch as batchmod
from . import hurst, multifractal, spectra, stable, tails

# a stats task raising one of these has a bug: the run stops (as common/hf.py treats them); others are recorded
CODE_ERRORS = (AttributeError, TypeError, NameError, ImportError, KeyError)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# ----------------------------------------------------------------------------- files
def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    from .runtime import to_jsonable
    atomic_write_text(path, json.dumps(to_jsonable(obj), default=str))


def atomic_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def stats_path(ck_dir: Path, name: str) -> Path:
    return Path(ck_dir) / "stats" / f"{name}.json"


def seed_for(base: int, *parts) -> list[int]:
    """Entropy for np.random.SeedSequence from the run seed and names (crc32: the same in every process)."""
    return [int(base)] + [zlib.crc32(str(p).encode()) for p in parts]


def rng_for(seed: list[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


# ----------------------------------------------------------------------------- the statistics
def noise_quantities(values: np.ndarray, B: int, n_blocks: int, T: dict, rng) -> dict:
    """Stable fit and both CSN tails of the minibatch noise at batch size B (block means of the projections)."""
    noise = stable.block_noise(values, B, max_blocks=n_blocks)
    out = {"n_blocks": len(noise)}
    if len(noise) < 20:
        return out
    f = stable.fit_stable(noise)
    out.update({"alpha": f.alpha, "beta": f.beta, "gamma": f.gamma})
    for side, v in (("right", noise[noise > 0]), ("left", -noise[noise < 0])):
        try:
            rd = tails.read_tail(v, n_boot=T["n_boot"], min_tail=min(T["min_tail"], max(10, len(v) // 10)),
                                 max_candidates=T["max_candidates"], hill_reps=T["hill_reps"],
                                 hill_tol=T["hill_tol"], hill_eps=T["hill_eps"], n_jobs=T.get("n_jobs", 1), rng=rng)
            out[f"zeta_{side}"] = rd.fit.zeta
            out[f"tail_label_{side}"] = rd.label
            out[f"tail_reason_{side}"] = rd.reason
            out[f"p_ks_{side}"] = rd.p_ks
        except ValueError:
            pass
    return out


def _samples(ck: Path) -> Path:
    return Path(ck) / "samples"


def task_grad(ck, p, rng):
    P = np.load(_samples(ck) / "grad.npy")
    return noise_quantities(P[:, 0], int(p["B"]), int(p["n_blocks"]), {**p["T"], "n_jobs": 1}, rng)


def task_esd(ck, p, rng):
    z = np.load(_samples(ck) / "esd.npz")
    names, shapes = list(z["names"]), z["shapes"].tolist()
    a, rows = spectra.alpha_esd_rows([(str(n), s, z[f"ev_{i}"]) for i, (n, s) in enumerate(zip(names, shapes))],
                                     rng=rng)
    return {"alpha_esd": a, "rows": rows}


def task_act(ck, p, rng):
    z = np.load(_samples(ck) / "act.npz")
    return {"layers": [spectra.activation_tail_fit(z[f"sub_{li}"] if f"sub_{li}" in z else None)
                       for li in range(int(p["n_layers"]))]}


def task_dh(ck, p, rng):
    X = np.load(_samples(ck) / "dh.npy")
    return multifractal.delta_h_rep(X, p["q_lo"], p["q_hi"], rng=rng, n_centres=min(5000, len(X)), device="cpu")


def task_htok(ck, p, rng):
    series = np.load(_samples(ck) / "htok.npy")
    return hurst.memory_claim(series, n_surr=int(p["n_surr"]), rng=rng, wavelet_j1=int(p["wavelet_j1"]))


def _h71_pool(ck, j):
    return np.asarray(np.load(_samples(ck) / "h71.npy", mmap_mode="r")[:, int(j)], dtype=float)


def task_h71_fit(ck, p, rng):
    return {"per_B": batchmod.fit_over_batches(_h71_pool(ck, p["j"]), p["Bs"], int(p["nb"]))}


def task_h71_boot(ck, p, rng):
    b, gap = batchmod.h71_bootstrap(_h71_pool(ck, p["j"]), p["Bs"], int(p["nb"]), int(p["n_boot"]), rng)
    return {"b": b, "gap": gap}


KINDS = {"grad": task_grad, "esd": task_esd, "act": task_act, "dh": task_dh, "htok": task_htok,
         "h71_fit": task_h71_fit, "h71_boot": task_h71_boot}


# ----------------------------------------------------------------------------- pool processes
def init_worker(density: str, grid_path: str | None, nice: int = 10) -> None:
    """Pool process start: no GPU, one thread per library, lower priority than the GPU pass, the parent's density."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        os.nice(nice)
    except (AttributeError, OSError):
        pass
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    stable.set_density(density, grid_path)


def run_task(spec: dict) -> dict:
    """Run one task: spec = {dir, name, kind, params, seed[, cpus]}; writes stats/<name>.json and returns it."""
    if spec.get("cpus"):
        try:
            os.sched_setaffinity(0, set(spec["cpus"]))
        except (AttributeError, OSError, ValueError):
            pass
    t0 = time.perf_counter()
    err = None
    try:
        result = KINDS[spec["kind"]](Path(spec["dir"]), spec["params"], rng_for(spec["seed"]))
    except CODE_ERRORS:
        raise
    except Exception as e:  # noqa: BLE001  a statistic that cannot be computed: recorded, the run goes on
        result, err = None, f"{type(e).__name__}: {str(e)[:300]}"
    rec = {"name": spec["name"], "kind": spec["kind"], "result": result, "error": err,
           "seconds": round(time.perf_counter() - t0, 3), "pid": os.getpid(),
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    path = stats_path(spec["dir"], spec["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, rec)
    return rec


# ----------------------------------------------------------------------------- merging into the checkpoint's rows
def assemble(record: dict, stats: dict, h71_defaults: dict | None = None) -> dict:
    """result.json of one checkpoint from row.json's record and its tasks' stats {name: stats record}.

    Returns {"row", "esd_rows", "act_rows", "h71_rows", "h71_summ", "errors"} in the columns and order of the
    rows p1_checkpoints.py wrote before the pool (2026-11-16)."""
    base = dict(record["row"])
    repo, step = base["repo"], base["step"]
    tail = {k: base.pop(k) for k in ("eval_loss", "gpu_pass_seconds") if k in base}
    row, errors = dict(base), {}
    esd_rows, act_rows, h71_rows, h71_summ = [], [], [], []

    def res(name):
        s = stats.get(name) or {}
        if s.get("error"):
            errors[name] = s["error"]
        return s.get("result")

    names = [t["name"] for t in record["tasks"]]
    if "esd" in names:
        r = res("esd")
        row["alpha_esd"] = r["alpha_esd"] if r else np.nan
        esd_rows = [{**x, "repo": repo, "step": step} for x in (r or {}).get("rows", [])]
    moments = record.get("act_moments")
    if moments is not None:
        r = res("act") or {}
        fits = r.get("layers") or [{} for _ in moments]
        for li, (m, f) in enumerate(zip(moments, fits)):
            act_rows.append({**m, **f, "layer": li, "repo": repo, "step": step})
        row["K"] = float(np.median([m["K"] for m in moments]))
        row["K_max"] = float(np.max([m["K"] for m in moments]))
        row["Kurt0"] = float(np.median([m["Kurt0"] for m in moments]))
        row["n_outlier_channels"] = int(np.sum([m["n_outlier_channels"] for m in moments]))
        row["act_zeta"] = float(np.nanmedian([f.get("act_zeta", np.nan) for f in fits])) if any(
            "act_zeta" in f for f in fits) else np.nan
    if "dh" in names:
        d = res("dh")
        if d:
            row.update({"dh_rep": d["delta_h_rep"], "dh_scope": d["scope"], "dh_D2": d["D2"],
                        "dh_clause_ok": d["clause_ok"], "dh_decades": d["decades_r"]})
    if "htok" in names:
        m = res("htok")
        if m:
            row.update({f"Htok_{k}": v for k, v in m.items()})
            row["H_dep"] = m["H_dep_dfa"]
    if "grad" in names:
        row.update(res("grad") or {})
    h = record.get("h71")
    if h:
        kw = h71_defaults or {}
        for d in h["directions"]:
            fit = res(f"h71_fit.{d}")
            boots_b, boots_gap = [], []
            for c in range(h["n_chunks"]):
                r = res(f"h71_boot.{d}.{c}") or {}
                boots_b += r.get("b", [])
                boots_gap += r.get("gap", [])
            meta = {"repo": repo, "commit": base.get("commit"), "size": base["size"], "step": step,
                    "stage": h["stage"], "step_frac": h["frac"], "direction": d}
            if not fit:
                continue
            rows_B = fit["per_B"]
            h71_rows.extend({**meta, **r} for r in rows_B)
            summ = batchmod.h71_summary(rows_B, boots_b, boots_gap, h["n_pool"], h["Bs"], h["nb"], **kw)
            h71_summ.append({**meta, **summ, "n_boot_requested": h["n_boot"]})
    for name, e in errors.items():
        row[f"error_{name}"] = e
    row.update(tail)
    return {"row": row, "esd_rows": esd_rows, "act_rows": act_rows, "h71_rows": h71_rows, "h71_summ": h71_summ,
            "errors": errors}
