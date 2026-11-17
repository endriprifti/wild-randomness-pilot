"""H_link (exploratory; added 2026-11-17 after the first results, fixed before computing it; common/hlink.py):
the Noah variable's magnitudes carry Joseph-type memory (volatility clustering), and the series are multifractal.

CPU only, from what the jobs saved: no GPU (CUDA_VISIBLE_DEVICES is emptied), worker processes at nice 19, one thread
each, without torch; the jobs' folders are only read. torch and transformers are loaded in the main process only, to
rebuild and verify the EOS positions of P1's windows (common/eos_rebuild.py: one small checkpoint per tokenizer on
the CPU, --eos-threads threads; --no-eos skips it).

Series:
  P1   the token-surprisal series of every P1 checkpoint whose GPU pass is done (row.json):
       checkpoints/<pass>/<repo>@<revision>/samples/htok.npy, 16 windows ("documents") of 2,048 tokens, 2,047
       surprisals each, one after another (the layout from the folder's run.json)
  P4   the gradient-noise series of every finished P4 run (a row in runs.csv or runs/<tag>.json):
       hgrad_series/<tag>.npy, the unclipped minibatch gradient projected on 8 fixed random directions per step,
       after warm-up as for H_grad (P4's measure); the run's statistics are means over the 8 directions
  calibration  synthetic series of known state (common/hlink.SYNTHETIC) at the lengths of the real series, and the
       control calibration (hlink.CONTROL_N: shifted documents, MRW in documents, i.i.d., fGn and MRW plus a drift)
Test per series (common/hlink.hlink_rows): H of |x - median| by DFA and wavelet against shuffled and phase-randomized
surrogates; MFDFA Delta h of x against IAAFT surrogates; pass = p_shuf < 0.05 by both estimators over two decades
AND p_iaaft < 0.05 (99 surrogates each). Two rows per series, on the same surrogates: scales "main" (DFA and MFDFA on
8 ... n/10, fixed before any value was computed) and "sensitivity" (10 ... n/10, the s7.2 range; P4's transformer
series fall below two decades there, see decades_flag). Series not saved yet (a run still queued or running, a
checkpoint without its GPU pass) are not listed; run the command again later and only the new ones are computed.
Confound controls (reported beside the main rows, the pass rule unchanged; common/hlink.py says why each setting):
  P1  p_doc_* and beyond_documents (99 shuffles within each window); with verified EOS positions also p_piece_* and
      beyond_pieces (99 shuffles within each document piece); a third row, scales "standardized": the same test and
      controls on the series minus its position profile in level, over its profile in spread (24 log-spaced bins)
  P4  a third row, scales "detrended": the projections minus their centred moving median of 201 steps; a fourth,
      "rescaled": those divided by their centred moving MAD over 4,001 steps

Outputs (results/h_link/, --out): calibration*.csv and calibration*_summary.csv (rates per kind and length; _controls
and _controls2: the confound-control calibrations), eos/<P1 folder>.json (the rebuilt EOS positions, the checks),
p1_tokens.csv, p4_grad.csv, summary.csv (pass rates per folder and group: model size in P1, model in P4, without
diverged runs), and per series one JSON in calibration/units/ or data/units/ with progress.json beside them
(measured hours per series, hours left; summed over workers). A rerun computes only the series not saved yet (or
whose file changed), so run it again after a job has finished more checkpoints or runs.

Commands (in the pilot folder inside the venv):
  python h_link.py --calibrate                      # the synthetic series (100 per kind and length; the results of
                                                    # 2026-11-17 are in reference/h_link_calibration*.csv, README)
  python h_link.py --dry-run                        # the P1 and P4 folders and series found, nothing computed
  nohup taskset -c 0-10 python h_link.py > logs/h_link.log 2>&1 &   # every P1 checkpoint and P4 run under results/
  python h_link.py --p1 results/p1_pythia --p4 results/p4_resnet18   # named folders only
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from common import hlink, runtime
from common import units as units_mod

ROOT = Path(__file__).resolve().parent
CAL_LENGTHS = [32768, 9064]      # about P1's token series (16 x 2,047 = 32,752); the shortest P4 series (transformer
#                                  after warm-up)
CAL_REPS = 100
MIN_LEN = 100                    # shorter series are skipped (smoke: 50)
EOS_TOL = {"median_abs": 0.02, "p99_abs": 0.25, "min_corr": 0.999}   # = common/eos_rebuild.TOL (no torch import here)


# ----------------------------------------------------------------------------- finding the series
def stamp(f: Path) -> str:
    """Size and modification time of a saved series: a unit computed from another file is computed again."""
    st = f.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _find(d, key):
    """The first value of `key` in a nested dict (P1's run.json config)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def p1_docs(folder: Path) -> tuple[int, int]:
    """(documents, surprisals per document) of the folder's H_tok series: rows of surprisal_len tokens give
    surprisal_len - 1 losses each (run.json's config; the full settings if run.json is missing)."""
    cfg = {}
    if (folder / "run.json").exists():
        try:
            cfg = json.loads((folder / "run.json").read_text()).get("config", {})
        except (ValueError, OSError):
            cfg = {}
    n_docs = _find(cfg, "surprisal_docs") or hlink.N_DOCS
    length = _find(cfg, "surprisal_len") or hlink.DOC_LEN + 1
    return int(n_docs), int(length) - 1


def p1_series(folder: Path) -> list[dict]:
    out = []
    n_docs, doc_len = p1_docs(folder)
    for rj in sorted((folder / "checkpoints").glob("*/*/row.json")):
        d = rj.parent
        f = d / "samples" / "htok.npy"
        if not f.exists():
            continue
        row = json.loads(rj.read_text()).get("row", {})
        meta = {k: row.get(k) for k in ("suite", "variant", "size", "seed", "repo", "step", "revision")}
        out.append({"source": "p1", "folder": folder.name, "key": f"{folder.name}/{d.parent.name}/{d.name}",
                    "path": str(f), "stamp": stamp(f), "warm": 0, "p1_pass": row.get("pass"), "group": row.get("size"),
                    "n_docs": n_docs, "doc_len": doc_len,
                    **meta})
    return out


def run_tag(model, seed, rule, p, matched) -> str:          # as p4_clipping.run_tag (without importing torch)
    return f"{model}_s{int(seed)}_{rule}{'' if p is None else p}{'_matched' if matched else ''}"


def p4_series(folder: Path) -> list[dict]:
    rows = {}
    units = folder / "runs"
    for j in sorted(units.glob("*.json")) if units.exists() else []:
        if j.name != "settings.json":
            rows[j.stem] = json.loads(j.read_text())["row"]
    csv = folder / "runs.csv"
    if csv.exists() and csv.stat().st_size:
        for r in pd.read_csv(csv).to_dict("records"):
            p = r.get("p_clip_target")
            p = None if p is None or (isinstance(p, float) and np.isnan(p)) else float(p)
            m = r.get("matched")
            m = str(m).lower() == "true" if not isinstance(m, (bool, np.bool_)) else bool(m)
            rows.setdefault(run_tag(r["model"], r["seed"], r["rule"], p, m), r)
    out = []
    for tag, r in rows.items():
        f = folder / "hgrad_series" / f"{tag}.npy"
        if not f.exists():
            continue
        warm = r.get("warmup_steps")
        meta = {k: r.get(k) for k in ("model", "seed", "rule", "p_clip_target", "matched", "diverged", "steps")}
        out.append({"source": "p4", "folder": folder.name, "key": f"{folder.name}/{tag}", "path": str(f),
                    "stamp": stamp(f), "warm": int(warm) if warm is not None and np.isfinite(float(warm)) else 0,
                    "tag": tag, "group": r.get("model"), **meta})
    return out


def discover(results: Path, smoke: bool) -> tuple[list[Path], list[Path]]:
    if smoke:
        return ([p for p in [results / "p1_checkpoints" / "smoke"] if p.exists()],
                [p for p in [results / "p4_clipping" / "smoke"] if p.exists()])
    p1 = [p for p in sorted(results.glob("p1_*"))
          if (p / "checkpoints").is_dir() and not p.name.endswith("_recomputed")]      # same samples as their source
    p4 = [p for p in sorted(results.glob("p4_*")) if (p / "hgrad_series").is_dir()]
    return p1, p4


# ----------------------------------------------------------------------------- worker side
def init_worker(nice: int) -> None:
    try:
        os.nice(nice)
    except (AttributeError, OSError):
        pass


def load_series(spec: dict) -> np.ndarray:
    x = np.load(spec["path"])
    x = np.asarray(x, dtype=float)
    if spec["source"] == "p4":
        warm = int(spec.get("warm") or 0)
        x = x[warm:] if len(x) > warm + 50 else x      # as P4's H_grad
    return x


def run_unit(spec: dict, test: dict, min_len: int = MIN_LEN) -> dict:
    t0 = time.perf_counter()
    rng = units_mod.rng_for(units_mod.seed_for(config.SEED, "h_link", spec["key"]))
    how = {}            # the confound controls: doc_len, n_docs, position, cuts (P1) or detrend, rescale (P4)
    meta = {k: v for k, v in spec.items() if k not in ("path", "eos_cuts")}
    if spec["source"] == "calibration":
        x = hlink.synthetic(spec["kind"], int(spec["n"]), rng)
    elif spec["source"] == "control":
        x, how = hlink.synthetic_control(spec["kind"], rng)
    elif spec["source"] == "control2":
        x, how = hlink.synthetic_control2(spec["kind"], rng)
    else:
        x = load_series(spec)
        if spec["source"] == "p4":
            how = {"detrend": True, "rescale": True}
        elif len(x) == int(spec["n_docs"]) * int(spec["doc_len"]):
            how = {"doc_len": int(spec["doc_len"]), "n_docs": int(spec["n_docs"]), "position": True,
                   "cuts": spec.get("eos_cuts")}
        else:
            meta["doc_note"] = f"{len(x)} values, not {spec['n_docs']} documents of {spec['doc_len']}: no control"
    if len(x) < min_len:
        return {**meta, "skipped": f"series of {len(x)} values (< {min_len})", "rows": [],
                "seconds": round(time.perf_counter() - t0, 2)}
    seg = ({"piece": hlink.piece_starts(how["cuts"], int(how["doc_len"]))} if how.get("cuts") is not None
           else None)
    # main (s_min 8) and sensitivity (s_min 10) rows; the controls draw their surrogates after the main ones, in
    # the order documents, pieces, then the standardized / detrended / rescaled rows
    rows = hlink.hlink_rows(x, rng, doc_len=how.get("doc_len"), segments=seg, **test)
    if how.get("position"):                                  # x minus its position profile, over its spread
        xs = hlink.position_standardize(x, int(how["n_docs"]), int(how["doc_len"]), hlink.POSITION_BINS)
        r = hlink.hlink_rows(xs, rng, doc_len=how["doc_len"], segments=seg,
                             **{**test, "s_min_sensitivity": None})[0]
        r.update(scales="standardized", position_bins=hlink.POSITION_BINS)
        rows.append(r)
    if how.get("detrend"):                                   # then the same test on x minus its moving median
        y = hlink.detrend(x, hlink.DETREND_WIDTH)
        r = hlink.hlink_rows(y, rng, **{**test, "s_min_sensitivity": None})[0]
        r.update(scales="detrended", detrend_width=hlink.DETREND_WIDTH)
        rows.append(r)
        if how.get("rescale"):                               # and on that over its moving MAD
            r = hlink.hlink_rows(hlink.rescale(y, hlink.MAD_WIDTH), rng, **{**test, "s_min_sensitivity": None})[0]
            r.update(scales="rescaled", detrend_width=hlink.DETREND_WIDTH, mad_width=hlink.MAD_WIDTH)
            rows.append(r)
    for r in rows:
        if "n_pieces" in r:
            r["n_cuts"] = int(sum(len(c) for c in how["cuts"]))     # document boundaries inside the windows
    return {**meta, "rows": rows, "seconds": round(time.perf_counter() - t0, 2)}


def table(units: list[dict]) -> pd.DataFrame:
    """One row per series and scale range (a skipped series: one row with `skipped`)."""
    out = []
    for u in units:
        meta = {k: v for k, v in u.items() if k not in ("rows", "stamp")}
        out += [{**meta, **r} for r in u["rows"]] or [meta]
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- main side
def run_all(specs: list[dict], units: units_mod.Units, test: dict, workers: int, nice: int,
            min_len: int = MIN_LEN) -> None:
    todo = [s for s in specs if not units.has(s["key"]) or units.get(s["key"]).get("stamp") != s.get("stamp")]
    if not todo:
        return
    units.write_progress()

    def save(res):
        units.t_start[res["key"]] = time.perf_counter() - float(res.get("seconds") or 0.0)   # the worker's hours
        units.put(res["key"], res)

    if workers <= 0:
        for s in todo:
            save(run_unit(s, test, min_len))
        return
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                                initargs=(nice,)) as ex:
        futs = [ex.submit(run_unit, s, test, min_len) for s in todo]
        for f in cf.as_completed(futs):
            save(f.result())


FLAGS = ["pass", "memory_pass", "width_pass", "memory_beyond_linear", "pass_ignoring_decades", "two_decades",
         "memory_beyond_documents", "width_beyond_documents", "beyond_documents",
         "memory_beyond_pieces", "width_beyond_pieces", "beyond_pieces"]
MEDIANS = [("H_abs_dfa_median", "H_abs_dfa"), ("H_abs_dfa_shuf_median", "H_abs_dfa_shuf_mean"),
           ("H_abs_dfa_doc_median", "H_abs_dfa_doc_mean"), ("H_abs_wav_median", "H_abs_wav"),
           ("H_abs_wav_shuf_median", "H_abs_wav_shuf_mean"), ("H_abs_wav_doc_median", "H_abs_wav_doc_mean"),
           ("delta_h_median", "delta_h"), ("delta_h_iaaft_median", "delta_h_iaaft_mean"),
           ("delta_h_doc_median", "delta_h_doc_mean"), ("H_abs_dfa_piece_median", "H_abs_dfa_piece_mean"),
           ("H_abs_wav_piece_median", "H_abs_wav_piece_mean"), ("delta_h_piece_median", "delta_h_piece_mean")]


def _flag(v) -> float:
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    return {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}.get(str(v), np.nan)


def rates(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Rates of every flag per group (over the rows that have it; NaN where none has it) and medians."""
    d = df[df["skipped"].isna()] if "skipped" in df else df
    if not len(d) or "pass" not in d:
        return pd.DataFrame()
    d = d.copy()
    spec = {"series": ("pass", "size")}
    for c in FLAGS:
        if c in d:
            d[c] = d[c].map(_flag)
            spec[f"{c}_rate"] = (c, "mean")
    spec["decades_dfa_min"] = ("decades_dfa", "min")
    spec.update({name: (col, "median") for name, col in MEDIANS if col in d})
    return d.groupby(by, dropna=False).agg(**spec).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="the synthetic series only")
    ap.add_argument("--p1", nargs="*", default=None, help="P1 folders (default: every results/p1_*/ with checkpoints)")
    ap.add_argument("--p4", nargs="*", default=None, help="P4 folders (default: every results/p4_*/ with hgrad_series)")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None, help="default results/h_link[/smoke]")
    ap.add_argument("--workers", type=int, default=2, help="processes (0: in this process)")
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--reps", type=int, default=None, help=f"calibration series per kind and length ({CAL_REPS})")
    ap.add_argument("--n-surr", type=int, default=None, help="surrogates per test (99)")
    ap.add_argument("--smoke", action="store_true", help="smoke folders, 9 surrogates, 3 short calibration series")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-eos", action="store_true", help="no EOS rebuild: P1 rows without the piece control")
    ap.add_argument("--eos-max-params", type=float, default=2e9,
                    help="largest checkpoint loaded on the CPU to verify a rebuilt stream (float32)")
    ap.add_argument("--eos-threads", type=int, default=4, help="torch threads for that check")
    ap.add_argument("--eos-retry", action="store_true", help="check again the groups that failed before")
    args = ap.parse_args(argv)
    try:
        os.nice(args.nice)
    except (AttributeError, OSError):
        pass
    out = Path(args.out) if args.out else Path(args.results) / "h_link" / ("smoke" if args.smoke else "")
    test = dict(hlink.TEST_DEFAULTS)
    if args.smoke:
        test.update(n_surr=9, s_min=4)
    if args.n_surr:
        test["n_surr"] = args.n_surr
    min_len = 50 if args.smoke else MIN_LEN
    log = lambda m: print(f"[h_link] {m}", flush=True)

    if args.calibrate:
        lengths = [2048] if args.smoke else CAL_LENGTHS
        reps = args.reps or (3 if args.smoke else CAL_REPS)
        specs = [{"source": "calibration", "key": f"calibration/{k}/n{n}/r{r}", "kind": k, "n": n, "replicate": r,
                  "expected": hlink.EXPECTED[k]}
                 for n in lengths for k in hlink.SYNTHETIC for r in range(reps)]
        log(f"calibration: {len(specs)} synthetic series ({', '.join(hlink.SYNTHETIC)} x lengths {lengths} x {reps})")
        if args.dry_run:
            return 0
        if args.smoke:                                      # smoke outputs are throwaway: start them afresh
            for sub in ("calibration", "calibration_controls", "calibration_controls2"):
                shutil.rmtree(out / sub, ignore_errors=True)
        units = units_mod.Units(out / "calibration", {"test": test, "synthetic": hlink.SYNTHETIC, "seed": config.SEED},
                                "h_link calibration", log=log)
        units.plan(s["key"] for s in specs)
        run_all(specs, units, test, args.workers, args.nice, min_len)
        df = table([units.get(s["key"]) for s in specs])
        runtime.write_csv(out / "calibration.csv", df.to_dict("records"))
        summ = rates(df, ["scales", "kind", "expected", "n"])
        runtime.write_csv(out / "calibration_summary.csv", summ.to_dict("records"))
        cols = ["scales", "kind", "n", "series", "pass_rate", "memory_pass_rate", "width_pass_rate",
                "memory_beyond_linear_rate", "pass_ignoring_decades_rate"]
        log("calibration pass rates:\n" + summ[cols].to_string(index=False))

        creps = args.reps or (2 if args.smoke else CAL_REPS)
        cspecs = [{"source": "control", "key": f"control/{k}/r{r}", "kind": k, "n": hlink.CONTROL_N[k],
                   "replicate": r, "expected": hlink.CONTROL_EXPECTED[k]}
                  for k in hlink.CONTROL_N for r in range(creps)]
        log(f"control calibration: {len(cspecs)} synthetic series ({', '.join(hlink.CONTROL_N)} x {creps})")
        cunits = units_mod.Units(out / "calibration_controls",
                                 {"test": test, "controls": hlink.CONTROLS, "synthetic": hlink.CONTROL_SYNTHETIC,
                                  "seed": config.SEED}, "h_link control calibration", log=log)
        cunits.plan(s["key"] for s in cspecs)
        run_all(cspecs, cunits, test, args.workers, args.nice, min_len)
        cdf = table([cunits.get(s["key"]) for s in cspecs])
        runtime.write_csv(out / "calibration_controls.csv", cdf.to_dict("records"))
        csumm = rates(cdf, ["scales", "kind", "expected", "n"])
        runtime.write_csv(out / "calibration_controls_summary.csv", csumm.to_dict("records"))
        ccols = [c for c in ["scales", "kind", "n", "series", "pass_rate", "memory_pass_rate", "width_pass_rate",
                             "memory_beyond_documents_rate", "width_beyond_documents_rate", "beyond_documents_rate"]
                 if c in csumm]
        log("control calibration rates:\n" + csumm[ccols].to_string(index=False))

        dspecs = [{"source": "control2", "key": f"control2/{k}/r{r}", "kind": k, "n": hlink.CONTROL2_N[k],
                   "replicate": r, "expected": hlink.CONTROL2_EXPECTED[k]}
                  for k in hlink.CONTROL2_N for r in range(creps)]
        log(f"second control calibration: {len(dspecs)} synthetic series ({', '.join(hlink.CONTROL2_N)} x {creps})")
        dunits = units_mod.Units(out / "calibration_controls2",
                                 {"test": test, "controls": hlink.CONTROLS, "controls2": hlink.CONTROLS2,
                                  "synthetic": hlink.CONTROL2_SYNTHETIC, "seed": config.SEED},
                                 "h_link second control calibration", log=log)
        dunits.plan(s["key"] for s in dspecs)
        run_all(dspecs, dunits, test, args.workers, args.nice, min_len)
        ddf = table([dunits.get(s["key"]) for s in dspecs])
        runtime.write_csv(out / "calibration_controls2.csv", ddf.to_dict("records"))
        dsumm = rates(ddf, ["scales", "kind", "expected", "n"])
        runtime.write_csv(out / "calibration_controls2_summary.csv", dsumm.to_dict("records"))
        dcols = [c for c in ["scales", "kind", "n", "series", "pass_rate", "memory_pass_rate", "width_pass_rate",
                             "beyond_documents_rate", "memory_beyond_pieces_rate", "width_beyond_pieces_rate",
                             "beyond_pieces_rate"] if c in dsumm]
        log("second control calibration rates:\n" + dsumm[dcols].to_string(index=False))
        return 0

    found1, found4 = discover(Path(args.results), args.smoke)
    p1_dirs = [Path(p) for p in args.p1] if args.p1 is not None else found1
    p4_dirs = [Path(p) for p in args.p4] if args.p4 is not None else found4
    specs = [s for f in p1_dirs for s in p1_series(f)] + [s for f in p4_dirs for s in p4_series(f)]
    if args.smoke and not args.dry_run:
        shutil.rmtree(out / "data", ignore_errors=True)
        shutil.rmtree(out / "eos", ignore_errors=True)
    eos = {}
    if p1_dirs and not args.dry_run and not args.no_eos:    # document boundaries for the piece control (CPU)
        from common import eos_rebuild                      # torch and transformers: this process only
        for f in p1_dirs:
            res = eos_rebuild.rebuild_folder(f, out / "eos" / f"{f.name}.json", args.eos_max_params,
                                             args.eos_threads, args.eos_retry, log)
            eos.update(res)
            st = collections.Counter(v["status"] for v in res.values())
            log(f"P1 {f}: EOS positions {dict(st)} (out/eos/{f.name}.json)")
            for why, k in collections.Counter(v["reason"] for v in res.values() if v["status"] != "verified").items():
                log(f"  {k} checkpoint(s) without the piece control: {why}")
    for s in specs:
        if s["source"] == "p1":
            e = eos.get(s["key"]) or {"status": "off" if args.no_eos else "not rebuilt", "reason": "", "cuts": None}
            s.update(eos_status=e["status"], eos_reason=e["reason"], eos_group=e.get("group"), eos_cuts=e["cuts"])
            s["stamp"] += "|eos:" + (hashlib.sha256(json.dumps(e["cuts"]).encode()).hexdigest()[:12]
                                     if e["cuts"] is not None else "none")
    units = (None if args.dry_run
             else units_mod.Units(out / "data", {"test": test, "controls": hlink.CONTROLS,
                                                 "controls2": hlink.CONTROLS2, "eos_tol": EOS_TOL, "seed": config.SEED},
                                  "h_link", log=log))
    done = sum(units.has(s["key"]) and units.get(s["key"]).get("stamp") == s["stamp"] for s in specs) if units else 0
    for f in p1_dirs:
        log(f"P1 {f}: {sum(s['folder'] == f.name and s['source'] == 'p1' for s in specs)} checkpoint series")
    for f in p4_dirs:
        log(f"P4 {f}: {sum(s['folder'] == f.name and s['source'] == 'p4' for s in specs)} run series "
            f"(8 directions each)")
    log(f"{len(specs)} series, {done} saved before; progress.json gives the measured hours per series once some "
        f"are done")
    if args.dry_run or not specs:
        return 0
    units.plan(s["key"] for s in specs)
    run_all(specs, units, test, args.workers, args.nice, min_len)
    res = [units.get(s["key"]) for s in specs]
    p1 = table([r for r in res if r["source"] == "p1"])
    p4 = table([r for r in res if r["source"] == "p4"])
    parts = []
    if len(p1):
        runtime.write_csv(out / "p1_tokens.csv", p1.to_dict("records"))
        parts.append(rates(p1, ["source", "folder", "group", "scales"]))          # group: the model size
    if len(p4):
        runtime.write_csv(out / "p4_grad.csv", p4.to_dict("records"))
        ok = ~p4["diverged"].astype(str).str.lower().eq("true") if "diverged" in p4 else slice(None)
        parts.append(rates(p4[ok], ["source", "folder", "group", "scales"]))      # group: the model; diverged
        #                                                                          runs are in p4_grad.csv only
    summ = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    runtime.write_csv(out / "summary.csv", summ.to_dict("records"))
    if len(summ):
        log("pass rates:\n" + summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
