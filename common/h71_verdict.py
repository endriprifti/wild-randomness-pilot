"""H7.1 verdict from the slope, the level and the scale exponent of alpha-hat(B) (sec. 7.5).

A flat alpha-hat(B) is what a stable law predicts, and also what a Gaussian gives once its blocks are Gaussian:
alpha-hat then sits at 2 for every B. The slope alone (h71_test's "reading") cannot tell the two apart, so the
verdict reads three things from each test's summary row:

  slope   b, the change of alpha-hat per doubling of B, with its bootstrap 90% interval [b_lo, b_hi]:
          flat if the interval lies inside +-inv_tol (0.03), drift if it lies above drift_tol (0.05).
  level   alpha_mean, the mean of alpha-hat over the batch sizes. alpha-hat cannot exceed 2, so alpha = 2 is
          tested one-sided against Gaussian pools of the same length, n_blocks and batch sizes: "below 2" if
          alpha_mean falls under the 5% quantile of their alpha_mean (alpha = 2 rejected at 5%; the bootstrap
          is not used for the level, since it fails at the boundary alpha = 2), "consistent with 2" otherwise.
  scale   s, the slope of log gamma-hat_B on log B. With alpha-hat below 2 it must be consistent with
          1/alpha - 1: the bootstrap 90% interval of s - (1/alpha-hat - 1) (scale_gap_lo, scale_gap_hi)
          contains 0. With alpha-hat consistent with 2 it must be near -1/2: inside the central 90% of s on the
          same Gaussian pools.

  invariant (wild)   slope flat, level below 2 and alpha_mean <= wild_max (1.9), s consistent with 1/alpha - 1
  flat at 2 (mild)   slope flat, level consistent with 2, s near -1/2
  drift              slope above drift_tol
  undecided          anything else; verdict_reason says which condition failed

The margin wild_max (Endri, 2026-11-14): between 1.9 and the Gaussian 5% quantile the verdict is undecided. Up to
B = 1,024, finite-variance noise with zeta of about 3 to 4 (Student t) keeps alpha-hat at 1.93 to 1.98 with a flat
slope, as a stable law with alpha that close to 2 does, so the test cannot call either one wild.

Every input is a column that h71_test already writes (b_lo, b_hi, alpha_mean, scale_exponent, scale_gap_lo,
scale_gap_hi, n_pool, n_blocks), so the verdict is recomputed from saved CSVs without refitting: P0's
block_check_bootstrap.csv, P0b's h71_trunc_tests.csv and h71_gauss_tests.csv, and the h71_summary.csv of P1 and P4.
The Gaussian reference depends only on (n_pool, n_blocks, batch sizes). reference/h71_gauss_null.csv holds it for
the settings of the full run (400 pools each at 1,600 and 200 blocks, B = 8 ... 1,024: P0b's h71null part, the
same seeds and fits), so P1 and P4 write the verdict into their h71_summary.csv as they run (verdict_columns,
2026-11-17); a row whose setting has no pools there reads undecided ("no Gaussian reference for this setting").
The command below simulates any missing setting and keeps those pools in --null (one row per Gaussian pool).

Command (from the repo root):
  python -m common.h71_verdict results/p1_pythia/h71_summary.csv results/p4_resnet18/h71_summary.csv --jobs 6
writes <name>_verdict.csv next to each input and prints the verdict counts.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .batch import _slopes, fit_over_batches
from .stable import block_noise

PILOT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = PILOT_ROOT / "reference" / "h71_gauss_null.csv"   # Gaussian pools of the full-run settings (in git)
VERDICTS = ("invariant (wild)", "flat at 2 (mild)", "drift", "undecided")
NULL_CODE = 71        # part code in the Gaussian pools' seeds
NULL_KEY = ["n_pool", "n_blocks", "batch_sizes"]
SUMMARY_COLUMNS = ["b_lo", "b_hi", "alpha_mean", "scale_exponent", "scale_gap_lo", "scale_gap_hi", "n_pool", "n_blocks"]
SEC_PER_FIT = {1600: 1.1, 200: 1.9}   # CPU seconds per stable fit on Gaussian block noise (4-core cloud CPU)


# ----------------------------------------------------------------------------- Gaussian reference
def bs_key(batch_sizes) -> str:
    return ",".join(str(int(b)) for b in sorted(int(b) for b in batch_sizes))


def bs_from_key(key: str) -> list[int]:
    return [int(b) for b in str(key).split(",") if b]


def null_replicate(seed: int, n_pool: int, n_blocks: int, batch_sizes, r: int) -> dict:
    """One Gaussian pool of the same length, cut into the same blocks and fitted as h71_test fits a real pool."""
    Bs = bs_from_key(bs_key(batch_sizes))
    rng = np.random.default_rng([int(seed), NULL_CODE, int(n_pool), int(n_blocks), *Bs, int(r)])
    pool = rng.standard_normal(int(n_pool))
    rows = fit_over_batches(pool, Bs, int(n_blocks))
    b, s, abar = _slopes(rows)
    return {"n_pool": int(n_pool), "n_blocks": int(n_blocks), "batch_sizes": bs_key(Bs), "seed": int(seed),
            "replicate": int(r), "alpha_mean": abar, "scale_exponent": s, "b": b,
            **{f"alpha_B{x['B']}": x["alpha"] for x in rows}}


def null_cost(n_blocks: int, batch_sizes) -> float:
    """Rough CPU seconds for one Gaussian pool (one stable fit per batch size)."""
    near = min(SEC_PER_FIT, key=lambda k: abs(math.log(k) - math.log(max(1, n_blocks))))
    return len(list(batch_sizes)) * SEC_PER_FIT[near]


def simulate_null(keys, reps: int, seed: int, jobs: int = 1, have: pd.DataFrame | None = None,
                  log=print) -> pd.DataFrame:
    """Gaussian pools for each (n_pool, n_blocks, batch sizes) key until each has `reps` of them.

    Pools already in `have` count; new ones are numbered after the highest replicate index for this seed, and
    each pool's seed comes from (seed, key, replicate), so the result does not depend on `jobs`."""
    have = have if have is not None else pd.DataFrame(columns=NULL_KEY + ["seed", "replicate"])
    todo = []
    for n_pool, n_blocks, bsk in keys:
        g = have[(have.n_pool == n_pool) & (have.n_blocks == n_blocks) & (have.batch_sizes == bsk)] if len(have) else have
        missing = reps - len(g)
        if missing <= 0:
            continue
        mine = g[g.seed == seed].replicate if len(g) else pd.Series(dtype=int)
        start = int(mine.max()) + 1 if len(mine) else 0
        todo += [(seed, n_pool, n_blocks, bs_from_key(bsk), r) for r in range(start, start + missing)]
    if not todo:
        return pd.DataFrame()
    est = sum(null_cost(t[2], t[3]) for t in todo)
    log(f"simulating {len(todo)} Gaussian pools for the H7.1 level reference: about {est / 3600:.2f} CPU-hours, "
        f"{est / 3600 / max(1, jobs):.2f} h on {jobs} process(es)")
    t0 = time.perf_counter()
    if jobs <= 1:
        rows = [null_replicate(*t) for t in todo]
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(null_replicate, *zip(*todo)))
    log(f"Gaussian pools done in {(time.perf_counter() - t0) / 60:.1f} min")
    return pd.DataFrame(rows)


def load_null(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame(columns=NULL_KEY + ["seed", "replicate", "alpha_mean", "scale_exponent", "b"])
    df = pd.read_csv(path)
    df["batch_sizes"] = df["batch_sizes"].astype(str)
    return df


def load_reference(*paths) -> pd.DataFrame:
    """The Gaussian pools of reference/h71_gauss_null.csv and of any further null CSVs (duplicates dropped)."""
    frames = [f for f in (load_null(p) for p in (REFERENCE, *paths) if p is not None) if len(f)]
    if not frames:
        return load_null(None)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=NULL_KEY + ["seed", "replicate"], keep="first").reset_index(drop=True)


_BOUNDS: dict = {}


def verdict_columns(summary: dict, batch_sizes, level: float = 0.90, inv_tol: float = 0.03, drift_tol: float = 0.05,
                    wild_max: float | None = 1.9, null: pd.DataFrame | None = None) -> dict:
    """The verdict of one h71_test summary (P1, P4) against the shipped Gaussian reference (or `null`): the columns
    apply_verdicts adds (reference bounds, the three readings, verdict, verdict_reason)."""
    bsk = bs_key(batch_sizes)
    key = (int(_f(summary["n_pool"])), int(_f(summary["n_blocks"])), bsk, float(level))
    if null is not None or key not in _BOUNDS:
        ref = null if null is not None else load_reference()
        g = ref[(ref.n_pool == key[0]) & (ref.n_blocks == key[1]) & (ref.batch_sizes == bsk)] if len(ref) else ref
        bnd = null_bounds(g, level) if len(g) else {"null_reps": 0, "alpha_null_q": np.nan,
                                                     "s_null_lo": np.nan, "s_null_hi": np.nan}
        if null is not None:
            return {**bnd, "batch_sizes": bsk, **classify(summary, bnd, inv_tol, drift_tol, wild_max)}
        _BOUNDS[key] = bnd
    bnd = _BOUNDS[key]
    return {**bnd, "batch_sizes": bsk, **classify(summary, bnd, inv_tol, drift_tol, wild_max)}


def null_bounds(null: pd.DataFrame, level: float = 0.90) -> dict:
    """5% quantile of alpha_mean and the central 90% of s on the Gaussian pools of one key."""
    q = (1 - level) / 2
    a = null.alpha_mean.dropna()
    s = null.scale_exponent.dropna()
    if len(a) < 20:
        return {"null_reps": int(len(a)), "alpha_null_q": np.nan, "s_null_lo": np.nan, "s_null_hi": np.nan}
    return {"null_reps": int(len(a)), "alpha_null_q": float(a.quantile(q)),
            "s_null_lo": float(s.quantile(q)), "s_null_hi": float(s.quantile(1 - q))}


# ----------------------------------------------------------------------------- the verdict
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def classify(row, bounds: dict, inv_tol: float = 0.03, drift_tol: float = 0.05,
             wild_max: float | None = 1.9) -> dict:
    """Verdict for one test's summary row (a dict or a pandas row) given its Gaussian reference.

    "invariant (wild)" also needs alpha_mean <= wild_max; between wild_max and the Gaussian quantile the verdict is
    undecided (see the module docstring). wild_max = None or 2 turns the margin off."""
    b_lo, b_hi = _f(row["b_lo"]), _f(row["b_hi"])
    abar, s = _f(row["alpha_mean"]), _f(row["scale_exponent"])
    g_lo, g_hi = _f(row["scale_gap_lo"]), _f(row["scale_gap_hi"])
    aq, s_lo, s_hi = bounds["alpha_null_q"], bounds["s_null_lo"], bounds["s_null_hi"]
    out = {"slope_reading": "undecided", "level_reading": "unknown", "scale_reading": "unknown"}

    if not (np.isfinite(b_lo) and np.isfinite(b_hi)):
        out.update(verdict="undecided", verdict_reason="no bootstrap interval for b")
        return out
    if b_lo >= -inv_tol and b_hi <= inv_tol:
        out["slope_reading"] = "flat"
    elif b_lo > drift_tol:
        out["slope_reading"] = "drift"
    if np.isfinite(abar) and np.isfinite(aq):
        out["level_reading"] = "below 2" if abar < aq else "consistent with 2"
        if out["level_reading"] == "below 2":
            ok = np.isfinite(g_lo) and np.isfinite(g_hi) and g_lo <= 0 <= g_hi
            out["scale_reading"] = "consistent with 1/alpha-1" if ok else "excludes 1/alpha-1"
        else:
            ok = np.isfinite(s) and s_lo <= s <= s_hi
            out["scale_reading"] = "near -1/2" if ok else "away from -1/2"

    if out["slope_reading"] == "drift":
        out.update(verdict="drift", verdict_reason=f"b interval [{b_lo:.3f}, {b_hi:.3f}] above {drift_tol:g}")
    elif out["slope_reading"] == "undecided":
        out.update(verdict="undecided",
                   verdict_reason=f"slope: b interval [{b_lo:.3f}, {b_hi:.3f}] neither inside +-{inv_tol:g} "
                                  f"nor above {drift_tol:g}")
    elif out["level_reading"] == "unknown":
        why = "no alpha_mean" if not np.isfinite(abar) else "no Gaussian reference for this setting"
        out.update(verdict="undecided", verdict_reason=f"slope flat, level unknown ({why})")
    elif out["level_reading"] == "below 2":
        if wild_max is not None and abar > wild_max:
            out.update(verdict="undecided",
                       verdict_reason=f"slope flat and alpha_mean {abar:.3f} below 2 ({aq:.3f}), but above {wild_max:g}: "
                                      f"near 2, where finite variance with zeta ~ 3-4 looks the same up to B = 1,024")
        elif out["scale_reading"] == "consistent with 1/alpha-1":
            out.update(verdict="invariant (wild)",
                       verdict_reason=f"slope flat; alpha_mean {abar:.3f} < {aq:.3f} (Gaussian 5%)"
                                      + (f" and <= {wild_max:g}; " if wild_max is not None else "; ") +
                                      f"s - (1/alpha-1) interval [{g_lo:.3f}, {g_hi:.3f}] contains 0")
        else:
            out.update(verdict="undecided",
                       verdict_reason=f"slope flat and alpha_mean {abar:.3f} below 2, but s = {s:.3f} excludes "
                                      f"1/alpha-1 = {1 / abar - 1:.3f} (gap interval [{g_lo:.3f}, {g_hi:.3f}])")
    else:
        if out["scale_reading"] == "near -1/2":
            out.update(verdict="flat at 2 (mild)",
                       verdict_reason=f"slope flat; alpha_mean {abar:.3f} >= {aq:.3f} (Gaussian 5%); "
                                      f"s = {s:.3f} in Gaussian range [{s_lo:.3f}, {s_hi:.3f}]")
        else:
            out.update(verdict="undecided",
                       verdict_reason=f"slope flat and alpha_mean {abar:.3f} consistent with 2, but s = {s:.3f} "
                                      f"outside the Gaussian range [{s_lo:.3f}, {s_hi:.3f}]")
    return out


def needed_keys(df: pd.DataFrame, batch_sizes) -> list[tuple]:
    bsk = bs_key(batch_sizes)
    keys = df[["n_pool", "n_blocks"]].dropna().astype(int).drop_duplicates()
    return [(int(p), int(nb), bsk) for p, nb in keys.itertuples(index=False)]


def apply_verdicts(df: pd.DataFrame, null: pd.DataFrame, batch_sizes, inv_tol: float = 0.03,
                   drift_tol: float = 0.05, level: float = 0.90, wild_max: float | None = 1.9) -> pd.DataFrame:
    """Add the reference bounds, the three readings and the verdict to every row of a summary table."""
    missing = [c for c in SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"not an h71_test summary: columns {missing} missing")
    bsk = bs_key(batch_sizes)
    out = []
    for _, row in df.iterrows():
        g = null[(null.n_pool == _f(row["n_pool"])) & (null.n_blocks == _f(row["n_blocks"]))
                 & (null.batch_sizes == bsk)] if len(null) else null
        bnd = null_bounds(g, level) if len(g) else {"null_reps": 0, "alpha_null_q": np.nan,
                                                     "s_null_lo": np.nan, "s_null_hi": np.nan}
        out.append({**bnd, "batch_sizes": bsk, **classify(row, bnd, inv_tol, drift_tol, wild_max)})
    extra = pd.DataFrame(out, index=df.index)
    keep = df.drop(columns=[c for c in extra.columns if c in df.columns])
    return pd.concat([keep, extra], axis=1)


def infer_batch_sizes(df: pd.DataFrame, path: Path | None = None) -> tuple[list[int] | None, str]:
    """Batch sizes of a summary file: alpha_B<B> columns (P0b), else the B column of h71_per_B.csv beside it."""
    cols = sorted(int(m.group(1)) for c in df.columns if (m := re.fullmatch(r"alpha_B(\d+)", str(c))))
    if cols:
        return cols, "alpha_B columns"
    if path is not None:
        side = Path(path).with_name("h71_per_B.csv")
        if side.exists() and side.stat().st_size > 0:
            Bs = sorted(int(b) for b in pd.read_csv(side, usecols=["B"]).B.dropna().unique())
            if Bs:
                return Bs, side.name
    return None, ""


def verdict_counts(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = [c for c in (by or []) if c in df.columns]
    t = df.groupby(by + ["verdict"], dropna=False).size().unstack(fill_value=0) if by else \
        df.verdict.value_counts().to_frame().T
    return t.reindex(columns=list(VERDICTS), fill_value=0)


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    import config   # repo root on the path when run as python -m common.h71_verdict

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="summary CSVs written by h71_test (h71_summary.csv, h71_*tests.csv, "
                                             "block_check_bootstrap.csv)")
    ap.add_argument("--null", default=str(PILOT_ROOT / "results" / "p0b_calibration" / "h71_gauss_null.csv"),
                    help="further Gaussian pools, read with reference/h71_gauss_null.csv; pools simulated for a "
                         "missing setting are added to this file")
    ap.add_argument("--null-reps", type=int, default=config.H71_NULL_REPS,
                    help="Gaussian pools per setting (default %(default)s)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for missing Gaussian pools")
    ap.add_argument("--batch-sizes", default=None,
                    help="comma-separated B values, when the file has no alpha_B columns and no h71_per_B.csv "
                         "beside it (default: config.BATCH_SIZES)")
    ap.add_argument("--inv-tol", type=float, default=config.H71["inv_tol"])
    ap.add_argument("--drift-tol", type=float, default=config.H71["drift_tol"])
    ap.add_argument("--level", type=float, default=config.H71["level"])
    ap.add_argument("--wild-max", type=float, default=config.H71.get("wild_max_alpha"),
                    help="invariant (wild) also needs mean alpha-hat <= this (default %(default)s; 2 turns it off)")
    ap.add_argument("--suffix", default="_verdict", help="output file: <input stem><suffix>.csv")
    args = ap.parse_args(argv)

    null_path = Path(args.null)
    extra = load_null(null_path)                 # this file only: what is written back
    null = load_reference(null_path)
    for f in args.files:
        path = Path(f)
        if not path.exists() or path.stat().st_size == 0:
            print(f"{path}: missing or empty, skipped")
            continue
        df = pd.read_csv(path)
        if args.batch_sizes:
            Bs, src = [int(b) for b in args.batch_sizes.split(",")], "--batch-sizes"
        else:
            Bs, src = infer_batch_sizes(df, path)
            if Bs is None:
                Bs, src = list(config.BATCH_SIZES), "config.BATCH_SIZES (no alpha_B columns or h71_per_B.csv)"
        new = simulate_null(needed_keys(df, Bs), args.null_reps, args.seed, args.jobs, null)
        if len(new):
            null = pd.concat([null, new], ignore_index=True)
            extra = pd.concat([extra, new], ignore_index=True)
            null_path.parent.mkdir(parents=True, exist_ok=True)
            extra.to_csv(null_path, index=False)
            print(f"Gaussian reference extended: {null_path}")
        out = apply_verdicts(df, null, Bs, args.inv_tol, args.drift_tol, args.level, args.wild_max)
        dest = path.with_name(path.stem + args.suffix + ".csv")
        out.to_csv(dest, index=False)
        by = [c for c in ("model", "size", "pool", "cutoff", "stage", "direction") if c in out.columns][:2]
        print(f"\n{path} -> {dest}  (B = {bs_key(Bs)} from {src})")
        print(verdict_counts(out, by).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
