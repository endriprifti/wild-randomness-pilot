"""Expected L40S hours of the 2026-11-15 reruns, from the first run's own L40S timings (no GPU needed).

Reads logs/progress.txt (job hours) and the first run's results/p5_escape/{grid,restarts}.csv and run.json
(wall-clock per grid cell), so every number comes from measurements on the same card, not from CPU timings.

  python rerun_hours.py                       # in the repository folder on the L40S, after run_all.sh
  python rerun_hours.py --root ~/pilot --alpha 1.2 --fine-check 10:250

How the P5 estimates are built:
  check cost      a cell that ran k steps did k / check_every checks, and each check interval (check_every steps
                  plus one check) costs at least one check, so wall_s / checks of any cell bounds one check's cost
                  from above; the smallest such value over the grid is used (c_max)
  extra checks    --fine-check 10:250 adds 250/10 - 1 = 24 checks per cell (25 checks up to step 250 instead of
                  one, then every 250 steps as before); the steps and their random numbers are unchanged
  p5 floor cells  = prologue + their first-run wall + cells x 24 x c_max (upper bound); the alpha = 1.2 cells are
                    left out, as in the floor command (--alphas 1.5,1.8,2.0)
  p5 alpha cells  = prologue + first-run wall + cells x 24 x c_max (upper bound); a cell that stopped before
                    k_cens in the first run is counted at the longest injected cell that ran to k_cens, since with
                    --second-reading a restart keeps stepping until both readings have fired
  p2 rerun        = the first P2 job's hours (bfloat16 runs at the float16 tensor-core rate on Ada; the NaN
                    Pythia rows had run the longest isotropic search, 21 perplexity passes per scheme)
  p5 refit, H3.1 flags, PopQA analysis: CPU only, minutes
  prologue        = the first P5 job's wall clock minus the sum of its cells (data, W_S, alpha at W_S)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["mode", "B", "eta", "g_max", "alpha_injected"]


def job_hours(progress: Path) -> dict[str, float]:
    """Hours of the last END line of each job (exit 0 or not)."""
    out = {}
    if not progress.exists():
        return out
    for line in progress.read_text().splitlines():
        f = line.split()
        if len(f) > 3 and f[1] == "END" and "elapsed" in f:
            try:
                out[f[2]] = float(f[f.index("elapsed") + 1].rstrip("h"))
            except ValueError:
                pass
    return out


def cells(folder: Path) -> tuple[pd.DataFrame, dict, float]:
    """One row per grid cell: wall_s, steps run (k_stop), checks, all_first (every restart exited at the first
    check), wall_s per check. Works on first-run files (k, exited only) and on new ones (k_stop)."""
    grid = pd.read_csv(folder / "grid.csv")
    rest = pd.read_csv(folder / "restarts.csv")
    run = json.loads((folder / "run.json").read_text())
    cfg = run["config"]
    every = int(cfg["check_every"])
    stop_col = "k_stop" if "k_stop" in rest else "k"
    per = rest.groupby(KEY, dropna=False).apply(
        lambda g: pd.Series({"k_stop": float(g[stop_col].max()),
                             "all_first": bool(g.exited.astype(bool).all() and (g.k <= every).all())}),
        include_groups=False).reset_index()
    g = grid.merge(per, on=KEY, how="left")
    g["checks"] = np.maximum(1, (g.k_stop // every).fillna(1)).astype(int)
    g["s_per_check"] = g.wall_s / g.checks
    wall = float(run.get("wall_clock_seconds", g.wall_s.sum()))
    return g, cfg, max(0.0, wall - float(g.wall_s.sum()))


def p5_estimates(folder: Path, alphas: list[float], fine: tuple[int, int] = (10, 250)) -> dict:
    g, cfg, prologue = cells(folder)
    every, k_cens = int(cfg["check_every"]), int(cfg["k_cens"])
    extra = fine[1] // fine[0] - fine[1] // every                     # 24 for 10:250 with checks every 250
    c_max = float(g.s_per_check.min())
    is_a = (g["mode"] == "injected") & g.alpha_injected.round(6).isin([round(a, 6) for a in alphas])
    floor = g[g.all_first.fillna(False).astype(bool) & ~is_a]
    sel = g[is_a]
    full = g[(g["mode"] == "injected") & (g.k_stop >= k_cens)]
    full_wall = float(full.wall_s.max()) if len(full) else float(sel.wall_s.max())
    a_first = float(sel.wall_s.sum())
    a_bound = float(np.where(sel.k_stop >= k_cens, sel.wall_s, full_wall).sum())
    return {"check_every": every, "k_cens": k_cens, "extra_checks": extra, "prologue_h": prologue / 3600,
            "c_max_s": c_max, "c_max_cell": g.loc[g.s_per_check.idxmin(), KEY].to_dict(),
            "n_floor": len(floor), "floor_first_run_h": float(floor.wall_s.sum()) / 3600,
            "floor_rerun_h_max": (prologue + floor.wall_s.sum() + len(floor) * extra * c_max) / 3600,
            "n_alpha_cells": len(sel), "n_alpha_short": int((sel.k_stop < k_cens).sum()),
            "alpha_first_run_h": a_first / 3600,
            "alpha_rerun_h_max": (prologue + a_bound + len(sel) * extra * c_max) / 3600,
            "floor_cells": floor[["mode", "B", "eta", "g_max", "alpha_injected", "wall_s"]].to_dict("records")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository folder with logs/ and results/")
    ap.add_argument("--alpha", default="1.2", help="injected alphas of the second-reading rerun (comma-separated)")
    ap.add_argument("--fine-check", default="10:250", help="EVERY:UNTIL of the reruns")
    args = ap.parse_args(argv)
    root = Path(args.root)
    fine = tuple(int(v) for v in args.fine_check.split(":"))
    hours = job_hours(root / "logs" / "progress.txt")
    print("first-run job hours (logs/progress.txt):", {k: hours[k] for k in sorted(hours)} or "none found")
    total = 0.0
    if "p2_w4a4" in hours:
        print(f"p2_w4a4 rerun (both models, bfloat16): at most about {hours['p2_w4a4']:.2f} h")
        total += hours["p2_w4a4"]
    p5 = root / "results" / "p5_escape"
    if (p5 / "grid.csv").exists():
        e = p5_estimates(p5, [float(a) for a in args.alpha.split(",")], fine)
        print(f"p5 prologue (data, W_S, alpha at W_S): {e['prologue_h']:.2f} h per job")
        print(f"one check costs at most {e['c_max_s']:.1f} s (cell {e['c_max_cell']}); --fine-check "
              f"{args.fine_check} adds {e['extra_checks']} checks per cell")
        print(f"p5 floor cells without alpha {args.alpha}: {e['n_floor']} (first run {e['floor_first_run_h']:.2f} h); "
              f"rerun at most {e['floor_rerun_h_max']:.2f} h")
        for c in e["floor_cells"]:
            print("   ", {k: v for k, v in c.items() if not (isinstance(v, float) and np.isnan(v))})
        print(f"p5 alpha={args.alpha} cells: {e['n_alpha_cells']}, {e['n_alpha_short']} of them stopped before "
              f"k_cens (first run {e['alpha_first_run_h']:.2f} h); rerun at most {e['alpha_rerun_h_max']:.2f} h")
        total += e["floor_rerun_h_max"] + e["alpha_rerun_h_max"]
    else:
        print(f"no {p5}/grid.csv: P5 estimates need the first run's grid")
    if "p5_escape_h31" in hours:
        print(f"p5_escape_h31 rerun (optional, adds lambda_min): about {hours['p5_escape_h31']:.2f} h")
    if "p4_transformer" in hours:
        print(f"p4_transformer: the failed run took {hours['p4_transformer']:.2f} h; the full job is still to be measured")
    print(f"total of the estimates above (without the optional H3.1 rerun and P4): at most {total:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
