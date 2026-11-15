"""Expected L40S hours of the 2026-11-15 reruns, from the first run's own L40S timings (no GPU needed).

Reads logs/progress.txt (job hours) and the first run's results/p5_escape/{grid,restarts}.csv and run.json
(wall-clock per grid cell), so every number comes from measurements on the same card, not from CPU timings.

  python rerun_hours.py                       # in the repository folder on the L40S, after run_all.sh
  python rerun_hours.py --root ~/pilot --alpha 1.2

Estimates (upper bounds where marked):
  p2 rerun        = the first P2 job's hours (bfloat16 runs at the float16 tensor-core rate on Ada; the NaN
                    Pythia rows had run the longest isotropic search, so this is an upper bound)
  p5 floor cells  = prologue + sum over floor cells of 25 x their first-run wall time (each did 250 steps and
                    one check; the rerun does the same 250 steps and up to 25 checks): upper bound
  p5 alpha cells  = prologue + (cells) x (the longest injected cell of the first run, i.e. one that ran to
                    k_cens): upper bound, since with --second-reading a cell stops only when every restart has
                    fired both readings
  p5 refit, H3.1 flags, PopQA analysis: CPU only, minutes
  prologue        = the first P5 job's wall clock minus the sum of its cells (data, W_S, alpha at W_S)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


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


def p5_estimates(folder: Path, alphas: list[float]) -> dict:
    grid = pd.read_csv(folder / "grid.csv")
    rest = pd.read_csv(folder / "restarts.csv")
    cfg = json.loads((folder / "run.json").read_text())
    every = int(cfg["config"]["check_every"])
    wall = float(cfg.get("wall_clock_seconds", grid.wall_s.sum()))
    prologue = max(0.0, wall - float(grid.wall_s.sum()))
    key = ["mode", "B", "eta", "g_max", "alpha_injected"]
    fl = rest.groupby(key, dropna=False).apply(lambda g: bool(g.exited.all() and (g.k <= every).all()),
                                               include_groups=False).rename("_floor_cell").reset_index()
    g = grid.merge(fl, on=key, how="left")
    floor = g[g._floor_cell.fillna(False).astype(bool)]
    inj = g[g["mode"] == "injected"]
    sel = inj[inj.alpha_injected.round(6).isin([round(a, 6) for a in alphas])]
    longest = float(inj.wall_s.max()) if len(inj) else float("nan")
    return {"check_every": every, "prologue_h": prologue / 3600, "n_floor": len(floor),
            "floor_first_run_h": float(floor.wall_s.sum()) / 3600,
            "floor_rerun_h_max": (prologue + 25 * float(floor.wall_s.sum())) / 3600,
            "n_alpha_cells": len(sel), "alpha_first_run_h": float(sel.wall_s.sum()) / 3600,
            "alpha_rerun_h_max": (prologue + len(sel) * longest) / 3600,
            "floor_cells": floor[["mode", "B", "eta", "g_max", "alpha_injected"]].to_dict("records")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository folder with logs/ and results/")
    ap.add_argument("--alpha", default="1.2", help="injected alphas that are rerun (comma-separated)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    hours = job_hours(root / "logs" / "progress.txt")
    print("first-run job hours (logs/progress.txt):", {k: hours[k] for k in sorted(hours)} or "none found")
    total = 0.0
    if "p2_w4a4" in hours:
        print(f"p2_w4a4 rerun (both models, bfloat16): about {hours['p2_w4a4']:.2f} h (upper bound)")
        total += hours["p2_w4a4"]
    p5 = root / "results" / "p5_escape"
    if (p5 / "grid.csv").exists():
        e = p5_estimates(p5, [float(a) for a in args.alpha.split(",")])
        print(f"p5 prologue (data, W_S, alpha at W_S): {e['prologue_h']:.2f} h per job")
        print(f"p5 floor cells: {e['n_floor']} (first run {e['floor_first_run_h']:.3f} h); "
              f"rerun with --fine-check 10:{e['check_every']}: at most {e['floor_rerun_h_max']:.2f} h")
        print(f"p5 alpha={args.alpha} cells: {e['n_alpha_cells']} (first run {e['alpha_first_run_h']:.2f} h); "
              f"rerun with --second-reading: at most {e['alpha_rerun_h_max']:.2f} h")
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
