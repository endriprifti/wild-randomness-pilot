"""P7: P4's transformer arm under SGD with momentum (the optimizer swap; Endri's proposal of 2026-11-27, fixed in the
README before any P7 run). Reported beside P4, never pooled with it.

Why: P4 held the data fixed and changed only the clipping rule. Under SGD, ResNet-18's fixed thresholds created bursts in
the gradient noise (H_link memory and width rose with clipping); under AdamW the 14M transformer's did not. The two arms
differ in optimizer, architecture, data and batch. P7 changes only the optimizer of the transformer arm.

Design (config.py "p7_transformer_sgd"; everything else is P4's transformer arm, p4_clipping.py and its settings):
  optimizer  torch.optim.SGD, momentum 0.9, Nesterov off, weight decay 5e-4 (P4's ResNet-18 setting); P4's warm-up over
             1% of steps (91), then a constant rate; P4's data (same pinned revision and cached token IDs), 9,155 steps
             of 64 x 512 tokens, eight fixed directions, unclipped projections before clipping (hgrad_series/), the
             divergence rule (factor 3), P4's measurements at the end without lambda_max, the Hessian trace and
             adaptive sharpness (P7 does not use them)
  Phase 0    the learning rate, by the rule fixed in config.py: seed 99, unclipped, rates 0.03, 0.1, 0.3, 1.0; a rate
             passes if the run does not diverge and its test loss <= L0 - 0.7 (L0 - 4.460), L0 = the run's first
             training loss, 4.460 = the mean test loss of P4's five unclipped AdamW runs; the passing rate with the
             lowest test loss is chosen. If none passes, two more rates half a decade beyond the better end of the
             grid (0.01 and 0.003 if the 0.03 run's test loss is at most the 1.0 run's, a diverged run counting as
             infinite; else 3.0 and 10), same rule. If none passes again, P7 stops before Phase 1 (phase0.json:
             status "stopped") and reads "SGD with momentum did not train the transformer to the bound at P4's
             budget". Seed 99 is left out of every analysis.
  Phase 1    seeds 0-4 (P4's: each run shares its initial weights, data order and directions with an AdamW run of P4),
             seed by seed: unclipped; fixed thresholds at targets 1, 10, 50% (the (1 - p) quantile of the same seed's
             unclipped post-warm-up gradient norms, P4's rule); the 10% target at matched update length (learning rate
             times the unclipped run's mean update length over the plain 10% run's, P4's rule). 25 runs.
The analysis (p7_stats.py) runs after h_link.py on the Phase 1 series (p7_queue.json).

Outputs (results/p7_transformer_sgd/[smoke/]): phase0.json and phase0/ (its runs, resumable), runs.csv,
hgrad_series/<tag>.npy (P4's names, e.g. transformer_s0_fixed0.1_matched), runs/<tag>.json (one per finished run, as
P4: h_link.py --p4 reads this folder like a P4 folder), progress.json, run.json; stopped.json after a STOP.

Resume: every run is saved as it ends; the same command goes on from the saved runs (runs/settings.json and
phase0/runs/settings.json refuse other settings). Phase 0 is decided again from its saved runs, so a restart never
changes the chosen rate.

Lack of credit (as P6b): touch results/p7_transformer_sgd/STOP. The job checks it before each Phase 0 run and before
each seed of Phase 1 (whole seeds only; the run or seed that is running finishes), writes stopped.json and exits 0;
the analysis runs on the complete seeds.

Commands (in the pilot folder inside the venv):
  python p7_transformer_sgd.py --stop-file results/p7_transformer_sgd/STOP     # the queue's job (p7_queue.json)
  python p7_transformer_sgd.py --smoke                                         # P4's smoke sizes, two seeds
  python p7_transformer_sgd.py --plan --device cpu                             # what is saved and what is left
"""

from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path

import numpy as np

import config
import p4_clipping as p4
from common import runtime
from common import units as units_mod

KIND = "transformer"
STOPPED_READING = "SGD with momentum did not train the transformer to the bound at P4's budget"


# ----------------------------------------------------------------------------- settings
def settings(smoke: bool) -> tuple[dict, dict]:
    """(P4's settings, P7's settings). P7 changes only the optimizer and the learning rate of P4's transformer arm."""
    return config.get("p4_clipping", smoke), config.get("p7_transformer_sgd", smoke)


def with_lr(cfg: dict, lr: float) -> dict:
    """P4's settings with the transformer's learning rate set to lr (train() uses lr * lr_scale)."""
    out = copy.deepcopy(cfg)
    out["transformer"]["lr"] = float(lr)
    return out


def arms(p7: dict) -> list[tuple[str, float, bool]]:
    return [(str(r), float(p), bool(m)) for r, p, m in p7["arms"]]


def run_tag(seed: int, rule: str, p: float, matched: bool) -> str:
    return p4.run_tag(KIND, seed, rule, p, matched)


# ----------------------------------------------------------------------------- Phase 0: the rule
def phase0_bound(first_loss: float, p0: dict) -> float:
    """The test loss a Phase 0 run must reach: L0 - capture (L0 - AdamW's mean test loss)."""
    return float(first_loss) - float(p0["capture"]) * (float(first_loss) - float(p0["adamw_test_loss"]))


def phase0_passes(row: dict, p0: dict) -> bool:
    if row.get("diverged") or not np.isfinite(row.get("test_loss", np.nan)):
        return False
    return bool(row["test_loss"] <= phase0_bound(row["first_train_loss"], p0))


def _loss_or_inf(row: dict | None) -> float:
    if row is None or row.get("diverged") or not np.isfinite(row.get("test_loss", np.nan)):
        return math.inf
    return float(row["test_loss"])


def better_end(rows: dict[float, dict], p0: dict) -> str:
    """'low' if the lowest rate of the grid did at least as well (test loss, a diverged run infinite) as the highest,
    else 'high'."""
    lrs = sorted(float(x) for x in p0["lrs"])
    return "low" if _loss_or_inf(rows.get(lrs[0])) <= _loss_or_inf(rows.get(lrs[-1])) else "high"


def phase0_decide(rows: dict[float, dict], p0: dict) -> dict:
    """The Phase 0 decision from the runs done so far (rate -> row). Returns {status, chosen_lr, next, ...}: status
    "chosen" (chosen_lr set), "run" (the rates in `next` still to run) or "stopped" (none passed after the extension)."""
    grid = [float(x) for x in p0["lrs"]]
    table = []
    for lr in sorted(rows):
        r = rows[lr]
        table.append({"lr": lr, "diverged": bool(r.get("diverged")), "first_train_loss": r.get("first_train_loss"),
                      "test_loss": r.get("test_loss"), "bound": phase0_bound(r["first_train_loss"], p0)
                      if r.get("first_train_loss") is not None and np.isfinite(r["first_train_loss"]) else None,
                      "passes": phase0_passes(r, p0), "stage": "grid" if lr in grid else "extension"})
    out = {"rule": {"grid": grid, "extend_low": [float(x) for x in p0["extend_low"]],
                    "extend_high": [float(x) for x in p0["extend_high"]], "capture": float(p0["capture"]),
                    "adamw_test_loss": float(p0["adamw_test_loss"]), "seed": int(p0["seed"])},
           "runs": table, "chosen_lr": None, "extension": None, "next": []}
    todo = [lr for lr in grid if lr not in rows]
    if todo:
        return {**out, "status": "run", "next": todo}

    def choose(stage_rows):
        ok = [t for t in stage_rows if t["passes"]]
        return min(ok, key=lambda t: (t["test_loss"], t["lr"]))["lr"] if ok else None
    chosen = choose([t for t in table if t["stage"] == "grid"])
    if chosen is not None:
        return {**out, "status": "chosen", "chosen_lr": chosen}
    side = better_end(rows, p0)
    ext = [float(x) for x in p0["extend_low" if side == "low" else "extend_high"]]
    out["extension"] = {"side": side, "lrs": ext}
    todo = [lr for lr in ext if lr not in rows]
    if todo:
        return {**out, "status": "run", "next": todo}
    chosen = choose([t for t in table if t["lr"] in ext])
    if chosen is not None:
        return {**out, "status": "chosen", "chosen_lr": chosen}
    return {**out, "status": "stopped", "reading": STOPPED_READING}


# ----------------------------------------------------------------------------- the job
def stop_requested(path: Path | None) -> bool:
    return path is not None and Path(path).exists()


def phase0_row(cfg, p7, data, lr, dev, run) -> dict:
    """One Phase 0 run: seed 99, unclipped, rate lr; train, then the test loss (nothing else is measured)."""
    seed = int(p7["phase0"]["seed"])
    model, info, _proj, _ = p4.train(KIND, with_lr(cfg, lr), data, "none", 0.0, seed, 1.0, dev, run,
                                     opt_cfg=p7["optimizer"])
    info.pop("norms_post_warmup")
    row = {"phase": 0, "model": KIND, "seed": seed, "rule": "fixed", "p_clip_target": 0.0, "matched": False,
           "lr_grid": float(lr), **info}
    if not info["diverged"]:
        bs = min(64, cfg["measure"].get("micro_batch", {}).get(KIND) or 64)
        row["test_loss"], _ = p4.mean_loss(KIND, model, data["test"], dev, bs=bs)
    else:
        row["test_loss"] = float("nan")
    del model
    return row


def main(argv=None):
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--stop-file", default=None, help="if this file exists, start nothing more (checked before "
                                                      "Phase 0 and before each seed); stopped.json, exit 0")
    ap.add_argument("--seeds", default=None, help="comma-separated Phase 1 seeds (default: config.py's, 0-4)")
    ap.add_argument("--plan", action="store_true", help="print the saved and remaining runs; run nothing")
    args = ap.parse_args(argv)
    cfg, p7 = settings(args.smoke)
    run = runtime.start_run("p7_transformer_sgd", {**cfg, "p7": p7}, args)
    dev = run.device
    stop_file = Path(args.stop_file) if args.stop_file else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in p7["seeds"]]
    out = run.out_dir
    base = {"p4_cfg": cfg, "p7": {k: v for k, v in p7.items() if k not in ("analysis", "hardware")},
            "seed": run.seed, "smoke": run.smoke}
    u0 = units_mod.Units(out / "phase0", base, "p7_phase0", sub="runs")

    def phase0_rows() -> dict[float, dict]:
        return {float(u0.get(k)["row"]["lr_grid"]): u0.get(k)["row"] for k in u0.saved()}

    if args.plan:
        dec = phase0_decide(phase0_rows(), p7["phase0"])
        print(f"Phase 0: {len(u0.saved())} run(s) saved; status {dec['status']}"
              + (f", chosen rate {dec['chosen_lr']:g}" if dec["chosen_lr"] else "")
              + (f", next {dec['next']}" if dec["next"] else ""))
        saved1 = sorted(p.stem for p in (out / "runs").glob("transformer_*.json")) if (out / "runs").exists() else []
        print(f"Phase 1: {len(saved1)} of {len(seeds) * len(arms(p7))} run(s) saved")
        return 0

    def stopped(where: str) -> int:
        units_mod.atomic_json(out / "stopped.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                     "where": where, "stop_file": str(stop_file)})
        run.note(f"{stop_file} exists: stopped {where}; saved runs stay (stopped.json)")
        runtime.finish_run(run, status="stopped", extra={"stopped": where})
        return 0

    def phase0_stop(dec: dict) -> str | None:
        """Where to stop if the STOP file exists and a Phase 0 run is still to start (else None)."""
        if dec["status"] != "run" or not stop_requested(stop_file):
            return None
        return "before Phase 0" if not u0.saved() else f"in Phase 0, before rate {dec['next'][0]:g}"

    # ---- Phase 0
    dec = phase0_decide(phase0_rows(), p7["phase0"])
    if phase0_stop(dec):
        return stopped(phase0_stop(dec))
    data = p4.transformer_data(cfg, run)
    while dec["status"] == "run":
        if phase0_stop(dec):
            return stopped(phase0_stop(dec))
        lr = dec["next"][0]
        tag = f"phase0_lr{lr:g}"
        u0.plan(f"phase0_lr{x:g}" for x in sorted(set(phase0_rows()) | set(dec["next"])))
        u0.start(tag)
        row = phase0_row(cfg, p7, data, lr, dev, run)
        run.note(f"Phase 0 rate {lr:g}: diverged={row['diverged']} first loss {row['first_train_loss']:.4f} "
                 f"test loss {row['test_loss']:.4f} (bound "
                 f"{phase0_bound(row['first_train_loss'], p7['phase0']):.4f})")
        u0.put(tag, {"row": row})
        dec = phase0_decide(phase0_rows(), p7["phase0"])
    units_mod.atomic_json(out / "phase0.json", dec)
    runtime.write_csv(out / "phase0" / "phase0_runs.csv", [phase0_rows()[k] for k in sorted(phase0_rows())])
    if dec["status"] == "stopped":
        run.note(f"Phase 0: no rate passed, the extension included; P7 stops before Phase 1: {STOPPED_READING}")
        runtime.finish_run(run, extra={"phase0": dec, "n_runs": 0})
        return 0
    lr = float(dec["chosen_lr"])
    run.note(f"Phase 0: chosen rate {lr:g} (phase0.json)")

    # ---- Phase 1
    cfg1 = with_lr(cfg, lr)
    skip = set(p7.get("skip_measures", []))
    (out / "hgrad_series").mkdir(exist_ok=True)
    units = units_mod.Units(out, {**base, "lr": lr, "batch_order": "shuffled"}, "p7_transformer_sgd", sub="runs")
    plan = [(s, r, p, m) for s in seeds for r, p, m in arms(p7)]
    units.plan(run_tag(s, r, p, m) for s, r, p, m in plan)
    units.write_progress()
    rows, stop_at = [], None
    for seed in seeds:
        if stop_requested(stop_file) and not all(units.get(run_tag(seed, r, p, m)) for r, p, m in arms(p7)):
            stop_at = seed
            break
        ref_len, ref_norms = None, None
        for rule, p, matched in arms(p7):
            tag = run_tag(seed, rule, p, matched)
            is_ref = rule == "fixed" and p == 0.0 and not matched
            norms_path = units.dir / f"{KIND}_s{seed}_ref_norms.npy"
            saved = units.get(tag)
            if saved is None:
                units.start(tag)
                scale = 1.0
                if matched:    # P4's rule: rate times the unclipped run's mean update length over the plain arm's
                    nominal = next(r for r in rows if r["seed"] == seed and r["rule"] == rule
                                   and r["p_clip_target"] == p and not r["matched"])
                    scale = ref_len / nominal["mean_update_len"] if nominal["mean_update_len"] > 0 else 1.0
                rng = units_mod.rng_for(units_mod.seed_for(run.seed, tag))   # P4's: the twin draws the same numbers
                model, info, proj, _ = p4.train(KIND, cfg1, data, rule if rule != "fixed" or p else "none", p or 0.0,
                                                seed, scale, dev, run, ref_norms=ref_norms, opt_cfg=p7["optimizer"])
                norms = info.pop("norms_post_warmup")
                np.save(out / "hgrad_series" / f"{tag}.npy", proj)
                row = {"model": KIND, "seed": seed, "rule": rule, "p_clip_target": p, "matched": matched,
                       "lr_scale": scale, "data_source": data["source"], "phase0_lr": lr,
                       "momentum": p7["optimizer"]["momentum"], "weight_decay": p7["optimizer"]["weight_decay"],
                       **info}
                if not info["diverged"]:
                    row.update(p4.measure(KIND, model, data, cfg1, run, rng, dev, proj, info["warmup_steps"],
                                          skip=skip))
                run.note(f"{tag}: lr={info['lr']:.4g} p_clip={info['p_clip_realized']:.3g} "
                         f"diverged={info['diverged']} test_loss={row.get('test_loss', float('nan')):.4f}")
                del model
                if is_ref:
                    np.save(norms_path, np.asarray(norms, dtype=float))
                saved = {"row": row, "h71_per_B": [], "h71_summary": []}
                units.put(tag, saved)
            rows.append(saved["row"])
            if is_ref:
                ref_len, ref_norms = saved["row"]["mean_update_len"], np.load(norms_path)
            runtime.write_csv(out / "runs.csv", rows)
    runtime.write_csv(out / "runs.csv", rows)
    if stop_at is not None:
        return stopped(f"before seed {stop_at} of Phase 1")
    runtime.finish_run(run, extra={"phase0": {k: dec[k] for k in ("status", "chosen_lr", "extension")},
                                   "n_runs": len(rows), "n_diverged": int(sum(bool(r["diverged"]) for r in rows))})
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
