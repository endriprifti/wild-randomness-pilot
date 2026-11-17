"""P3: H4.1, accuracy against state size n_s at matched parameters.

Arms: fractional block (modes tied to (nu, lambda)), S4 with HiPPO-LegS initialization, S4D,
Mamba (pure-PyTorch reference scan), optional frozen HiPPO-LegS. For every n_s the channel
width of each arm is set so that its parameter count matches the reference arm (S4D at H_ref),
and every arm gets the same learning-rate grid (equal tuning budget), steps and seeds.

Tasks:
  powerlaw     y_t = sum_k kappa_k g_{t-k}, kappa_k ~ (k+1)^-(1+nu*), lags 1..L (power-law memory)
  exponential  kappa_k ~ exp(-k / (tau_frac L))                        (exponential memory)
  lra_image    Long Range Arena image task: CIFAR-10 in grayscale as 1,024-step sequences, in one of the
               designs of config.py's lra_designs: "reduced" (the default since 2026-11-17: 3 epochs, n_s 2, 8
               and 32, seeds 1-3, tuned at n_s = 8 on seed 0) or "full" (--lra-design full, the first plan:
               20 epochs, all six n_s, seeds 0-4, tuned at n_s = 16)
Accuracy is R^2 on held-out sequences for the two regression tasks and classification
accuracy for lra_image. H4.1 is refuted if the baseline accuracy-state curves lie on or above
the fractional block's on the power-law task, or if any advantage does not depend on task type.

Every run is saved as it finishes, and a restart with the same --out goes on from the saved runs
(results/p3_fractional_memory/[smoke/] by default):
  runs/<task>/<phase>_<arch>_ns<n_s>_seed<seed>_lr<lr>_steps<steps>.json   one finished run, written atomically
  runs/<task>/settings.json   the settings that decide a run's result; a restart with others stops (new --out)
  runs.csv          one row per run incl. tuning trials, with params, FLOPs proxy, peak memory and wall-clock
  curves.csv        mean accuracy per task, arm and n_s over the finished main runs (n = seeds so far)
  states_to_target.csv, and run.json at the end
runs.csv, curves.csv and states_to_target.csv are rewritten after every run. A run's result depends only on
its settings: each run seeds itself from its seed, and the synthetic data are drawn in config order from the run
seed (as before 2026-11-17), whatever --tasks says; so restarts and the order of runs change nothing. Main runs
go seed by seed (every arm and state size of one seed, then the next seed), so a P3 stopped early leaves whole
seeds. A run stopped in the middle is lost and runs again from its start.

Commands:
  python p3_fractional_memory.py
  python p3_fractional_memory.py --plan              # saved and remaining runs, hours by measured step times
  python p3_fractional_memory.py --tasks powerlaw,exponential      # drop LRA first if time runs short
  python p3_fractional_memory.py --smoke
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from common import gpu_locks, hf, runtime, ssm

SYNTHETIC = ("powerlaw", "exponential")
CIFAR10_TRAIN = 50_000          # images in CIFAR-10's train split (checked when the data are loaded)
# the settings that decide one run's result besides its key (phase, arch, n_s, seed, lr, steps)
RESULT_KEYS = ("batch", "n_layers", "H_ref", "ref_arch", "weight_decay")


def memory_task(kind, L, n, cfg_t, rng):
    g = rng.standard_normal((n, L)).astype(np.float32)
    k = np.arange(L, dtype=np.float64)
    if kind == "powerlaw":
        kappa = (k + 1) ** (-(1 + cfg_t["nu_star"]))
    else:
        kappa = np.exp(-k / (cfg_t["tau_frac"] * L))
    kappa /= np.sqrt((kappa ** 2).sum())
    y = np.real(np.fft.irfft(np.fft.rfft(g, 2 * L) * np.fft.rfft(kappa, 2 * L), 2 * L))[:, :L]
    return torch.tensor(g[..., None]), torch.tensor(y[..., None].astype(np.float32))


def synthetic_data(cfg, seed, wanted):
    """{task: (Xtr, Ytr, Xte, Yte)} for the synthetic tasks in `wanted`. All of them are drawn, in config order,
    from one generator seeded with the run seed, as the code before 2026-11-17 did for a full run: a task's data
    do not depend on which tasks this run trains."""
    rng = runtime.rng(seed)
    out = {}
    for task, ct in cfg["tasks"].items():
        if task not in SYNTHETIC:
            continue
        Xtr, Ytr = memory_task(task, ct["L"], ct["n_train"], ct, rng)
        Xte, Yte = memory_task(task, ct["L"], ct["n_test"], ct, rng)
        if task in wanted:
            out[task] = (Xtr, Ytr, Xte, Yte)
    return out


def lra_image(cfg_t, run, rng):
    """CIFAR-10 as grayscale sequences (LRA image). The Hugging Face copy first, then torchvision."""
    def to_seq(imgs):
        x = torch.tensor(np.asarray(imgs), dtype=torch.float32)
        if x.ndim == 4 and x.shape[-1] == 3:
            x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
        x = x.reshape(len(x), -1, 1) / 255.0
        return (x - 0.5) / 0.25
    try:
        Xtr, ytr, Xte, yte, src = hf.load_cifar10(run, str(runtime.PILOT_ROOT / "data"),
                                                  cfg_t.get("limit_train"), cfg_t.get("limit_test"))
        Xtr, ytr, Xte, yte = to_seq(Xtr), torch.tensor(ytr), to_seq(Xte), torch.tensor(yte)
    except RuntimeError as e:
        if not run.offline_ok:
            raise
        run.fallback(f"CIFAR-10 for LRA image: random images with random labels (smoke only; {str(e)[:160]})")
        n_tr, n_te = cfg_t.get("limit_train", 256), cfg_t.get("limit_test", 64)
        Xtr = torch.randn(n_tr, 1024, 1); ytr = torch.randint(0, 10, (n_tr,))
        Xte = torch.randn(n_te, 1024, 1); yte = torch.randint(0, 10, (n_te,))
        src = "fallback"
    if "limit_train" in cfg_t:
        Xtr, ytr = Xtr[:cfg_t["limit_train"]], ytr[:cfg_t["limit_train"]]
        Xte, yte = Xte[:cfg_t["limit_test"]], yte[:cfg_t["limit_test"]]
    return (Xtr, ytr, Xte, yte), src


def evaluate(model, X, Y, task, device, batch=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            preds.append(model(X[s:s + batch].to(device)).float().cpu())
    P = torch.cat(preds)
    model.train()
    if task == "lra_image":
        return float((P.argmax(-1) == Y).float().mean())
    resid = ((P - Y) ** 2).sum()
    tot = ((Y - Y.mean()) ** 2).sum()
    return float(1 - resid / tot)


def train_one(arch, task, n_s, H, data, cfg, lr, seed, device, steps):
    runtime.set_all_seeds(seed)
    Xtr, Ytr, Xte, Yte = data
    L = Xtr.shape[1]
    cls = task == "lra_image"
    model = ssm.SSMModel(arch, 1, 10 if cls else 1, H, n_s, cfg["n_layers"], L, pool=cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    diverged = False
    for step in range(steps):
        idx = torch.randint(0, len(Xtr), (cfg["batch"],), generator=g)
        x, y = Xtr[idx].to(device), Ytr[idx].to(device)
        out = model(x)
        loss = F.cross_entropy(out, y) if cls else F.mse_loss(out, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    wall = time.perf_counter() - t0
    acc = evaluate(model, Xte, Yte, task, device) if not diverged else float("nan")
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
    return {"acc": acc, "params": ssm.n_params(model), "steps": steps, "wall_s": wall, "peak_mem_mb": peak,
            "diverged": diverged, "tokens_seen": steps * cfg["batch"] * L,
            "flops_proxy": 6 * ssm.n_params(model) * steps * cfg["batch"] * L}


# ----------------------------------------------------------------------------- the plan
def task_settings(cfg: dict, task: str, design: str) -> dict:
    """State sizes, seeds, tuning and steps per run of one task: the shared settings, and for lra_image those of
    its design (epochs x train images // batch steps per run, as before)."""
    s = {k: cfg[k] for k in ("n_states", "seeds", "tune_n_states", "tune_seed", "lr_grid")}
    if task == "lra_image":
        d = cfg["lra_designs"][design]
        s.update({k: d[k] for k in ("n_states", "seeds", "tune_n_states", "tune_seed") if k in d})
        n_train = cfg["tasks"][task].get("limit_train") or CIFAR10_TRAIN
        s.update(design=design, epochs=d["epochs"], n_train=n_train,
                 steps=d["epochs"] * max(1, n_train // cfg["batch"]))
    else:
        s.update(design="", steps=cfg["steps"])
    return s


def result_settings(cfg: dict, task: str, seed: int, smoke: bool) -> dict:
    """What decides a run's result apart from its key: the task's data and the shared training settings."""
    return json.loads(json.dumps({"task": cfg["tasks"][task], **{k: cfg[k] for k in RESULT_KEYS},
                                  "seed": int(seed), "smoke": bool(smoke)}))


def run_key(spec: dict) -> str:
    return (f"{spec['phase']}_{spec['arch']}_ns{spec['n_s']}_seed{spec['seed']}_lr{spec['lr']:g}"
            f"_steps{spec['steps']}")


def plan_task(task: str, s: dict, archs: list[str]) -> tuple[list[dict], list[dict]]:
    """Tuning runs (every arm at every LR, tuning seed and state size), then main runs seed by seed; a main run's
    lr is the best of its arm's tuning runs (filled in once they are saved)."""
    tune = [{"task": task, "phase": "tune", "arch": a, "n_s": s["tune_n_states"], "seed": s["tune_seed"], "lr": lr,
             "steps": s["steps"]} for a in archs for lr in s["lr_grid"]]
    main = [{"task": task, "phase": "main", "arch": a, "n_s": n, "seed": sd, "lr": None, "steps": s["steps"]}
            for sd in s["seeds"] for n in s["n_states"] for a in archs]
    return tune, main


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def best_lr(tune_rows: list[dict]) -> float:
    """The tuning rule of the first P3: the LR with the highest accuracy (a diverged run counts as -inf; ties go
    to the larger LR)."""
    return max((_num(r["acc"]) if np.isfinite(_num(r["acc"])) else -np.inf, r["lr"]) for r in tune_rows)[1]


class Widths:
    """Matched widths per (task kind, n_s): (target params, {arch: (H, params)}), computed once."""

    def __init__(self, cfg, archs):
        self.cfg, self.archs, self.cache = cfg, archs, {}

    def __call__(self, task: str, n_s: int):
        cls, L = task == "lra_image", self.cfg["tasks"][task]["L"]
        if (cls, L, n_s) not in self.cache:
            c = self.cfg
            ref = ssm.SSMModel(c["ref_arch"], 1, 10 if cls else 1, c["H_ref"], n_s, c["n_layers"], L, pool=cls)
            target = ssm.n_params(ref)
            self.cache[(cls, L, n_s)] = (target, {a: ssm.width_for_budget(a, 1, 10 if cls else 1, n_s, c["n_layers"],
                                                                          L, cls, target) for a in self.archs})
        return self.cache[(cls, L, n_s)]


# ----------------------------------------------------------------------------- saved runs
def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(runtime.to_jsonable(obj), default=str))
    os.replace(tmp, path)


class RunStore:
    """runs/<task>/<key>.json, one per finished run, and runs/<task>/settings.json."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir) / "runs"
        self.rows: dict[tuple[str, str], dict] = {}
        for p in sorted(self.dir.glob("*/*.json")):
            if p.name != "settings.json":
                self.rows[(p.parent.name, p.stem)] = json.loads(p.read_text())

    def check_settings(self, task: str, fp: dict) -> None:
        path = self.dir / task / "settings.json"
        if path.exists():
            old = json.loads(path.read_text())
            diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
            if diff:
                raise SystemExit(f"{path} was written with other settings ({', '.join(diff)}: saved {[old.get(k) for k in diff]}, "
                                 f"now {[fp.get(k) for k in diff]}); its runs would not match. Use another --out.")
        else:
            atomic_json(path, fp)

    def get(self, task: str, key: str) -> dict | None:
        return self.rows.get((task, key))

    def put(self, task: str, key: str, row: dict) -> None:
        atomic_json(self.dir / task / f"{key}.json", row)
        self.rows[(task, key)] = json.loads(json.dumps(runtime.to_jsonable(row), default=str))


def step_seconds(rows) -> dict:
    """Median seconds per training step by (arch, n_s) over the saved runs that did not diverge (the step costs
    the same in every task: L = 1,024 and the same batch everywhere at full size)."""
    by: dict = {}
    for r in rows:
        if not r.get("diverged") and r.get("steps") and np.isfinite(_num(r.get("wall_s"))):
            by.setdefault((r["arch"], int(r["n_s"])), []).append(r["wall_s"] / r["steps"])
    return {k: float(np.median(v)) for k, v in by.items()}


def hours_for(specs: list[dict], per: dict) -> tuple[float | None, bool]:
    """Hours of `specs` by the measured step times: (hours, True if some (arch, n_s) had no measurement and took
    the arm's nearest state size, or the median of all); (None, True) without any measurement."""
    if not per:
        return (None, True) if specs else (0.0, False)
    rough, total = False, 0.0
    every = float(np.median(list(per.values())))
    for sp in specs:
        v = per.get((sp["arch"], sp["n_s"]))
        if v is None:
            rough = True
            same = [(abs(math.log2(n) - math.log2(sp["n_s"])), t) for (a, n), t in per.items() if a == sp["arch"]]
            v = min(same)[1] if same else every
        total += v * sp["steps"]
    return total / 3600, rough


# ----------------------------------------------------------------------------- outputs
def planned_rows(store: RunStore, plans: dict, archs: list[str]) -> list[dict]:
    """The saved rows of the plan, in plan order; main runs of an arm count once its tuning runs are all saved."""
    rows = []
    for task, (s, tune, main) in plans.items():
        t_rows = [store.get(task, run_key(sp)) for sp in tune]
        rows += [r for r in t_rows if r is not None]
        lrs = {a: best_lr([r for sp, r in zip(tune, t_rows) if sp["arch"] == a])
               for a in archs if all(r is not None for sp, r in zip(tune, t_rows) if sp["arch"] == a)}
        for sp in main:
            if sp["arch"] in lrs:
                r = store.get(task, run_key({**sp, "lr": lrs[sp["arch"]]}))
                if r is not None:
                    rows.append(r)
    return rows


def write_outputs(out_dir: Path, rows: list[dict], cfg: dict) -> None:
    runtime.write_csv(out_dir / "runs.csv", rows)
    main_runs = pd.DataFrame([r for r in rows if r["phase"] == "main"])
    if main_runs.empty:
        runtime.write_csv(out_dir / "curves.csv", [])
        runtime.write_csv(out_dir / "states_to_target.csv", [])
        return
    curves = main_runs.groupby(["task", "arch", "n_s"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                             params=("params", "mean"), n=("acc", "size")).reset_index()
    runtime.write_csv(out_dir / "curves.csv", curves.to_dict("records"))
    tgt = []
    for (task, arch), g in curves.groupby(["task", "arch"]):
        thr = cfg["target_accuracy"][task]
        hit = g[g.acc_mean >= thr].n_s
        tgt.append({"task": task, "arch": arch, "target_acc": thr, "min_n_s": int(hit.min()) if len(hit) else None})
    runtime.write_csv(out_dir / "states_to_target.csv", tgt)


def plan_table(store: RunStore, plans: dict, archs: list[str]) -> tuple[list[dict], list[dict]]:
    """(one line per task and phase with runs, saved, left, steps and hours left, the specs still to run)."""
    per = step_seconds(store.rows.values())
    lines, todo_all = [], []
    for task, (s, tune, main) in plans.items():
        t_rows = [store.get(task, run_key(sp)) for sp in tune]
        lrs = {a: best_lr([r for sp, r in zip(tune, t_rows) if sp["arch"] == a])
               for a in archs if all(r is not None for sp, r in zip(tune, t_rows) if sp["arch"] == a)}
        for phase, specs in (("tune", tune), ("main", main)):
            todo = []
            for sp in specs:
                if phase == "main" and sp["arch"] in lrs:
                    sp = {**sp, "lr": lrs[sp["arch"]]}
                if sp["lr"] is None or store.get(task, run_key(sp)) is None:
                    todo.append(sp)
            h, rough = hours_for(todo, per)
            lines.append({"task": task, "design": s["design"], "phase": phase, "runs": len(specs),
                          "saved": len(specs) - len(todo), "left": len(todo), "steps_per_run": s["steps"],
                          "hours_left": h, "rough": rough})
            todo_all += todo
    return lines, todo_all


def print_plan(store: RunStore, plans: dict, archs: list[str], out_dir: Path) -> None:
    lines, todo = plan_table(store, plans, archs)
    per = step_seconds(store.rows.values())
    print(f"P3 plan in {out_dir}:")
    print(f"  {'task':12} {'design':8} {'phase':5} {'runs':>5} {'saved':>6} {'left':>5} {'steps/run':>10} {'hours left':>11}")
    for ln in lines:
        h = "?" if ln["hours_left"] is None else f"{ln['hours_left']:.1f}{' ~' if ln['rough'] and ln['left'] else ''}"
        print(f"  {ln['task']:12} {ln['design'] or '-':8} {ln['phase']:5} {ln['runs']:5d} {ln['saved']:6d} {ln['left']:5d} "
              f"{ln['steps_per_run']:10,d} {h:>11}")
    h, rough = hours_for(todo, per)
    n_meas = sum(1 for r in store.rows.values() if not r.get("diverged"))
    print(f"  left: {len(todo)} runs, " + ("no measured step times yet" if h is None else
          f"about {h:.1f} h by the median step time per arm and n_s of {n_meas} saved runs"
          + (" (~: some arm and n_s not measured yet; nearest n_s of the arm used)" if rough else "")))
    if per:
        ns = sorted({n for _, n in per})
        print("  seconds per step, measured:  " + "  ".join(f"n_s={n}" for n in ns))
        for a in archs:
            print(f"    {a:8} " + "  ".join(f"{per[(a, n)]:.3f}" if (a, n) in per else "  -  " for n in ns))


# ----------------------------------------------------------------------------- main
def gpu_lock(args, device: str):
    """P3 holds logs/.p3_gpu.lock while it runs on the GPU; P1's olmo7b takes the same lock (common/gpu_locks.py),
    so the two never share the L40S. Not under --smoke, --plan or on the CPU."""
    if args.smoke or args.plan or not device.startswith("cuda"):
        return None, 0.0
    path = runtime.PILOT_ROOT / "logs" / ".p3_gpu.lock"
    return gpu_locks.take(path, f"[p3_fractional_memory] {path} is held (P1's olmo7b, or another P3); waiting")


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--archs", default=None)
    ap.add_argument("--lra-design", default=None, choices=["reduced", "full"],
                    help="lra_image design from config.py lra_designs (default: config's lra_design, reduced)")
    ap.add_argument("--plan", action="store_true",
                    help="print the saved and the remaining runs with hours by measured step times; train nothing")
    args = ap.parse_args()
    cfg = config.get("p3_fractional_memory", args.smoke)
    design = args.lra_design or cfg["lra_design"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _held, waited = gpu_lock(args, device)       # before CUDA starts; held as long as this process runs
    run = runtime.start_run("p3_fractional_memory", cfg, args)
    if waited > 1:
        run.note(f"waited {waited / 3600:.2f} h for logs/.p3_gpu.lock")
    dev = run.device
    tasks = args.tasks.split(",") if args.tasks else list(cfg["tasks"])
    archs = args.archs.split(",") if args.archs else cfg["archs"]
    store = RunStore(run.out_dir)
    widths = Widths(cfg, archs)
    settings = {t: task_settings(cfg, t, design) for t in cfg["tasks"]}
    plans = {t: (settings[t], *plan_task(t, settings[t], archs)) for t in cfg["tasks"]}
    for t in tasks:
        store.check_settings(t, result_settings(cfg, t, run.seed, run.smoke))
    if args.plan:
        print_plan(store, {t: plans[t] for t in tasks}, archs, run.out_dir)
        return

    def _stop(signum, frame):                    # SIGTERM (kill, run_all.sh, run_lane.sh): saved runs stay
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _stop)

    todo_total = len(plan_table(store, {t: plans[t] for t in tasks}, archs)[1])
    done_here = 0
    synth = synthetic_data(cfg, run.seed, [t for t in tasks if t in SYNTHETIC])
    rng = runtime.rng(run.seed)
    lines, _ = plan_table(store, {t: plans[t] for t in tasks}, archs)
    for ln in lines:
        run.note(f"plan: {ln['task']} {ln['phase']}: {ln['saved']}/{ln['runs']} runs saved, {ln['left']} to run "
                 f"({ln['steps_per_run']:,} steps each)")
    try:
        for task in tasks:
            s, tune, main_specs = plans[task]
            if not plan_table(store, {task: plans[task]}, archs)[1]:
                run.note(f"{task}: every run saved")
                continue
            if task == "lra_image":
                data, src = lra_image(cfg["tasks"][task], run, rng)
                if src != "fallback" and len(data[0]) != s["n_train"]:
                    raise SystemExit(f"lra_image: {len(data[0])} train images, the plan assumed {s['n_train']}")
            else:
                data, src = synth[task], "synthetic"

            def ensure(sp):
                nonlocal done_here
                key = run_key(sp)
                target, w = widths(task, sp["n_s"])
                H = w[sp["arch"]][0]
                saved = store.get(task, key)
                if saved is not None:
                    if int(saved["H"]) != H or int(saved["target_params"]) != target:
                        raise SystemExit(f"saved run {task}/{key} has H={saved['H']} (target {saved['target_params']}),"
                                         f" the widths now give H={H} (target {target}): the code changed; use "
                                         f"another --out")
                    return saved
                r = train_one(sp["arch"], task, sp["n_s"], H, data, cfg, sp["lr"], sp["seed"], dev, sp["steps"])
                row = {"task": task, "arch": sp["arch"], "n_s": sp["n_s"], "H": H, "lr": sp["lr"], "seed": sp["seed"],
                       "phase": sp["phase"], "target_params": target, "data_source": src, **r,
                       "design": s["design"], "origin": "this run",
                       "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                store.put(task, key, row)
                write_outputs(run.out_dir, planned_rows(store, plans, archs), cfg)
                done_here += 1
                _, left = plan_table(store, {t: plans[t] for t in tasks}, archs)
                h, rough = hours_for(left, step_seconds(store.rows.values()))
                print(f"[p3_fractional_memory] {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}  "
                      f"{done_here}/{todo_total}  {task} {sp['phase']} {sp['arch']} n_s={sp['n_s']} seed={sp['seed']} "
                      f"lr={sp['lr']:g}: acc {r['acc']:.4f}, {r['steps']:,} steps in {r['wall_s']:.0f} s "
                      f"({r['wall_s'] / max(1, r['steps']):.3f} s/step); {len(left)} runs left"
                      + ("" if h is None else f", about {h:.1f} h{' (rough)' if rough else ''}"), flush=True)
                return store.get(task, key)

            t_rows = [ensure(sp) for sp in tune]
            lrs = {a: best_lr([r for sp, r in zip(tune, t_rows) if sp["arch"] == a]) for a in archs}
            run.note(f"{task}: tuned learning rates {lrs}")
            for sp in main_specs:
                ensure({**sp, "lr": lrs[sp["arch"]]})
            for n_s in s["n_states"]:
                target, w = widths(task, n_s)
                run.note(f"{task} n_s={n_s}: " + ", ".join(f"{a} H={w[a][0]} params={w[a][1]}" for a in archs)
                         + f" (target {target})")
    except SystemExit:
        print(f"[p3_fractional_memory] stopped; {len(store.rows)} runs saved in {store.dir} (a restart goes on from "
              f"them)", flush=True)
        raise
    rows = planned_rows(store, plans, archs)
    write_outputs(run.out_dir, rows, cfg)
    runtime.finish_run(run, extra={"n_runs": len(rows), "trained_now": done_here, "lra_design": design})


if __name__ == "__main__":
    main()
