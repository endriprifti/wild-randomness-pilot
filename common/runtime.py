"""Run bookkeeping shared by every pilot script.

Every script calls `start_run(test_name, cfg, args)` first and `finish_run(...)` last.
That fixes all random seeds, picks the device, creates results/<test>/ (or
results/<test>/smoke/ under --smoke) and writes run.json with the config, the git
commit, the GPU type, library versions and the wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PILOT_ROOT = Path(__file__).resolve().parent.parent


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true",
                   help="tiny settings that finish in under five minutes on a small GPU or CPU")
    p.add_argument("--seed", type=int, default=None, help="override the base seed in config.py")
    p.add_argument("--device", default=None, help="cuda, cuda:0 or cpu (default: cuda if available)")
    p.add_argument("--out", default=None, help="results directory (default results/<test>[/smoke])")
    p.add_argument("--offline-ok", action="store_true",
                   help="allow synthetic fallbacks when Hugging Face or dataset downloads fail "
                        "(always on under --smoke; never use for real results)")
    p.add_argument("--require-pins", action="store_true",
                   help="fail if a model or revision is not already pinned to a commit in pins.json")
    return p


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Fix Python, NumPy and PyTorch seeds; request deterministic kernels where they exist."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PILOT_ROOT, capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0:
            return "not-a-git-repo"
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PILOT_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:  # git missing
        return "git-unavailable"


def gpu_info() -> dict:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "name": prop.name,
                    "capability": f"sm_{prop.major}{prop.minor}",
                    "memory_gb": round(prop.total_memory / 2**30, 2),
                })
            info["compiled_arch_list"] = torch.cuda.get_arch_list()
    except ImportError:
        info["torch"] = None
    return info


def library_versions() -> dict:
    """Installed versions, read from package metadata (no heavy imports)."""
    from importlib import metadata
    names = ["numpy", "scipy", "pandas", "torch", "transformers", "datasets", "accelerate",
             "huggingface_hub", "torchvision", "PyWavelets", "bitsandbytes", "emmy-ml"]
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclasses.dataclass
class Run:
    test: str
    cfg: dict
    out_dir: Path
    device: str
    seed: int
    smoke: bool
    offline_ok: bool
    t0: float
    started: str
    notes: list = dataclasses.field(default_factory=list)
    fallbacks: list = dataclasses.field(default_factory=list)
    pins: dict = dataclasses.field(default_factory=dict)    # "repo@revision" -> commit actually loaded
    require_pins: bool = False

    def note(self, msg: str) -> None:
        print(f"[{self.test}] {msg}", flush=True)
        self.notes.append(msg)

    def fallback(self, what: str) -> None:
        """Record that a synthetic stand-in replaced real data or weights."""
        msg = f"FALLBACK: {what}"
        print(f"[{self.test}] {msg}", flush=True)
        self.fallbacks.append(what)


def start_run(test: str, cfg: dict, args: argparse.Namespace) -> Run:
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    set_all_seeds(seed)
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    out = Path(args.out) if args.out else PILOT_ROOT / "results" / test / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    run = Run(test=test, cfg=cfg, out_dir=out, device=device, seed=seed, smoke=bool(args.smoke),
              offline_ok=bool(args.smoke or args.offline_ok), t0=time.perf_counter(),
              require_pins=bool(getattr(args, "require_pins", False)),
              started=_dt.datetime.now(_dt.timezone.utc).isoformat())
    run.note(f"device={device} seed={seed} smoke={run.smoke} out={out}")
    return run


def write_csv(path: Path, rows: Iterable[dict]) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: to_jsonable(r.get(k)) for k in keys})
    return path


def finish_run(run: Run, status: str = "ok", extra: dict | None = None) -> Path:
    elapsed = time.perf_counter() - run.t0
    meta = {
        "test": run.test,
        "status": status,
        "smoke": run.smoke,
        "seed": run.seed,
        "device": run.device,
        "started_utc": run.started,
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "wall_clock_seconds": round(elapsed, 2),   # elapsed time from time.perf_counter (monotonic, so a
                                                   # clock jump, e.g. WSL resyncing its clock, cannot distort it)
        "git_commit": git_commit(),
        "gpu": gpu_info(),
        "versions": library_versions(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "config": to_jsonable(run.cfg),
        "fallbacks": run.fallbacks,
        "pinned_commits": run.pins,
        "notes": run.notes,
    }
    if extra:
        meta["results"] = to_jsonable(extra)
    path = run.out_dir / "run.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    run.note(f"finished status={status} in {elapsed:.1f}s; wrote {path}")
    return path
