"""Saved units of a resumable job (P2, P4, P5): one JSON file per finished unit, a settings file that refuses a
resume with other settings, and progress.json with the measured hours of every unit and the hours left at the
measured mean.

A unit is the piece of a job that is saved as a whole when it ends: a P4 training run with its measurements, a
P2 quantization scheme with its isotropic control, a P5 grid cell. A job stopped in the middle of a unit loses
that unit only; started again with the same --out, it reads the saved units and runs the others. Each unit draws
its random numbers from its own seed (seed_for), so the order of units and restarts change nothing.

progress.json (read by run_queue.py --status):
  {"job", "units_total", "units_done", "unit_hours": {unit: hours}, "measured_units", "mean_unit_hours",
   "hours_left", "updated_utc"}
Hours count only units run to the end in a process that ran them from their start; units loaded from disk keep
the hours they had when they were run.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import zlib
from pathlib import Path

import numpy as np

from common import runtime


def seed_for(base: int, *parts) -> list[int]:
    """Entropy for np.random.SeedSequence from the run seed and names (crc32: the same in every process)."""
    return [int(base)] + [zlib.crc32(str(p).encode()) for p in parts]


def rng_for(seed: list[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(runtime.to_jsonable(obj), default=str))
    os.replace(tmp, path)


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]", "_", key)


def fmt_hours(h: float) -> str:
    return f"{h:.2f} h" if h >= 0.1 else f"{h * 60:.1f} min"


def _diff(old: dict, new: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}")
    return out


class Units:
    """<out>/<sub>/<unit>.json for every finished unit, <out>/<sub>/settings.json, <out>/progress.json."""

    def __init__(self, out_dir: Path, settings: dict, job: str, sub: str = "units", total: int | None = None,
                 log=print):
        self.out = Path(out_dir)
        self.dir = self.out / sub
        self.job, self.total, self.log = job, total, log
        self.dir.mkdir(parents=True, exist_ok=True)
        fp = json.loads(json.dumps(runtime.to_jsonable(settings), default=str))
        path = self.dir / "settings.json"
        if path.exists():
            old = json.loads(path.read_text())
            diff = _diff(old, fp)
            if diff:
                raise SystemExit(f"{path} was written with other settings ({', '.join(diff[:8])}"
                                 f"{' ...' if len(diff) > 8 else ''}); its units would not match. Use another --out, "
                                 f"or delete {self.dir} to start again.")
        else:
            atomic_json(path, fp)
        prog = self.out / "progress.json"
        self.hours: dict[str, float] = {}
        if prog.exists():
            try:
                self.hours = {k: float(v) for k, v in json.loads(prog.read_text()).get("unit_hours", {}).items()}
            except (ValueError, OSError):
                self.hours = {}
        self.t_start: dict[str, float] = {}
        self.planned: set[str] | None = None

    def plan(self, keys) -> None:
        """The units of this session's plan: progress counts the saved ones among them."""
        self.planned = {safe_name(k) for k in keys}
        self.total = len(self.planned)

    def path(self, key: str) -> Path:
        return self.dir / f"{safe_name(key)}.json"

    def get(self, key: str):
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def saved(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json") if p.name != "settings.json")

    def start(self, key: str) -> None:
        self.t_start[key] = time.perf_counter()

    def put(self, key: str, obj) -> float:
        """Save a finished unit; returns its hours (nan if start() was not called for it)."""
        t0 = self.t_start.pop(key, None)
        h = (time.perf_counter() - t0) / 3600 if t0 is not None else float("nan")
        atomic_json(self.path(key), obj)
        if np.isfinite(h):
            self.hours[safe_name(key)] = h
        self.write_progress(last=(key, h))
        return h

    def write_progress(self, last=None) -> dict:
        saved = set(self.saved())
        done = len(saved & self.planned) if self.planned is not None else len(saved)
        meas = [v for k, v in self.hours.items() if np.isfinite(v) and (self.planned is None or k in self.planned)]
        mean = float(np.mean(meas)) if meas else None
        left = max(0, (self.total or done) - done)
        d = {"job": self.job, "units_total": self.total, "units_done": done, "unit_hours": self.hours,
             "measured_units": len(meas), "mean_unit_hours": mean,
             "hours_left": None if mean is None else mean * left,
             "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        atomic_json(self.out / "progress.json", d)
        if last is not None:
            key, h = last
            msg = f"[{self.job}] unit {done}/{self.total if self.total else '?'} saved: {key}"
            if np.isfinite(h):
                msg += f", {fmt_hours(h)}"
            if mean is not None:
                msg += (f"; measured mean {fmt_hours(mean)} over {len(meas)} unit(s); left {left} unit(s), "
                        f"about {fmt_hours(mean * left)} at that mean")
            self.log(msg)
        return d
