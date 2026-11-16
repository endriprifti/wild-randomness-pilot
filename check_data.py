"""Open one item of every dataset split that a full-config job reads, at the revision it will read.

Run it before a cloud run or a rerun (a few minutes; it streams only the first row of each split, plus the
Fashion-MNIST download of about 30 MB):

  python check_data.py                    # every job; pins new dataset revisions in pins.json
  python check_data.py --jobs p1_checkpoints,p2_w4a4
  python check_data.py --no-pin           # do not write pins.json (unpinned revisions are resolved, not saved)

The list comes from config.py itself: every entry named *dataset (a [repo, config, split] list or a dict of
common.hf.DataSpec fields), PopQA from the P2 config, CIFAR-10 for P3 (LRA image) and P4 (ResNet-18), and
Fashion-MNIST (torchvision) for P5. Each split is read exactly as the job reads it (same repository, config,
split, data_files and pinned commit), so a wrong split or a missing file fails here, not 20 minutes into a
job. Writes results/check_data/check_data.csv and exits 1 if any split fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import config
from common import hf, runtime

JOBS = tuple(config.CONFIG)          # every job in config.py


def _walk(obj, where=""):
    """(where, spec) for every *dataset entry in a config tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{where}.{k}" if where else k
            if k.endswith("dataset") and isinstance(v, (list, tuple)) and len(v) == 3 and isinstance(v[0], str):
                yield here, hf.data_spec(v)
            elif k.endswith("dataset") and isinstance(v, dict) and "repo" in v:
                yield here, hf.data_spec(v)
            else:
                yield from _walk(v, here)


def planned(jobs=JOBS, smoke: bool = False) -> list[tuple[str, str, object]]:
    """(job, config entry, spec) for every split the jobs read; spec is a DataSpec or ('torchvision', name, split)."""
    out = []
    for job in jobs:
        if job not in config.CONFIG:
            continue
        cfg = config.get(job, smoke=smoke)
        out += [(job, where, spec) for where, spec in _walk(cfg)]
        if job == "p2_w4a4":
            from common import popqa
            out.append((job, "popqa", hf.data_spec(popqa.spec(cfg["popqa"]))))
        if (job == "p3_fractional_memory" and "lra_image" in cfg.get("tasks", {})) or \
                (job == "p4_clipping" and "resnet18" in cfg):
            out += [(job, "cifar10", hf.data_spec(hf.CIFAR10, split=s)) for s in ("train", "test")]
        if job == "p5_escape":
            out += [(job, "fmnist", ("torchvision", "FashionMNIST", s)) for s in ("train", "test")]
    return out


def _key(spec):
    if isinstance(spec, tuple):
        return spec
    return (spec.repo, spec.config, spec.split, str(spec.data_files), spec.revision)


def _summary(item: dict) -> str:
    parts = []
    for k, v in list(item.items())[:6]:
        if isinstance(v, str):
            parts.append(f"{k}: str[{len(v)}]")
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {type(v).__name__}")
    return ", ".join(parts)


def check_one(spec) -> dict:
    t0 = time.perf_counter()
    if isinstance(spec, tuple):                                      # torchvision
        _, name, split = spec
        row = {"dataset": f"torchvision {name}", "config": "", "split": split, "data_files": "", "revision": "",
               "commit": ""}
        try:
            import torchvision
            ds = getattr(torchvision.datasets, name)(str(runtime.PILOT_ROOT / "data"), train=split == "train",
                                                     download=True)
            img, label = ds[0]
            row.update(status="ok", n_rows=len(ds), first_item=f"image {getattr(img, 'size', '')}, label {label}")
        except Exception as e:  # noqa: BLE001
            row.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:300]}")
    else:
        row = {"dataset": spec.repo, "config": spec.config or "", "split": spec.split,
               "data_files": "" if not spec.data_files else str(spec.data_files[spec.split]),
               "revision": spec.revision, "commit": ""}
        try:
            item, sha = hf.first_item(spec)
            row.update(commit=sha or "", status="ok", first_item=_summary(item))
            if spec.field and not item.get(spec.field):
                row.update(status="FAIL", error=f"first item has no {spec.field!r} field: {sorted(item)}")
        except Exception as e:  # noqa: BLE001
            row.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:300]}",
                       transient=hf.is_transient(e))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default=",".join(JOBS), help="comma-separated jobs (default: all)")
    ap.add_argument("--no-pin", action="store_true", help="do not write new dataset pins to pins.json")
    ap.add_argument("--out", default=str(runtime.PILOT_ROOT / "results" / "check_data"))
    args = ap.parse_args(argv)
    if args.no_pin:
        hf.PINS.save = lambda: None                                  # resolve, but leave pins.json untouched
    plan = planned([j.strip() for j in args.jobs.split(",") if j.strip()])
    by_split: dict = {}
    for job, where, spec in plan:
        by_split.setdefault(_key(spec), [spec, []])[1].append(f"{job}:{where}")
    rows = []
    for _, (spec, users) in by_split.items():
        row = check_one(spec)
        row["used_by"] = "; ".join(users)
        rows.append(row)
        files = f" [{row['data_files']}]" if row["data_files"] else ""
        cfgname = f"/{row['config']}" if row["config"] else ""
        what = f"{row['dataset']}{cfgname}:{row['split']}{files}"
        at = f" @{row['commit'][:10]}" if row.get("commit") else ""
        detail = row.get("first_item") if row["status"] == "ok" else row.get("error")
        print(f"{row['status']:<4} {what}{at}  ({row['seconds']} s)  {detail}\n     used by {row['used_by']}",
              flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runtime.write_csv(out / "check_data.csv", rows)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} splits ok; wrote {out / 'check_data.csv'}"
          + ("" if args.no_pin else f"; dataset pins in {hf.PINS.path.name}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
