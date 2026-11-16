"""Merge pins.json entries from another copy into this repository's pins.json, losing none.

For pulling on a machine whose pins.json has local additions from its runs (names pinned on first use):

  cp pins.json ~/pins.local.json               # 1. keep the local file
  git checkout -- pins.json                    # 2. drop the local edit, so that the pull applies cleanly
  git pull --ff-only                           # 3. the committed pins.json (with its new entries)
  python merge_pins.py ~/pins.local.json       # 4. add every local entry back

The result is the union of both files. A name present in both with the same commit keeps the committed entry
(which may carry more fields, e.g. a name-switch check). A name present in both with different commits keeps the
LOCAL entry, since that machine's results loaded it, and is listed as a conflict (exit code 2) so it can be
reported; the committed entry is printed next to it. Nothing is written if the merge would change nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def merge(base: dict, other: dict) -> tuple[dict, list[str], list[str]]:
    """(merged, keys added from other, conflicting keys)."""
    out = dict(base)
    added, conflicts = [], []
    for k, v in other.items():
        if k not in base:
            out[k] = v
            added.append(k)
        elif base[k].get("commit") != v.get("commit"):
            out[k] = v
            conflicts.append(k)
        else:
            out[k] = {**v, **base[k]}           # same commit: keep every field, the committed ones first
    return out, sorted(added), sorted(conflicts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("other", help="the other pins.json (e.g. the local copy saved before git pull)")
    ap.add_argument("--into", default=str(ROOT / "pins.json"), help="the file to merge into (default: ./pins.json)")
    args = ap.parse_args(argv)
    into = Path(args.into)
    base = json.loads(into.read_text()) if into.exists() else {}
    other = json.loads(Path(args.other).read_text())
    merged, added, conflicts = merge(base, other)
    for k in conflicts:
        print(f"CONFLICT {k}: kept local commit {other[k].get('commit')}, committed was {base[k].get('commit')}")
    if merged != base:
        tmp = into.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(sorted(merged.items())), indent=1) + "\n")   # the format Pins.save writes
        os.replace(tmp, into)
    print(f"{into.name}: {len(base)} entries before, {len(merged)} after; {len(added)} added from {args.other}, "
          f"{len(conflicts)} conflicts")
    for k in added[:20]:
        print(f"  + {k}")
    if len(added) > 20:
        print(f"  ... and {len(added) - 20} more")
    return 2 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
