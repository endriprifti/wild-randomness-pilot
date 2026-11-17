"""Recover the finished runs of a P3 process started with the code before 2026-11-17.

That code keeps every finished run in a list in memory (main()'s `rows`) and writes runs.csv only after the last
task, lra_image; stopping it loses every run. This script attaches gdb to the process and runs a few lines of
Python inside it, which write that list, with the process's config and seed, to a JSON file. The process keeps
running and nothing in it changes (it pauses for about a second). The new P3 then takes the runs over:

  python p3_fractional_memory.py --out results/p3_fractional --adopt results/p3_fractional/rescued_runs.json --plan

Usage, in the pilot folder (needs gdb: sudo apt-get install -y gdb):
  python p3_rescue.py                                   # finds the process; writes results/p3_fractional/rescued_runs.json
  python p3_rescue.py --pid 12345 --out results/p3_fractional/rescued_runs.json
If gdb says "ptrace: Operation not permitted" (Ubuntu lets a user attach only to its own child processes), run
it with sudo: sudo "$(which python)" p3_rescue.py. The file is written by the P3 process itself, so it belongs
to P3's user.

Only the standard library is used here, and inside P3 only json, os, sys and threading, which it has loaded.
How: gdb attaches for a moment and, in P3's main thread, calls PyGILState_Ensure,
PyRun_SimpleString("exec(open(<snippet>).read())") and PyGILState_Release; the snippet walks the main thread's
Python frames to main() of p3_fractional_memory.py (or, failing that, searches the garbage collector's objects
for the list of runs). It does so only at a safe moment: the main thread has released the GIL (PyTorch releases it
in every operation and while backward() runs, so Python's own state is consistent) and is not inside the C
allocator. Otherwise gdb detaches at once and the script tries again a quarter of a second later (--tries).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SNIPPET = r'''
def _p3_rescue(out):
    import json, os, sys, threading, time
    found = {}
    try:
        f = sys._current_frames().get(threading.main_thread().ident)
        while f is not None:
            if f.f_code.co_name == "main" and f.f_code.co_filename.endswith("p3_fractional_memory.py"):
                loc = f.f_locals
                found = {"rows": loc.get("rows"), "cfg": loc.get("cfg"), "run": loc.get("run"),
                         "tasks": loc.get("tasks"), "archs": loc.get("archs"), "how": "main() frame"}
                break
            f = f.f_back
        if found.get("rows") is None:
            import gc
            need = {"task", "arch", "n_s", "phase", "acc", "steps", "H", "lr", "seed"}
            cands = [o for o in gc.get_objects() if isinstance(o, list) and o and isinstance(o[0], dict)
                     and need <= set(o[0])]
            found = {"rows": max(cands, key=len) if cands else None, "how": "gc search"}
        if found.get("rows") is None:
            raise RuntimeError("no list of runs found")
        run = found.get("run")
        data = {"rows": [dict(r) for r in found["rows"]], "cfg": found.get("cfg"),
                "seed": getattr(run, "seed", None), "smoke": getattr(run, "smoke", None),
                "out_dir": str(getattr(run, "out_dir", "")), "tasks": found.get("tasks"),
                "archs": found.get("archs"), "how": found["how"], "pid": os.getpid(),
                "python": sys.version.split()[0],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, default=str)
        os.replace(tmp, out)
    except BaseException as e:
        with open(out + ".error", "w") as fh:
            fh.write(type(e).__name__ + ": " + str(e))
_p3_rescue(OUT)
del _p3_rescue
'''

GDB_SCRIPT = r'''
import re
import gdb

PID, SNIPPET = {pid}, {snippet!r}
UNSAFE = re.compile(r"malloc|free|realloc|calloc|memalign|lll_lock|PyMem_|PyObject_Malloc|PyObject_Free")


def attempt():
    gdb.execute("set pagination off", to_string=True)
    try:
        gdb.execute("sharedlibrary libc\\.so", to_string=True)      # libc's names, to see malloc in the stack
    except gdb.error:
        pass
    main = [t for t in gdb.selected_inferior().threads() if t.ptid[1] == PID]
    if not main:
        return "nomain"
    main[0].switch()
    try:
        holds = int(gdb.parse_and_eval("(int) PyGILState_Check()"))
    except gdb.error as e:
        return "nosym " + str(e).replace(chr(10), " ")
    if holds:
        return "busy (main thread holds the GIL)"
    names, f = [], gdb.newest_frame()
    while f is not None and len(names) < 40:
        n = f.name()
        if n is None:
            break
        names.append(n)
        try:
            f = f.older()
        except gdb.error:
            break
    bad = [n for n in names if UNSAFE.search(n)]
    if bad:
        return "busy (in " + bad[0] + ")"
    st = int(gdb.parse_and_eval("(int) PyGILState_Ensure()"))
    rc = int(gdb.parse_and_eval("(int) PyRun_SimpleString(\"exec(open('%s').read())\")" % SNIPPET))
    gdb.parse_and_eval("(void) PyGILState_Release(%d)" % st)
    return "ran rc=%d at %s" % (rc, names[0] if names else "?")


try:
    print("P3RESCUE " + attempt())
finally:
    gdb.execute("detach", to_string=True)
'''


def find_pid() -> int:
    me = os.getpid()
    found = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"p3_fractional_memory.py") for a in argv) and b"python" in argv[0]:
            found.append(int(d.name))
    if len(found) != 1:
        raise SystemExit(f"p3_rescue: {'no' if not found else len(found)} P3 processes found {found}; pass --pid")
    return found[0]


def rescue(pid: int, out: Path, timeout: float = 120.0, tries: int = 400) -> dict:
    gdb = shutil.which("gdb")
    if gdb is None:
        raise SystemExit("p3_rescue: gdb not found (sudo apt-get install -y gdb)")
    cmd = (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
    if "p3_fractional_memory.py" not in cmd:
        raise SystemExit(f"p3_rescue: process {pid} is not P3: {cmd}")
    out = out.resolve()
    made = not out.parent.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    st = os.stat(f"/proc/{pid}")
    if made and os.geteuid() == 0 and st.st_uid != 0:
        os.chown(out.parent, st.st_uid, st.st_gid)   # the P3 process writes the file: a folder made here is its own
    if "'" in str(out) or '"' in str(out):
        raise SystemExit("p3_rescue: the output path may not contain quotes")
    for p in (out, Path(str(out) + ".error")):
        p.unlink(missing_ok=True)
    snippet = out.with_name(out.name + ".snippet.py")
    snippet.write_text(f"OUT = {str(out)!r}\n" + SNIPPET)
    snippet.chmod(0o644)
    script = out.with_name(out.name + ".gdb.py")
    script.write_text(GDB_SCRIPT.format(pid=pid, snippet=str(snippet)))
    args = [gdb, "-nx", "-batch", "-iex", "set debuginfod enabled off", "-iex", "set auto-solib-add off",
            "-p", str(pid), "-x", str(script)]
    t0, said, r = time.time(), [], None
    try:
        for _ in range(tries):
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            said = [x for x in (r.stdout + r.stderr).splitlines() if x.startswith("P3RESCUE ")]
            if not said or not said[-1].startswith("P3RESCUE busy"):
                break
            time.sleep(0.25)
    finally:
        snippet.unlink(missing_ok=True)
        script.unlink(missing_ok=True)
    if said and said[-1].startswith("P3RESCUE busy"):
        raise SystemExit(f"p3_rescue: P3 was never at a safe moment in {tries} tries ({said[-1][9:]}); nothing "
                         f"was run inside it. Try again, or with --tries 2000.")
    err = Path(str(out) + ".error")
    if not out.exists():
        tail = "\n".join(((r.stdout + r.stderr) if r else "").strip().splitlines()[-15:])
        why = err.read_text() if err.exists() else "no file written"
        raise SystemExit(f"p3_rescue: nothing recovered ({why}). gdb said:\n{tail}")
    data = json.loads(out.read_text())
    data["_seconds"] = time.time() - t0
    return data


def summary(data: dict) -> str:
    rows = data["rows"]
    by: dict = {}
    for r in rows:
        k = (r["task"], r["phase"])
        by.setdefault(k, [0, 0.0])
        by[k][0] += 1
        by[k][1] += float(r.get("wall_s") or 0.0)
    lines = [f"recovered {len(rows)} runs from pid {data['pid']} (Python {data['python']}, {data['how']}, "
             f"seed {data['seed']}) in {data['_seconds']:.1f} s:"]
    lines += [f"  {t:12} {ph:5} {n:4d} runs, {h / 3600:6.2f} h of training" for (t, ph), (n, h) in by.items()]
    if rows:
        last = rows[-1]
        lines.append(f"  last: {last['task']} {last['phase']} {last['arch']} n_s={last['n_s']} seed={last['seed']} "
                     f"acc={last['acc']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, default=None, help="the P3 process (default: the one running "
                                                          "p3_fractional_memory.py)")
    ap.add_argument("--out", default=str(ROOT / "results" / "p3_fractional" / "rescued_runs.json"))
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one gdb attempt")
    ap.add_argument("--tries", type=int, default=400, help="attempts to find P3 at a safe moment (0.25 s apart)")
    args = ap.parse_args(argv)
    pid = args.pid or find_pid()
    data = rescue(pid, Path(args.out), args.timeout, args.tries)
    print(summary(data))
    print(f"wrote {args.out}; process {pid} is " + ("still running" if Path(f'/proc/{pid}').exists() else "gone"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
