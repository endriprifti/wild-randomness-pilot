"""P6b (2026-11-21, before any P6b run): the replication's design and queue file, its stop file, the complete-seed
rule, the positive-control rule (P6's readings untouched), the analysis end to end on smoke runs, and the waiter that
starts P6b only after both the P4 transformer and p1_olmo7b have ended."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p6_stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = [4, 5, 7, 8, 9, 10, 11, 12, 13, 14]


# ----------------------------------------------------------------------------- design and queue file
def test_p6b_design_is_the_small_design_with_new_seeds():
    import p6_noah_joseph as p6
    d, small = p6.DESIGNS["p6b_small"], p6.DESIGNS["small"]
    assert d["seeds"] == SEEDS == p6.P6B_SEEDS
    assert 6 not in d["seeds"] and p6.FEASIBILITY["seed"] == 6          # the feasibility check's seed is left out
    assert not set(d["seeds"]) & {1, 2, 3}                                # P6's runs
    assert {k: v for k, v in d.items() if k != "seeds"} == {k: v for k, v in small.items() if k != "seeds"}
    assert p6.settings(False, d) == p6.settings(False, small)            # the same streams, training, measurements
    runs = p6.plan(d)
    assert len(runs) == 40 and [r["seed"] for r in runs] == [s for s in SEEDS for _ in range(4)]   # seed by seed
    fz = p6.run_key("14m", *p6.FEASIBILITY["cell"], p6.FEASIBILITY["seed"])
    assert fz not in {r["key"] for r in runs}
    assert small["seeds"] == [1, 2, 3, 4, 5]                             # P6's design as it was


def test_p6b_queue_runs_the_seeds_one_after_another():
    jobs = json.loads((ROOT / "p6b_queue_small.json").read_text())
    seeds = jobs[:-1]
    assert [j["name"] for j in seeds] == [f"p6b_small_14m_s{k}" for k in SEEDS]
    for i, j in enumerate(seeds):
        a = j["args"]
        assert a[:3] == ["p6_noah_joseph.py", "--design", "p6b_small"]
        assert a[a.index("--seeds") + 1] == str(SEEDS[i]) and a[a.index("--sizes") + 1] == "14m"
        assert "--budget-hours" not in a and "--timing-json" not in a and "--raised" not in a
        assert a[a.index("--stop-file") + 1] == "results/p6b_small/STOP"
        assert j["out"] == f"results/p6b_small/14m_s{SEEDS[i]}" and j["gpu"] == 7.0
        assert j.get("after", []) == ([seeds[i - 1]["name"]] if i else [])
    an = jobs[-1]
    assert an["name"] == "p6b_small_analysis" and an["gpu"] == 0.0
    assert an["args"] == ["p6_noah_joseph.py", "--design", "p6b_small", "--analyse", "--from", "results/p6b_small"]
    assert set(an["after"]) == {j["name"] for j in seeds}
    for f in ("p6_queue_small.json", "p6_queue_full.json", "p6_queue_small_raised.json"):   # P6's folders untouched
        assert not any("p6b" in j["out"] for j in json.loads((ROOT / f).read_text()))


def test_p6b_queue_dry_run(tmp_path):
    r = subprocess.run([PY, str(ROOT / "run_queue.py"), "--root", str(tmp_path), "--jobs-file",
                        str(ROOT / "p6b_queue_small.json"), "--dry-run"], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "RUN_QUEUE_FREE_GIB": "40"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "p6b_small_14m_s4" in r.stdout and "waits for p6b_small_14m_s13" in r.stdout


def test_p6b_refuses_the_budget_rule(tmp_path):
    import p6_noah_joseph as p6
    for extra in (["--budget-hours", "12"], ["--raised"]):
        with pytest.raises(SystemExit, match="P6b"):
            p6.main(["--design", "p6b_small", "--seeds", "4", "--out", str(tmp_path / "x"), *extra])


def test_stop_file_starts_no_run(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    out = tmp_path / "s4"
    r = subprocess.run([PY, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--design", "p6b_small", "--device", "cpu",
                        "--seeds", "1", "--stop-file", str(stop), "--out", str(out)], capture_output=True, text=True,
                       timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.loads((out / "stopped.json").read_text())["seeds"] == [1]
    assert not (out / "runs").exists() and not (out / "units").exists()
    res = json.loads((out / "run.json").read_text())["results"]
    assert res["runs"] == 0 and res["stopped_by"] == str(stop)


# ----------------------------------------------------------------------------- the rules
def test_complete_seeds():
    d = {"sizes": ["14m"], "noah": ["light", "zipf1.0"], "joseph": ["H0.5", "H0.9"], "seeds": [4, 5, 7]}
    units = [{"seed": s, "size": "14m", "noah": n, "joseph": j} for s in (4, 5) for n in d["noah"]
             for j in d["joseph"]]
    units += [{"seed": 7, "size": "14m", "noah": "light", "joseph": "H0.5"}]
    assert p6_stats.complete_seeds(units, d) == ([4, 5], [7])
    assert p6_stats.complete_seeds(units[1:], d) == ([5], [4, 7])


def _df(effects, rng, seeds=(4, 5, 7, 8, 9), noise=0.01):
    rows = []
    for s in seeds:
        b = rng.normal(0, 0.05)
        for n in (0, 2):
            for j in (0, 2):
                nn, jj = n / 2 - 0.5, j / 2 - 0.5
                row = {"seed": s, "noah_score": n, "joseph_score": j, "size": "14m", "stage": 1.0}
                for col, (bN, bJ, bNJ) in effects.items():
                    row[col] = b + bN * nn + bJ * jj + bNJ * nn * jj + rng.normal(0, noise)
                rows.append(row)
    return pd.DataFrame(rows)


def test_positive_control_check_is_the_manipulation_check(rng):
    df = _df({"variant_Htok_excess": (0, 0.05, 0)}, rng)
    man = p6_stats.manipulation_checks(df, rng, 199)
    assert p6_stats.POSITIVE_CONTROL in set(man["check"])
    assert p6_stats.positive_control(man)["passed"] is True
    r1 = np.random.default_rng(1)                                    # a null that misses (p 0.885)
    flat = p6_stats.manipulation_checks(_df({"variant_Htok_excess": (0, 0, 0)}, r1), r1, 199)
    pc = p6_stats.positive_control(flat)
    assert pc["passed"] is False and pc["detail"].startswith("estimate")
    assert p6_stats.positive_control(pd.DataFrame([{"check": "x", "value": True}]))["passed"] is False


def test_positive_control_rule(rng):
    cols = {c: (0.0, 0.0, 0.0) for _, c, *_ in p6_stats.FAMILY}
    cols["K"] = (0.0, 0.08, 0.0)                                  # Joseph -> Noah link as predicted
    cols["grad_main_dh_excess"] = (0.08, 0.0, 0.0)                # Noah -> Joseph link as predicted
    cols[p6_stats.REFEREE] = (0.0, 0.0, 0.0)
    fam = p6_stats.family_tests(_df(cols, rng), rng, 999)
    missed = {"passed": False, "detail": "estimate +0.004, p 0.31"}
    out = p6_stats.positive_control_rule(fam, missed)
    keep = ["outcome", "term", "estimate", "p_perm", "p_holm", "direction", "name"]
    pd.testing.assert_frame_equal(out[keep], fam[keep])                # numbers and Holm unchanged
    nj = out["direction"] == "Noah->Joseph"
    before = fam["reading"]
    assert (out.loc[~nj, "reading"] == before[~nj]).all()             # Joseph -> Noah read as usual
    nolink = nj & before.str.startswith("no link detected")
    assert nolink.sum() >= 8
    assert out.loc[nolink, "reading"].str.startswith("uninformative (the positive control missed: estimate").all()
    g = out[(out["name"] == "dh_grad_excess") & (out["term"] == "N")].iloc[0]
    assert g["reading"] == "link as predicted; positive control missed"
    assert not out["positive_control_passed"].any()
    v = p6_stats.verdict_p6b(out, missed)
    assert v.startswith("Link(s): Joseph->Noah: K J") and "Noah->Joseph: dh_grad_excess N" in v
    assert "that detected no link are uninformative: the positive control missed (estimate +0.004, p 0.31)" in v
    passed = {"passed": True, "detail": "estimate +0.05, p 0.001"}
    same = p6_stats.positive_control_rule(fam, passed)
    assert (same["reading"] == fam["reading"]).all() and same["positive_control_passed"].all()
    assert p6_stats.verdict_p6b(same, passed) == p6_stats.verdict(fam) + " The positive control passed."
    null = p6_stats.family_tests(_df({c: (0, 0, 0) for _, c, *_ in p6_stats.FAMILY}, rng), rng, 199)
    vn = p6_stats.verdict_p6b(p6_stats.positive_control_rule(null, missed), missed)
    assert vn.startswith("No Joseph -> Noah link detected (4 tests). The Noah -> Joseph tests (10)")
    # P6's own summary: the default arguments give what they gave before (title P6, P6's verdict)
    man = pd.DataFrame([{"check": "x", "value": True, "detail": ""}])
    md = p6_stats.to_markdown(fam, man, "h")
    assert md.startswith("# P6 factorial Noah-Joseph test\n") and f"**{p6_stats.verdict(fam)}**" in md


def test_a_note_in_one_test_leaves_the_others_readings(rng):
    # 2026-11-21: with one test "not measured", the other tests' readings came out NaN (a NaN note is truthy)
    fam = p6_stats.family_tests(_df({"K": (0, 0.08, 0)}, rng), rng, 999)     # 999: 14 x 0.001 < 0.05
    k = fam[(fam["name"] == "K") & (fam["term"] == "J")].iloc[0]
    assert k["reading"] == "link as predicted"
    assert fam["reading"].map(lambda x: isinstance(x, str)).all()
    assert (fam[fam["name"] == "alpha"]["reading"] == "not measured").all()


# ----------------------------------------------------------------------------- the analysis on smoke runs
@pytest.mark.slow
def test_p6b_analysis_uses_complete_seeds_and_leaves_p6s_analysis_alone(tmp_path):
    runs = tmp_path / "runs_folder"
    for s in (1, 2):
        r = subprocess.run([PY, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--device", "cpu", "--seeds", str(s),
                            "--out", str(runs / f"tiny_s{s}")], capture_output=True, text=True, timeout=1800,
                           cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    def analyse(design, out):
        r = subprocess.run([PY, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--device", "cpu", "--design", design,
                            "--analyse", "--from", str(runs), "--out", str(out)], capture_output=True, text=True,
                           timeout=600, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return out

    p6, p6b = analyse("small", tmp_path / "a_p6"), analyse("p6b_small", tmp_path / "a_p6b")
    for f in ("checkpoints.csv", "manipulation_checks.csv", "secondary.csv"):
        assert (p6 / f).read_bytes() == (p6b / f).read_bytes(), f            # the same rows, the same random stream
    seeds = json.loads((p6b / "p6b_seeds.json").read_text())
    assert seeds["complete"] == [1, 2] and seeds["incomplete"] == [] and not (p6 / "p6b_seeds.json").exists()
    f6, f6b = pd.read_csv(p6 / "family.csv"), pd.read_csv(p6b / "family.csv")
    pd.testing.assert_frame_equal(f6.drop(columns="reading"), f6b.drop(columns=["reading", "positive_control_passed"]))
    assert (p6 / "summary.md").read_text().startswith("# P6 factorial")
    assert (p6b / "summary.md").read_text().startswith("# P6b factorial")
    # a seed with a corner missing is left out whole
    unit = runs / "tiny_s2" / "units" / "tiny_zipf1.0_H0.9_s2.json"
    assert unit.exists()
    unit.rename(tmp_path / "moved.json")
    part = analyse("p6b_small", tmp_path / "a_part")
    seeds = json.loads((part / "p6b_seeds.json").read_text())
    assert seeds["complete"] == [1] and seeds["incomplete"] == [2]
    assert set(pd.read_csv(part / "checkpoints.csv")["seed"]) == {1}
    # a run whose CPU statistics are not all there does not count as finished
    (tmp_path / "moved.json").rename(unit)
    st = next((runs / "tiny_s1" / "runs").glob("tiny_light_H0.5_s1/checkpoints/*/stats/*.json"))
    shutil.move(str(st), str(tmp_path / "stat.json"))
    seeds = json.loads((analyse("p6b_small", tmp_path / "a_stat") / "p6b_seeds.json").read_text())
    assert seeds["complete"] == [2] and seeds["incomplete"] == [1]


# ----------------------------------------------------------------------------- the waiter
WAITER = str(ROOT / "p6b_after_olmo.py")


def standin(tmp_path, script, *args):
    """A python process whose command line runs `script` (a stand-in that sleeps until killed)."""
    f = tmp_path / "bin" / script
    f.parent.mkdir(exist_ok=True)
    f.write_text("import time\nwhile True:\n    time.sleep(0.05)\n")
    return subprocess.Popen([PY, str(f), *args])


def progress(logs, *lines):
    with open(logs / "progress.txt", "a") as f:
        for ln in lines:
            f.write(ln + "\n")


def setup(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    end = tmp_path / "job_end"
    jobs = [{"name": "p6b_standin", "args": ["-c", f"open({str(end)!r}, 'w').write('ran')"],
             "out": "results/p6b_small/x", "gpu": 0.0}]
    jf = tmp_path / "p6b_queue_small.json"
    jf.write_text(json.dumps(jobs))
    return logs, jf, end


def waiter(tmp_path, jf, *extra):
    return subprocess.Popen([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--poll", "0.2", *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for(cond, timeout=30.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.05)


def read_until(p, text, timeout=30.0):
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        ln = p.stdout.readline()
        if not ln:
            break
        lines.append(ln)
        if text in ln:
            return "".join(lines)
    raise AssertionError(f"{text!r} not in {''.join(lines)!r}")


def hold(path):
    p = subprocess.Popen([PY, "-c", "import fcntl, os, sys, time; f = open(sys.argv[1], 'a'); "
                          "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True)\n"
                          "while not os.path.exists(sys.argv[1] + '.release'): time.sleep(0.02)", str(path)],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def kill(*ps):
    for p in ps:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait(10)


def test_p6b_starts_only_after_p4_and_then_olmo7b_have_ended(tmp_path):
    logs, jf, end = setup(tmp_path)
    queue = hold(logs / ".run_lane.queue.lock")                       # the queue running P4
    p4 = standin(tmp_path, "p4_clipping.py", "--model", "transformer", "--out", "results/p4_transformer")
    olmo = standin(tmp_path, "p1_checkpoints.py", "--suite", "olmo7b", "--out", "results/p1_olmo7b")
    progress(logs, "2026-11-20T10:00:00Z  START  p1_olmo7b  total 1.00 h",
             "2026-11-21T09:00:00Z  START  p4_transformer  total 1.00 h  (queue)")
    w = None
    try:
        w = waiter(tmp_path, jf)
        read_until(w, "waiting until the P4 transformer and p1_olmo7b have both ended")
        time.sleep(1.0)
        # P4 ends and its queue ends: OLMo-7B now runs, P6b still waits
        kill(p4)
        progress(logs, "2026-11-24T16:30:00Z  END    p4_transformer  exit 0  elapsed 72.00 h  total 73.00 h  (queue)")
        (logs / ".run_lane.queue.lock.release").write_text("")
        queue.wait(10)
        time.sleep(1.5)
        assert not end.exists() and w.poll() is None
        # OLMo-7B's process ends but run_all.sh has not written its END line yet: still waiting
        kill(olmo)
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None
        progress(logs, "2026-11-25T03:00:00Z  END    p1_olmo7b  exit 0  elapsed 10.00 h  total 83.00 h")
        out, _ = w.communicate(timeout=120)
        assert w.returncode == 0, out
        assert end.read_text() == "ran" and (logs / "p6b_standin.done").exists()
        assert "both have ended" in out and "the P6b queue ended with exit 0" in out
    finally:
        kill(w, p4, olmo, queue)


def test_p6b_waits_while_olmo7b_runs_after_a_failed_p4(tmp_path):
    logs, jf, end = setup(tmp_path)
    olmo = standin(tmp_path, "p1_checkpoints.py", "--suite", "olmo7b")
    progress(logs, "2026-11-20T10:00:00Z  START  p1_olmo7b  total 1.00 h",
             "2026-11-21T09:00:00Z  START  p4_transformer  total 1.00 h  (queue)",
             "2026-11-21T10:00:00Z  END    p4_transformer  exit 1  elapsed 1.00 h  total 2.00 h  (queue)")
    w = None
    try:
        w = waiter(tmp_path, jf)
        read_until(w, "p1_olmo7b: running or waiting")
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None
        kill(olmo)
        progress(logs, "2026-11-21T12:00:00Z  END    p1_olmo7b  exit 130 (interrupted)  elapsed 2.00 h  total 4.00 h")
        out, _ = w.communicate(timeout=120)
        assert w.returncode == 0 and end.exists(), out
    finally:
        kill(w, olmo)


def test_waiter_refusals_check_and_stop(tmp_path):
    logs, jf, end = setup(tmp_path)
    # nothing known about P4 or OLMo-7B: it would wait for ever, so it refuses
    r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2 and "neither running nor ended" in r.stdout and "neither running or waiting" in r.stdout
    r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(tmp_path / "none.json")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "is missing" in r.stdout
    olmo = standin(tmp_path, "p1_checkpoints.py", "--suite", "olmo7b")
    progress(logs, "2026-11-21T09:00:00Z  START  p4_transformer  total 1.00 h  (queue)",
             "2026-11-21T10:00:00Z  END    p4_transformer  exit 0  elapsed 1.00 h  total 2.00 h  (queue)",
             "2026-11-21T10:00:01Z  START  p1_olmo7b  total 2.00 h")
    w = None
    try:
        before = sorted(p.name for p in logs.iterdir())
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "P4 transformer: ended" in r.stdout and "p1_olmo7b: running or waiting" in r.stdout
        assert "a new waiter would wait, then start P6b" in r.stdout
        assert sorted(p.name for p in logs.iterdir()) == before          # --check changes nothing
        w = waiter(tmp_path, jf)
        read_until(w, "waiting until")
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "another p6b_after_olmo.py runs" in r.stdout
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert "a waiter runs" in r.stdout
        w.terminate()
        out, _ = w.communicate(timeout=30)
        assert w.returncode == 130 and "stopped while waiting; nothing started" in out and not end.exists()
        (logs / "p6b_standin.done").write_text("x")
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "every P6b job is done already" in r.stdout
    finally:
        kill(w, olmo)


def test_last_event_reads_every_kind_of_progress_line(tmp_path):
    import p6b_after_olmo as wo
    logs = tmp_path
    assert wo.last_event(logs, "p1_olmo7b") == (None, "")
    progress(logs, "2026-11-20T10:00:00Z  START  p1_olmo7b  total 1.00 h",
             "2026-11-20T11:00:00Z  START  p4_transformer  total 1.00 h  (queue)",
             "2026-11-20T12:00:00Z  END    p1_olmo7b  exit 0  elapsed 2.00 h  total 3.00 h",
             "2026-11-20T13:00:00Z  SKIP   p4_transformer  (finished OK: x; delete logs/p4_transformer.done to rerun)")
    assert wo.last_event(logs, "p1_olmo7b")[0] == "END"
    assert wo.last_event(logs, "p4_transformer")[0] == "SKIP"
    assert wo.last_event(logs, "p1_olmo")[0] is None                    # not a prefix match
