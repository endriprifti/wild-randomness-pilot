"""P2, P4 and P5 save every finished unit and a restart runs only the others, with the same results (common/units.py);
other settings in the folder stop the job."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from common import units

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_units_refuse_other_settings_and_report_progress(tmp_path):
    log = []
    u = units.Units(tmp_path, {"cfg": {"a": 1, "b": {"c": 2}}, "seed": 0}, "job", log=log.append)
    u.plan(["x", "y", "z"])
    u.start("x")
    u.put("x", {"v": 1})
    u.put("y", {"v": 2})                        # loaded or not timed: saved without hours
    d = json.loads((tmp_path / "progress.json").read_text())
    assert d["units_done"] == 2 and d["units_total"] == 3 and d["measured_units"] == 1
    assert d["hours_left"] == pytest.approx(d["mean_unit_hours"])
    assert "unit 1/3 saved: x" in log[0] and "left 2 unit(s)" in log[0]
    again = units.Units(tmp_path, {"cfg": {"a": 1, "b": {"c": 2}}, "seed": 0}, "job")
    assert again.get("x") == {"v": 1} and again.has("y") and not again.has("z")
    with pytest.raises(SystemExit, match="b.c"):
        units.Units(tmp_path, {"cfg": {"a": 1, "b": {"c": 3}}, "seed": 0}, "job")
    assert units.seed_for(5, "t") == units.seed_for(5, "t") != units.seed_for(5, "u")


def _run(script, out, *extra, env=None):
    r = subprocess.run([PY, str(ROOT / script), "--smoke", "--device", "cpu", "--out", str(out), *extra],
                       capture_output=True, text=True, timeout=900, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def _same(a: Path, b: Path, files):
    for f in files:
        x, y = pd.read_csv(a / f), pd.read_csv(b / f)
        cols = [c for c in x if c != "wall_s"]
        pd.testing.assert_frame_equal(x[cols], y[cols], obj=f)


def _resume(script, tmp_path, sub, drop, files, *extra, env=None):
    out = tmp_path / "out"
    _run(script, out, *extra, env=env)
    ref = tmp_path / "ref"
    ref.mkdir()
    for f in files:
        (ref / f).write_bytes((out / f).read_bytes())
    for k in drop:
        (out / sub / f"{k}.json").unlink()            # as if the job had stopped before these units
    r = _run(script, out, *extra, env=env)
    assert r.stdout.count(" saved: ") == len(drop), r.stdout[-2000:]
    _same(ref, out, files)
    return out, r


@pytest.mark.slow
def test_p5_cells_resume(tmp_path):
    out, r = _resume("p5_escape.py", tmp_path, "cells", ["cell001", "cell004"], ["restarts.csv", "grid.csv", "fits.csv"])
    assert "read from" in r.stdout and json.loads((out / "progress.json").read_text())["units_done"] == 6


@pytest.mark.slow
def test_p2_units_resume(tmp_path):
    m = "EleutherAI__pythia-14m"
    out, _ = _resume("p2_w4a4.py", tmp_path, "units", [f"{m}__baseline", f"{m}__w4a4__rotation"],
                     ["schemes.csv", "deciles.csv", "popqa_items.csv", "activations.csv", "trends.csv"])
    r = _run("p2_w4a4.py", out)
    assert "all 8 units saved; not loaded" in r.stdout
    s = out / "units" / "settings.json"
    d = json.loads(s.read_text())
    d["cfg"]["calib"]["n_seqs"] = 3
    s.write_text(json.dumps(d))
    r = subprocess.run([PY, str(ROOT / "p2_w4a4.py"), "--smoke", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode != 0 and "cfg.calib.n_seqs" in r.stderr


@pytest.mark.slow
def test_p4_runs_resume(tmp_path):
    env = dict(os.environ, P4_TOKEN_CACHE=str(tmp_path / "cache"))
    out, r = _resume("p4_clipping.py", tmp_path, "runs", ["transformer_s0_fixed0.0", "transformer_s0_fixed0.1_matched"],
                     ["runs.csv"], "--model", "transformer", env=env)
    assert "token IDs read from" in r.stdout and (out / "runs" / "transformer_s0_ref_norms.npy").exists()
