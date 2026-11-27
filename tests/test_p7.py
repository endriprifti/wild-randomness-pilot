"""P7 (2026-11-27, before any P7 run): P4's transformer arm under SGD with momentum. P4's own outputs unchanged by the
optimizer option, the optimizer and settings, the Phase 0 rule, the job end to end (resume, STOP), the analysis
(trend tests, Holm, gates, readings) on synthetic tables, the queue file, and the waiter that starts P7 only after
P6c has ended."""
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

import config
import p7_stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PRE_P7 = "b1e4472"      # the last commit before P7 (p4_clipping.py as P4's runs used it)


# ----------------------------------------------------------------------------- settings and optimizer
def test_p7_settings_change_only_the_optimizer_and_leave_p4s_alone():
    p7 = config.get("p7_transformer_sgd")
    assert p7["optimizer"] == {"name": "sgd", "momentum": 0.9, "nesterov": False, "weight_decay": 5e-4}
    assert p7["phase0"]["lrs"] == [0.03, 0.1, 0.3, 1.0] and p7["phase0"]["seed"] == 99
    assert p7["phase0"]["extend_low"] == [0.01, 0.003] and p7["phase0"]["extend_high"] == [3.0, 10.0]
    assert p7["phase0"]["capture"] == 0.7 and p7["phase0"]["adamw_test_loss"] == 4.460
    assert p7["seeds"] == [0, 1, 2, 3, 4]
    assert [tuple(a) for a in p7["arms"]] == [("fixed", 0.0, False), ("fixed", 0.01, False), ("fixed", 0.1, False),
                                              ("fixed", 0.5, False), ("fixed", 0.1, True)]
    assert p7["analysis"]["scores"] == {"0.0": -1.5, "0.01": -0.5, "0.1": 0.5, "0.5": 1.5}
    assert p7["analysis"]["n_perm"] == 9999 and p7["analysis"]["alpha"] == 0.05
    # amended 2026-11-27 before any P7 run: the 50% arm's range was [0.25, 0.75]
    assert p7["analysis"]["manipulation"] == {"min_10": 0.03, "range_50": [0.25, 0.95]}
    # P4's transformer arm: P7 uses it as it is (P4's data, steps, warm-up, directions)
    p4 = config.get("p4_clipping")
    assert p4["transformer"]["weight_decay"] == 0.1          # AdamW's, not used by P7's SGD


def test_p4s_section_of_config_is_unchanged():
    """P4's units refuse a resume with other settings: its config section must hash as before P7."""
    import hashlib
    r = subprocess.run(["git", "show", f"{PRE_P7}:config.py"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("git history not available")
    old = {}
    exec(compile(r.stdout, "config_old.py", "exec"), old)
    assert set(config.CONFIG) - set(old["CONFIG"]) == {"p7_transformer_sgd"}
    for name in old["CONFIG"]:              # P4's, and every other section (P6 reads P1's)
        for smoke in (False, True):
            a = json.dumps(old["get"](name, smoke), sort_keys=True, default=str)
            b = json.dumps(config.get(name, smoke), sort_keys=True, default=str)
            assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest(), (name, smoke)


def test_make_optimizer():
    import torch
    import p4_clipping as p4
    m = torch.nn.Linear(3, 2)
    sgd = p4.make_optimizer(m.parameters(), 0.3, config.get("p7_transformer_sgd")["optimizer"])
    g = sgd.param_groups[0]
    assert isinstance(sgd, torch.optim.SGD)
    assert (g["lr"], g["momentum"], g["nesterov"], g["weight_decay"], g["dampening"]) == (0.3, 0.9, False, 5e-4, 0)
    adamw = p4.make_optimizer(m.parameters(), 1e-3, {"name": "adamw", "weight_decay": 0.1})
    ref = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    a, b = adamw.param_groups[0], ref.param_groups[0]
    assert {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}
    with pytest.raises(ValueError):
        p4.make_optimizer(m.parameters(), 0.1, {"name": "lion"})


def _p4_smoke(script: Path, out: Path, cache: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "P4_TOKEN_CACHE": str(cache), "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([PY, str(script), "--smoke", "--model", "transformer", "--no-h71", "--device", "cpu",
                           "--out", str(out)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)


@pytest.mark.slow
def test_p4_smoke_is_byte_identical_with_the_file_before_p7(tmp_path):
    """P4's own runs: the same rows (wall time aside) and the same gradient series with the file before P7."""
    r = subprocess.run(["git", "show", f"{PRE_P7}:p4_clipping.py"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("git history not available")
    old = tmp_path / "old" / "p4_clipping.py"
    old.parent.mkdir()
    old.write_text(r.stdout)
    cache = tmp_path / "cache"
    outs = {}
    for name, script in (("old", old), ("new", ROOT / "p4_clipping.py")):
        p = _p4_smoke(script, tmp_path / name / "out", cache)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        outs[name] = tmp_path / name / "out"
    a = pd.read_csv(outs["old"] / "runs.csv").drop(columns=["wall_s"])
    b = pd.read_csv(outs["new"] / "runs.csv").drop(columns=["wall_s"])
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    fa = sorted(p.name for p in (outs["old"] / "hgrad_series").glob("*.npy"))
    assert fa and fa == sorted(p.name for p in (outs["new"] / "hgrad_series").glob("*.npy"))
    for f in fa:
        assert (outs["old"] / "hgrad_series" / f).read_bytes() == (outs["new"] / "hgrad_series" / f).read_bytes()
    for f in sorted((outs["old"] / "runs").glob("*.json")):
        ja, jb = json.loads(f.read_text()), json.loads((outs["new"] / "runs" / f.name).read_text())
        for j in (ja, jb):
            j.get("row", {}).pop("wall_s", None)
            j.pop("seconds", None)
            j.pop("saved_utc", None)
        assert ja == jb, f.name


# ----------------------------------------------------------------------------- Phase 0
def _p0():
    return dict(config.get("p7_transformer_sgd")["phase0"])


def _row(loss, first=10.0, diverged=False):
    return {"first_train_loss": first, "test_loss": float("nan") if diverged else loss, "diverged": diverged}


def test_phase0_rule():
    import p7_transformer_sgd as p7
    p0 = _p0()
    # bound: L0 - 0.7 (L0 - 4.46); L0 = 10 -> 6.122
    assert p7.phase0_bound(10.0, p0) == pytest.approx(10 - 0.7 * (10 - 4.46))
    # nothing run yet: the grid, in order
    d = p7.phase0_decide({}, p0)
    assert d["status"] == "run" and d["next"] == [0.03, 0.1, 0.3, 1.0]
    d = p7.phase0_decide({0.03: _row(7.0)}, p0)
    assert d["status"] == "run" and d["next"] == [0.1, 0.3, 1.0]
    # two pass: the lower test loss is chosen; a diverged run never passes
    rows = {0.03: _row(7.0), 0.1: _row(6.0), 0.3: _row(5.5), 1.0: _row(1.0, diverged=True)}
    d = p7.phase0_decide(rows, p0)
    assert d["status"] == "chosen" and d["chosen_lr"] == 0.3 and d["extension"] is None
    assert [t["passes"] for t in d["runs"]] == [False, True, True, False]


def test_phase0_rule_bound_is_per_run():
    import p7_transformer_sgd as p7
    p0 = _p0()
    rows = {0.03: _row(6.5, first=11.0), 0.1: _row(6.3, first=10.0), 0.3: _row(6.6, first=12.0), 1.0: _row(9.0)}
    # bounds: 6.422, 6.122, 6.722: only 0.3 passes (6.6 <= 6.722), although 0.1 has the lowest loss
    d = p7.phase0_decide(rows, p0)
    assert [t["passes"] for t in d["runs"]] == [False, False, True, False]
    assert d["status"] == "chosen" and d["chosen_lr"] == 0.3


def test_phase0_extension_and_stop():
    import p7_transformer_sgd as p7
    p0 = _p0()
    low = {0.03: _row(7.0), 0.1: _row(7.5), 0.3: _row(8.0), 1.0: _row(1.0, diverged=True)}
    d = p7.phase0_decide(low, p0)
    assert d["status"] == "run" and d["extension"] == {"side": "low", "lrs": [0.01, 0.003]}
    assert d["next"] == [0.01, 0.003]
    high = {0.03: _row(9.0), 0.1: _row(8.0), 0.3: _row(7.5), 1.0: _row(7.0)}
    d = p7.phase0_decide(high, p0)
    assert d["extension"] == {"side": "high", "lrs": [3.0, 10.0]} and d["next"] == [3.0, 10.0]
    # the extension passes: chosen among the extension's rates only
    d = p7.phase0_decide({**high, 3.0: _row(6.0), 10.0: _row(5.0)}, p0)
    assert d["status"] == "chosen" and d["chosen_lr"] == 10.0
    d = p7.phase0_decide({**high, 3.0: _row(6.0)}, p0)
    assert d["status"] == "run" and d["next"] == [10.0]
    # none passes again: P7 stops before Phase 1, with the fixed reading
    d = p7.phase0_decide({**low, 0.01: _row(7.0), 0.003: _row(8.0)}, p0)
    assert d["status"] == "stopped" and d["reading"] == p7.STOPPED_READING
    assert p7.STOPPED_READING == "SGD with momentum did not train the transformer to the bound at P4's budget"
    # both ends diverged: infinite at both, the low end counts as better
    both = {0.03: _row(1, diverged=True), 0.1: _row(8.0), 0.3: _row(8.0), 1.0: _row(1, diverged=True)}
    assert p7.phase0_decide(both, p0)["extension"]["side"] == "low"


def test_phase0_rule_is_the_one_in_p7_stats():
    import p7_transformer_sgd as p7
    assert p7_stats.PATTERN.startswith("ResNet-18's clipping pattern")
    assert p7.KIND == "transformer"


# ----------------------------------------------------------------------------- the job end to end
def _p7(tmp_path, *args, timeout=1800):
    env = {**os.environ, "P4_TOKEN_CACHE": str(tmp_path / "cache"), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([PY, str(ROOT / "p7_transformer_sgd.py"), "--smoke", "--device", "cpu",
                           "--out", str(tmp_path / "p7"), *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)


def test_stop_file_before_phase0_starts_nothing(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    r = _p7(tmp_path, "--stop-file", str(stop))
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads((tmp_path / "p7" / "stopped.json").read_text())
    assert st["where"] == "before Phase 0"
    assert not (tmp_path / "p7" / "phase0.json").exists() and not (tmp_path / "p7" / "runs.csv").exists()


@pytest.mark.slow
def test_p7_smoke_end_to_end_resume_and_stop(tmp_path):
    out = tmp_path / "p7"
    r = _p7(tmp_path, "--seeds", "0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ph = json.loads((out / "phase0.json").read_text())
    assert ph["status"] == "chosen" and ph["chosen_lr"] in (0.1, 1.0)
    assert {t["lr"] for t in ph["runs"]} == {0.1, 1.0}
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 5 and set(runs["optimizer"]) == {"sgd"} and set(runs["seed"]) == {0}
    assert (runs["phase0_lr"] == ph["chosen_lr"]).all() and (runs["momentum"] == 0.9).all()
    assert "lambda_max" not in runs.columns and "hessian_trace" not in runs.columns
    plain = runs[~runs["matched"]]
    assert np.allclose(plain["lr"], ph["chosen_lr"])
    m = runs[runs["matched"]].iloc[0]
    assert m["lr"] == pytest.approx(ph["chosen_lr"] * m["lr_scale"])
    tags = sorted(p.stem for p in (out / "hgrad_series").glob("*.npy"))
    assert tags == sorted(["transformer_s0_fixed0.0", "transformer_s0_fixed0.01", "transformer_s0_fixed0.1",
                           "transformer_s0_fixed0.5", "transformer_s0_fixed0.1_matched"])
    # h_link reads the folder like a P4 folder
    import h_link
    specs = h_link.p4_series(out)
    assert sorted(s["tag"] for s in specs) == tags and {s["folder"] for s in specs} == {"p7"}
    # resume: nothing is trained again (Phase 0 is decided again from its saved runs)
    t0 = time.time()
    r = _p7(tmp_path, "--seeds", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((out / "phase0.json").read_text())["chosen_lr"] == ph["chosen_lr"]
    pd.testing.assert_frame_equal(pd.read_csv(out / "runs.csv").drop(columns=["wall_s"]),
                                  runs.drop(columns=["wall_s"]))
    # STOP before a seed that is not saved: seed 0 (saved) stays, seed 1 is not started
    stop = tmp_path / "STOP"
    stop.write_text("")
    r = _p7(tmp_path, "--seeds", "0,1", "--stop-file", str(stop))
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((out / "stopped.json").read_text())["where"] == "before seed 1 of Phase 1"
    assert set(pd.read_csv(out / "runs.csv")["seed"]) == {0}
    # STOP with a Phase 0 run still to do: stopped there, before that run
    (out / "phase0" / "runs" / "phase0_lr1.json").unlink()
    r = _p7(tmp_path, "--seeds", "0", "--stop-file", str(stop))
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((out / "stopped.json").read_text())["where"] == "in Phase 0, before rate 1"
    assert time.time() - t0 < 600


# ----------------------------------------------------------------------------- the analysis
SCORES = {0.0: -1.5, 0.01: -0.5, 0.1: 0.5, 0.5: 1.5}
CLIP = {0.0: 0.0, 0.01: 0.012, 0.1: 0.11, 0.5: 0.48}


def _tables(tmp_path, rng, b_h=0.0, b_w=0.0, b_iaaft=0.0, seeds=range(5), clip=None, diverged=(), drop=(),
            phase0="chosen", noise=0.002):
    """runs.csv, phase0.json and h_link's p4_grad.csv of a synthetic P7: H(|x|) = 0.55 + b_h score + seed + noise;
    delta_h = 0.04 + (b_w + b_iaaft) score; IAAFT mean = 0.03 + b_iaaft score (so the excess has slope b_w)."""
    runs_dir = tmp_path / "p7_transformer_sgd"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "phase0.json").write_text(json.dumps(
        {"status": phase0, "chosen_lr": 0.3 if phase0 == "chosen" else None, "extension": None,
         "reading": None if phase0 == "chosen" else "SGD with momentum did not train the transformer to the bound "
                                                    "at P4's budget"}))
    clip = clip or CLIP
    runs, hl = [], []
    for s in seeds:
        se = rng.normal(0, 0.01)
        for p, m in [(0.0, False), (0.01, False), (0.1, False), (0.5, False), (0.1, True)]:
            tag = f"transformer_s{s}_fixed{p}{'_matched' if m else ''}"
            if tag in drop:
                continue
            sc = SCORES[p]
            div = tag in diverged
            runs.append({"model": "transformer", "seed": s, "rule": "fixed", "p_clip_target": p, "matched": m,
                         "diverged": div, "p_clip_realized": clip[p] + rng.normal(0, 0.001) * (p > 0),
                         "lr": 0.3, "lr_scale": 1.0, "alpha": 1.9 + rng.normal(0, 0.01), "zeta": 3.0,
                         "test_loss": 4.6 + rng.normal(0, 0.01), "train_loss": 4.5})
            h = 0.55 + b_h * sc + se + rng.normal(0, noise)
            ia = 0.03 + b_iaaft * sc + rng.normal(0, noise)
            dh = 0.04 + (b_w + b_iaaft) * sc + rng.normal(0, noise)
            for scales in ("main", "sensitivity", "detrended", "rescaled"):
                hl.append({"source": "p4", "folder": "p7_transformer_sgd", "tag": tag, "scales": scales, "seed": s,
                           "rule": "fixed", "p_clip_target": p, "matched": m, "diverged": div, "warm": 91,
                           "H_abs_dfa": h, "delta_h": dh, "delta_h_iaaft_mean": ia, "delta_h_iaaft_q95": ia + 0.003,
                           "p_iaaft": 0.5, "H_abs_dfa_shuf_mean": 0.5, "H_abs_dfa_phase_mean": 0.53,
                           "memory_pass": h > 0.56, "width_pass": dh > ia + 0.003, "pass": False})
    pd.DataFrame(runs).to_csv(runs_dir / "runs.csv", index=False)
    (tmp_path / "hl").mkdir(exist_ok=True)
    pd.DataFrame(hl).to_csv(tmp_path / "hl" / "p4_grad.csv", index=False)
    return runs_dir, tmp_path / "hl" / "p4_grad.csv"


def _analyse(tmp_path, runs_dir, hl, n_perm=999):
    an = dict(config.get("p7_transformer_sgd")["analysis"])
    an["n_perm"] = n_perm
    return p7_stats.analyse(runs_dir, hl, tmp_path / "analysis", an, an["scores"])


def test_rise_in_memory_and_width(tmp_path, rng):
    s = _analyse(tmp_path, *_tables(tmp_path, rng, b_h=0.03, b_w=0.01))
    assert s["reading"] == p7_stats.PATTERN + " in memory and width"
    t1, t2 = s["tests"]
    assert t1["p_holm"] < 0.05 and t2["p_holm"] < 0.05 and t1["direction"] == t2["direction"] == "rise"
    assert t1["rise"] == pytest.approx(3 * 0.03, abs=0.01)
    assert s["gates"]["clipping manipulation"] and s["gates"]["observed width rises (for T2)"]
    for f in ("tests.csv", "gates.csv", "arms.csv", "exploratory.csv", "twins.csv", "summary.json", "summary.md"):
        assert (tmp_path / "analysis" / f).exists()
    arms = pd.read_csv(tmp_path / "analysis" / "arms.csv")
    assert list(arms["arm"]) == ["fixed0.0", "fixed0.01", "fixed0.1", "fixed0.5", "fixed0.1_matched"]
    ex = pd.read_csv(tmp_path / "analysis" / "exploratory.csv")
    assert {"arm minus unclipped", "matched against plain 10%", "trend on the detrended rows",
            "trend on the rescaled rows"} <= set(ex["part"])


def test_rise_in_memory_only_and_no_effect(tmp_path, rng):
    s = _analyse(tmp_path / "a", *_tables(tmp_path / "a", rng, b_h=0.03))
    assert s["reading"] == p7_stats.PATTERN + " (T1)"
    s = _analyse(tmp_path / "b", *_tables(tmp_path / "b", rng))
    assert s["reading"].startswith(p7_stats.NONE + ": " + p7_stats.NO_POWER)
    assert "detectable" in s["reading"]
    assert all(t["direction"] == "none" for t in s["tests"])
    assert s["tests"][0]["mde_rise"] > 0 and s["tests"][0]["mde_rise_holm1"] > s["tests"][0]["mde_rise"]


def test_fall_reads_opposite(tmp_path, rng):
    s = _analyse(tmp_path, *_tables(tmp_path, rng, b_h=-0.03))
    assert s["reading"] == "T1: " + p7_stats.OPPOSITE
    assert s["tests"][0]["reading"] == p7_stats.OPPOSITE


def test_width_excess_from_a_falling_baseline_is_not_width(tmp_path, rng):
    # the IAAFT mean falls with clipping while delta_h stays: the excess rises, the observed width does not
    s = _analyse(tmp_path, *_tables(tmp_path, rng, b_w=0.01, b_iaaft=-0.01))
    t2 = s["tests"][1]
    assert t2["p_holm"] < 0.05 and t2["slope"] > 0
    assert t2["direction"] == "baseline" and t2["reading"] == p7_stats.BASELINE
    assert s["reading"] == "T2: " + p7_stats.BASELINE
    assert not s["gates"]["observed width rises (for T2)"]


def test_manipulation_gate(tmp_path, rng):
    bad = {**CLIP, 0.5: 0.97}                     # the 50% arm clips 97% of steps
    s = _analyse(tmp_path / "a", *_tables(tmp_path / "a", rng, clip=bad))
    assert not s["gates"]["clipping manipulation"] and s["reading"] == p7_stats.UNINFORMATIVE
    assert all(t["reading"] == p7_stats.UNINFORMATIVE for t in s["tests"])
    # a test that passes keeps its reading
    s = _analyse(tmp_path / "b", *_tables(tmp_path / "b", rng, clip=bad, b_h=0.03))
    assert s["tests"][0]["direction"] == "rise" and s["tests"][1]["reading"] == p7_stats.UNINFORMATIVE
    assert s["reading"] == p7_stats.PATTERN + " (T1)"
    g = p7_stats.manipulation_gate
    rule = config.get("p7_transformer_sgd")["analysis"]["manipulation"]
    d = pd.DataFrame({"p_clip_target": [0.0, 0.01, 0.1, 0.5], "p_clip_realized": [0.0, 0.01, 0.02, 0.5]})
    assert not g(d, rule)["ok_10"]
    d["p_clip_realized"] = [0.0, 0.2, 0.1, 0.5]
    assert not g(d, rule)["ordered"]
    # the amended range: ResNet-18's 68.9% and P7's smoke 77 to 93% pass, all steps or nearly all do not
    for f50, ok in ((0.689, True), (0.77, True), (0.93, True), (0.95, True), (0.96, False), (0.24, False)):
        d["p_clip_realized"] = [0.0, 0.012, 0.11, f50]
        assert g(d, rule)["ok_50"] is ok and g(d, rule)["passed"] is ok, f50


def test_phase0_stopped_gives_its_reading_and_no_tests(tmp_path, rng):
    s = _analyse(tmp_path, *_tables(tmp_path, rng, phase0="stopped"))
    assert s["reading"] == "SGD with momentum did not train the transformer to the bound at P4's budget"
    assert "tests" not in s and (tmp_path / "analysis" / "summary.md").exists()


def test_without_h_link_nothing_is_tested(tmp_path, rng):
    runs_dir, hl = _tables(tmp_path, rng)
    hl.unlink()
    s = _analyse(tmp_path, runs_dir, hl)
    assert s["reading"].startswith("no test: T1: not testable") and "T2: not testable" in s["reading"]
    assert all(t["direction"] == "untestable" for t in s["tests"])


def test_incomplete_seeds_and_diverged_runs_are_left_out(tmp_path, rng):
    runs_dir, hl = _tables(tmp_path, rng, b_h=0.03, drop=("transformer_s4_fixed0.5",),
                           diverged=("transformer_s3_fixed0.01",))
    s = _analyse(tmp_path, runs_dir, hl)
    assert s["seeds_complete"] == [0, 1, 2, 3]
    assert s["diverged_left_out"] == ["transformer_s3_fixed0.01"] and s["n_runs_tests"] == 15
    assert not s["gates"]["no Phase 1 run diverged"]


def test_trend_fit_is_least_squares_with_seed_blocks(rng):
    seeds = np.repeat(np.arange(5), 4)
    score = np.tile([-1.5, -0.5, 0.5, 1.5], 5)
    y = 0.2 * score + rng.normal(0, 1, 5)[seeds] + rng.normal(0, 0.1, 20)
    X = np.column_stack([np.ones(20)] + [(seeds == b).astype(float) for b in range(1, 5)] + [score])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(resid @ resid / (20 - 6) * np.linalg.inv(X.T @ X)[-1, -1])
    f = p7_stats.trend_fit(y, score, seeds)
    assert f["slope"] == pytest.approx(beta[-1]) and f["se"] == pytest.approx(se) and f["df"] == 14
    # unbalanced (one run left out) too
    keep = np.arange(20) != 7
    X2, y2 = X[keep], y[keep]
    b2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    assert p7_stats.trend_fit(y2, score[keep], seeds[keep])["slope"] == pytest.approx(b2[-1])


def test_permutation_p(rng):
    seeds = np.repeat(np.arange(5), 4)
    score = np.tile([-1.5, -0.5, 0.5, 1.5], 5)
    y_null = rng.normal(0, 1, 20)
    p = p7_stats.perm_p(y_null, score, seeds, np.random.default_rng(1), 999)
    assert 0.001 <= p <= 1
    y = 1.0 * score + rng.normal(0, 0.1, 20)
    assert p7_stats.perm_p(y, score, seeds, np.random.default_rng(1), 999) == pytest.approx(1 / 1000)
    # the smallest p within-seed label permutation can give is 2 / 24^5 (two-sided), far below 1 / 10,000
    ps = [p7_stats.perm_p(rng.normal(0, 1, 20), score, seeds, np.random.default_rng(k), 199) for k in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10


def test_holm():
    assert p7_stats.holm([0.01, 0.04]) == [0.02, 0.04]
    assert p7_stats.holm([0.04, 0.01]) == [0.04, 0.02]
    assert p7_stats.holm([0.03, 0.03]) == [0.06, 0.06]


def test_twins_pair_each_sgd_run_with_its_adamw_run(tmp_path, rng):
    runs_dir, hl = _tables(tmp_path, rng)
    p4 = tmp_path / "p4_transformer"
    p4.mkdir()
    r = pd.read_csv(runs_dir / "runs.csv")
    r["test_loss"] = 4.46
    r.to_csv(p4 / "runs.csv", index=False)
    h = pd.read_csv(hl)
    h["folder"] = "p4_transformer"
    h["H_abs_dfa"] = 0.5
    (tmp_path / "hl4").mkdir()
    h.to_csv(tmp_path / "hl4" / "p4_grad.csv", index=False)
    an = dict(config.get("p7_transformer_sgd")["analysis"], n_perm=99)
    s = p7_stats.analyse(runs_dir, hl, tmp_path / "analysis", an, an["scores"], p4_runs_path=p4,
                         p4_hlink=tmp_path / "hl4" / "p4_grad.csv")
    tw = pd.read_csv(tmp_path / "analysis" / "twins.csv")
    assert s["twins"] == 25 and len(tw) == 25
    assert np.allclose(tw["H_abs_dfa_adamw"], 0.5) and np.allclose(tw["test_loss_adamw"], 4.46)
    assert np.allclose(tw["H_abs_dfa_diff"], tw["H_abs_dfa_sgd"] - 0.5)


def test_p7_stats_cli_on_missing_inputs(tmp_path):
    r = subprocess.run([PY, str(ROOT / "p7_stats.py"), "--results", str(tmp_path), "--out", str(tmp_path / "a")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Phase 0 not decided" in r.stdout


# ----------------------------------------------------------------------------- the queue file
def test_p7_queue_file():
    import run_queue
    jobs = [run_queue.Job(**{**d, "after": tuple(d.get("after", ())), "needs": tuple(d.get("needs", ()))})
            for d in json.loads((ROOT / "p7_queue.json").read_text())]
    names = [j.name for j in jobs]
    assert names == ["p7_transformer_sgd", "p7_h_link", "p7_decomposition", "p7_analysis"]
    j = {x.name: x for x in jobs}
    assert j["p7_transformer_sgd"].args == ["p7_transformer_sgd.py", "--stop-file", "results/p7_transformer_sgd/STOP"]
    assert j["p7_transformer_sgd"].out == "results/p7_transformer_sgd" and j["p7_transformer_sgd"].gpu > 0
    assert j["p7_h_link"].args[:4] == ["h_link.py", "--p1", "--p4", "results/p7_transformer_sgd"]
    assert j["p7_h_link"].out == "results/h_link_p7_transformer_sgd"          # not results/h_link (P4's CSVs)
    assert j["p7_h_link"].after == ("p7_transformer_sgd",) and j["p7_h_link"].gpu == 0
    assert j["p7_decomposition"].after == ("p7_h_link",) and j["p7_decomposition"].gpu == 0
    assert set(j["p7_analysis"].after) == {"p7_h_link", "p7_decomposition"} and j["p7_analysis"].gpu == 0
    for x in jobs:
        assert (ROOT / x.args[0]).exists()
    # h_link's default discovery (results/p4_*) does not pick up P7's folder: P4's CSVs stay P4's
    assert not "p7_transformer_sgd".startswith("p4_")


def test_p7_queue_dry_run(tmp_path):
    r = subprocess.run([PY, str(ROOT / "run_queue.py"), "--root", str(tmp_path), "--jobs-file",
                        str(ROOT / "p7_queue.json"), "--dry-run"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    for n in ("p7_transformer_sgd", "p7_h_link", "p7_decomposition", "p7_analysis"):
        assert n in r.stdout


# ----------------------------------------------------------------------------- the waiter
WAITER = str(ROOT / "p7_after_p6c.py")
P6C_JOBS = ["p6c_small_14m_s15", "p6c_small_14m_s16", "p6c_small_analysis"]


def standin(tmp_path, script, *args):
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
    jobs = [{"name": "p7_standin", "args": ["-c", f"open({str(end)!r}, 'w').write('ran')"],
             "out": "results/p7_transformer_sgd", "gpu": 0.0}]
    jf = tmp_path / "p7_queue.json"
    jf.write_text(json.dumps(jobs))
    af = tmp_path / "p6c_queue_small.json"
    af.write_text(json.dumps([{"name": n, "args": ["x"], "out": "results/p6c_small/x"} for n in P6C_JOBS]))
    return logs, jf, af, end


def waiter(tmp_path, jf, af, *extra):
    return subprocess.Popen([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--after-jobs-file",
                             str(af), "--poll", "0.2", *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


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


def release(logs, name, p):
    (logs / (name + ".release")).write_text("")
    p.wait(10)


def kill(*ps):
    for p in ps:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait(10)


def test_p7_starts_only_after_p6c_has_ended(tmp_path):
    logs, jf, af, end = setup(tmp_path)
    p6c_waiter = hold(logs / ".p6c_after_olmo.lock")                 # P6c's waiter, waiting for OLMo-7B
    w = q = queue = None
    try:
        w = waiter(tmp_path, jf, af)
        read_until(w, "waiting until P6c has ended")
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None
        # P6c's queue starts (its waiter still holds its lock) and runs its jobs
        queue = hold(logs / ".run_lane.queue.lock")
        q = standin(tmp_path, "run_queue.py", "--jobs-file", str(af))
        progress(logs, "2026-11-28T10:00:00Z  START  p6c_small_14m_s15  total 1.00 h  (queue)",
                 "2026-11-28T11:00:00Z  END    p6c_small_14m_s15  exit 0  elapsed 1.00 h  total 2.00 h  (queue)",
                 "2026-11-28T11:00:01Z  START  p6c_small_14m_s16  total 2.00 h  (queue)",
                 "2026-11-28T12:00:00Z  END    p6c_small_14m_s16  exit 0  elapsed 1.00 h  total 3.00 h  (queue)",
                 "2026-11-28T12:00:01Z  START  p6c_small_analysis  total 3.00 h  (queue)",
                 "2026-11-28T12:10:00Z  END    p6c_small_analysis  exit 0  elapsed 0.10 h  total 3.10 h  (queue)")
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None                  # its queue process still runs
        kill(q)
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None                  # the queue lock is still held
        release(logs, ".run_lane.queue.lock", queue)
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None                  # P6c's waiter still holds its lock
        release(logs, ".p6c_after_olmo.lock", p6c_waiter)
        out, _ = w.communicate(timeout=120)
        assert w.returncode == 0, out
        assert end.read_text() == "ran" and (logs / "p7_standin.done").exists()
        assert "P6c has ended" in out and "the P7 queue ended with exit 0" in out
    finally:
        kill(w, q, queue, p6c_waiter)


def test_p6c_stopped_for_lack_of_credit_starts_nothing(tmp_path):
    logs, jf, af, end = setup(tmp_path)
    progress(logs, "2026-11-28T10:00:00Z  START  p6c_small_14m_s15  total 1.00 h  (queue)",
             "2026-11-28T11:00:00Z  END    p6c_small_14m_s15  exit 0  elapsed 1.00 h  total 2.00 h  (queue)")
    (tmp_path / "results" / "p6c_small").mkdir(parents=True)
    (tmp_path / "results" / "p6c_small" / "STOP").write_text("")
    r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--after-jobs-file", str(af),
                        "--poll", "0.2"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "P7 not started" in r.stdout and not end.exists()


def test_p7_waiter_refusals_and_check(tmp_path):
    logs, jf, af, end = setup(tmp_path)
    run = lambda *a: subprocess.run([PY, WAITER, "--root", str(tmp_path), *a], capture_output=True, text=True,
                                    timeout=60)
    r = run("--jobs-file", str(tmp_path / "none.json"), "--after-jobs-file", str(af))
    assert r.returncode == 2 and "is missing" in r.stdout
    r = run("--jobs-file", str(jf), "--after-jobs-file", str(tmp_path / "none.json"))
    assert r.returncode == 2 and "P6c's jobs file" in r.stdout
    r = run("--jobs-file", str(jf), "--after-jobs-file", str(af))
    assert r.returncode == 2 and "P6c is neither waiting" in r.stdout
    p6c_waiter = hold(logs / ".p6c_after_olmo.lock")
    w = None
    try:
        before = sorted(p.name for p in logs.iterdir())
        r = run("--jobs-file", str(jf), "--after-jobs-file", str(af), "--check")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "P6c: waiting to start" in r.stdout and "a new waiter would wait, then start P7" in r.stdout
        assert sorted(p.name for p in logs.iterdir()) == before          # --check changes nothing
        w = waiter(tmp_path, jf, af)
        read_until(w, "waiting until")
        r = run("--jobs-file", str(jf), "--after-jobs-file", str(af))
        assert r.returncode == 2 and "another p7_after_p6c.py runs" in r.stdout
        w.terminate()
        out, _ = w.communicate(timeout=30)
        assert w.returncode == 130 and "stopped while waiting; nothing started" in out and not end.exists()
        (logs / "p7_standin.done").write_text("x")
        r = run("--jobs-file", str(jf), "--after-jobs-file", str(af))
        assert r.returncode == 2 and "every P7 job is done already" in r.stdout
    finally:
        kill(w, p6c_waiter)


def test_waiter_state(tmp_path):
    import p7_after_p6c as wa
    logs = tmp_path / "logs"
    logs.mkdir()
    st = wa.state(tmp_path, {}, set(), P6C_JOBS)
    assert not st["known"] and not st["ready"]
    # a failed seed job: the jobs after it never start, and P6c has still ended once nothing of it runs
    progress(logs, "2026-11-28T10:00:00Z  START  p6c_small_14m_s15  total 1.00 h  (queue)",
             "2026-11-28T10:30:00Z  END    p6c_small_14m_s15  exit 1  elapsed 0.50 h  total 1.50 h  (queue)",
             "2026-11-28T10:30:00Z  START  p6c_small_14m_s1  total 1.50 h  (queue)")    # another job, not P6c's
    st = wa.state(tmp_path, {}, set(), P6C_JOBS)
    assert st["ended"] and st["ready"] and st["started"] == ["p6c_small_14m_s15"]
    # a START with no process (a queue killed with -9): waited out
    progress(logs, "2026-11-28T11:00:00Z  START  p6c_small_14m_s16  total 1.50 h  (queue)")
    st = wa.state(tmp_path, {}, set(), P6C_JOBS)
    assert st["lost"] == ["p6c_small_14m_s16"] and not st["ready"]
    assert "p6c_small_14m_s16" in st["last_line"]
