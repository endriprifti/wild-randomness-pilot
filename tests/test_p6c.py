"""P6c (2026-11-27, before any P6c run): training is bit-identical with H7.1 off (the condition for leaving it out),
the design and queue file, the confirmatory family of four tests with its gates and readings (P6's and P6b's
readings untouched), the analysis end to end on smoke runs, and the waiter that starts P6c once p1_olmo7b has
ended."""
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p6_stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = list(range(15, 35))


# ----------------------------------------------------------------------------- H7.1 off: the same training
def _digest(model) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _train(tmp_path, tag, h71_stages, cell, monkeypatch):
    """One smoke run of `cell` with H7.1 at `h71_stages`, as a fresh job starts it (runtime's seeds set first);
    returns the weights' digest before and after each stage's measurement pass, the run folder and its tasks."""
    import config
    import p6_noah_joseph as p6
    from common import runtime
    st = p6.settings(True, {**p6.SMOKE_DESIGN, "h71_stages": list(h71_stages)})
    out = tmp_path / tag
    out.mkdir()
    noah, joseph = cell
    key = p6.run_key("tiny", noah, joseph, 1)
    c = {"key": key, "run": key, "size": "tiny", "noah": noah, "joseph": joseph, "seed": 1,
         "dir": out / "runs" / key}
    args = types.SimpleNamespace(micro_batch=None, loss_chunk=None, eval_batch=4, jvp_batch=32, hvp_batch=None,
                                 keep_state=True)
    run = types.SimpleNamespace(device="cpu", out_dir=out)
    digests, tasks = [], []
    real = p6.gpu_pass

    def gpu_pass(model, ck, *a, **kw):
        before = _digest(model)
        rec = real(model, ck, *a, **kw)
        digests.append((float(ck["stage"]), int(ck["step"]), before, _digest(model)))
        return rec
    monkeypatch.setattr(p6, "gpu_pass", gpu_pass)
    pool = types.SimpleNamespace(submit=tasks.append)
    prog = p6.Progress(out)
    runtime.set_all_seeds(config.SEED)
    res = p6.train_run(c, st, args, run, pool, prog)
    monkeypatch.setattr(p6, "gpu_pass", real)
    return digests, Path(c["dir"]), tasks, res


@pytest.mark.slow
@pytest.mark.parametrize("cell", [("light", "H0.5"), ("zipf1.0", "H0.9")])
def test_training_is_bit_identical_with_h71_off(tmp_path, monkeypatch, cell):
    """The condition of the proposal (section 2): with H7.1 off, the same train_log.csv, the same weights at each
    stage (and after each stage's measurements), the same hgrad series and the same final optimizer state, on smoke
    runs. H7.1 runs mid-training here (at the 50% stage as well as the end), where it could disturb what follows."""
    import torch
    on, d_on, t_on, r_on = _train(tmp_path, "on", [0.5, 1.0], cell, monkeypatch)
    off, d_off, t_off, r_off = _train(tmp_path, "off", [], cell, monkeypatch)
    # H7.1 ran at both stages in one run and at none in the other
    assert {t["name"].split(".")[0] for t in t_on} >= {"h71_fit", "h71_boot"}
    assert not any(t["name"].startswith("h71") for t in t_off)
    for stage in ("0.5", "1"):
        assert (d_on / "checkpoints" / f"stage{stage}" / "samples" / "h71.npy").exists()
        assert not (d_off / "checkpoints" / f"stage{stage}" / "samples" / "h71.npy").exists()
    # the training: identical to the bit
    assert [x[:2] for x in on] == [x[:2] for x in off] == [(0.5, 12), (1.0, 24)]
    assert on == off                                             # weights before and after every measurement pass
    assert all(before == after for _, _, before, after in on)     # no measurement changes the weights
    assert (d_on / "train_log.csv").read_bytes() == (d_off / "train_log.csv").read_bytes()
    assert (d_on / "hgrad.npy").read_bytes() == (d_off / "hgrad.npy").read_bytes()
    s_on = torch.load(d_on / "train_state.pt", weights_only=False)
    s_off = torch.load(d_off / "train_state.pt", weights_only=False)
    assert s_on["step"] == s_off["step"] == 24 and s_on["losses"] == s_off["losses"]
    for k, v in s_on["model"].items():
        assert torch.equal(v, s_off["model"][k]), k
    for i, sv in s_on["opt"]["state"].items():
        for k, v in sv.items():
            assert torch.equal(v, s_off["opt"]["state"][i][k]), (i, k)
    keep = [k for k in r_on if k != "train_hours"]
    assert {k: r_on[k] for k in keep} == {k: r_off[k] for k in keep}
    # and every other measurement is the same: the samples of every stage but H7.1's
    for stage in ("0.5", "1"):
        a, b = d_on / "checkpoints" / f"stage{stage}", d_off / "checkpoints" / f"stage{stage}"
        files = sorted(p.relative_to(a) for p in a.rglob("*.np[yz]") if p.name != "h71.npy")
        assert files == sorted(p.relative_to(b) for p in b.rglob("*.np[yz]"))
        for f in files:
            assert (a / f).read_bytes() == (b / f).read_bytes(), f


# ----------------------------------------------------------------------------- design and queue file
def test_p6c_design_is_p6bs_design_with_new_seeds_and_no_h71():
    import p6_noah_joseph as p6
    d, b = p6.DESIGNS["p6c_small"], p6.DESIGNS["p6b_small"]
    assert d["seeds"] == SEEDS == p6.P6C_SEEDS and d["h71_stages"] == []
    used = set(p6.DESIGNS["small"]["seeds"]) | set(b["seeds"]) | {p6.FEASIBILITY["seed"]}
    assert used == set(range(1, 15)) and not used & set(d["seeds"])           # seeds 1-14 are used
    assert {k: v for k, v in d.items() if k not in ("seeds", "h71_stages")} == \
        {k: v for k, v in b.items() if k not in ("seeds", "h71_stages")}
    s_c, s_b = p6.settings(False, d), p6.settings(False, b)
    assert s_c["measure"].pop("h71_stages") == [] and s_b["measure"].pop("h71_stages") == [0.5, 1.0]
    assert s_c == s_b                                         # the same streams, training and other measurements
    runs = p6.plan(d)
    assert len(runs) == 80 and [r["seed"] for r in runs] == [s for s in SEEDS for _ in range(4)]   # seed by seed
    assert p6.smoke_design("p6c_small")["h71_stages"] == [] and p6.smoke_design("p6b_small") is p6.SMOKE_DESIGN
    assert p6.smoke_design("small") is p6.SMOKE_DESIGN and p6.SMOKE_DESIGN["h71_stages"] == [1.0]
    assert b["seeds"] == [4, 5, 7, 8, 9, 10, 11, 12, 13, 14] and b["h71_stages"] == [0.5, 1.0]   # P6b as it was


def test_p6c_queue_runs_the_seeds_one_after_another():
    jobs = json.loads((ROOT / "p6c_queue_small.json").read_text())
    seeds = jobs[:-1]
    assert [j["name"] for j in seeds] == [f"p6c_small_14m_s{k}" for k in SEEDS]
    for i, j in enumerate(seeds):
        a = j["args"]
        assert a[:3] == ["p6_noah_joseph.py", "--design", "p6c_small"]
        assert a[a.index("--seeds") + 1] == str(SEEDS[i]) and a[a.index("--sizes") + 1] == "14m"
        assert "--budget-hours" not in a and "--timing-json" not in a and "--raised" not in a
        assert a[a.index("--stop-file") + 1] == "results/p6c_small/STOP"
        assert j["out"] == f"results/p6c_small/14m_s{SEEDS[i]}" and j["gpu"] == 7.0
        assert j.get("after", []) == ([seeds[i - 1]["name"]] if i else [])
    an = jobs[-1]
    assert an["name"] == "p6c_small_analysis" and an["gpu"] == 0.0 and an["out"] == "results/p6c_small/analysis"
    assert an["args"] == ["p6_noah_joseph.py", "--design", "p6c_small", "--analyse", "--from", "results/p6c_small"]
    assert set(an["after"]) == {j["name"] for j in seeds}
    for f in ("p6_queue_small.json", "p6_queue_full.json", "p6_queue_small_raised.json", "p6b_queue_small.json"):
        assert not any("p6c" in j["out"] for j in json.loads((ROOT / f).read_text()))


def test_p6c_queue_dry_run(tmp_path):
    r = subprocess.run([PY, str(ROOT / "run_queue.py"), "--root", str(tmp_path), "--jobs-file",
                        str(ROOT / "p6c_queue_small.json"), "--dry-run"], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "RUN_QUEUE_FREE_GIB": "40"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "p6c_small_14m_s15" in r.stdout and "waits for p6c_small_14m_s33" in r.stdout


def test_p6c_refuses_the_budget_rule(tmp_path):
    import p6_noah_joseph as p6
    for extra in (["--budget-hours", "12"], ["--raised"]):
        with pytest.raises(SystemExit, match="P6c"):
            p6.main(["--design", "p6c_small", "--seeds", "15", "--out", str(tmp_path / "x"), *extra])


# ----------------------------------------------------------------------------- the family, gates and readings
def _df(effects, rng, seeds=tuple(range(15, 35)), noise=0.01, stages=(1.0,)):
    rows = []
    for stage in stages:
        for s in seeds:
            b = rng.normal(0, 0.05)
            for n, noah in ((0, "light"), (2, "zipf1.0")):
                for j, joseph in ((0, "H0.5"), (2, "H0.9")):
                    nn, jj = n / 2 - 0.5, j / 2 - 0.5
                    row = {"seed": s, "noah_score": n, "joseph_score": j, "noah": noah, "joseph": joseph,
                           "size": "14m", "stage": stage}
                    for col, (bN, bJ, bNJ) in effects.items():
                        row[col] = b + bN * nn + bJ * jj + bNJ * nn * jj + rng.normal(0, noise)
                    rows.append(row)
    return pd.DataFrame(rows)


GATES_OK = {"variant_Htok_excess": (0, 0.05, 0), "K": (0.05, 0, 0)}


def _family(effects, rng, n_perm=999, **kw):
    df = _df({**GATES_OK, "alpha_esd": (0, 0, 0), "alpha_esd_qkv": (0, 0, 0), "grad_main_dh": (0, 0, 0),
              "grad_main_dh_iaaft_mean": (0, 0, 0), **effects}, rng, **kw)
    df["grad_main_dh_excess"] = df["grad_main_dh"] - df["grad_main_dh_iaaft_mean"]
    fam = p6_stats.family_p6c(df, rng, n_perm)
    man = p6_stats.manipulation_checks(df, rng, n_perm)
    gates = p6_stats.p6c_gates(man)
    return p6_stats.p6c_readings(fam, gates), gates, df


def test_p6c_family_is_four_tests_with_holm_over_four(rng):
    fam, gates, df = _family({"alpha_esd": (0, -0.03, 0.04), "alpha_esd_qkv": (0, -0.08, 0),
                              "grad_main_dh": (-0.03, 0, 0), "grad_main_dh_iaaft_mean": (0.04, 0, 0)}, rng)
    assert list(fam["test"]) == ["C1", "C2", "C3", "C4"]
    assert list(zip(fam["outcome"], fam["term"], fam["predicted_sign"])) == [
        ("alpha_esd", "J", -1), ("alpha_esd_qkv", "J", -1), ("alpha_esd", "NJ", 1), ("grad_main_dh_excess", "N", -1)]
    assert gates["joseph"]["passed"] and gates["noah"]["passed"]
    assert gates["joseph"]["check"] == p6_stats.POSITIVE_CONTROL and gates["noah"]["check"] == "Noah -> K"
    assert np.allclose(fam["p_holm"], p6_stats.holm(fam["p_perm"].tolist()))
    assert (fam["n_perm"] == 999).all() and (fam["df"] == 80 - 23).all()      # 20 seed blocks: df 57
    assert fam["reading"].tolist() == [
        "confirmed: Joseph lowers alpha_ESD",
        "confirmed: Joseph lowers alpha_ESD of the attention input projections",
        "confirmed: the Zipf law weakens Joseph's effect on alpha_ESD",
        "confirmed: P6b's link replicates (Noah lowers the width of the gradient series beyond its IAAFT surrogates)"]
    c4 = fam[fam["test"] == "C4"].iloc[0]                                     # its two parts beside it
    assert c4["observed_estimate"] == pytest.approx(-0.03, abs=0.01)
    assert c4["iaaft_mean_estimate"] == pytest.approx(0.04, abs=0.01) and c4["iaaft_mean_p_perm"] < 0.05
    # the detectable effect is at the first threshold of four tests, not of P6's fourteen
    se = fam["se"].to_numpy()
    from scipy import stats as sst
    mde = (sst.t.ppf(1 - 0.05 / 4 / 2, 57) + sst.t.ppf(0.8, 57)) * se
    assert np.allclose(fam["mde_80"], mde)
    v = p6_stats.verdict_p6c(fam, gates)
    assert v.startswith("C1 confirmed: Joseph lowers alpha_ESD, Holm p ")
    assert "Joseph gate passed; Noah gate passed" in v


def test_p6c_readings_no_effect_opposite_and_gates(rng):
    fam, gates, _ = _family({"alpha_esd": (0, +0.03, 0)}, rng)               # C1 passes with the other sign
    r = dict(zip(fam["test"], fam["reading"]))
    assert r["C1"] == "effect opposite to the prediction"
    assert r["C2"].startswith("no effect detected (estimate ") and "95% CI" in r["C2"] and "80% power" in r["C2"]
    # the Joseph gate misses: C1 to C3 uninformative unless they pass; C4 read as usual (its gate is Noah's)
    fam, gates, _ = _family({"variant_Htok_excess": (0, 0, 0), "alpha_esd_qkv": (0, -0.08, 0)},
                            np.random.default_rng(3))
    assert not gates["joseph"]["passed"] and gates["noah"]["passed"]
    r = dict(zip(fam["test"], fam["reading"]))
    assert r["C1"].startswith("uninformative (the Joseph gate missed: estimate ") and "detectable" in r["C1"]
    assert r["C2"] == ("confirmed: Joseph lowers alpha_ESD of the attention input projections; Joseph gate missed")
    assert r["C4"].startswith("no effect detected (")
    assert list(fam["gate_passed"]) == [False, False, False, True]
    # the Noah gate misses: C4 uninformative
    fam, gates, _ = _family({"K": (0, 0, 0)}, np.random.default_rng(4))
    assert not gates["noah"]["passed"]
    assert dict(zip(fam["test"], fam["reading"]))["C4"].startswith("uninformative (the Noah gate missed: ")
    # a missing outcome keeps its place in Holm's four (p = 1) and reads "not measured"
    df = _df({**GATES_OK, "alpha_esd": (0, -0.05, 0)}, rng)
    fam = p6_stats.p6c_readings(p6_stats.family_p6c(df, rng, 999),
                                p6_stats.p6c_gates(p6_stats.manipulation_checks(df, rng, 999)))
    assert fam["reading"][1] == fam["reading"][3] == "not measured"
    assert fam["reading"][0].startswith("confirmed: ") and fam["reading"][2].startswith("no effect detected (")
    c1 = fam[fam["test"] == "C1"].iloc[0]
    assert c1["p_holm"] == pytest.approx(min(1.0, 4 * c1["p_perm"]))


def test_p6c_columns_from_the_result_files():
    names = [f"gpt_neox.layers.{i}.{t}.weight" for i in range(6) for t in
             ("attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")]
    rows = [{"layer": n, "alpha_esd": 2.0 + k / 10, "esd_xmin": 0.1 * k, "esd_n_tail": k, "esd_ks": 0.01 * k}
            for k, n in enumerate(names)]
    rows[0]["alpha_esd"] = float("nan")                                       # a failed fit is left out
    out = p6_stats.esd_columns({"result": {"alpha_esd": 3.0, "rows": rows}})
    qkv = [2.0 + k / 10 for k in range(4, 24, 4)]                             # the other five qkv matrices
    assert out["alpha_esd_qkv"] == pytest.approx(np.median(qkv)) and out["esd_n_qkv"] == 5
    assert out["alpha_esd_mlp_out"] == pytest.approx(np.median([2.0 + k / 10 for k in range(3, 24, 4)]))
    assert out["alpha_esd_layer2"] == pytest.approx(np.median([2.0 + k / 10 for k in range(8, 12)]))
    assert out["esd_n_tail_median"] == pytest.approx(11.5) and "alpha_esd_layer5" in out
    assert p6_stats.esd_columns(None)["esd_n_qkv"] == 0
    g = {"result": {"rows": [{"scales": "main", "delta_h": 0.03, "delta_h_iaaft_mean": 0.05},
                             {"scales": "sensitivity", "delta_h": 9, "delta_h_iaaft_mean": 9}]}}
    assert p6_stats.c4_part_columns(g) == {"grad_main_dh": 0.03, "grad_main_dh_iaaft_mean": 0.05}
    assert p6_stats.c4_part_columns(None) == {}


def test_p6c_exploratory_list(rng):
    df = _df({**GATES_OK, "alpha_esd": (0, -0.05, 0), "alpha_esd_qkv": (0, -0.1, 0), "alpha_esd_attn_out": (0, 0, 0),
              "alpha_esd_layer0": (0, 0, 0), "alpha_esd_layer10": (0, 0, 0), "loss_gap_oracle": (0, 0, 0),
              "esd_ks_median": (0, 0, 0), "grad_main_dh_excess": (0, 0, 0), "grad_main_dh": (0, 0, 0)},
             rng, seeds=tuple(range(15, 21)), stages=(0.01, 0.1, 0.5, 1.0))
    ex = p6_stats.exploratory_p6c(df, rng, 199)
    w = ex.groupby("what").size()
    assert w["simple effect of Joseph at Noah = light"] == 2 and w["simple effect of Joseph at Noah = zipf1.0"] == 2
    assert w["C1 at 50% of training"] == 1 and w["C3 at 1% of training"] == 1
    assert w["alpha_ESD by matrix type"] == 6 and w["alpha_ESD by layer"] == 6
    assert list(ex[ex["what"] == "alpha_ESD by layer"]["outcome"].unique()) == ["alpha_esd_layer0", "alpha_esd_layer10"]
    assert w["C2 with the loss gap to the oracle as a covariate"] == 1
    se = ex[ex["what"] == "simple effect of Joseph at Noah = light"]
    assert (se["df"] == 12 - 7).all() and (se["estimate"] < 0).all()        # 12 runs: intercept, 5 blocks, J
    assert (ex[ex["what"].str.startswith("C4")]["outcome"].unique().tolist()
            == ["grad_main_dh_excess", "grad_main_dh"])
    assert "h71" not in " ".join(ex["outcome"])


# ----------------------------------------------------------------------------- the analysis on smoke runs
@pytest.mark.slow
def test_p6c_analysis_end_to_end_and_p6s_analysis_untouched(tmp_path):
    runs = tmp_path / "runs_folder"
    for s in (1, 2):
        r = subprocess.run([PY, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--device", "cpu", "--design", "p6c_small",
                            "--seeds", str(s), "--out", str(runs / f"tiny_s{s}")], capture_output=True, text=True,
                           timeout=1800, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert not list(runs.glob("**/h71.npy")) and not list(runs.glob("**/stats/h71_*.json"))     # H7.1 off
    assert list(runs.glob("**/stats/esd.json"))

    def analyse(design, out):
        r = subprocess.run([PY, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--device", "cpu", "--design", design,
                            "--analyse", "--from", str(runs), "--out", str(out)], capture_output=True, text=True,
                           timeout=600, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return out
    c, b = analyse("p6c_small", tmp_path / "a_p6c"), analyse("p6b_small", tmp_path / "a_p6b")
    for f in ("checkpoints.csv", "family.csv", "manipulation_checks.csv", "secondary.csv", "exploratory.csv",
              "summary.md", "p6c_seeds.json"):
        assert (c / f).exists(), f
    assert json.loads((c / "p6c_seeds.json").read_text())["complete"] == [1, 2]
    fam = pd.read_csv(c / "family.csv")
    assert list(fam["test"]) == ["C1", "C2", "C3", "C4"] and fam["reading"].notna().all()
    assert (c / "summary.md").read_text().startswith("# P6c: a confirmatory test")
    cc, cb = pd.read_csv(c / "checkpoints.csv"), pd.read_csv(b / "checkpoints.csv")
    assert set(cb.columns) < set(cc.columns) and "alpha_esd_qkv" in cc.columns
    pd.testing.assert_frame_equal(cc[cb.columns], cb)                         # P6's columns: the same rows
    assert not (c / "p6b_seeds.json").exists() and not (b / "p6c_seeds.json").exists()
    assert (b / "summary.md").read_text().startswith("# P6b factorial")
    sec = pd.read_csv(c / "secondary.csv")
    assert not sec["outcome"].str.startswith("h71").any()


# ----------------------------------------------------------------------------- the waiter
WAITER = str(ROOT / "p6c_after_olmo.py")


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
    jobs = [{"name": "p6c_standin", "args": ["-c", f"open({str(end)!r}, 'w').write('ran')"],
             "out": "results/p6c_small/x", "gpu": 0.0}]
    jf = tmp_path / "p6c_queue_small.json"
    jf.write_text(json.dumps(jobs))
    return logs, jf, end


def waiter(tmp_path, jf, *extra):
    return subprocess.Popen([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--poll", "0.2", *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


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


def test_p6c_starts_only_after_olmo7b_has_ended_and_no_queue_runs(tmp_path):
    logs, jf, end = setup(tmp_path)
    olmo = standin(tmp_path, "p1_checkpoints.py", "--suite", "olmo7b", "--out", "results/p1_olmo7b")
    progress(logs, "2026-11-26T10:00:00Z  START  p1_olmo7b  total 1.00 h")
    w = queue = None
    try:
        w = waiter(tmp_path, jf)
        read_until(w, "waiting until p1_olmo7b has ended and no queue runs")
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None
        # OLMo-7B's process ends, but run_all.sh has not written its END line yet: still waiting
        kill(olmo)
        time.sleep(1.0)
        assert not end.exists() and w.poll() is None
        # its END line comes while another queue runs: still waiting
        queue = hold(logs / ".run_lane.queue.lock")
        progress(logs, "2026-11-27T20:00:00Z  END    p1_olmo7b  exit 0  elapsed 34.00 h  total 35.00 h")
        time.sleep(1.5)
        assert not end.exists() and w.poll() is None
        (logs / ".run_lane.queue.lock.release").write_text("")
        queue.wait(10)
        out, _ = w.communicate(timeout=120)
        assert w.returncode == 0, out
        assert end.read_text() == "ran" and (logs / "p6c_standin.done").exists()
        assert "OLMo-7B has ended and no queue runs" in out and "the P6c queue ended with exit 0" in out
    finally:
        kill(w, olmo, queue)


def test_p6c_waiter_refusals_check_and_stop(tmp_path):
    logs, jf, end = setup(tmp_path)
    r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2 and "neither running or waiting nor ended" in r.stdout
    r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(tmp_path / "none.json")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "is missing" in r.stdout
    olmo = standin(tmp_path, "p1_checkpoints.py", "--suite", "olmo7b")
    progress(logs, "2026-11-26T10:00:00Z  START  p1_olmo7b  total 1.00 h")
    w = None
    try:
        before = sorted(p.name for p in logs.iterdir())
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "p1_olmo7b: running or waiting" in r.stdout and "a new waiter would wait, then start P6c" in r.stdout
        assert sorted(p.name for p in logs.iterdir()) == before          # --check changes nothing
        w = waiter(tmp_path, jf)
        read_until(w, "waiting until")
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "another p6c_after_olmo.py runs" in r.stdout
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf), "--check"],
                           capture_output=True, text=True, timeout=60)
        assert "a waiter runs" in r.stdout
        w.terminate()
        out, _ = w.communicate(timeout=30)
        assert w.returncode == 130 and "stopped while waiting; nothing started" in out and not end.exists()
        (logs / "p6c_standin.done").write_text("x")
        r = subprocess.run([PY, WAITER, "--root", str(tmp_path), "--jobs-file", str(jf)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "every P6c job is done already" in r.stdout
    finally:
        kill(w, olmo)
