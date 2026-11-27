"""The surrogate decomposition (step 1, results brief 11): the brief's computation and analysis unchanged, paths
relative to results/, resume, the readings of its procedure, and the comparison with the brief's 242 runs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import surrogate_decomposition as sd
from common import hurst
from common import units as units_mod

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REF = ROOT / "reference" / "surrogate_decomposition_brief11.csv"


def brief_run(spec, X, n_surr):
    """decomp.run of the brief (2026-11-27), as it was run, on an array instead of a path."""
    warm = spec["warm"]
    X = X[warm:] if len(X) > warm + 50 else X
    X = X[np.isfinite(X).all(axis=1)]
    n, d = X.shape
    cols = [X[:, j] for j in range(d)]
    scales = hurst.log_scales(n, 8)
    rng = units_mod.rng_for(units_mod.seed_for(config.SEED, "step1_decomposition", spec["key"]))
    width = lambda cs: float(np.mean([hurst.delta_h(v, 0.5, 4.0, scales)["delta_h"] for v in cs]))
    obs = width(cols)
    do_iaaft = spec["key"] in sd.CHECK
    sur = {"shuf": [], "phase": []} | ({"iaaft": []} if do_iaaft else {})
    for _ in range(n_surr):
        sur["shuf"].append(width([hurst.shuffle_surrogate(v, rng) for v in cols]))
        sur["phase"].append(width([hurst.phase_surrogate(v, rng) for v in cols]))
        if do_iaaft:
            sur["iaaft"].append(width([hurst.iaaft_surrogate(v, rng) for v in cols]))
    res = {"dh": obs}
    for k, v in sur.items():
        v = np.asarray(v)
        res[f"{k}_mean"], res[f"{k}_q95"] = float(v.mean()), float(np.quantile(v, 0.95))
    return res


def fake_results(tmp_path, rng, n=700):
    """A results/ tree with two P6b seeds (four corners each) and a P4 transformer folder with H_link's CSV."""
    res = tmp_path / "results"
    for s in (4, 5):
        for noah in ("light", "zipf1.0"):
            for joseph in ("H0.5", "H0.9"):
                run = f"14m_{noah}_{joseph}_s{s}"
                d = res / "p6b_small" / f"14m_s{s}" / "runs" / run
                (d / "hlink" / "stats").mkdir(parents=True)
                X = rng.standard_t(3 if noah == "zipf1.0" else 30, size=(n, 3))
                np.save(d / "hgrad.npy", X)
                main = {"scales": "main", "delta_h": 0.1, "delta_h_iaaft_mean": 0.09, "delta_h_iaaft_q95": 0.12,
                        "p_iaaft": 0.4}
                (d / "hlink" / "stats" / "hlink_grad.json").write_text(json.dumps(
                    {"result": {"folder": run, "warm": 20, "rows": [main, {**main, "scales": "sensitivity"}]}}))
    p4 = res / "p4_transformer" / "hgrad_series"
    p4.mkdir(parents=True)
    rows = []
    for s in (0, 1):
        for p in (0.0, 0.01, 0.1, 0.5):
            tag = f"transformer_s{s}_fixed{p}"
            np.save(p4 / f"{tag}.npy", rng.normal(size=(n, 3)))
            for scales in ("main", "detrended"):
                rows.append({"folder": "p4_transformer", "tag": tag, "scales": scales, "warm": 30, "seed": s,
                             "rule": "fixed", "p_clip_target": p, "matched": False, "delta_h": 0.05,
                             "delta_h_iaaft_mean": 0.04, "delta_h_iaaft_q95": 0.06, "p_iaaft": 0.3})
    rows.append({**rows[0], "folder": "p4_resnet18", "tag": "resnet18_s0_fixed0.0"})     # another folder's row
    (res / "h_link").mkdir()
    pd.DataFrame(rows).to_csv(res / "h_link" / "p4_grad.csv", index=False)
    return res


def test_specs_paths_are_relative_to_results(tmp_path, rng):
    res = fake_results(tmp_path, rng)
    S = sd.specs(res, ["P6b", "P4T", "P6c", "P7"])           # P6c and P7 not run yet: no runs, no error
    assert len(S) == 16
    assert all(not s["path"].startswith("/") and (res / s["path"]).exists() for s in S)
    p6b = [s for s in S if s["design"] == "P6b"]
    assert {s["key"] for s in p6b} == {f"P6b/14m_{a}_{b}_s{k}" for a in ("light", "zipf1.0")
                                       for b in ("H0.5", "H0.9") for k in (4, 5)}
    s = [x for x in p6b if x["key"] == "P6b/14m_zipf1.0_H0.9_s5"][0]
    assert (s["noah"], s["joseph"], s["seed"], s["warm"]) == ("zipf1.0", "H0.9", 5, 20)
    assert s["path"] == "p6b_small/14m_s5/runs/14m_zipf1.0_H0.9_s5/hgrad.npy"
    p4 = [x for x in S if x["design"] == "P4T"]
    assert len(p4) == 8 and all(x["path"].startswith("p4_transformer/hgrad_series/") for x in p4)
    with pytest.raises(SystemExit):
        sd.specs(res, ["P9"])


def test_compute_is_the_briefs_computation(tmp_path, rng):
    res = fake_results(tmp_path, rng)
    S = {s["key"]: s for s in sd.specs(res, ["P6b", "P4T"])}
    for key in ("P6b/14m_light_H0.5_s4", "P4T/transformer_s0_fixed0.1", "P4T/transformer_s1_fixed0.5"):
        spec = S[key]
        X = np.load(res / spec["path"])
        X[spec["warm"] + 5, 1] = np.nan                      # a non-finite row after the warm-up is dropped
        np.save(res / spec["path"], X)
        got = sd.compute(spec, res, n_surr=7)
        want = brief_run(spec, X, 7)
        for k, v in want.items():
            assert got[k] == v, (key, k)
        assert ("iaaft_mean" in got) == (key in sd.CHECK)
        assert got["n"] == len(X) - spec["warm"] - 1


def test_main_resume_tables_and_compare(tmp_path, rng):
    res = fake_results(tmp_path, rng)
    out = tmp_path / "dec"
    args = ["--results", str(res), "--out", str(out), "--designs", "P6b,P4T", "--workers", "0", "--n-surr", "5"]
    assert sd.main(args) == 0
    assert len(list((out / "units").glob("*.json"))) == 16
    for f in ("units.csv", "P6b_units.csv", "P4T_units.csv", "noah_effects.csv", "cells.csv", "arms.csv",
              "readings.csv", "iaaft_check.csv", "report.txt"):
        assert (out / f).exists(), f
    u = pd.read_csv(out / "units.csv")
    assert set(u["path"].str[:1]) != {"/"} and (u["iaaft_file"] == u["ref_iaaft"]).all()
    eff = pd.read_csv(out / "noah_effects.csv")
    assert list(eff["quantity"]) == sd.QUANTITIES and set(eff["design"]) == {"P6b"}
    # the heavier marginal (zipf: t with 3 degrees of freedom) shows in the kurtosis
    assert eff.set_index("quantity").loc["kurt", "N"] > 0
    rd = pd.read_csv(out / "readings.csv")
    assert set(rd["design"]) == {"P6b", "P4T"}
    # resume: nothing computed again
    stamp = {f.name: f.stat().st_mtime_ns for f in (out / "units").glob("*.json")}
    assert sd.main(args) == 0
    assert {f.name: f.stat().st_mtime_ns for f in (out / "units").glob("*.json")} == stamp
    # compare with a reference built from the saved runs: no difference
    ref = tmp_path / "ref.csv"
    pd.read_csv(out / "units.csv").to_csv(ref, index=False)
    assert sd.main(args + ["--analyse-only", "--compare", str(ref)]) == 0
    c = pd.read_csv(out / "compare.csv")
    assert (c["max_abs_diff"].fillna(0) == 0).all()


def test_the_analysis_of_the_briefs_242_runs():
    """analyse() on brief 11's table gives the brief's numbers (the text analyse.py printed, then the readings)."""
    U = pd.read_csv(REF)
    assert len(U) == 242 and U["design"].value_counts().to_dict() == {"P4R": 95, "P4T": 95, "P6b": 40, "P6": 12}
    assert not U["path"].str.startswith("/").any()
    T, txt = sd.analyse(U)
    e = T["noah_effects"].set_index(["design", "quantity"])
    assert e.loc[("P6", "iaaft_file"), "N"] == pytest.approx(0.0400, abs=5e-5)
    assert e.loc[("P6b", "shuf_mean"), "N"] == pytest.approx(0.0552, abs=5e-5)
    assert e.loc[("P6b", "phase_mean"), "N"] == pytest.approx(-0.0076, abs=5e-5)
    assert "fixed 1-50% (30 runs): dh > q95 of shuffled 30, phase 8, IAAFT 5" in txt          # P4T
    assert "all runs: shuffled 95, phase 24, IAAFT 16 of 95" in txt
    assert len(T["iaaft_check"]) == 9 and (T["iaaft_check"]["z"].abs() < 3).all()
    r = T["readings"].set_index("design")["reading"]
    assert r["P6"].startswith("the baseline effect sits in the marginal")
    assert r["P6b"].startswith("the baseline effect sits in the marginal")
    assert r["P4R"].startswith("width beyond the marginal and beyond the spectrum")
    # the guard (2026-11-27): the transformer's width exceeds the IAAFT quantile in 5 of 30 runs, so no reading
    assert r["P4T"].startswith("no reading: the width exceeds the IAAFT quantile in only 5 of 30 runs (dh > q95: "
                               "shuffled 30, phase-randomized 8, IAAFT 5); medians by arm, dh / phase-randomized "
                               "mean: fixed 1% 0.0299 / 0.0277; fixed 1% matched 0.0291 / 0.0272;")
    a = T["arms"].set_index(["design", "arm", "matched"])            # the same medians as the table
    assert round(a.loc[("P4T", "fixed 1%", False), "dh"], 4) == 0.0299
    assert round(a.loc[("P4T", "fixed 1%", False), "phase"], 4) == 0.0277


def test_readings():
    assert sd.p6_reading({"iaaft_file": (0.04, 0.01, 0.07), "shuf_mean": (0.05, 0.01, 0.08),
                          "phase_mean": (0.001, -0.002, 0.004)}).startswith("the baseline effect sits in the marginal")
    assert sd.p6_reading({"iaaft_file": (0.04, 0.01, 0.07), "shuf_mean": (0.01, -0.01, 0.03),
                          "phase_mean": (0.03, 0.01, 0.05)}).startswith("the baseline effect sits in the spectrum")
    assert sd.p6_reading({"iaaft_file": (0.04, 0.01, 0.07), "shuf_mean": (0.05, 0.01, 0.08),
                          "phase_mean": (0.03, 0.01, 0.05)}).startswith("not assigned")
    def sel(n_shuf, n_phase, n_iaaft, n=30):
        k = np.arange(n)
        arm = np.array(["fixed 1%", "fixed 10%", "fixed 50%"])[k % 3]
        return pd.DataFrame({"arm": arm, "matched": (k // 3) % 2 == 1, "dh": 0.03 + 0.001 * (k % 3),
                             "phase_mean": 0.028, "gt_shuf": k < n_shuf, "gt_phase": k < n_phase,
                             "gt_iaaft": k < n_iaaft})
    assert sd.p4_reading(sel(30, 30, 30)).startswith("width beyond")
    assert sd.p4_reading(sel(30, 30, 15)).startswith("width beyond")                 # exactly half: the rule applies
    assert "kept by the spectrum" in sd.p4_reading(sel(30, 12, 15))
    assert "the marginal (shuffled) and the spectrum" in sd.p4_reading(sel(14, 12, 15))
    r = sd.p4_reading(sel(30, 8, 14))                                                # fewer than half: no reading
    assert r.startswith("no reading: the width exceeds the IAAFT quantile in only 14 of 30 runs")
    assert "fixed 1% 0.0300 / 0.0280; fixed 1% matched 0.0300 / 0.0280; fixed 10% 0.0310 / 0.0280" in r
    assert sd.p4_reading(sel(0, 0, 0, n=0)).startswith("no reading: no runs")


def test_list_and_queue_job(tmp_path, rng):
    res = fake_results(tmp_path, rng)
    r = subprocess.run([PY, str(ROOT / "surrogate_decomposition.py"), "--results", str(res), "--designs",
                        "P6b,P4T,P7", "--list"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "P6b: 8 runs found, 8 series present, 0 saved" in r.stdout and "P7: 0 runs found" in r.stdout
    job = [j for j in json.loads((ROOT / "p7_queue.json").read_text()) if j["name"] == "p7_decomposition"][0]
    assert job["args"][:3] == ["surrogate_decomposition.py", "--designs", "P7"]
