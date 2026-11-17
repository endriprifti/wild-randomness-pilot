"""H_link (exploratory, 2026-11-17): the synthetic series, the test (main row on 8 ... n/10, sensitivity row on
10 ... n/10, same surrogates), the two confound controls (within-document shuffles for P1, rows on the projections
minus a moving median for P4), and h_link.py's search for saved series (only finished ones; the folders are only
read; a unit is computed again when its file changed)."""
import json

import numpy as np
import pandas as pd
import pytest

import h_link
import p4_clipping as p4
from common import hlink, hurst


def test_synthetic_series():
    rng = np.random.default_rng(0)
    for kind in hlink.SYNTHETIC:
        x = hlink.synthetic(kind, 3000, rng)
        assert x.shape == (3000,) and np.isfinite(x).all(), kind
    assert (hlink.synthetic("cascade", 3000, rng) > 0).all()
    H = hurst.hurst_dfa(hlink.synthetic("fgn", 2 ** 14, rng), hurst.log_scales(2 ** 14, 8))["H"]
    assert abs(H - 0.75) < 0.07
    m = hlink.synthetic("mrw", 2 ** 14, rng)            # Gaussian times a log-normal volatility: heavier than Gaussian
    assert ((m - m.mean()) ** 4).mean() / m.var() ** 2 > 3.2
    with pytest.raises(ValueError):
        hlink.synthetic("other", 10, rng)


def test_p_value():
    sur = np.array([0.1, 0.2, 0.3, np.nan])
    assert hlink._p_greater(0.5, sur) == pytest.approx(1 / 4)
    assert hlink._p_greater(0.0, sur) == pytest.approx(1.0)
    assert hlink._p_greater(0.2, sur) == pytest.approx(3 / 4)       # ties count against the observed value
    assert np.isnan(hlink._p_greater(np.nan, sur))


@pytest.mark.slow
def test_rows_main_and_sensitivity():
    rng = np.random.default_rng(1)
    x = hlink.synthetic("mrw", 9064, rng)
    main, sens = hlink.hlink_rows(x, rng, n_surr=39)
    assert (main["scales"], main["s_min"], sens["scales"], sens["s_min"]) == ("main", 8, "sensitivity", 10)
    assert main["s_max"] == sens["s_max"] == 906
    # the shortest P4 series: two decades at 8 ... n/10, not at 10 ... n/10 (flagged, and no pass there)
    assert main["decades_dfa"] >= 2 and main["two_decades"] and main["decades_flag"] == ""
    assert sens["decades_dfa"] == pytest.approx(np.log10(906 / 10)) and not sens["two_decades"]
    assert sens["decades_flag"].startswith("below 2 decades") and not sens["pass"] and not sens["memory_pass"]
    # the wavelet estimator has no s_min: same value and same surrogates in both rows
    for k in ("H_abs_wav", "p_shuf_wav", "p_phase_wav", "decades_wav"):
        assert main[k] == sens[k]
    # volatility clustering of the MRW: the magnitudes' H far above their shuffled value
    assert main["H_abs_dfa"] > main["H_abs_dfa_shuf_q95"] and main["memory_pass"]
    assert main["p_shuf_dfa"] == main["p_shuf_wav"] == pytest.approx(1 / 40)
    assert sens["pass_ignoring_decades"] == (sens["p_shuf_dfa"] < 0.05 and sens["p_shuf_wav"] < 0.05
                                             and sens["p_iaaft"] < 0.05)
    assert main["pass"] == (main["memory_pass"] and main["width_pass"])


def test_iid_magnitudes_near_half():
    rng = np.random.default_rng(2)
    r = hlink.hlink_test(rng.normal(size=4096), rng, n_surr=9)
    assert r["scales"] == "main"
    assert abs(r["H_abs_dfa_shuf_mean"] - 0.5) < 0.08 and abs(r["H_abs_dfa"] - 0.5) < 0.1
    assert abs(r["delta_h"]) < 0.15 and abs(r["delta_h_iaaft_mean"]) < 0.15


def test_matrix_is_the_mean_over_columns():
    rng = np.random.default_rng(3)
    X = np.column_stack([hlink.synthetic("fgn", 1500, rng), rng.standard_t(3, size=1500)])
    X[5, 1] = np.nan                                           # a row with a non-finite value is dropped
    r = hlink.hlink_test(X, rng, n_surr=3)
    Xf = np.delete(X, 5, axis=0)
    sc = hurst.log_scales(len(Xf), 8)
    assert (r["n"], r["n_series"]) == (1499, 2)
    assert r["H_abs_dfa"] == pytest.approx(np.mean([hurst.hurst_dfa(hlink.magnitudes(Xf[:, j]), sc)["H"]
                                                    for j in range(2)]))
    assert r["delta_h"] == pytest.approx(np.mean([hurst.delta_h(Xf[:, j], 0.5, 4.0, sc)["delta_h"] for j in range(2)]))


# ----------------------------------------------------------------------------- h_link.py on saved series
def _p1_folder(root, rng):
    f = root / "p1_test"
    for name, done in (("EleutherAI__pythia-14m@step1000", True), ("EleutherAI__pythia-14m@step2000", True),
                       ("EleutherAI__pythia-70m@step1000", False)):          # 70m: GPU pass not done (no row.json)
        (f / "checkpoints").mkdir(parents=True, exist_ok=True)
        d = f / "checkpoints" / "grid" / name
        (d / "samples").mkdir(parents=True)
        np.save(d / "samples" / "htok.npy", rng.gamma(2.0, size=600).astype(np.float32))
        if done:
            row = {"suite": "pythia", "variant": "main", "size": name.split("-")[1].split("@")[0], "seed": 0,
                   "repo": "EleutherAI/" + name.split("__")[1].split("@")[0], "step": int(name.split("step")[1]),
                   "revision": name.split("@")[1], "pass": "grid", "source": "hub"}
            (d / "row.json").write_text(json.dumps({"row": row, "tasks": {}}))
    # 3 windows of 201 tokens: 3 documents of 200 surprisals (as 16 windows of 2,048 give 16 x 2,047)
    (f / "run.json").write_text(json.dumps({"config": {"P1": {"surprisal_docs": 3, "surprisal_len": 201}}}))
    return f


def _p4_folder(root, rng):
    f = root / "p4_test"
    (f / "runs").mkdir(parents=True)
    (f / "hgrad_series").mkdir()
    (f / "runs" / "settings.json").write_text("{}")
    rows = [dict(model="resnet18", seed=0, rule="fixed", p_clip_target=0.0, matched=False, warmup_steps=40, steps=640,
                 diverged=False),
            dict(model="resnet18", seed=0, rule="autoclip", p_clip_target=None, matched=True, warmup_steps=40,
                 steps=640, diverged=True),
            dict(model="resnet18", seed=1, rule="fixed", p_clip_target=0.1, matched=False, warmup_steps=40, steps=640)]
    tags = [p4.run_tag(r["model"], r["seed"], r["rule"], r["p_clip_target"], r["matched"]) for r in rows]
    for r, t in zip(rows[:2], tags[:2]):                       # two finished runs
        (f / "runs" / f"{t}.json").write_text(json.dumps({"row": r}))
        np.save(f / "hgrad_series" / f"{t}.npy", rng.normal(size=(640, 3)))
    np.save(f / "hgrad_series" / f"{tags[2]}.npy", rng.normal(size=(640, 3)))   # run 3: series, no row yet
    np.save(f / "hgrad_series" / "transformer_s0_fixed0.0.npy", rng.normal(size=(640, 3)))   # no row either
    return f, tags


def test_finds_only_finished_series(tmp_path):
    rng = np.random.default_rng(4)
    f1 = _p1_folder(tmp_path, rng)
    s1 = h_link.p1_series(f1)
    assert [s["key"] for s in s1] == ["p1_test/grid/EleutherAI__pythia-14m@step1000",
                                      "p1_test/grid/EleutherAI__pythia-14m@step2000"]
    assert s1[0]["p1_pass"] == "grid" and s1[0]["group"] == "14m" and s1[0]["source"] == "p1"
    assert (s1[0]["n_docs"], s1[0]["doc_len"]) == (3, 200)
    assert h_link.p1_docs(tmp_path) == (16, 2047)                # no run.json: the full settings
    f4, tags = _p4_folder(tmp_path, rng)
    s4 = h_link.p4_series(f4)
    assert sorted(s["tag"] for s in s4) == sorted(tags[:2]) and all(s["warm"] == 40 for s in s4)
    assert all(h_link.run_tag(m, s, r, p, mt) == p4.run_tag(m, s, r, p, mt)
               for m, s, r, p, mt in [("transformer", 3, "fixed", 0.25, False),
                                      ("resnet18", 0, "autoclip", None, True)])
    x = h_link.load_series(s4[0])
    assert x.shape == (600, 3)                                 # after warm-up, as P4's H_grad
    # discovery: P1 folders with checkpoints (not the *_recomputed copies), P4 folders with hgrad_series
    (tmp_path / "p1_test_recomputed" / "checkpoints").mkdir(parents=True)
    (tmp_path / "p4_transformer").mkdir()                      # queued: no series yet
    p1s, p4s = h_link.discover(tmp_path, smoke=False)
    assert [p.name for p in p1s] == ["p1_test"] and [p.name for p in p4s] == ["p4_test"]


def test_main_resume_and_changed_file(tmp_path, capsys):
    rng = np.random.default_rng(5)
    f1 = _p1_folder(tmp_path / "res", rng)
    f4, tags = _p4_folder(tmp_path / "res", rng)
    before = sorted((p.relative_to(tmp_path), p.stat().st_mtime_ns) for p in (tmp_path / "res").rglob("*"))
    out = tmp_path / "out"
    argv = ["--results", str(tmp_path / "res"), "--out", str(out), "--workers", "0", "--n-surr", "3", "--nice", "0"]
    assert h_link.main(argv) == 0
    assert sorted((p.relative_to(tmp_path), p.stat().st_mtime_ns) for p in (tmp_path / "res").rglob("*")) == before
    p1 = pd.read_csv(out / "p1_tokens.csv")
    p4r = pd.read_csv(out / "p4_grad.csv")
    assert len(p1) == 6 and len(p4r) == 8                       # two series each: P1 3 rows, P4 4 rows each
    assert sorted(p1["scales"].unique()) == ["main", "sensitivity", "standardized"]
    assert sorted(p4r["scales"].unique()) == ["detrended", "main", "rescaled", "sensitivity"]
    assert set(p1["eos_status"]) == {"unverified"} and "p_piece_wav" not in p1      # run.json lacks the data config
    assert set(p1["doc_len"]) == {200} and p1["p_doc_dfa"].between(0, 1).all() and "beyond_documents" in p1
    assert "p_doc_dfa" not in p4r and set(p4r.loc[p4r["scales"] == "detrended", "detrend_width"]) == {201}
    assert set(p4r["n_series"]) == {3} and set(p4r["n"]) == {600}
    assert set(p4r["diverged"].astype(str)) == {"True", "False"}
    summ = pd.read_csv(out / "summary.csv")
    assert summ.loc[summ["source"] == "p4", "series"].tolist() == [1, 1, 1, 1]    # the diverged run is left out
    assert set(zip(summ["source"], summ["group"], summ["scales"])) == {
        ("p1", "14m", "main"), ("p1", "14m", "sensitivity"), ("p1", "14m", "standardized"), ("p4", "resnet18", "main"),
        ("p4", "resnet18", "sensitivity"), ("p4", "resnet18", "detrended"), ("p4", "resnet18", "rescaled")}
    assert summ.loc[summ["source"] == "p1", "beyond_documents_rate"].notna().all()
    saved = sorted((out / "data" / "units").glob("*.json"))
    stamps = {p.name: p.stat().st_mtime_ns for p in saved}
    capsys.readouterr()
    assert h_link.main(argv) == 0                               # resume: nothing computed again
    assert "4 saved before" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in saved} == stamps
    # a series file that changed (e.g. a rerun in the same folder) is computed again
    np.save(f4 / "hgrad_series" / f"{tags[0]}.npy", rng.normal(size=(700, 3)))
    assert h_link.main(argv) == 0
    p4r = pd.read_csv(out / "p4_grad.csv")
    assert sorted(p4r.loc[p4r["tag"] == tags[0], "n"]) == [660, 660, 660, 660]


def test_short_series_skipped(tmp_path):
    rng = np.random.default_rng(6)
    f4, tags = _p4_folder(tmp_path, rng)
    np.save(f4 / "hgrad_series" / f"{tags[0]}.npy", rng.normal(size=(80, 3)))
    spec = [s for s in h_link.p4_series(f4) if s["tag"] == tags[0]][0]
    res = h_link.run_unit(spec, dict(hlink.TEST_DEFAULTS, n_surr=3))
    assert res["skipped"].startswith("series of 80 values") and res["rows"] == []
    df = h_link.table([res])
    assert len(df) == 1 and "pass" not in df


@pytest.mark.slow
def test_calibration_smoke(tmp_path):
    assert h_link.main(["--calibrate", "--smoke", "--reps", "1", "--workers", "0", "--nice", "0",
                        "--out", str(tmp_path)]) == 0
    cal = pd.read_csv(tmp_path / "calibration.csv")
    assert len(cal) == 8 and set(cal["kind"]) == set(hlink.SYNTHETIC) and set(cal["n"]) == {2048}
    summ = pd.read_csv(tmp_path / "calibration_summary.csv")
    assert set(summ["scales"]) == {"main", "sensitivity"} and set(summ["series"]) == {1}
    ctl = pd.read_csv(tmp_path / "calibration_controls.csv")        # 2 document kinds x 2 rows, 3 drift kinds x 3
    assert len(ctl) == 13 and set(ctl["kind"]) == set(hlink.CONTROL_N)
    assert set(ctl.loc[ctl["kind"].str.endswith("_drift"), "scales"]) == {"main", "sensitivity", "detrended"}
    assert ctl.loc[ctl["kind"].str.startswith(("docs", "mrw_docs")), "p_doc_wav"].notna().all()
    csumm = pd.read_csv(tmp_path / "calibration_controls_summary.csv")
    assert {"beyond_documents_rate", "pass_rate"} <= set(csumm.columns)


# ----------------------------------------------------------------------------- the confound controls
def test_within_document_shuffle():
    rng = np.random.default_rng(7)
    v = np.arange(23.0)
    w = hlink.within_doc_shuffle(v, 5, rng)
    for k in range(4):                                         # each document keeps its values, in its place
        assert sorted(w[5 * k:5 * k + 5]) == list(v[5 * k:5 * k + 5])
    assert sorted(w[20:]) == [20.0, 21.0, 22.0] and not np.array_equal(w, v)


def test_detrend_removes_a_slow_drift_and_keeps_the_noise():
    rng = np.random.default_rng(8)
    t = np.arange(4000.0)
    d = 5 * np.exp(-t / 500) + np.cos(2 * np.pi * t / 4000)
    assert np.abs(hlink.detrend(d)[300:-300]).max() < 1e-2      # a median of a locally monotone window: its centre
    e = rng.standard_t(3, size=(4000, 2))
    r = hlink.detrend(d[:, None] + e)
    assert r.shape == (4000, 2)
    assert np.corrcoef(r[300:-300, 0], e[300:-300, 0])[0, 1] > 0.99
    x, how = hlink.synthetic_control("iid_drift", rng)
    assert how == {"detrend": True} and len(x) == hlink.CONTROL_N["iid_drift"]
    dr = x - hlink.detrend(x)                                  # what the median took out: about the drift
    assert 0.6 < dr.std() / x.std() < 0.8                      # drift sd = noise sd: drift share 1/sqrt(2)


@pytest.mark.slow
def test_documents_control():
    rng = np.random.default_rng(9)
    x, how = hlink.synthetic_control("docs_shift", rng)
    assert how == {"doc_len": 2047} and len(x) == 16 * 2047
    means = x.reshape(16, 2047).mean(axis=1)
    assert means.std() > 0.25                                  # document levels differ (sd 0.5 in theory)
    main = hlink.hlink_rows(x, rng, n_surr=19, doc_len=2047)[0]
    # shifted documents fool the shuffled baseline, not the within-document one
    assert main["H_abs_dfa"] > main["H_abs_dfa_shuf_q95"]
    assert abs(main["H_abs_dfa"] - main["H_abs_dfa_doc_mean"]) < 0.05
    m, _ = hlink.synthetic_control("mrw_docs", rng)
    r = hlink.hlink_rows(m, rng, n_surr=19, doc_len=2047)[0]
    assert r["H_abs_dfa"] > r["H_abs_dfa_doc_q95"] and r["H_abs_wav"] > r["H_abs_wav_doc_q95"]
    assert r["beyond_documents"] == (r["memory_beyond_documents"] and r["width_beyond_documents"])


def test_documents_control_leaves_the_main_columns():
    x = hlink.synthetic_control("docs_shift", np.random.default_rng(10))[0][:4094]
    a = hlink.hlink_rows(x, np.random.default_rng(11), n_surr=5)
    b = hlink.hlink_rows(x, np.random.default_rng(11), n_surr=5, doc_len=2047)
    for ra, rb in zip(a, b):
        assert {k: rb[k] for k in ra} == ra and "p_doc_width" in rb and "p_doc_width" not in ra


def test_p4_unit_has_detrended_rows(tmp_path):
    rng = np.random.default_rng(12)
    f4, tags = _p4_folder(tmp_path, rng)
    spec = [s for s in h_link.p4_series(f4) if s["tag"] == tags[0]][0]
    res = h_link.run_unit(spec, dict(hlink.TEST_DEFAULTS, n_surr=3))
    assert [r["scales"] for r in res["rows"]] == ["main", "sensitivity", "detrended", "rescaled"]
    assert res["rows"][2]["detrend_width"] == hlink.DETREND_WIDTH and res["rows"][2]["s_min"] == 8
