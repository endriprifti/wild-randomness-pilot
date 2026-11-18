"""P6 (2026-11-18): the synthetic streams and their checks, the training loss and surprisal split, the pre-stated
factorial tests and the queue files."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import p6_stats
import p6_streams as ps
from common import hurst, lm

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(copy.deepcopy(ps.P6), vocab=512, K=4, R=128, block=16, p_bar=1 / 32, seq_len=128)


# ----------------------------------------------------------------------------- rank laws, floors, fGn
def test_light_control_is_entropy_matched():
    tau = ps.light_tau(1.0, ps.P6["R"])
    assert abs(tau - 205.87) < 0.01
    assert abs(ps.entropy(ps.exp_law(tau, ps.P6["R"])) - ps.entropy(ps.zipf_law(1.0, ps.P6["R"]))) < 1e-9
    f_l, f_z = ps.floors("light"), ps.floors("zipf1.0")
    assert abs(f_l["oracle_floor"] - f_z["oracle_floor"]) < 1e-9
    assert abs(f_z["topic_information"] - 1.1285) < 1e-3
    assert ps.P6["R"] * ps.P6["K"] == ps.P6["vocab"] == 50304


def test_fgn_matches_hurst_and_survives_long_series():
    for H in (0.5, 0.7, 0.9):
        a = ps.fgn(4096, H, np.random.default_rng(3))
        b = hurst.fgn(4096, H, np.random.default_rng(3))
        assert np.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        hurst.fgn(4194304, 0.9, np.random.default_rng(0))      # the cancellation this module avoids
    g = ps.fgn(4194304, 0.9, np.random.default_rng(0))
    assert abs(hurst.hurst_wavelet(g)["H"] - 0.9) < 0.03


# ----------------------------------------------------------------------------- streams
def test_stream_tokens_and_common_random_numbers():
    a = ps.Stream("zipf1.0", "H0.9", 1, "train", 64, SMALL)
    b = ps.Stream("light", "H0.9", 1, "train", 64, SMALL)
    c = ps.Stream("zipf1.0", "H0.5", 1, "train", 64, SMALL)
    d = ps.Stream("zipf1.0", "H0.9", 2, "train", 64, SMALL)
    X = a.windows(range(64))
    assert X.min() >= 0 and X.max() < SMALL["vocab"] and X.shape == (64, 128)
    # the Noah level changes only the ranks: same switches, topics and variants
    assert np.array_equal(a.switch_pos, b.switch_pos)
    ra, va, ta = a.window_parts(5)
    rb, vb, tb = b.window_parts(5)
    assert np.array_equal(va, vb) and np.array_equal(ta, tb) and not np.array_equal(ra, rb)
    # the Joseph level changes only the latent process: same ranks
    rc, _, _ = c.window_parts(5)
    assert np.array_equal(ra, rc)
    # another seed: other numbers
    assert not np.array_equal(a.window(5), d.window(5))
    # random access: a window is the same read alone, in a fresh object, or in a batch
    a2 = ps.Stream("zipf1.0", "H0.9", 1, "train", 64, SMALL)
    assert np.array_equal(a2.window(17), X[17])
    # tokens decode to (rank, variant)
    r, v, t = a.window_parts(9)
    assert np.array_equal(X[9], (r - 1) * SMALL["K"] + v)
    # topics change only at switches, always to another topic
    tt = a.topics(0, a.n_tokens)
    ch = np.flatnonzero(np.diff(tt) != 0) + 1
    assert np.array_equal(ch, a.switch_pos)


def test_switch_rate_is_fixed_per_stream():
    for jl in ("H0.5", "H0.9"):
        s = ps.Stream("zipf1.0", jl, 3, "train", 4096, ps.P6)
        assert abs(s.hazard.mean() / ps.P6["p_bar"] - 1) < 1e-3       # the path's own mean, at every H
        assert abs(len(s.switch_pos) / s.n_tokens / ps.P6["p_bar"] - 1) < 0.02


def test_rank_checks_find_the_set_law_and_reject_others(rng):
    R = ps.P6["R"]
    n = 20_000_000
    for noah in ("zipf1.0", "zipf1.25", "light"):
        counts = rng.multinomial(n, ps.rank_law(noah))
        assert ps.rank_checks(counts, noah)["rank_ok"], noah
    counts = rng.multinomial(n, ps.rank_law("zipf1.25"))
    assert not ps.rank_checks(counts, "zipf1.0")["rank_ok"]
    counts = rng.multinomial(n, ps.zipf_law(1.0, R))
    assert not ps.rank_checks(counts, "light")["rank_ok"]
    s, _ = ps.zipf_mle(rng.multinomial(n, ps.zipf_law(1.1, R)), 100)
    assert abs(s - 1.1) < 0.01


def test_check_stream_passes_a_correct_stream_and_names_failures():
    s = ps.Stream("zipf1.0", "H0.7", 2, "train", 8192, ps.P6)
    ck = ps.check_stream(s, ref=Path("/nonexistent"), every=8)
    assert ck["rank_ok"] and ck["variant_ok"] and ck["rate_ok"] and ck["g_ok"], ck
    assert ck["count_ok"] is None and ck["ok"]                 # no calibration given: reported, not checked
    bad = dict(copy.deepcopy(ps.P6))
    bad["check"] = dict(bad["check"], g_tol=1e-6)
    s2 = ps.Stream("zipf1.0", "H0.7", 2, "train", 1024, bad)
    ck2 = ps.check_stream(s2, ref=Path("/nonexistent"), every=8)
    assert not ck2["ok"] and "fGn H" in ck2["failed"]


def test_count_band_reads_the_calibration(tmp_path):
    ref = tmp_path / "c.csv"
    pd.DataFrame({"H": [0.9] * 4 + [0.5] * 4, "n_blocks": [100] * 8,
                  "H_count": [0.88, 0.87, 0.89, 0.88, 0.5, 0.51, 0.49, 0.5]}).to_csv(ref, index=False)
    lo, hi = ps.count_band(0.9, 120, ref)
    sd = np.std([0.88, 0.87, 0.89, 0.88], ddof=1)
    assert lo < 0.88 < hi and hi - lo == pytest.approx(2 * max(0.03, 5 * sd))    # max(0.03, 5 sd) on each side
    assert ps.count_band(0.7, 100, ref) is None


def test_shipped_calibration_covers_both_lengths_and_levels():
    df = pd.read_csv(ps.CHECK_REF)
    assert set(np.round(df["H"], 2)) == {0.5, 0.7, 0.9}
    assert set(df["n_windows"]) == {131072, ps.P6["heldout_windows"], 4194304}       # 2^33 tokens: option A
    assert set(np.round(df[df["n_windows"] == 4194304]["H"], 2)) == {0.5, 0.9}          # the small design's levels
    assert (df.groupby(["H", "n_windows"]).size() >= 50).all()
    nb = 2 ** 33 // ps.P6["block"]
    b5, b9 = ps.count_band(0.5, nb, exact=True), ps.count_band(0.9, nb, exact=True)
    assert b5[1] < b9[0] and ps.count_band(0.7, nb, exact=True) is None
    # the fGn H check holds with room in every calibrated stream
    assert (np.abs(df["H_g_dfa"] - df["H"]) <= ps.P6["check"]["g_tol"]).mean() > 0.99
    assert (np.abs(df["H_g_wav"] - df["H"]) <= ps.P6["check"]["g_tol"]).mean() > 0.99
    # each level's count band excludes the other levels
    for (H, nw), d in df.groupby(["H", "n_windows"]):
        lo, hi = ps.count_band(H, int(d["n_blocks"].iloc[0]))
        for H2 in set(df["H"]) - {H}:
            m2 = df[(df["H"] == H2) & (df["n_windows"] == nw)]["H_count"].mean()
            assert not (lo <= m2 <= hi), (H, H2, nw)


def test_heldout_layout_is_disjoint_and_fits():
    import config
    import p6_noah_joseph as p6
    st = p6.settings(False)
    lay = ps.heldout_layout(ps.P6, st["measure"])
    spans = sorted(lay[k] for k in ("pool", "eval", "surprisal", "probe"))
    for (a, n), (b, _) in zip(spans, spans[1:]):
        assert a + n <= b
    assert lay["n_windows"] <= ps.P6["heldout_windows"]
    assert lay["surprisal"][1] == p6.MEASURE["surprisal_docs"] == 64
    p1 = config.get("p1_checkpoints")
    assert lay["n_pool"] == p1["h71"]["coarse_blocks"] * max(p1["batch_sizes_h71"])


# ----------------------------------------------------------------------------- loss, surprisal split, attention
@pytest.fixture
def tiny_model():
    import p6_noah_joseph as p6
    torch.manual_seed(0)
    m = p6.build_model("tiny", 1, 512)
    return m


def test_chunked_loss_equals_one_pass_and_its_gradient(tiny_model):
    import p6_noah_joseph as p6
    x = torch.randint(0, 512, (3, 64))
    ref = lm.per_example_lm_loss(lambda **kw: tiny_model(**kw), x).mean()
    g_ref = torch.autograd.grad(ref, list(tiny_model.parameters()))
    loss = p6.lm_loss(tiny_model, x, chunk=37)
    g = torch.autograd.grad(loss, list(tiny_model.parameters()))
    assert torch.allclose(loss, ref, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g, g_ref))


def test_sdpa_and_eager_agree(tiny_model):
    import p6_noah_joseph as p6
    x = torch.randint(0, 512, (2, 64))
    p6.set_attention(tiny_model, "eager")
    a = p6.lm_loss(tiny_model, x, 1000).detach()
    p6.set_attention(tiny_model, "sdpa")
    b = p6.lm_loss(tiny_model, x, 1000).detach()
    p6.set_attention(tiny_model, "eager")
    assert torch.allclose(a, b, atol=1e-5)


def test_surprisal_parts_add_up(tiny_model):
    import p6_noah_joseph as p6
    x = torch.randint(0, 512, (2, 32))
    whole, rank, var = p6.surprisal_parts(tiny_model, x, "cpu", K=4, batch=1)
    ref = lm.token_losses(tiny_model, x, "cpu", batch=2).numpy()
    assert np.allclose(whole, ref, atol=1e-5)
    assert np.allclose(whole, rank + var, atol=1e-5)
    assert (rank >= -1e-6).all() and (var >= -1e-6).all()


def test_flat_directions_are_unit_and_fixed(tiny_model):
    import p6_noah_joseph as p6
    params = list(tiny_model.parameters())
    D = p6.flat_directions(params, 3, 7, "cpu")
    assert D.shape == (3, sum(p.numel() for p in params))
    assert torch.allclose(D.norm(dim=1), torch.ones(3))
    assert torch.equal(D, p6.flat_directions(params, 3, 7, "cpu"))


def test_hessian_direction_in_micro_batches_is_p1s(tiny_model):
    import p1_checkpoints
    import p6_noah_joseph as p6
    ids = torch.randint(0, 512, (8, 32))
    _, ref = p1_checkpoints.directions(tiny_model, ids, "cpu", 2, 1, 5)
    _, same = p6.directions(tiny_model, ids, "cpu", 2, 1, 5, None)       # no micro-batches: P1's function
    _, big = p6.directions(tiny_model, ids, "cpu", 2, 1, 5, 8)           # one micro-batch covers the batch
    _, mb = p6.directions(tiny_model, ids, "cpu", 2, 1, 5, 3)            # micro-batches of 3, 3 and 2

    def flat(d):
        return torch.cat([d[n].reshape(-1) for n in sorted(d)])
    assert [(k, i) for k, i, _ in ref] == [(k, i) for k, i, _ in mb] == [("random", 0), ("random", 1), ("hessian", 0)]
    for (_, _, a), (_, _, b), (_, _, c), (_, _, e) in zip(ref, same, big, mb):
        assert torch.equal(flat(a), flat(b)) and torch.equal(flat(a), flat(c))
        assert float(flat(a) @ flat(e) / (flat(a).norm() * flat(e).norm())) > 1 - 1e-5


def test_memory_fits_the_queue_and_the_laptop():
    import p6_noah_joseph as p6
    st = p6.settings(False, p6.DESIGNS["full"])
    peaks = {}
    for name in ("p6_queue_small.json", "p6_queue_full.json", "p6_queue_small_raised.json"):
        for j in json.loads((ROOT / name).read_text()):
            if j["gpu"] > 0:
                a = j["args"]
                size = a[a.index("--sizes") + 1] if "--sizes" in a else a[a.index("--size") + 1]
                if size not in peaks:
                    r = p6.prior_peak_gib(size, st)
                    peaks[size] = max(r["train"], r["measure"])
                assert peaks[size] <= j["gpu"], (j["name"], peaks[size])     # the queue's figure covers the job
    assert set(peaks) == {"14m", "31m"}
    # the Hessian-vector products on all 64 sequences at once (P1's way) would need about 16 GiB at 14M
    assert p6.measure_peaks("14m", st, hb=64)["measure"] > 15
    # the laptop timing flags fit a 4 GB card with room for the CUDA context
    lap = p6.prior_peak_gib("14m", st, mb=2, chunk=1024, eb=1, jb=8, hb=4)
    assert max(lap["train"], lap["measure"]) < 2.5


def test_plans_and_designs():
    import p6_noah_joseph as p6
    small, full = p6.plan(p6.DESIGNS["small"]), p6.plan(p6.DESIGNS["full"])
    assert len(small) == 20 and len(full) == 90
    keys = {r["key"] for r in small}
    assert keys <= {r["key"] for r in full}          # the small design is the full design's 14M corners
    assert p6.steps_of(p6.settings(False)) == 8192
    assert p6.stage_steps(p6.settings(False)) == {0.01: 82, 0.1: 819, 0.5: 4096, 1.0: 8192}
    assert p6.init_seed(1) == p6.init_seed(1) != p6.init_seed(2)


# ----------------------------------------------------------------------------- the pre-stated tests
def _df(effects: dict, rng, seeds=(1, 2, 3, 4, 5), noise=0.01, seed_sd=0.05):
    rows = []
    for s in seeds:
        b = rng.normal(0, seed_sd)
        for n in (0, 2):
            for j in (0, 2):
                nn, jj = n / 2 - 0.5, j / 2 - 0.5
                row = {"seed": s, "noah_score": n, "joseph_score": j, "size": "14m", "stage": 1.0}
                for col, (bN, bJ, bNJ) in effects.items():
                    row[col] = b + bN * nn + bJ * jj + bNJ * nn * jj + rng.normal(0, noise)
                rows.append(row)
    return pd.DataFrame(rows)


def test_holm():
    assert p6_stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert p6_stats.holm([0.5]) == [0.5]


def test_term_test_finds_planted_effects_and_controls_the_null(rng):
    df = _df({"y": (0.0, 0.05, 0.0)}, rng)
    r = p6_stats.term_test(df, "y", "J", rng, 999)
    assert r["p_perm"] < 0.01 and r["estimate"] == pytest.approx(0.05, abs=0.02)
    assert r["df"] == 20 - 8
    # a large seed effect (blocks) does not hide it, and does not make null terms significant
    rej = 0
    for _ in range(40):
        d = _df({"y": (0.0, 0.0, 0.0)}, rng, seed_sd=1.0)
        rej += p6_stats.term_test(d, "y", "NJ", rng, 199)["p_perm"] < 0.05
    assert rej <= 6


def test_family_readings(rng):
    cols = {c: (0.0, 0.0, 0.0) for _, c, *_ in p6_stats.FAMILY}
    cols["K"] = (0.0, 0.08, 0.0)                     # Joseph -> Noah as predicted
    cols["alpha"] = (0.0, 0.08, 0.0)                 # opposite to the prediction (alpha up)
    cols["Htok_excess"] = (-0.08, 0.0, 0.0)          # negative Noah effect without the variant part: dilution
    cols[p6_stats.REFEREE] = (0.0, 0.0, 0.0)
    df = _df(cols, rng)
    fam = p6_stats.family_tests(df, rng, 999)
    assert len(fam) == 14
    get = lambda name, term: fam[(fam["name"] == name) & (fam["term"] == term)].iloc[0]   # noqa: E731
    assert get("K", "J")["reading"] == "link as predicted"
    assert get("alpha", "J")["reading"] == "link, opposite to the prediction"
    assert get("Htok_excess", "N")["reading"].startswith("dilution")
    assert get("dh_grad_excess", "N")["reading"].startswith("no link detected")
    assert p6_stats.verdict(fam).startswith("Link(s):")
    null = p6_stats.family_tests(_df({c: (0, 0, 0) for _, c, *_ in p6_stats.FAMILY}, rng), rng, 199)
    assert (null["p_holm"] >= 0).all()
    md = p6_stats.to_markdown(fam, pd.DataFrame([{"check": "x", "value": True}]), "h")
    assert "| Joseph->Noah | K | J |" in md


def test_missing_outcome_keeps_its_place_in_the_family(rng):
    df = _df({"K": (0, 0.08, 0)}, rng)
    fam = p6_stats.family_tests(df, rng, 199)
    assert len(fam) == 14 and fam["p_holm"].notna().sum() == 14
    k = fam[(fam["name"] == "K") & (fam["term"] == "J")].iloc[0]
    assert k["p_holm"] == pytest.approx(min(1.0, 14 * k["p_perm"]))       # the other 13 count as p = 1


# ----------------------------------------------------------------------------- the queue files
@pytest.mark.parametrize("name", ["p6_queue_small.json", "p6_queue_full.json", "p6_queue_small_raised.json"])
def test_queue_files(name):
    import run_queue
    jobs = [run_queue.Job(**{**d, "after": tuple(d.get("after", ())), "needs": tuple(d.get("needs", ()))})
            for d in json.loads((ROOT / name).read_text())]
    names = [j.name for j in jobs]
    assert len(set(names)) == len(names) and len({j.out for j in jobs}) == len(jobs)
    for j in jobs:
        assert j.args[0] == "p6_noah_joseph.py" and "--out" not in j.args
        assert set(j.after) <= set(names[: names.index(j.name)])
    assert jobs[0].name.split("_")[1] == "timing" and jobs[-1].name.endswith("analysis") and jobs[-1].gpu == 0
    runs = [j for j in jobs if "--seeds" in j.args]
    assert all("--budget-hours" in j.args and "--timing-json" in j.args for j in runs)


def test_budget_picks_the_seeds_from_the_timing(tmp_path):
    import argparse

    import p6_noah_joseph as p6
    t = {"size": "14m", "l40s_ratio": [1.0, 1.0], "train_seconds_per_run": 900.0,
         "measure_seconds_per_stage_without_h71": 30.0, "h71_seconds_full_estimate": 276.0}
    f = tmp_path / "timing.json"
    f.write_text(json.dumps(t))
    per_seed = 4 * (900 + 4 * 30 + 2 * 276) / 3600          # small design: 4 cells, H7.1 at two stages
    a = argparse.Namespace(budget_hours=12.0, timing_json=str(f), timing_json_extra=None)
    assert p6.budget_seeds(a, p6.DESIGNS["small"]) == [1, 2, 3, 4, 5]
    a.budget_hours = 3.2 * per_seed
    assert p6.budget_seeds(a, p6.DESIGNS["small"]) == [1, 2, 3]
    a.budget_hours = 2.5 * per_seed
    with pytest.raises(SystemExit):
        p6.budget_seeds(a, p6.DESIGNS["small"])
    a.budget_hours = None
    assert p6.budget_seeds(a, p6.DESIGNS["small"]) is None
    # the full design needs the 31M timing too (or scales it from 14M)
    a.budget_hours = 1000.0
    assert p6.budget_seeds(a, p6.DESIGNS["full"]) == [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------- feasibility check (2026-11-18)
def test_feasibility_rule():
    import p6_noah_joseph as p6
    v = p6.feasibility_verdict

    def rows(k0, km0, n0, k1, km1, n1):
        return [{"stage": 0.0, "K": k0, "K_max": km0, "n_outlier_channels": n0},
                {"stage": 0.5, "K": 50.0, "K_max": 90.0, "n_outlier_channels": 9},     # only the final stage counts
                {"stage": 1.0, "K": k1, "K_max": km1, "n_outlier_channels": n1}]
    assert v(rows(0.1, 0.3, 0, 0.6, 1.2, 0))["verdict"] == "raise"           # within 1 of step 0, no channel
    assert v(rows(0.1, 0.3, 0, -0.9, -0.1, 0))["verdict"] == "raise"         # a drop is not a rise
    assert v(rows(0.1, 0.3, 0, 0.1, 0.3, 1))["verdict"] == "keep"            # one outlier channel
    assert v(rows(0.1, 0.3, 0, 1.2, 0.3, 0))["verdict"] == "keep"            # K up by more than 1
    r = v(rows(0.1, 0.3, 0, 0.5, 1.4, 0))
    assert r["verdict"] == "keep" and "K_max" in r["reason"]
    assert v(rows(0.1, 0.3, 2, 0.1, 0.3, 2))["verdict"] == "raise"           # no more channels than at step 0
    assert v(rows(0, 0, 0, 0, 0, 0), diverged=True)["verdict"] == "diverged"
    assert v(rows(0, 0, 0, 0, 0, 0)[:2])["verdict"] == "incomplete"
    assert p6.FEASIBILITY["seed"] not in p6.DESIGNS["small"]["seeds"] + p6.DESIGNS["full"]["seeds"]
    assert p6.FEASIBILITY["cell"] == ("zipf1.0", "H0.9")
    r = p6.FEASIBILITY["raise"]
    st = p6.settings(False, p6.DESIGNS["small"])
    assert r["tokens"] // (r["batch_seqs"] * st["stream"]["seq_len"]) == p6.steps_of(st) == 8192


def test_feasibility_run_measures_only_k_and_outliers(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "feas"
    r = subprocess.run([sys.executable, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--feasibility", "--device", "cpu",
                        "--micro-batch", "2", "--eval-batch", "1", "--hvp-batch", "4", "--out", str(out)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    df = pd.read_csv(out / "feasibility.csv")
    assert list(df.columns) == ["stage", "step", "tokens", "K", "K_max", "n_outlier_channels"]
    assert list(df["stage"]) == [0.0, 0.5, 1.0] and df["step"].iloc[0] == 0
    lay = pd.read_csv(out / "feasibility_layers.csv")
    assert set(lay.columns) == {"stage", "step", "layer", "K", "n_outlier_channels"}
    j = json.loads((out / "feasibility.json").read_text())
    assert j["verdict"] in ("keep", "raise") and j["cell"] == "tiny_zipf1.0_H0.9_s6"
    run = out / "runs" / "tiny_zipf1.0_H0.9_s6"
    # no other measurement: no samples, no CPU statistics, no H_link
    assert not list(run.rglob("samples")) and not list(run.rglob("stats")) and not (run / "hlink").exists()
    # K and the outlier count agree with P1's aggregation of the same moments (median, max, sum over layers)
    fin = lay[lay["stage"] == 1.0]
    assert df["K"].iloc[-1] == pytest.approx(np.median(fin["K"])) and df["K_max"].iloc[-1] == pytest.approx(fin["K"].max())
    assert df["n_outlier_channels"].iloc[-1] == fin["n_outlier_channels"].sum()


# ----------------------------------------------------------------------------- option A: the raised design (2026-11-18)
def test_lean_fgn_and_dfa_give_the_same_numbers():
    for n in (1000, 3 * 2 ** 15, 2 ** 18):
        for H in (0.5, 0.7, 0.9):
            r0, r1 = np.random.default_rng(7), np.random.default_rng(7)
            a, b = ps.fgn(n, H, r0), ps.fgn_lean(n, H, r1)
            assert np.abs(a - b).max() < 1e-12 * max(1.0, np.abs(a).max())
            assert r0.random() == r1.random()                        # the same draws, in the same order
    x = np.random.default_rng(0).standard_normal(300_017)
    for s_min in (10, 100):
        sc = hurst.log_scales(len(x), s_min)
        assert ps.dfa_lean(x, sc) == hurst.hurst_dfa(x, sc)["H"]


@pytest.mark.parametrize("joseph", ["H0.5", "H0.9"])
def test_lean_stream_is_the_same_stream(joseph):
    a = ps.Stream("zipf1.0", joseph, 2, "train", 1024, lean=False)
    b = ps.Stream("zipf1.0", joseph, 2, "train", 1024, lean=True)
    assert not a.lean and b.lean and b.hazard is None
    assert np.abs(a.g - b.g).max() < 1e-12
    assert np.array_equal(a.switch_pos, b.switch_pos) and np.array_equal(a.seg_topic, b.seg_topic)
    assert b.hazard_capped == int((a.hazard >= ps.P6["p_max"]).sum())
    assert np.array_equal(a.windows([0, 7, 1023]), b.windows([0, 7, 1023]))
    la, lb = ps.latent_checks(a, Path("/nonexistent")), ps.latent_checks(b, Path("/nonexistent"))
    for k in ("H_g_dfa", "H_g_wav", "H_count", "switch_rate", "n_switches", "hazard_capped", "g_ok", "rate_ok"):
        assert la[k] == pytest.approx(lb[k], rel=1e-12, abs=1e-12), k
    b.release_latent()
    assert b.g is None and np.array_equal(a.windows([5]), b.windows([5]))
    with pytest.raises(ValueError):
        ps.latent_checks(b)
    # only streams above LEAN_FROM_BLOCKS are lean by default: never the design's (2^28 tokens: 4.2 million blocks)
    assert ps.Stream("light", "H0.5", 1, "heldout", 64).lean is False
    assert 2 ** 28 // ps.P6["block"] <= ps.LEAN_FROM_BLOCKS < 2 ** 33 // ps.P6["block"]


def test_lean_count_band_needs_its_own_length(tmp_path):
    s = ps.Stream("zipf1.0", "H0.9", 3, "train", 256, SMALL, lean=True)
    ref = tmp_path / "c.csv"
    pd.DataFrame({"H": [0.9] * 3, "n_blocks": [10 * s.n_blocks] * 3, "H_count": [0.8, 0.81, 0.82]}).to_csv(ref)
    assert ps.count_band(0.9, s.n_blocks, ref) is not None                 # the nearest length (the design's rule)
    assert ps.count_band(0.9, s.n_blocks, ref, exact=True) is None
    ck = ps.latent_checks(s, ref)
    assert ck["count_ok"] is False and ck["count_band"] is None            # lean: no band borrowed from another length
    assert ps.latent_checks(s, Path("/nonexistent"))["count_ok"] is None   # no calibration file: reported only
    pd.DataFrame({"H": [0.9] * 3, "n_blocks": [s.n_blocks] * 3, "H_count": [0.8, 0.81, 0.82]}).to_csv(ref)
    assert ps.latent_checks(s, ref)["count_band"] == [round(0.81 - 0.05, 4), round(0.81 + 0.05, 4)]


def test_raised_settings_are_not_the_default():
    import p6_noah_joseph as p6
    d, r = p6.settings(False, p6.DESIGNS["small"]), p6.settings(False, p6.DESIGNS["small"], raised=True)
    assert d["train"]["tokens"] == 2 ** 28 and d["train"]["batch_seqs"] == 16 and p6.TRAIN["tokens"] == 2 ** 28
    assert r["train"]["tokens"] == 2 ** 33 and r["train"]["batch_seqs"] == 512
    assert p6.steps_of(d) == p6.steps_of(r) == 8192 and p6.stage_steps(d) == p6.stage_steps(r)
    assert {k: v for k, v in r["train"].items() if k not in ("tokens", "batch_seqs")} == \
        {k: v for k, v in d["train"].items() if k not in ("tokens", "batch_seqs")}
    assert r["measure"] == d["measure"] and r["stream"] == d["stream"]
    assert p6.FEASIBILITY["raise"] == p6.RAISED_TRAIN
    assert not p6.lean_stream(d) and p6.lean_stream(r)
    s = p6.settings(True, raised=True)
    assert p6.steps_of(s) == p6.steps_of(p6.settings(True)) and s["train"]["batch_seqs"] == 16
    # the queue file: every job raised, its own names and folders, budget 90 h, the raised timing file
    small = json.loads((ROOT / "p6_queue_small.json").read_text())
    raised = json.loads((ROOT / "p6_queue_small_raised.json").read_text())
    assert len(raised) == len(small) and not {j["name"] for j in raised} & {j["name"] for j in small}
    assert not {j["out"] for j in raised} & {j["out"] for j in small}
    timing = raised[0]
    assert "--timing" in timing["args"]
    for j in raised:
        a = j["args"]
        assert a[:2] == ["p6_noah_joseph.py", "--raised"] and j["gpu"] == next(
            k["gpu"] for k in small if k["name"] == j["name"].replace("p6r_", "p6_", 1))
        if "--budget-hours" in a:
            assert float(a[a.index("--budget-hours") + 1]) == p6.BUDGET_HOURS_RAISED["small"] == 90.0
            assert a[a.index("--timing-json") + 1] == timing["out"] + "/timing.json"
            assert j["out"].startswith("results/p6_small_raised/")
    assert raised[-1]["args"][raised[-1]["args"].index("--from") + 1] == "results/p6_small_raised"


def test_budget_refuses_a_timing_of_another_batch(tmp_path):
    import argparse

    import p6_noah_joseph as p6
    st = p6.settings(False, p6.DESIGNS["small"], raised=True)
    comp = {k: v * 3600 for k, v in p6.prior_component_hours("14m", st).items()}
    t = {"size": "14m", "l40s_ratio": [1.0, 1.0], "train_seconds_per_run": comp["train"],
         "measure_seconds_per_stage_without_h71": comp["stage"], "h71_seconds_full_estimate": comp["h71"],
         "batch_seqs": 512, "train_steps_per_run": 8192}
    f = tmp_path / "timing.json"
    f.write_text(json.dumps(t))
    a = argparse.Namespace(budget_hours=90.0, timing_json=str(f), timing_json_extra=None)
    assert p6.budget_seeds(a, p6.DESIGNS["small"], st) == [1, 2, 3]           # the prior: 28.6 h per seed
    f.write_text(json.dumps({**t, "batch_seqs": 16}))                          # the design's timing file
    with pytest.raises(SystemExit):
        p6.budget_seeds(a, p6.DESIGNS["small"], st)
    with pytest.raises(SystemExit):                                            # and the other way round
        f.write_text(json.dumps({**t, "batch_seqs": 512}))
        p6.budget_seeds(a, p6.DESIGNS["small"], p6.settings(False, p6.DESIGNS["small"]))


def test_lean_train_stream_build(tmp_path, monkeypatch):
    import p6_noah_joseph as p6
    st = p6.settings(True, raised=True)
    monkeypatch.setattr(ps, "LEAN_FROM_BLOCKS", 10)
    assert p6.lean_stream(st)

    class Prog:
        def __init__(self):
            self.lines = []

        def say(self, m):
            self.lines.append(m)
    c = {"key": "tiny_zipf1.0_H0.9_s1", "noah": "zipf1.0", "joseph": "H0.9", "seed": 1}
    rd = tmp_path / "runs" / c["key"]
    rd.mkdir(parents=True)
    train, latent = p6.build_lean_train(c, st, rd, tmp_path, Prog())
    assert train.lean and train.g is None and set(latent) == {"train"}
    assert (tmp_path / ".p6_stream_build.lock").exists()
    ref = ps.Stream("zipf1.0", "H0.9", 1, "train", p6.steps_of(st) * st["train"]["batch_seqs"], st["stream"],
                    lean=False)
    assert np.array_equal(train.switch_pos, ref.switch_pos)
    assert latent["train"]["H_count"] == pytest.approx(ps.latent_checks(ref)["H_count"], abs=1e-12)
    checks = p6.stream_checks(rd, {"train": train}, enforce=False, prog=Prog(), latent=latent)
    assert checks["train"]["H_count"] == latent["train"]["H_count"]
    # resumed: the saved check is kept, and the latent checks are not run again
    train2, latent2 = p6.build_lean_train(c, st, rd, tmp_path, Prog())
    assert latent2 == {}


def test_raised_smoke_run(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "raised"
    r = subprocess.run([sys.executable, str(ROOT / "p6_noah_joseph.py"), "--smoke", "--raised", "--device", "cpu",
                        "--seeds", "1", "--out", str(out)], capture_output=True, text=True, timeout=1200, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    run = json.loads((out / "run.json").read_text())
    assert run["config"]["design"] == "smoke_raised"
    assert run["config"]["train"]["tokens"] == 4 * 24 * 4 * 128 and run["config"]["train"]["batch_seqs"] == 16
    df = pd.read_csv(out / "checkpoints.csv")
    assert set(df["stage"]) == {0.5, 1.0} and len(df) == 8
    log = pd.read_csv(next((out / "runs").glob("*/train_log.csv")))
    assert len(log) == 24                                                   # the same steps as the design
