"""H7.1 verdict: slope, level against Gaussian pools, scale exponent; recomputed from saved CSVs."""
import numpy as np
import pandas as pd
import pytest

from common import h71_verdict as hv

BOUNDS = {"null_reps": 400, "alpha_null_q": 1.99, "s_null_lo": -0.51, "s_null_hi": -0.49}


def row(**kw):
    base = {"b_lo": -0.01, "b_hi": 0.02, "alpha_mean": 1.5, "scale_exponent": -0.33, "scale_gap_lo": -0.02,
            "scale_gap_hi": 0.03, "n_pool": 1000, "n_blocks": 100}
    base.update(kw)
    return base


@pytest.mark.parametrize("kw, verdict, reason", [
    ({}, "invariant (wild)", "contains 0"),
    ({"alpha_mean": 1.997, "scale_exponent": -0.50}, "flat at 2 (mild)", "Gaussian range"),
    ({"b_lo": 0.06, "b_hi": 0.09}, "drift", "above 0.05"),
    ({"b_lo": 0.06, "b_hi": 0.09, "alpha_mean": np.nan}, "drift", "above 0.05"),
    ({"b_lo": 0.01, "b_hi": 0.06}, "undecided", "slope"),
    ({"b_lo": -0.08, "b_hi": -0.06}, "undecided", "slope"),
    ({"scale_gap_lo": 0.01, "scale_gap_hi": 0.05}, "undecided", "excludes"),
    ({"alpha_mean": 1.997, "scale_exponent": -0.40}, "undecided", "outside the Gaussian range"),
    ({"b_lo": np.nan, "b_hi": np.nan}, "undecided", "no bootstrap interval"),
])
def test_classify_branches(kw, verdict, reason):
    out = hv.classify(row(**kw), BOUNDS)
    assert out["verdict"] == verdict
    assert reason in out["verdict_reason"]


def test_margin_keeps_near_two_undecided():
    near = row(alpha_mean=1.96, scale_exponent=-0.49, scale_gap_lo=-0.01, scale_gap_hi=0.01)
    out = hv.classify(near, BOUNDS)                                                  # margin 1.9 by default
    assert out["verdict"] == "undecided" and "above 1.9" in out["verdict_reason"]
    assert hv.classify(near, BOUNDS, wild_max=2.0)["verdict"] == "invariant (wild)"  # 2 turns it off
    assert hv.classify(near, BOUNDS, wild_max=None)["verdict"] == "invariant (wild)"
    assert hv.classify(row(alpha_mean=1.9), BOUNDS)["verdict"] == "invariant (wild)"  # 1.9 itself is wild
    assert "<= 1.9" in hv.classify(row(), BOUNDS)["verdict_reason"]


def test_flat_at_two_is_not_read_as_invariant():
    # the case the slope alone got wrong: alpha-hat flat at 2, scale B^(-1/2)
    out = hv.classify(row(alpha_mean=2.0, scale_exponent=-0.5, scale_gap_lo=-0.01, scale_gap_hi=0.01), BOUNDS)
    assert out["slope_reading"] == "flat" and out["level_reading"] == "consistent with 2"
    assert out["verdict"] == "flat at 2 (mild)"


def test_no_reference_leaves_flat_slope_undecided():
    nob = {"null_reps": 0, "alpha_null_q": np.nan, "s_null_lo": np.nan, "s_null_hi": np.nan}
    out = hv.classify(row(), nob)
    assert out["verdict"] == "undecided" and "level unknown" in out["verdict_reason"]


def test_null_bounds_are_quantiles():
    g = pd.DataFrame({"alpha_mean": np.linspace(1.9, 2.0, 101), "scale_exponent": np.linspace(-0.6, -0.4, 101)})
    b = hv.null_bounds(g, 0.90)
    assert b["alpha_null_q"] == pytest.approx(1.905) and b["s_null_lo"] == pytest.approx(-0.59)
    assert b["s_null_hi"] == pytest.approx(-0.41) and b["null_reps"] == 101
    assert np.isnan(hv.null_bounds(g.head(10))["alpha_null_q"])     # too few pools: no reference


@pytest.fixture
def cheap_fits(monkeypatch):
    """Stand-in for the stable fits (seconds each): alpha from the sample kurtosis, gamma from the SD."""
    def fake(pool, Bs, n_blocks, symmetric=False):
        rows = []
        for B in Bs:
            x = hv.block_noise(pool, int(B), n_blocks)
            k = float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2)
            rows.append({"B": int(B), "n_blocks": len(x), "alpha": min(2.0, 6.0 / max(k, 3.0)),
                         "gamma": float(np.std(x) / np.sqrt(2)), "beta": 0.0})
        return rows
    monkeypatch.setattr(hv, "fit_over_batches", fake)


def test_null_replicate_is_reproducible_and_keyed(cheap_fits):
    a = hv.null_replicate(1, 32 * 2000, 2000, [8, 16, 32], 0)
    b = hv.null_replicate(1, 32 * 2000, 2000, (32, 8, 16), 0)        # order of B does not matter
    c = hv.null_replicate(1, 32 * 2000, 2000, [8, 16, 32], 1)
    assert a == b and a["scale_exponent"] != c["scale_exponent"]
    assert a["scale_exponent"] == pytest.approx(-0.5, abs=0.05)     # Gaussian blocks: gamma_B ~ B^(-1/2)
    assert a["batch_sizes"] == "8,16,32" and set(a) >= {"alpha_B8", "alpha_B32", "b", "scale_exponent"}


def test_simulate_null_tops_up_only_what_is_missing(cheap_fits):
    key = (32 * 30, 30, "8,16,32")
    have = pd.DataFrame([hv.null_replicate(5, *key[:2], [8, 16, 32], r) for r in range(2)])
    assert hv.simulate_null([key], 2, 5, have=have, log=lambda *_: None).empty
    new = hv.simulate_null([key], 3, 5, have=have, log=lambda *_: None)
    assert list(new.replicate) == [2]


def test_apply_verdicts_matches_rows_to_their_setting():
    null = pd.concat([pd.DataFrame({"n_pool": n, "n_blocks": nb, "batch_sizes": "8,16", "seed": 0,
                                    "replicate": range(40), "alpha_mean": np.linspace(lo, 2.0, 40),
                                    "scale_exponent": np.linspace(-0.52, -0.48, 40)})
                      for n, nb, lo in ((1600, 100, 1.98), (200, 12, 1.90))], ignore_index=True)
    df = pd.DataFrame([row(n_pool=1600, n_blocks=100, alpha_mean=1.95, scale_exponent=-0.49),
                       row(n_pool=200, n_blocks=12, alpha_mean=1.95, scale_exponent=-0.49),
                       row(n_pool=999, n_blocks=7)])
    out = hv.apply_verdicts(df, null, [8, 16])
    # 1.95 is below 2 with 100 blocks per B but consistent with 2 with 12
    assert list(out.level_reading) == ["below 2", "consistent with 2", "unknown"]
    assert out.verdict.iloc[1] == "flat at 2 (mild)" and out.verdict.iloc[2] == "undecided"
    assert out.null_reps.tolist() == [40, 40, 0]
    with pytest.raises(ValueError):
        hv.apply_verdicts(df.drop(columns=["b_lo"]), null, [8, 16])


def test_infer_batch_sizes(tmp_path):
    df = pd.DataFrame({"alpha_B16": [1.5], "alpha_B8": [1.5], "b_lo": [0.0]})
    assert hv.infer_batch_sizes(df)[0] == [8, 16]
    p = tmp_path / "h71_summary.csv"
    pd.DataFrame({"B": [8, 64, 8, 64], "alpha": 1.5}).to_csv(tmp_path / "h71_per_B.csv", index=False)
    assert hv.infer_batch_sizes(pd.DataFrame({"b_lo": [0.0]}), p)[0] == [8, 64]
    assert hv.infer_batch_sizes(pd.DataFrame({"b_lo": [0.0]}), tmp_path / "x" / "s.csv")[0] is None


def test_command_line_recomputes_from_saved_csvs(tmp_path, capsys):
    run_dir = tmp_path / "p1_pythia"
    run_dir.mkdir()
    pd.DataFrame([{**row(), "size": "14m", "stage": 1.0, "direction": "random0", "reading": "invariant"},
                  {**row(alpha_mean=2.0, scale_exponent=-0.5), "size": "14m", "stage": 1.0,
                   "direction": "random1", "reading": "invariant"}]).to_csv(run_dir / "h71_summary.csv", index=False)
    pd.DataFrame({"B": [8, 16]}).to_csv(run_dir / "h71_per_B.csv", index=False)
    null = pd.DataFrame({"n_pool": 1000, "n_blocks": 100, "batch_sizes": "8,16", "seed": 0, "replicate": range(30),
                         "alpha_mean": np.linspace(1.97, 2.0, 30), "scale_exponent": np.linspace(-0.52, -0.48, 30),
                         "b": 0.0})
    null.to_csv(tmp_path / "null.csv", index=False)
    assert hv.main([str(run_dir / "h71_summary.csv"), "--null", str(tmp_path / "null.csv"), "--null-reps", "30"]) == 0
    out = pd.read_csv(run_dir / "h71_summary_verdict.csv")
    # the run's own slope-only reading is kept; the verdict tells the two rows apart
    assert list(out.reading) == ["invariant", "invariant"]
    assert list(out.verdict) == ["invariant (wild)", "flat at 2 (mild)"]
    assert "B = 8,16 from h71_per_B.csv" in capsys.readouterr().out
