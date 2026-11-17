"""H_link's second set of confound controls (Endri, 2026-11-17, same test, fixed before any value on real data):
the position profile of P1's windows (standardized rows), the document pieces between EOS tokens (piece shuffles,
EOS positions rebuilt and verified on the CPU), and P4's slow changes in the noise level (rescaled rows)."""
import json

import numpy as np
import pandas as pd
import pytest

import h_link
from common import eos_rebuild, hf, hlink, lm


def test_segment_shuffle_keeps_each_segment():
    rng = np.random.default_rng(0)
    v = np.arange(30.0)
    starts = [0, 4, 5, 12, 20]
    w = hlink.segment_shuffle(v, starts, rng)
    for a, b in zip(starts, starts[1:] + [30]):
        assert sorted(w[a:b]) == list(v[a:b])
    assert not np.array_equal(w, v)
    assert np.array_equal(hlink.segment_shuffle(v, range(30), rng), v)      # one value per segment: unchanged


def test_pieces_and_their_starts():
    rng = np.random.default_rng(1)
    cuts = hlink.pieces(16, 2047, 1000.0, rng)
    assert len(cuts) == 16 and all(0 < c < 2047 for cs in cuts for c in cs)
    assert 1.0 < np.mean([len(c) for c in cuts]) < 3.5                    # about 2,046 / 1,000 per window
    st = hlink.piece_starts([[3, 7], [], [1]], 10)
    assert st.tolist() == [0, 3, 7, 10, 20, 21]


def test_position_profile_removes_a_power_law_profile():
    rng = np.random.default_rng(2)
    D, L = 16, 2047
    j = np.arange(1, L + 1, dtype=float)
    z = (rng.gamma(2.0, size=(D, L)) - 2) / np.sqrt(2)
    x = (3 * j ** -0.5 + (1 + j ** -0.5) * z).ravel()
    lev, spr = hlink.position_profile(x.reshape(D, L))
    assert lev.shape == spr.shape == (L,)
    assert lev[0] - lev[-1] > 1.5 and spr[0] > 1.5 * spr[-1]               # the profile is found
    s = hlink.position_standardize(x, D, L).reshape(D, L)
    early, late = s[:, :20], s[:, 1000:]
    assert abs(np.median(early) - np.median(late)) < 0.3                   # level and spread flat in position
    assert 0.7 < np.median(np.abs(early - np.median(early))) / np.median(np.abs(late - np.median(late))) < 1.4
    assert abs(np.median(x.reshape(D, L)[:, :20]) - np.median(x.reshape(D, L)[:, 1000:])) > 1


def test_rescale_removes_a_slow_spread_drift_and_keeps_short_clustering():
    rng = np.random.default_rng(3)
    x, how = hlink.synthetic_control2("iid_spread", rng)
    assert how == {"detrend": True, "rescale": True} and len(x) == 9064
    y = hlink.rescale(hlink.detrend(x))
    sd_x = np.array([b.std() for b in np.array_split(x, 9)])
    sd_y = np.array([b.std() for b in np.array_split(y, 9)])
    assert sd_x.max() / sd_x.min() > 1.5 and sd_y.max() / sd_y.min() < sd_x.max() / sd_x.min()
    assert np.std(np.log(sd_y)) < 0.5 * np.std(np.log(sd_x))
    # clustering on scales well below the window survives: bursts of 200 steps keep their sd ratio
    burst = np.where((np.arange(9064) // 200) % 2 == 0, 3.0, 1.0) * rng.normal(size=9064)
    r = hlink.rescale(burst)
    hi = (np.arange(9064) // 200) % 2 == 0
    assert r[hi].std() / r[~hi].std() > 2.5
    assert hlink.rescale(np.column_stack([x, 2 * x])).shape == (9064, 2)


def test_second_control_series():
    rng = np.random.default_rng(4)
    for kind in ("docs_profile", "mrw_profile"):
        x, how = hlink.synthetic_control2(kind, rng)
        assert len(x) == 16 * 2047 and np.isfinite(x).all()
        assert set(how) == {"doc_len", "n_docs", "position", "cuts"} and len(how["cuts"]) == 16
        X = x.reshape(16, 2047)
        assert np.median(X[:, :5]) > np.median(X[:, 1000:]) + 1             # the position profile
    x, how = hlink.synthetic_control2("mrw_spread", rng)
    assert len(x) == 9064 and how == {"detrend": True, "rescale": True}
    with pytest.raises(ValueError):
        hlink.synthetic_control2("other", rng)


def test_piece_control_leaves_the_other_columns():
    x, how = hlink.synthetic_control2("docs_profile", np.random.default_rng(5))
    x = x[:4094]
    st = hlink.piece_starts(how["cuts"][:2], 2047)
    a = hlink.hlink_rows(x, np.random.default_rng(6), n_surr=4, doc_len=2047)
    b = hlink.hlink_rows(x, np.random.default_rng(6), n_surr=4, doc_len=2047, segments={"piece": st})
    for ra, rb in zip(a, b):
        assert {k: rb[k] for k in ra} == ra
        assert rb["n_pieces"] == len(st) and 0 < rb["p_piece_wav"] <= 1 and "beyond_pieces" in rb
        assert rb["beyond_pieces"] == (rb["memory_beyond_pieces"] and rb["width_beyond_pieces"])


# ----------------------------------------------------------------------------- units in h_link.py
def _p1_spec(tmp_path, x, cuts=None):
    f = tmp_path / "htok.npy"
    np.save(f, x.astype(np.float32))
    return {"source": "p1", "folder": "p1_x", "key": "p1_x/grid/a@b", "path": str(f), "stamp": "s", "warm": 0,
            "n_docs": 16, "doc_len": 2047, "group": "14m", "eos_cuts": cuts, "eos_status": "verified"}


def test_p1_unit_has_standardized_rows_and_the_piece_control(tmp_path):
    x, how = hlink.synthetic_control2("docs_profile", np.random.default_rng(7))
    test = dict(hlink.TEST_DEFAULTS, n_surr=2)
    res = h_link.run_unit(_p1_spec(tmp_path, x, how["cuts"]), test)
    assert [r["scales"] for r in res["rows"]] == ["main", "sensitivity", "standardized"]
    n_cuts = sum(len(c) for c in how["cuts"])
    for r in res["rows"]:
        assert r["doc_len"] == 2047 and r["n_pieces"] == 16 + n_cuts and r["n_cuts"] == n_cuts
        assert "p_doc_width" in r and "p_piece_width" in r
    st = res["rows"][2]
    assert st["position_bins"] == hlink.POSITION_BINS and st["s_min"] == 8 and "eos_cuts" not in res
    res0 = h_link.run_unit(_p1_spec(tmp_path, x, None), test)               # no verified EOS positions
    assert [r["scales"] for r in res0["rows"]] == ["main", "sensitivity", "standardized"]
    assert all("p_piece_wav" not in r and "p_doc_wav" in r for r in res0["rows"])
    for r0, r in zip(res0["rows"], res["rows"]):                            # the pieces come last: rest unchanged
        same = [k for k in r0 if k.startswith(("H_abs_", "p_shuf", "p_phase", "p_iaaft", "p_doc", "delta_h"))
                and "piece" not in k]
        if r0["scales"] != "standardized":
            assert {k: r0[k] for k in same} == {k: r[k] for k in same}


def test_p4_unit_has_rescaled_rows(tmp_path):
    rng = np.random.default_rng(8)
    f = tmp_path / "s.npy"
    np.save(f, rng.normal(size=(700, 2)))
    spec = {"source": "p4", "folder": "p4_x", "key": "p4_x/t", "path": str(f), "stamp": "s", "warm": 40,
            "group": "resnet18"}
    res = h_link.run_unit(spec, dict(hlink.TEST_DEFAULTS, n_surr=2))
    assert [r["scales"] for r in res["rows"]] == ["main", "sensitivity", "detrended", "rescaled"]
    assert res["rows"][3]["mad_width"] == hlink.MAD_WIDTH and res["rows"][3]["detrend_width"] == 201


@pytest.mark.slow
def test_second_control_calibration_smoke(tmp_path):
    assert h_link.main(["--calibrate", "--smoke", "--reps", "1", "--workers", "0", "--nice", "0",
                        "--out", str(tmp_path)]) == 0
    c2 = pd.read_csv(tmp_path / "calibration_controls2.csv")   # 2 profile kinds x 3 rows, 2 spread kinds x 4
    assert len(c2) == 14 and set(c2["kind"]) == set(hlink.CONTROL2_N)
    assert set(c2.loc[c2["kind"].str.endswith("_profile"), "scales"]) == {"main", "sensitivity", "standardized"}
    assert set(c2.loc[c2["kind"].str.endswith("_spread"), "scales"]) == {"main", "sensitivity", "detrended",
                                                                         "rescaled"}
    assert c2.loc[c2["kind"].str.endswith("_profile"), "p_piece_wav"].notna().all()
    s2 = pd.read_csv(tmp_path / "calibration_controls2_summary.csv")
    assert {"beyond_pieces_rate", "beyond_documents_rate", "pass_rate"} <= set(s2.columns)


# ----------------------------------------------------------------------------- the EOS rebuild
def test_stream_matches_tokenize_stream_and_the_cut_rule():
    tok = hf.ByteTokenizer(512)
    texts = ["a" * 250, "b" * 300, "c" * 700, "d" * 900]
    arr, pos = eos_rebuild.stream(texts, tok, 200, 6)
    assert np.array_equal(arr, hf.tokenize_stream(texts, tok, 200, 6).numpy())
    for w in range(len(arr)):
        assert sorted(np.flatnonzero(arr[w] == tok.eos_token_id).tolist()) == pos[w]
    assert pos[1] == [50] and pos[2] == [151]                  # EOS after 250 and after 250 + 1 + 300 tokens
    assert eos_rebuild.cuts_from_eos([[0, 5, 198, 199]], 200) == [[5, 198]]
    # loss index e predicts token e + 1, the first token of the next document
    assert arr[1][50 + 1] == tok.encode("b")[0] and arr[1][50 - 1] == tok.encode("a")[0]


FAKE = [("EleutherAI__pythia-14m@step1000", "step1000"), ("EleutherAI__pythia-14m@step2000", "step2000")]


def _eos_folder(root, corrupt=False):
    """A smoke-like P1 folder: byte tokenizer, Zipfian texts and stand-in models, as P1 builds them offline."""
    f = root / "p1_eos"
    cfg = {"eval_dataset": {"repo": "monology/pile-uncopyrighted", "split": "validation",
                            "data_files": {"validation": "val.jsonl.zst"}},
           "eval_seqs": 2, "eval_len": 128, "surprisal_len": 512, "surprisal_docs": 16, "dtype": "float32"}
    f.mkdir(parents=True, exist_ok=True)
    (f / "run.json").write_text(json.dumps({"smoke": True, "seed": 5, "config": cfg}))
    tok = hf.ByteTokenizer(hf.TINY_NEOX["vocab_size"])
    ids = hf.tokenize_stream(hf.zipf_text(16, 400, 6)[::-1], tok, 512, 16)
    for name, rev in FAKE:
        d = f / "checkpoints" / "grid" / name
        (d / "samples").mkdir(parents=True, exist_ok=True)
        row = {"repo": "EleutherAI/pythia-14m", "revision": rev, "commit": None, "source": "fallback",
               "n_params": 165632, "size": "14m", "pass": "grid"}
        (d / "row.json").write_text(json.dumps({"row": row}))
        model, _ = eos_rebuild.load_model(row)
        x = lm.token_losses(model, ids, "cpu").numpy().reshape(-1)
        if corrupt:
            x = np.roll(x, 1)
        np.save(d / "samples" / "htok.npy", x)
    return f, ids, tok


def test_eos_rebuild_verified_on_a_smoke_like_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "hub_reachable", lambda: False)
    f, ids, tok = _eos_folder(tmp_path)
    out = tmp_path / "eos.json"
    res = eos_rebuild.rebuild_folder(f, out, log=lambda m: None)
    assert set(res) == {f"p1_eos/grid/{n}" for n, _ in FAKE}
    assert {v["status"] for v in res.values()} == {"verified"} and len({v["group"] for v in res.values()}) == 1
    cuts = res[f"p1_eos/grid/{FAKE[0][0]}"]["cuts"]
    assert sum(len(c) for c in cuts) >= 2
    arr = ids.numpy()
    for w, cs in enumerate(cuts):
        for c in cs:
            assert arr[w, c] == tok.eos_token_id                # a piece starts at the loss after an EOS
    doc = json.loads(out.read_text())
    g = next(iter(doc["groups"].values()))
    assert g["texts"] == "fallback" and g["check"]["max_abs"] < 1e-5 and g["n_checkpoints"] == 2
    assert g["checkpoint"].endswith("step1000")                 # the smallest (ties: by key)
    # a later run reuses the verified group
    monkeypatch.setattr(eos_rebuild, "verify_group", lambda *a, **k: pytest.fail("verified again"))
    assert eos_rebuild.rebuild_folder(f, out, log=lambda m: None) == res


def test_eos_rebuild_refuses_a_stream_that_does_not_reproduce_the_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "hub_reachable", lambda: False)
    f, _, _ = _eos_folder(tmp_path, corrupt=True)                # losses shifted by one token
    out = tmp_path / "eos.json"
    res = eos_rebuild.rebuild_folder(f, out, log=lambda m: None)
    assert {v["status"] for v in res.values()} == {"unverified"} and all(v["cuts"] is None for v in res.values())
    assert all(v["reason"].startswith("mismatch") for v in res.values())
    _eos_folder(tmp_path)                                        # the right losses: checked again only on request
    assert {v["status"] for v in eos_rebuild.rebuild_folder(f, out, log=lambda m: None).values()} == {"unverified"}
    assert {v["status"] for v in eos_rebuild.rebuild_folder(f, out, retry=True, log=lambda m: None).values()} == {
        "verified"}


def test_eos_rebuild_too_large_and_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "hub_reachable", lambda: False)
    f, _, _ = _eos_folder(tmp_path)
    res = eos_rebuild.rebuild_folder(f, tmp_path / "a.json", max_params=1e5, log=lambda m: None)
    assert {v["reason"][:9] for v in res.values()} == {"too large"}
    res = eos_rebuild.rebuild_folder(f, tmp_path / "a.json", max_params=1e6, log=lambda m: None)   # limit raised
    assert {v["status"] for v in res.values()} == {"verified"}
    # two tokenizers among one repo's revisions that give different streams ("w" is byte 119: 121 or 23)
    monkeypatch.setattr(eos_rebuild, "fingerprint",
                        lambda c, cache, log: "bytes:512" if c["revision"] == "step1000" else "bytes:100")
    res = eos_rebuild.rebuild_folder(f, tmp_path / "b.json", log=lambda m: None)
    assert {v["reason"][:9] for v in res.values()} == {"ambiguous"}


def test_main_attaches_the_eos_cuts(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "hub_reachable", lambda: False)
    f, _, _ = _eos_folder(tmp_path / "res")
    out = tmp_path / "out"
    argv = ["--results", str(tmp_path / "res"), "--p1", str(f), "--p4", "--out", str(out), "--workers", "0",
            "--n-surr", "2", "--nice", "0"]
    assert h_link.main(argv) == 0
    p1 = pd.read_csv(out / "p1_tokens.csv")
    assert set(p1["scales"]) == {"main", "sensitivity", "standardized"} and set(p1["eos_status"]) == {"verified"}
    assert (p1["n_cuts"] >= 2).all() and p1["p_piece_wav"].notna().all()
    assert (out / "eos" / "p1_eos.json").exists()
    assert h_link.main(argv + ["--no-eos"]) == 0                 # without the rebuild: computed again, no pieces
    p1 = pd.read_csv(out / "p1_tokens.csv")
    assert set(p1["eos_status"]) == {"off"} and "p_piece_wav" not in p1
