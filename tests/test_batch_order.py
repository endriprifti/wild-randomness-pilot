"""Batches from a fixed random pool order (2026-11-17): the H7.1 main fit and the coupling alpha describe random
batches, as the i.i.d. bootstrap does; P1's tasks made before keep the old order; p1_recompute.py and the in-place
upgrade of p1_checkpoints.py give what the current code gives; Delta h_rep NaN rule; the verdict in the summaries."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import config
import p1_checkpoints as p1
import p1_recompute
import p4_clipping as p4
from common import batch, h71_verdict, multifractal, p1_tasks, runtime, stable

KEY = "grid/EleutherAI__pythia-14m@step1000"
SEED = 7


@pytest.fixture
def grid():
    """P1's stable density: interpolated from a fixed table, so a fit does not depend on the fits before it (the FFT
    density keeps a table per alpha rounded to 1e-6, so results differ by about 1e-6 with the order of the fits)."""
    with stable.using_density("grid"):
        yield str(stable._GRID.path)


def _cheap(pool, Bs, n_blocks, symmetric=False):
    """Stand-in for the stable fits (seconds each): alpha from the kurtosis of the block noise, gamma from its SD."""
    rows = []
    for B in Bs:
        x = stable.block_noise(pool, int(B), n_blocks)
        k = float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2)
        rows.append({"B": int(B), "n_blocks": len(x), "alpha": min(2.0, 6.0 / max(k, 3.0)),
                     "gamma": float(np.std(x) / np.sqrt(2)), "beta": 0.0})
    return rows


def _documents(rng, n, length=16):
    """A pool like P1's: consecutive chunks of the same documents share the document's level."""
    docs = rng.standard_t(5, n // length + 1)
    return np.repeat(docs, length)[:n] + 0.5 * rng.standard_t(5, n)


def test_shuffled_main_fit_lies_in_its_bootstrap_interval(monkeypatch):
    monkeypatch.setattr(batch, "fit_over_batches", _cheap)
    inside = {"consecutive": 0, "shuffled": 0}
    for rep in range(20):
        rng = np.random.default_rng(rep)
        pool = _documents(rng, 400 * 64)
        for name, seed in (("consecutive", None), ("shuffled", [SEED, rep])):
            s = batch.h71_test(pool, [8, 16, 32, 64], 400, n_boot=200, rng=np.random.default_rng(rep + 1000),
                               order_seed=seed)["summary"]
            inside[name] += s["b_lo"] <= s["b"] <= s["b_hi"]
    # the 90% interval covers the shuffled fit about 18 times in 20; consecutive blocks of documents miss it
    assert inside["shuffled"] >= 15 and inside["consecutive"] <= 10, inside


def test_order_changes_the_main_fit_only(monkeypatch, rng):
    monkeypatch.setattr(batch, "fit_over_batches", _cheap)
    pool = _documents(rng, 50 * 32)
    a = batch.h71_test(pool, [8, 16, 32], 50, n_boot=5, rng=np.random.default_rng(1))
    b = batch.h71_test(pool, [8, 16, 32], 50, n_boot=5, rng=np.random.default_rng(1), order_seed=[3])
    assert a["per_B"] == _cheap(pool, [8, 16, 32], 50)                    # no order_seed: as before
    assert b["per_B"] == _cheap(batch.shuffled(pool, [3]), [8, 16, 32], 50)
    assert a["summary"]["b_lo"] == b["summary"]["b_lo"] and a["summary"]["n_boot"] == b["summary"]["n_boot"]
    o = batch.pool_order(1000, [3])
    assert sorted(o) == list(range(1000)) and (o == batch.pool_order(1000, [3])).all()
    P = rng.normal(size=(100, 3))
    assert (batch.shuffled(P, [3]) == P[batch.pool_order(100, [3])]).all()   # rows: every direction alike


# ----------------------------------------------------------------------------- P1
def _old_job(root, rng):
    """A P1 job's folder as the code before 2026-11-17 leaves it: tasks without order_seed, every statistic saved."""
    ckroot = root / "checkpoints"
    d = ckroot / KEY
    (d / "samples").mkdir(parents=True)
    pool = _documents(rng, 30 * 32)
    p1_tasks.atomic_npy(d / "samples" / "h71.npy", np.stack([pool, _documents(rng, len(pool))], 1).astype(np.float32))
    p1_tasks.atomic_npy(d / "samples" / "grad.npy", np.stack([_documents(rng, 400), rng.normal(size=400)], 1))
    p1_tasks.atomic_npy(d / "samples" / "htok.npy", rng.normal(size=4096).astype(np.float32))
    ev = [np.sort(rng.pareto(2.0, 64) + 1.0), np.sort(rng.pareto(3.0, 32) + 1.0)]
    p1_tasks.atomic_npz(d / "samples" / "esd.npz", names=np.array(["a", "b"]), shapes=np.array([[64, 128], [32, 64]]),
                        ev_0=ev[0], ev_1=ev[1])
    p1_tasks.atomic_npy(d / "samples" / "dh.npy", rng.normal(size=(300, 3)).astype(np.float32))
    T = {**config.TAILS, "n_boot": 20, "hill_reps": 20}

    def t(name, kind, params):
        return {"name": name, "kind": kind, "params": params, "seed": p1_tasks.seed_for(SEED, KEY, name)}
    tasks = [t("esd", "esd", {}), t("dh", "dh", {"q_lo": 0.5, "q_hi": 4.0}),
             t("htok", "htok", {"n_surr": 9, "wavelet_j1": 3}), t("grad", "grad", {"B": 8, "n_blocks": 50, "T": T})]
    for dname, j in (("random0", 0), ("hessian0", 1)):
        tasks += [t(f"h71_fit.{dname}", "h71_fit", {"j": j, "Bs": [8, 16, 32], "nb": 30}),
                  t(f"h71_boot.{dname}.0", "h71_boot", {"j": j, "Bs": [8, 16, 32], "nb": 30, "n_boot": 3})]
    row = {"suite": "pythia", "variant": "base", "size": "14m", "seed": 0, "repo": "EleutherAI/pythia-14m",
           "step": 1000, "revision": "step1000", "commit": "c", "pass": "grid", "probe_loss": 3.0,
           "n_nonembed": 1000, "eval_loss": 2.9, "gpu_pass_seconds": 1.0}
    record = {"row": row, "tasks": tasks,
              "h71": {"directions": ["random0", "hessian0"], "stage": 0.01, "frac": 0.007, "nb": 30, "Bs": [8, 16, 32],
                      "n_pool": len(pool), "n_boot": 3, "n_chunks": 1}}
    p1_tasks.atomic_json(d / "row.json", record)
    stats = {x["name"]: p1_tasks.run_task({**x, "dir": str(d)}) for x in tasks}
    p1_tasks.atomic_json(d / "result.json", p1_tasks.assemble(record, stats, p1.h71_kw()))
    fp = p1.fingerprint(config.get("p1_checkpoints", smoke=True), SEED, set(), "grid", 25)
    fp.pop("batch_order")                                                   # the fingerprint before 2026-11-17
    p1_tasks.atomic_json(ckroot / "fingerprint.json", fp)
    return d, record


def _current(tmp_path, record, sd):
    """What the current code computes for the same checkpoint: the record's tasks with the order, run and merged."""
    rec = json.loads(json.dumps(record))
    assert sorted(p1_tasks.upgrade_tasks(rec, SEED)) == ["grad", "h71_fit.hessian0", "h71_fit.random0"]
    out = tmp_path / "current"
    stats = {x["name"]: p1_tasks.run_task({**x, "dir": str(out), "samples_dir": str(sd)}) for x in rec["tasks"]}
    return p1_tasks.assemble(rec, stats, p1.h71_kw())


def test_old_tasks_keep_their_order_in_new_processes(tmp_path, rng, grid):
    """A job started with the earlier code sends tasks without order_seed; pool processes started after a pull must
    compute what its other tasks computed."""
    d, record = _old_job(tmp_path / "src", rng)
    grad = next(x for x in record["tasks"] if x["name"] == "grad")
    old = p1_tasks.run_task({**grad, "dir": str(tmp_path / "again"), "samples_dir": str(d)})
    P = np.load(d / "samples" / "grad.npy")
    ref = p1_tasks.noise_quantities(P[:, 0], 8, 50, {**grad["params"]["T"], "n_jobs": 1}, p1_tasks.rng_for(grad["seed"]))
    assert old["result"] == json.loads(json.dumps(ref, default=float))
    new = p1_tasks.run_task({**grad, "params": {**grad["params"], "order_seed": p1_tasks.order_seed(SEED)},
                             "dir": str(tmp_path / "new"), "samples_dir": str(d)})
    assert new["result"]["alpha"] != old["result"]["alpha"]


def test_recompute_matches_the_current_code(tmp_path, rng, grid):
    src = tmp_path / "p1_pythia"
    sd, record = _old_job(src, rng)
    want = _current(tmp_path, record, sd)
    before = {p.name: p.read_bytes() for p in (sd / "stats").glob("*.json")}
    assert p1_recompute.main([str(src), "--dry-run"]) == 0
    assert not (tmp_path / "p1_pythia_recomputed" / "checkpoints").exists()
    assert p1_recompute.main([str(src), "--workers", "0"]) == 0
    dst = tmp_path / "p1_pythia_recomputed"
    got = json.loads((dst / "checkpoints" / KEY / "result.json").read_text())
    assert _same(got["row"], want["row"])
    assert _same(got["h71_summ"], want["h71_summ"]) and _same(got["h71_rows"], want["h71_rows"])
    assert got["row"]["alpha_batch_order"] == "shuffled" and {s["batch_order"] for s in got["h71_summ"]} == {"shuffled"}
    old = json.loads((sd / "result.json").read_text())
    assert old["row"]["alpha"] != got["row"]["alpha"]                        # the order matters for this pool
    assert {k: v for k, v in old["h71_summ"][0].items() if k in ("b_lo", "b_hi", "n_boot")} == \
           {k: v for k, v in got["h71_summ"][0].items() if k in ("b_lo", "b_hi", "n_boot")}   # bootstrap kept
    # SRC is only read
    assert {p.name: p.read_bytes() for p in (sd / "stats").glob("*.json")} == before
    assert not (sd / "stats" / "consecutive").exists()
    df = pd.read_csv(dst / "h71_summary.csv")
    assert {"verdict", "verdict_reason", "slope_reading", "batch_order"} <= set(df.columns) and "reading" not in df
    assert (dst / "checkpoints.csv").exists() and (dst / "run.json").exists()
    # a second run keeps what is done
    mtime = (dst / "checkpoints" / KEY / "result.json").stat().st_mtime_ns
    assert p1_recompute.main([str(src), "--workers", "0"]) == 0
    assert (dst / "checkpoints" / KEY / "result.json").stat().st_mtime_ns == mtime


def _same(a, b) -> bool:
    """Equal rows (or lists of rows), NaN equal to NaN, after the JSON round trip of result.json."""
    a, b = (json.loads(json.dumps(runtime.to_jsonable(x), default=str)) for x in (a, b))
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


def test_resume_upgrades_an_old_folder_in_place(tmp_path, rng, grid):
    src = tmp_path / "p1_pythia"
    sd, record = _old_job(src, rng)
    want = _current(tmp_path, record, sd)
    cfg = config.get("p1_checkpoints", smoke=True)
    fp = p1.fingerprint(cfg, SEED, set(), "grid", 25)
    with pytest.raises(SystemExit, match="stable_density"):                # other settings are still refused
        p1.check_fingerprint(src / "checkpoints", p1.fingerprint(cfg, SEED, set(), "fft", 25))
    assert p1.check_fingerprint(src / "checkpoints", fp) is True
    saved = json.loads((src / "checkpoints" / "fingerprint.json").read_text())
    assert saved["batch_order"] == "shuffled" and saved["upgraded_from"] == "consecutive"
    assert p1.check_fingerprint(src / "checkpoints", fp) is False
    old_grad = (sd / "stats" / "grad.json").read_bytes()
    redo = p1.upgrade_checkpoint(sd, SEED)
    assert sorted(redo) == ["grad", "h71_fit.hessian0", "h71_fit.random0"]
    assert not (sd / "result.json").exists() and (sd / "stats" / "consecutive" / "result.json").exists()
    assert (sd / "stats" / "consecutive" / "grad.json").read_bytes() == old_grad
    assert p1.upgrade_checkpoint(sd, SEED) == []                            # once
    # the resume then runs the tasks without stats in the pool and merges (Store.pending / complete)
    prog = p1.Progress(tmp_path)
    run = type("R", (), {"notes": [], "note": lambda self, m: self.notes.append(m)})()
    pool = p1.StatPool(lambda: {0, 1}, 1, "grid", grid, prog, {})
    store = p1.Store(tmp_path, prog, run, keep_samples=True)
    store.h71_kw = p1.h71_kw()
    c = {"key": KEY, "dir": sd, "label": "x"}
    try:
        store.register(c)
        store.pending(c, json.loads((sd / "row.json").read_text()), pool)
        pool.wait_all(store.task_done)
    finally:
        pool.shutdown()
    got = json.loads((sd / "result.json").read_text())
    assert _same(got["row"], want["row"]) and _same(got["h71_summ"], want["h71_summ"])


# ----------------------------------------------------------------------------- reported values
def test_dh_rep_is_nan_when_the_clause_fails_or_the_width_is_negative():
    ok = {"delta_h_rep": 0.4, "clause_ok": True, "D_max": 2.0, "log10_n": 4.0, "decades_r": 2.5}
    assert multifractal.reported_width(ok) == (0.4, "")
    w, why = multifractal.reported_width({**ok, "clause_ok": False, "D_max": 5.1})
    assert np.isnan(w) and "clause" in why and "5.10" in why
    w, why = multifractal.reported_width({**ok, "delta_h_rep": -4.2})
    assert np.isnan(w) and why.startswith("negative width")
    record = {"row": {"repo": "r", "step": 1, "size": "70m"}, "tasks": [{"name": "dh", "kind": "dh", "params": {}}]}
    res = p1_tasks.assemble(record, {"dh": {"result": {**ok, "delta_h_rep": -3.7, "scope": "global", "D2": 1.0}}})
    assert np.isnan(res["row"]["dh_rep"]) and res["row"]["dh_rep_raw"] == -3.7


def test_verdict_columns_use_the_shipped_reference():
    ref = h71_verdict.load_reference()
    bs = h71_verdict.bs_key(config.BATCH_SIZES)
    for n_blocks in (1600, 200):
        g = ref[(ref.n_pool == n_blocks * 1024) & (ref.n_blocks == n_blocks) & (ref.batch_sizes == bs)]
        assert len(g) == config.H71_NULL_REPS and g.replicate.nunique() == len(g)
    s = {"b_lo": -0.01, "b_hi": 0.01, "alpha_mean": 1.5, "scale_exponent": -0.33, "scale_gap_lo": -0.02,
         "scale_gap_hi": 0.03, "n_pool": 1600 * 1024, "n_blocks": 1600}
    v = h71_verdict.verdict_columns(s, config.BATCH_SIZES, wild_max=1.9)
    assert v["verdict"] == "invariant (wild)" and v["null_reps"] == config.H71_NULL_REPS
    assert 1.98 < v["alpha_null_q"] < 2.0
    v = h71_verdict.verdict_columns({**s, "n_pool": 12345}, config.BATCH_SIZES)
    assert v["verdict"] == "undecided" and "no Gaussian reference" in v["verdict_reason"]


def test_p4_pool_is_a_fixed_random_draw():
    a = p4.pool_ids("transformer", 10_000, 640, 0)
    assert len(set(a.tolist())) == 640 and (a == p4.pool_ids("transformer", 10_000, 640, 0)).all()
    assert not (a[:-1] < a[1:]).all()                                       # drawn order, not sorted
    assert sorted(p4.pool_ids("resnet18", 500, 500, 0).tolist()) == list(range(500))
    rows = p4.h71_summary_rows([{"b": 0.0, "b_lo": -0.01, "b_hi": 0.01, "reading": "invariant", "alpha_mean": 1.5,
                                 "scale_exponent": -0.33, "scale_gap_lo": -0.02, "scale_gap_hi": 0.03,
                                 "n_pool": 1600 * 1024, "n_blocks": 1600}], config.get("p4_clipping"))
    assert "reading" not in rows[0] and rows[0]["verdict"] == "invariant (wild)"
