"""P1's CPU statistics in a process pool: one H7.1 checkpoint per stage, the pool's CPUs, task seeds that do not
depend on order, results merged as before, resume and the settings check."""
import json
import time

import numpy as np
import pytest

import config
import p1_checkpoints as p1
from common import batch, h71_verdict, p1_tasks, stable

PYTHIA = {s: f"step{s}" for s in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000))}


def test_h71_runs_once_per_stage():
    cfg = config.get("p1_checkpoints")
    st = p1.h71_stages(cfg, "pythia", "14m", PYTHIA)
    assert st == {1000: 0.01, 14000: 0.1, 71000: 0.5, 143000: 1.0} or st == {1000: 0.01, 14000: 0.1, 72000: 0.5,
                                                                             143000: 1.0}
    chosen = [s for s, _ in p1.hf.nearest_steps(PYTHIA, p1.wanted_steps(cfg, "pythia", "14m", PYTHIA))]
    assert set(st) <= set(chosen)                                           # every stage step is read
    old = [s for s in chosen if any(abs(s / 143000 - x) < 0.02 or (x >= 0.999 and s == 143000) for x in cfg["stages"])]
    assert len(old) == 10 and len(st) == 4                                   # the rule before 2026-11-16: ten
    assert p1.h71_stages(cfg, "pythia", "410m", PYTHIA) == {}
    assert p1.h71_stages(cfg, "polypythia", "14m", PYTHIA) == {}


def test_cpus_follow_the_rule(tmp_path):
    assert p1.parse_cpus("0-3,7") == {0, 1, 2, 3, 7} and p1.fmt_cpus({0, 1, 2, 3, 7, 9, 10}) == "0-3,7,9-10"
    (tmp_path / "logs").mkdir()
    prog = tmp_path / "logs" / "progress.txt"
    rule = p1.cpu_rule("p3_fractional=0-10", {7, 8, 9, 10}, root=tmp_path)
    assert rule() == {7, 8, 9, 10}
    prog.write_text("2026-11-15T00:00:00Z  START  p3_fractional  total 1.00 h\n"
                    "2026-11-15T05:00:00Z  END    p3_fractional  exit 1  elapsed 5.00 h  total 6.00 h\n"
                    "2026-11-16T00:56:00Z  START  p3_fractional  total 6.00 h\n")
    assert rule() == {7, 8, 9, 10}                                          # an earlier END, then started again
    with prog.open("a") as f:
        f.write("2026-11-17T00:00:00Z  END    p3_fractional  exit 0  elapsed 23.0 h  total 29.0 h\n")
    assert rule() == set(range(11))
    (tmp_path / "logs" / "p1_cpus").write_text("2-5\n")
    assert rule() == {2, 3, 4, 5}                                            # the file wins


def test_seeds_depend_on_names_only():
    a = p1_tasks.seed_for(5, "grid/EleutherAI__pythia-14m@step1000", "grad")
    assert a == p1_tasks.seed_for(5, "grid/EleutherAI__pythia-14m@step1000", "grad")
    assert a != p1_tasks.seed_for(5, "grid/EleutherAI__pythia-14m@step1000", "htok")
    assert p1_tasks.rng_for(a).integers(1 << 30) == p1_tasks.rng_for(list(a)).integers(1 << 30)


def _checkpoint(tmp_path, rng):
    """A checkpoint folder as the GPU pass leaves it: samples and row.json's record."""
    d = tmp_path / "ck"
    (d / "samples").mkdir(parents=True)
    pool = stable.rvs_stable(1.6, 0.0, 1.0, 0.0, 30 * 32, rng)
    p1_tasks.atomic_npy(d / "samples" / "h71.npy", np.stack([pool, rng.normal(size=len(pool))], 1).astype(np.float32))
    p1_tasks.atomic_npy(d / "samples" / "grad.npy", stable.rvs_stable(1.7, 0, 1, 0, (400, 2), rng).astype(np.float32))
    p1_tasks.atomic_npy(d / "samples" / "htok.npy", rng.normal(size=4096).astype(np.float32))
    ev = [np.sort(rng.pareto(2.0, 64) + 1.0), np.sort(rng.pareto(3.0, 32) + 1.0)]
    p1_tasks.atomic_npz(d / "samples" / "esd.npz", names=np.array(["a", "b"]), shapes=np.array([[64, 128], [32, 64]]),
                        ev_0=ev[0], ev_1=ev[1])
    p1_tasks.atomic_npz(d / "samples" / "act.npz", sub_0=np.abs(rng.standard_t(2, 5000)) + 1e-3)
    p1_tasks.atomic_npy(d / "samples" / "dh.npy", rng.normal(size=(300, 3)).astype(np.float32))
    key, seed = "grid/x@step1", 7
    T = {**config.TAILS, "n_boot": 20, "hill_reps": 20}

    def t(name, kind, params):
        return {"name": name, "kind": kind, "params": params, "seed": p1_tasks.seed_for(seed, key, name)}
    tasks = [t("esd", "esd", {}), t("act", "act", {"n_layers": 2}), t("dh", "dh", {"q_lo": 0.5, "q_hi": 4.0}),
             t("htok", "htok", {"n_surr": 9, "wavelet_j1": 3}),
             t("grad", "grad", {"B": 8, "n_blocks": 50, "T": T}),
             t("h71_fit.random0", "h71_fit", {"j": 0, "Bs": [8, 16, 32], "nb": 30}),
             t("h71_boot.random0.0", "h71_boot", {"j": 0, "Bs": [8, 16, 32], "nb": 30, "n_boot": 2}),
             t("h71_boot.random0.1", "h71_boot", {"j": 0, "Bs": [8, 16, 32], "nb": 30, "n_boot": 2})]
    row = {"suite": "pythia", "size": "14m", "repo": "x", "step": 1, "commit": "c", "pass": "grid", "probe_loss": 3.0,
           "eval_loss": 2.9, "gpu_pass_seconds": 1.0}
    moments = [{"K": 1.0, "Kurt0": 4.0, "n_outlier_channels": 1, "max_abs": 9.0},
               {"K": 3.0, "Kurt0": 6.0, "n_outlier_channels": 0, "max_abs": 2.0}]
    record = {"row": row, "tasks": tasks, "act_moments": moments,
              "h71": {"directions": ["random0"], "stage": 0.01, "frac": 0.007, "nb": 30, "Bs": [8, 16, 32],
                      "n_pool": len(pool), "n_boot": 4, "n_chunks": 2}}
    return d, key, record


def test_tasks_merge_into_the_rows_written_before(tmp_path, rng):
    d, key, record = _checkpoint(tmp_path, rng)
    stats = {t["name"]: p1_tasks.run_task({**t, "dir": str(d)}) for t in reversed(record["tasks"])}
    assert all(s["error"] is None for s in stats.values()), {k: s["error"] for k, s in stats.items()}
    assert all(p1_tasks.stats_path(d, n).exists() for n in stats)
    again = p1_tasks.run_task({**record["tasks"][4], "dir": str(d)})              # grad, run a second time
    assert again["result"] == stats["grad"]["result"]
    res = p1_tasks.assemble(record, stats, {"level": 0.9, "inv_tol": 0.03, "drift_tol": 0.05})
    row = res["row"]
    assert list(row)[-2:] == ["eval_loss", "gpu_pass_seconds"] and row["K"] == 2.0 and row["K_max"] == 3.0
    assert np.isfinite(row["alpha_esd"]) and np.isfinite(row["act_zeta"]) and np.isfinite(row["H_dep"])
    assert "dh_rep" in row and "alpha" in row and row["n_blocks"] == 50
    assert len(res["esd_rows"]) == 2 and len(res["act_rows"]) == 2 and len(res["h71_rows"]) == 3
    s = res["h71_summ"][0]
    assert s["stage"] == 0.01 and s["n_boot"] <= 4 and s["n_boot_requested"] == 4
    # the summary of the chunks is the summary of their replicates together
    b = stats["h71_boot.random0.0"]["result"]["b"] + stats["h71_boot.random0.1"]["result"]["b"]
    g = stats["h71_boot.random0.0"]["result"]["gap"] + stats["h71_boot.random0.1"]["result"]["gap"]
    ref = batch.h71_summary(stats["h71_fit.random0"]["result"]["per_B"], b, g, record["h71"]["n_pool"], [8, 16, 32], 30)
    ref.pop("reading")                          # slope only: the verdict's slope_reading replaces it (2026-11-17)
    assert {k: v for k, v in s.items() if k in ref} == pytest.approx(ref, nan_ok=True)
    assert "reading" not in s and s["verdict"] in h71_verdict.VERDICTS and s["batch_order"] == "consecutive"


def test_a_failed_statistic_is_recorded_and_a_bug_stops(tmp_path, rng):
    d, key, record = _checkpoint(tmp_path, rng)
    (d / "samples" / "dh.npy").write_bytes(b"not an array")                  # ValueError: recorded
    rec = p1_tasks.run_task({**record["tasks"][2], "dir": str(d)})
    assert rec["result"] is None and rec["error"].startswith(("ValueError", "UnpicklingError", "OSError"))
    res = p1_tasks.assemble(record, {"dh": rec}, None)
    assert "error_dh" in res["row"] and "dh" in res["errors"]
    with pytest.raises(KeyError):                                             # a bug: raised
        p1_tasks.run_task({**record["tasks"][0], "kind": "esd", "params": {}, "dir": str(d),
                           "name": "esd2"} | {"kind": "nonexistent"})


def test_pool_runs_tasks_and_store_resumes(tmp_path, rng):
    d, key, record = _checkpoint(tmp_path, rng)
    prog = p1.Progress(tmp_path)
    run = type("R", (), {"notes": [], "note": lambda self, m: self.notes.append(m)})()
    pool = p1.StatPool(lambda: {0, 1}, 0, "fft", None, prog, {})
    store = p1.Store(tmp_path, prog, run, keep_samples=False)
    store.h71_kw = {}
    c = {"key": key, "dir": d, "label": "x@step1"}
    try:
        store.register(c)
        p1_tasks.atomic_json(d / "row.json", record)
        store.pending(c, record, pool)
        pool.wait_all(store.task_done)
    finally:
        pool.shutdown()
    assert (d / "result.json").exists() and not (d / "samples" / "dh.npy").exists()   # big samples removed
    assert (d / "samples" / "h71.npy").exists()
    lines = (tmp_path / "timings.csv").read_text().splitlines()
    assert lines[0] == "utc,checkpoint,measurement,where,seconds" and len(lines) == 1 + len(record["tasks"])
    first = json.loads((d / "result.json").read_text())
    rows = (tmp_path / "checkpoints.csv").read_text()
    assert "probe_loss" in rows and (tmp_path / "h71_summary.csv").exists()
    # a restart: the folder's result.json is read back, nothing runs again
    store2 = p1.Store(tmp_path, prog, run, keep_samples=False)
    store2.register(c)
    assert store2.load_result(c)["row"] == first["row"]


def test_resume_refuses_other_settings(tmp_path):
    cfg = config.get("p1_checkpoints", smoke=True)
    fp = p1.fingerprint(cfg, 1, set(), "grid", 25)
    p1.check_fingerprint(tmp_path / "checkpoints", fp)
    p1.check_fingerprint(tmp_path / "checkpoints", fp)                        # same settings: fine
    with pytest.raises(SystemExit, match="stable_density"):
        p1.check_fingerprint(tmp_path / "checkpoints", p1.fingerprint(cfg, 1, set(), "fft", 25))
    with pytest.raises(SystemExit, match="config_sha256"):
        p1.check_fingerprint(tmp_path / "checkpoints", p1.fingerprint(config.get("p1_checkpoints"), 1, set(), "grid", 25))
