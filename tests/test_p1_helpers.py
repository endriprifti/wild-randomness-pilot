"""P1 bookkeeping: commit pins, step choice, the name-switch flag, the loss-spike check, seed variance."""
import json
import types

import numpy as np
import pandas as pd
import pytest

import config
import p1_checkpoints as p1
from common import hf

SHA = "a" * 40


def test_pins_reuse_recorded_commit_and_require_pins(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"EleutherAI/pythia-14m@step1000": {"commit": SHA, "commit_date": "2026-03-01T00:00:00+00:00"}}))
    pins = hf.Pins(path)
    run = types.SimpleNamespace(pins={}, require_pins=True)
    assert pins.resolve("EleutherAI/pythia-14m", "step1000", run) == SHA       # no Hub call: already pinned
    assert run.pins == {"EleutherAI/pythia-14m@step1000": SHA}
    assert pins.resolve("EleutherAI/pythia-14m", "b" * 40, run) == "b" * 40    # a commit is its own pin
    with pytest.raises(KeyError):
        pins.resolve("EleutherAI/pythia-14m", "step2000", run)                # unpinned under --require-pins


def test_pins_save_merges_with_file(tmp_path):
    path = tmp_path / "pins.json"
    a, b = hf.Pins(path), hf.Pins(path)
    a.data["x@main"] = {"commit": SHA}
    a.save()
    b.data["y@main"] = {"commit": SHA}
    b.save()                                                                  # must not drop x
    assert set(json.loads(path.read_text())) == {"x@main", "y@main"}


def test_wanted_steps_spike_grid_and_olmo_fractions():
    cfg = config.get("p1_checkpoints")
    avail = {s: f"step{s}" for s in range(0, 143001, 1000)}
    w, only = p1.wanted_steps(cfg, "polypythia", "410m", avail)
    assert set(cfg["spike_check"]["extra_steps"]) <= set(w)
    assert 64000 not in only and 3000 in only                                 # regular steps stay full reads
    w14, only14 = p1.wanted_steps(cfg, "polypythia", "14m", avail)
    assert not only14 and 3000 not in w14
    olmo = {s: f"stage1-step{s}-tokens{s // 250}B" for s in range(0, 1_900_001, 10_000)}
    wo, _ = p1.wanted_steps(cfg, "olmo", "1b", olmo)
    assert max(wo) == 1_900_000 and min(wo) >= 1
    assert "olmo7b" not in p1.default_suites(cfg) and "olmo" in p1.default_suites(cfg)


def test_name_switch_flag():
    cfg = config.get("p1_checkpoints")
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", "2025-11-01T00:00:00").startswith("CHECK")
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-31m", "2026-02-15T00:00:00") == "after switch"
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-70m", "2020-01-01") == ""
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", None).startswith("unknown")


def _spike_frame(jump_seed=3, jump_at=5000, size=1.0):
    rng = np.random.default_rng(0)
    steps = [1000, 2000, 3000, 5000, 6000, 12000]
    rows = []
    for seed in range(1, 10):
        level = 0.0
        for st in steps:
            level += rng.normal(0, 0.01)
            if seed == jump_seed and st == jump_at:
                level += size
            rows.append({"suite": "polypythia", "size": "410m", "seed": seed, "step": st, "alpha_esd": 3 + level})
    return pd.DataFrame(rows)


def test_spike_check_finds_a_jump_only_where_it_is():
    sc = {"suite": "polypythia", "size": "410m", "flagged_seeds": [3, 4], "quantities": ["alpha_esd", "K"], "robust_z": 3.5}
    out = pd.DataFrame(p1.spike_check(_spike_frame(), sc))
    hit = out[out.jump]
    assert len(hit) == 1 and hit.iloc[0].seed == 3 and hit.iloc[0].step_from == 3000 and hit.iloc[0].step_to == 5000
    assert (out.others_n == 7).all() and set(out.quantity) == {"alpha_esd"}   # K absent: skipped
    quiet = pd.DataFrame(p1.spike_check(_spike_frame(size=0.0), sc))
    assert not quiet.jump.any()


def test_seed_variance_compares_variants_at_shared_steps():
    rows = []
    for var, sd in (("joint", 1.0), ("data", 0.5), ("weight", 2.0)):
        for seed, v in enumerate([-1, 0, 1] if var != "joint" else [-1, -0.5, 0, 0.5, 1]):
            rows.append({"variant": var, "size": "160m", "step": 1000, "seed": seed, "K": sd * v})
    rows.append({"variant": "joint", "size": "410m", "step": 1000, "seed": 0, "K": 1.0})   # one variant only
    out = pd.DataFrame(p1.seed_variance(pd.DataFrame(rows)))
    assert set(out["size"]) == {"160m"}
    r = out.set_index("variant")
    assert r.loc["joint", "sd_ratio_to_joint"] == pytest.approx(1.0)
    assert r.loc["data", "n_seeds"] == 3
    joint_sd = np.std([-1, -0.5, 0, 0.5, 1], ddof=1)
    assert r.loc["weight", "sd"] == pytest.approx(2.0)
    assert r.loc["data", "sd_ratio_to_joint"] == pytest.approx(0.5 / joint_sd)
