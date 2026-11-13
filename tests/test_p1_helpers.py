"""P1 bookkeeping: commit pins, step choice, the name-switch checks (date and content), the loss-spike check,
seed variance."""
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


def test_wanted_steps_and_scan_grid():
    cfg = config.get("p1_checkpoints")
    avail = {s: f"step{s}" for s in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000))}
    w = p1.wanted_steps(cfg, "polypythia", "410m", avail)
    assert w == sorted(cfg["steps"])                                           # no spike steps in the main grid
    sc = cfg["spike_check"]
    steps = p1.scan_steps(avail, sc)
    assert steps[0] == 1000 and steps[-1] == 143000 and len(steps) == 143      # every 1,000-step checkpoint
    olmo = {s: f"stage1-step{s}-tokens{s // 250}B" for s in range(0, 1_900_001, 10_000)}
    wo = p1.wanted_steps(cfg, "olmo", "1b", olmo)
    assert max(wo) == 1_900_000 and min(wo) >= 1
    assert "olmo7b" not in p1.default_suites(cfg) and "olmo" in p1.default_suites(cfg)


def _loss_frame(spike_seed=3, spike_at=23000, size=0.5, seeds=(1, 3, 4)):
    rng = np.random.default_rng(1)
    rows = []
    for seed in seeds:
        for st in range(1000, 61001, 1000):
            loss = 2.5 + 8.0 / np.sqrt(st / 1000) + rng.normal(0, 0.005)
            if seed == spike_seed and spike_at <= st < spike_at + 2000:        # spike, recovered 2,000 steps later
                loss += size
            rows.append({"suite": "polypythia", "size": "410m", "seed": seed, "step": st, "probe_loss": loss})
    return pd.DataFrame(rows)


def test_loss_jumps_find_the_spike_onset_net_of_the_stable_seed():
    sc = {"suite": "polypythia", "size": "410m", "flagged_seeds": [3, 4], "reference_seeds": [1], "robust_z": 3.5}
    out = pd.DataFrame(p1.loss_jumps(_loss_frame(), sc))
    hit = out[out.jump]
    assert list(zip(hit.seed, hit.step_from, hit.step_to)) == [(3, 22000, 23000)]   # the onset, not the recovery
    assert not out[(out.seed == 4)].jump.any()                                     # the steep early descent is no jump
    assert not pd.DataFrame(p1.loss_jumps(_loss_frame(size=0.0), sc)).jump.any()


def test_merge_windows():
    assert p1.merge_windows([(22000, 23000)], 5000) == [(17000, 28000, [(22000, 23000)])]
    w = p1.merge_windows([(30000, 31000), (22000, 23000), (60000, 61000)], 5000)
    assert w == [(17000, 36000, [(22000, 23000), (30000, 31000)]), (55000, 66000, [(60000, 61000)])]
    assert p1.merge_windows([], 5000) == []


def test_weights_bytes_prefers_safetensors():
    f = lambda n, b: types.SimpleNamespace(rfilename=n, size=b)  # noqa: E731
    assert hf.weights_bytes([f("model.safetensors", 911), f("pytorch_model.bin", 911), f("tokenizer.json", 2)]) == 911
    assert hf.weights_bytes([f("pytorch_model.bin", 500), f("config.json", 1)]) == 500
    assert hf.weights_bytes([f("config.json", 1)]) is None


def test_name_switch_flag():
    cfg = config.get("p1_checkpoints")
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", "2025-11-01T00:00:00").startswith("CHECK")
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-31m", "2026-02-15T00:00:00").startswith("CHECK")
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", "2026-02-23T23:59:59").startswith("CHECK")  # before 24 Feb
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", "2026-02-24T10:00:00") == "after switch"  # re-creation day
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-31m", "2026-02-27T09:00:00") == "after switch"
    assert p1.name_switch_flag(cfg, "EleutherAI/pythia-14m", "2026-03-02T00:00:00") == "after switch"
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


def test_pass_b_spike_check_finds_a_jump_only_where_it_is():
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


DEDUP = "c" * 64


def _lookup(table, calls=None):
    """Stand-in for hf.hub_lfs: (repo, revision) -> (commit, {file: sha256}); a missing key raises like the Hub."""
    def lookup(repo, rev):
        if calls is not None:
            calls.append((repo, rev))
        if (repo, rev) not in table:
            raise FileNotFoundError(f"{repo}@{rev}")
        return table[(repo, rev)]
    return lookup


def test_dedup_content_flag_flags_only_identical_weights():
    cfg = config.get("p1_checkpoints")
    repo, dd = "EleutherAI/pythia-14m", "EleutherAI/pythia-14m-deduped"
    same = _lookup({(repo, SHA): (SHA, {"model.safetensors": DEDUP, "tokenizer.json": "t"}),
                    (dd, "step1000"): ("d" * 40, {"model.safetensors": DEDUP})})
    r = p1.dedup_content_flag(cfg, repo, "step1000", SHA, same)
    assert r["dedup_check"].startswith("CHECK") and dd in r["dedup_check"]
    assert r["dedup_commit"] == "d" * 40 and r["weights_sha256"] == r["dedup_sha256"] == DEDUP
    diff = _lookup({(repo, SHA): (SHA, {"model.safetensors": "e" * 64}),
                    (dd, "step1000"): ("d" * 40, {"model.safetensors": DEDUP})})
    assert p1.dedup_content_flag(cfg, repo, "step1000", SHA, diff)["dedup_check"].startswith("differs")
    # the pinned commit, not the moving branch name, is looked up on our side
    assert p1.dedup_content_flag(cfg, repo, "step1000", None, diff)["dedup_check"].startswith("unknown")


def test_dedup_content_flag_unknown_without_a_common_file_or_branch():
    cfg = config.get("p1_checkpoints")
    repo, dd = "EleutherAI/pythia-31m", "EleutherAI/pythia-31m-deduped"
    bin_only = _lookup({(repo, SHA): (SHA, {"model.safetensors": DEDUP}),
                        (dd, "step1"): ("d" * 40, {"pytorch_model.bin": DEDUP})})
    r = p1.dedup_content_flag(cfg, repo, "step1", SHA, bin_only)
    assert r["dedup_check"].startswith("unknown: no weights file in common")
    both_bin = _lookup({(repo, SHA): (SHA, {"pytorch_model.bin": DEDUP}),
                        (dd, "step1"): ("d" * 40, {"pytorch_model.bin": DEDUP})})
    r = p1.dedup_content_flag(cfg, repo, "step1", SHA, both_bin)      # second file of content_files
    assert r["dedup_check"].startswith("CHECK") and r["weights_file"] == "pytorch_model.bin"
    no_branch = _lookup({(repo, SHA): (SHA, {"model.safetensors": DEDUP})})
    assert p1.dedup_content_flag(cfg, repo, "step1", SHA, no_branch)["dedup_check"].startswith("unknown: Hub lookup")
    assert p1.dedup_content_flag(cfg, "EleutherAI/pythia-70m", "step1", SHA, no_branch) == {}


def test_content_check_is_cached_in_pins(tmp_path):
    cfg = config.get("p1_checkpoints")
    repo, dd = "EleutherAI/pythia-14m", "EleutherAI/pythia-14m-deduped"
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({f"{repo}@step1000": {"commit": SHA, "commit_date": "2026-02-24T00:00:00+00:00"}}))
    calls = []
    lookup = _lookup({(repo, SHA): (SHA, {"model.safetensors": "e" * 64}),
                      (dd, "step1000"): ("d" * 40, {"model.safetensors": DEDUP})}, calls)
    first = p1.content_check(cfg, repo, "step1000", hf.Pins(path), lookup)
    assert first["dedup_check"].startswith("differs") and len(calls) == 2
    stored = json.loads(path.read_text())[f"{repo}@step1000"]["deduped_check"]
    assert stored["dedup_check"] == first["dedup_check"] and stored["dedup_sha256"] == DEDUP
    again = p1.content_check(cfg, repo, "step1000", hf.Pins(path), lookup)
    assert again == first and len(calls) == 2                       # read from pins.json, no Hub call
    failing = _lookup({})
    p1.content_check(cfg, repo, "step2000", hf.Pins(path), failing)  # unpinned and unreachable: not cached
    assert "EleutherAI/pythia-14m@step2000" not in json.loads(path.read_text())


def test_lfs_sha256s_reads_hub_metadata():
    lfs = types.SimpleNamespace
    sib = [lfs(rfilename="model.safetensors", lfs={"sha256": DEDUP, "size": 1, "pointer_size": 1}),
           lfs(rfilename="pytorch_model.bin", lfs=lfs(sha256="f" * 64, size=1, pointer_size=1)),
           lfs(rfilename="config.json", lfs=None)]
    assert hf.lfs_sha256s(sib) == {"model.safetensors": DEDUP, "pytorch_model.bin": "f" * 64}
    assert hf.compare_weights({"model.safetensors": DEDUP}, {"model.safetensors": DEDUP}) == ("model.safetensors", True)
    assert hf.compare_weights({"model.safetensors": DEDUP}, {}) == (None, None)
