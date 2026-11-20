"""P1 checkpoints missing on the Hub (2026-11-20): EleutherAI/pythia-410m-seed9@step40000 (commit db4323e5) holds only
tokenizer files. Such a checkpoint is skipped with a run note and listed in run.json; a resume skips it again and goes
on with the next one; every other load failure still stops the job; pass (b) of the loss-spike check compares the
flagged seeds with one seed fewer over the two intervals that adjoin the missing step."""
import json
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

import config
import p1_checkpoints as p1
from common import hf, p1_tasks

COMMIT = "db4323e5d96ae607d64df2a06154076cda03f5f7"
TOKENIZER_ONLY = [".gitattributes", "special_tokens_map.json", "tokenizer_config.json"]
COMPLETE = [".gitattributes", "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json"]
UNRECOGNIZED = "Unrecognized model in {}. Should have a `model_type` key in its config.json"


class _Run:
    def __init__(self):
        self.notes = []

    def note(self, m):
        self.notes.append(m)


@pytest.fixture
def pins(tmp_path, monkeypatch):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"EleutherAI/pythia-410m-seed9@step40000": {"commit": COMMIT}}))
    p = hf.Pins(path)
    monkeypatch.setattr(hf, "PINS", p)
    return p


# ----------------------------------------------------------------------------- point 1: which failures are skipped
def test_lacking_names_config_and_weights():
    assert p1.lacking(TOKENIZER_ONLY) == ["config.json", "weights file"]
    assert p1.lacking(["config.json", "tokenizer.json"]) == ["weights file"]
    assert p1.lacking(["model.safetensors", "tokenizer.json"]) == ["config.json"]
    assert p1.lacking(COMPLETE) == []
    assert p1.lacking(["config.json", "pytorch_model.bin"]) == []
    assert p1.lacking(["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"]) == []
    assert p1.lacking(["config.json", "training_args.bin", "sub/model.safetensors"]) == ["weights file"]


def test_load_failure_with_tokenizer_files_only_is_missing_on_hub(pins):
    run, listed = _Run(), []
    repo, rev = "EleutherAI/pythia-410m-seed9", "step40000"

    def load(r, v, run_, **kw):
        raise ValueError(UNRECOGNIZED.format(r))

    def files(r, c, run_):
        listed.append((r, c))
        return TOKENIZER_ONLY
    with pytest.raises(p1.MissingOnHub) as ei:
        p1.load_checkpoint(repo, rev, run, load, files, dtype=None, device="cpu")
    info = ei.value.info
    assert listed == [(repo, COMMIT)]                                   # the pinned commit, not the branch name
    assert p1.missing_entry(info) == {"repo": repo, "rev": rev, "commit": COMMIT, "files": TOKENIZER_ONLY}
    assert info["lacks"] == ["config.json", "weights file"] and info["load_error"].startswith("ValueError: Unrecognized")


def test_every_other_load_failure_still_stops(pins):
    run = _Run()
    repo, rev = "EleutherAI/pythia-410m-seed9", "step40000"
    err = ValueError(UNRECOGNIZED.format(repo))

    def load(r, v, run_, **kw):
        raise err
    # config.json and weights are there: the load's own error, unchanged
    with pytest.raises(ValueError) as ei:
        p1.load_checkpoint(repo, rev, run, load, lambda r, c, run_: COMPLETE)
    assert ei.value is err and not isinstance(ei.value, p1.MissingOnHub)

    # no file list from the Hub: the load's error, with a note
    def offline(r, c, run_):
        raise ConnectionError("no network")
    with pytest.raises(ValueError) as ei:
        p1.load_checkpoint(repo, rev, run, load, offline)
    assert ei.value is err and "could not be read" in run.notes[-1]

    # not pinned (no commit to look at): raised, the file list is not asked for
    def never(r, c, run_):
        raise AssertionError("file list asked for an unpinned name")
    with pytest.raises(ValueError):
        p1.load_checkpoint("EleutherAI/pythia-410m-seed9", "step41000", run, load, never)

    # a load that works never asks for the file list
    assert p1.load_checkpoint(repo, rev, run, lambda r, v, run_, **kw: ("m", "t", "hub"), never) == ("m", "t", "hub")


def test_marker_is_read_only_for_its_commit(tmp_path):
    p1_tasks.atomic_json(tmp_path / p1.MISSING_FILE, {"repo": "r", "rev": "step1", "commit": COMMIT, "files": [],
                                                      "lacks": ["config.json"]})
    assert p1.missing_marker(tmp_path, COMMIT)["rev"] == "step1"
    assert p1.missing_marker(tmp_path, "f" * 40) is None                 # another commit: measure again
    assert p1.missing_marker(tmp_path, None) is None
    assert p1.missing_marker(tmp_path / "nothing", COMMIT) is None


# ----------------------------------------------------------------------------- points 1 and 2 through main()
MISSING = ("EleutherAI/pythia-14m-seed1", "step2000")     # smoke pass (a): seeds 3 then 1, steps 1,000-4,000 -> 6/8


def _p1_main(tmp_path, monkeypatch, files_of_missing):
    """One P1 smoke run (spike passes only, probe loss only) with stand-in models, offline. Returns (run.json, the
    (repo, rev) of every model load, the commits whose file list was read)."""
    loads, listed = [], []

    def load(name, revision, run, dtype=None, device="cpu", **kw):
        loads.append((name, revision))
        if (name, revision) == MISSING:
            raise ValueError(UNRECOGNIZED.format(name))
        model = hf.neox_from_config(hf.TINY_NEOX, seed=zlib.crc32(f"{name}@{revision}".encode()) % (2 ** 31))
        return model.to(dtype).to(device).eval(), hf.ByteTokenizer(hf.TINY_NEOX["vocab_size"]), "fallback"

    def files(repo, commit, run=None):
        listed.append((repo, commit))
        return files_of_missing

    monkeypatch.setattr(hf, "load_causal_lm", load)
    monkeypatch.setattr(p1, "hub_files", files)
    monkeypatch.setattr(hf, "list_steps", lambda repo, run, default, branch_regex=None: {int(s): f"step{int(s)}"
                                                                                        for s in default})
    monkeypatch.setattr(hf, "load_texts", lambda spec, n, run, seed=0, **kw: (hf.zipf_text(n, 400, seed), "fallback"))
    for v in p1_tasks.THREAD_VARS:
        monkeypatch.setenv(v, "1")                # main() sets them for its pool; undone after the test
    monkeypatch.setattr(sys, "argv", ["p1_checkpoints.py", "--smoke", "--suite", "polypythia", "--spike-only",
                                      "--skip", "grad,esd,act,dh,htok,h71,scaling,coupling", "--device", "cpu",
                                      "--stable-density", "fft", "--out", str(tmp_path / "out")])
    p1.main()
    return json.loads((tmp_path / "out" / "run.json").read_text()), loads, listed


@pytest.fixture
def smoke_pins(tmp_path, monkeypatch):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({f"{MISSING[0]}@{MISSING[1]}": {"commit": COMMIT}}))
    monkeypatch.setattr(hf, "PINS", hf.Pins(path))


def _key_dirs(tmp_path, tag):
    return tmp_path / "out" / "checkpoints" / tag


def test_missing_checkpoint_is_skipped_with_a_note_and_no_row(tmp_path, monkeypatch, smoke_pins):
    meta, loads, listed = _p1_main(tmp_path, monkeypatch, TOKENIZER_ONLY)
    assert meta["status"] == "ok"
    assert meta["results"]["missing_on_hub"] == [{"repo": MISSING[0], "rev": MISSING[1], "commit": COMMIT,
                                                  "files": TOKENIZER_ONLY}]   # once, even if pass (b) meets it too
    notes = meta["notes"]
    hit = [m for m in notes if "missing on the Hub" in m]
    assert hit and hit[0].startswith(f"{MISSING[0]}@{MISSING[1]} [spike_a] 6/8: missing on the Hub: pinned commit "
                                     f"{COMMIT} has no config.json and no weights file")
    assert any(m.startswith(f"{MISSING[0]}@step3000 [spike_a] 7/8: GPU pass") for m in notes)   # and goes on
    assert set(listed) == {(MISSING[0], COMMIT)}
    df = pd.read_csv(tmp_path / "out" / "checkpoints.csv")
    assert not ((df.repo == MISSING[0]) & (df.revision == MISSING[1])).any()              # no row
    a = df[df["pass"] == "spike_a"]
    assert len(a) == 7 and sorted(a.seed.unique()) == [1, 3]
    marker = _key_dirs(tmp_path, "spike_a") / f"EleutherAI__pythia-14m-seed1@{MISSING[1]}" / p1.MISSING_FILE
    assert json.loads(marker.read_text())["commit"] == COMMIT


def test_resume_after_the_stop_skips_it_and_goes_on_with_the_next(tmp_path, monkeypatch, smoke_pins):
    # 1) as on the server before the fix: the job stops at 6/8 (here, because the file list shows config.json and
    #    weights, so the failure is not "missing on the Hub" and must stop the job)
    with pytest.raises(ValueError, match="Unrecognized model"):
        _p1_main(tmp_path, monkeypatch, COMPLETE)
    a_dir = _key_dirs(tmp_path, "spike_a")
    done = sorted(d.name for d in a_dir.iterdir() if (d / "result.json").exists())
    assert len(done) == 5 and not (a_dir / f"EleutherAI__pythia-14m-seed1@{MISSING[1]}" / p1.MISSING_FILE).exists()

    # 2) the resume with the fix: 1-5 read back, 6/8 skipped, 7/8 and 8/8 measured
    meta, loads, listed = _p1_main(tmp_path, monkeypatch, TOKENIZER_ONLY)
    assert meta["status"] == "ok" and meta["results"]["resumed_complete"] >= 5
    notes = meta["notes"]
    assert any(m.startswith(f"{MISSING[0]}@{MISSING[1]} [spike_a] 6/8: missing on the Hub") for m in notes)
    gpu = [m.split(": GPU pass")[0] for m in notes if ": GPU pass" in m and "[spike_a]" in m]
    assert gpu == [f"{MISSING[0]}@step3000 [spike_a] 7/8", f"{MISSING[0]}@step4000 [spike_a] 8/8"]
    assert loads[0] == MISSING                                    # the first model loaded is the one that stopped
    first = pd.read_csv(tmp_path / "out" / "checkpoints.csv")

    # 3) a second resume: skipped again from its missing_on_hub.json, without loading it or asking the Hub
    meta2, loads2, listed2 = _p1_main(tmp_path, monkeypatch, TOKENIZER_ONLY)
    assert MISSING not in loads2 and listed2 == []
    assert any("6/8: missing on the Hub" in m and "recorded in an earlier run" in m for m in meta2["notes"])
    assert meta2["results"]["missing_on_hub"] == meta["results"]["missing_on_hub"]
    assert not any(": GPU pass" in m for m in meta2["notes"])     # everything else is complete
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "out" / "checkpoints.csv"), first)


# ----------------------------------------------------------------------------- point 3: pass (b) with a missing seed
def _pass_b_rows(missing=(9, 40000), jump=(3, 43000, 1.0)):
    """Pass (b) rows of the nine 410M seeds in the window 35,000-45,000, one seed absent at one step. The flagged
    seeds 3 and 4 move less than the others, except for seed 3's jump."""
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(1, 10):
        level = 0.0
        for st in range(35000, 45001, 1000):
            level += rng.normal(0, 0.001 if seed in (3, 4) else 0.01)
            if (seed, st) == jump[:2]:
                level += jump[2]
            if (seed, st) == missing:
                continue
            rows.append({"suite": "polypythia", "size": "410m", "seed": seed, "step": st, "pass": "spike_b",
                         "spike_window": "35000-45000", "alpha": 1.8 + level, "alpha_esd": 3.0 + level,
                         "K": 5.0 + level})
    return rows


def test_pass_b_with_one_seed_missing_at_one_step():
    sc = config.get("p1_checkpoints")["spike_check"]
    assert sc["flagged_seeds"] == [3, 4]
    out = pd.DataFrame(p1.spike_b_rows(_pass_b_rows(), [(35000, 45000, [(39000, 40000)])], sc))
    assert set(out.quantity) == {"alpha", "alpha_esd", "K"} and set(out.seed) == {3, 4}
    near = out.step_to.isin([40000, 41000])                       # 39,000->40,000 and 40,000->41,000
    assert (out[near].others_n == 6).all() and (out[~near].others_n == 7).all()
    assert len(out) == 3 * 2 * 10                                 # every interval of both flagged seeds is there
    assert np.isfinite(out.robust_z).all()
    hit = out[out.jump]                                           # seed 3's jump is still found, nothing else
    assert set(zip(hit.seed, hit.step_to)) == {(3, 43000)} and len(hit) == 3
    # without the missing checkpoint the same seven others everywhere
    full = pd.DataFrame(p1.spike_b_rows(_pass_b_rows(missing=(0, 0)), [(35000, 45000, [(39000, 40000)])], sc))
    assert (full.others_n == 7).all()
    same = out[~near].reset_index(drop=True)
    ref = full[~full.step_to.isin([40000, 41000])].reset_index(drop=True)
    pd.testing.assert_frame_equal(same, ref)                      # the other intervals do not change
