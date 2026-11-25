"""OLMo 2 7B branch names (2026-11-25): p1_olmo7b failed on 2026-11-24 in hf.list_steps with "no step branches",
because the olmo7b branch_regex expected "step{N}-tokens{T}B" while allenai/OLMo-2-1124-7B names its pretraining
branches "stage1-step{N}-tokens{T}B" like OLMo 2 1B (a few early ones once lacked the "stage1-" prefix). The regex
now takes both forms and still leaves out stage-2 and soup branches."""
import re
import types

import huggingface_hub
import pytest

import config
import p1_checkpoints as p1
from common import hf

REPO = "allenai/OLMo-2-1124-7B"
FINAL = 928646
OLD_REGEX = r"^step(\d+)-tokens"


def _tokens_b(step: int) -> int:
    return round(step * 1024 * 4096 / 1e9)             # 4.19M tokens per step; only for realistic-looking names


def _branches(unprefixed_below: int = 0) -> list[str]:
    """Stage-1 checkpoints every 1,000 steps plus the final one, the early ones (step < unprefixed_below) without the
    "stage1-" prefix; stage-2 ingredients, a soup-like name and main, which must not match."""
    steps = list(range(1000, FINAL, 1000)) + [FINAL]
    names = [("" if s < unprefixed_below else "stage1-") + f"step{s}-tokens{_tokens_b(s)}B" for s in steps]
    names += [f"stage2-ingredient{i}-step{s}-tokens{round(s * 4.19 / 1000)}B"
              for i in (1, 2, 3) for s in (1000, 5000, 11931)]
    names += ["stage2-ingredient1-step11931-tokens50B-soup", "main"]
    return names


@pytest.fixture
def refs(monkeypatch):
    """hf.list_steps reads huggingface_hub.list_repo_refs; this returns the branch list set by the test."""
    state = {"names": []}

    def list_repo_refs(repo, **kw):
        assert repo == REPO
        return types.SimpleNamespace(branches=[types.SimpleNamespace(name=n, target_commit="0" * 40)
                                               for n in state["names"]])
    monkeypatch.setattr(huggingface_hub, "list_repo_refs", list_repo_refs)
    return state


def _run():
    return types.SimpleNamespace(offline_ok=False, smoke=False, notes=[], note=lambda m: None,
                                 fallback=lambda m: None)


def _regex():
    return config.get("p1_checkpoints")["suites"]["olmo7b"]["branch_regex"]


@pytest.mark.parametrize("unprefixed_below", [0, 5000], ids=["all prefixed", "early ones unprefixed"])
def test_olmo7b_regex_takes_stage1_branches_in_both_forms(refs, unprefixed_below):
    refs["names"] = _branches(unprefixed_below)
    avail = hf.list_steps(REPO, _run(), [1], branch_regex=_regex())
    stage1 = [n for n in refs["names"] if re.fullmatch(r"(stage1-)?step\d+-tokens\d+B", n)]
    assert sorted(avail.values()) == sorted(stage1)                       # every stage-1 branch, nothing else
    assert max(avail) == FINAL and avail[FINAL] == f"stage1-step{FINAL}-tokens{_tokens_b(FINAL)}B"
    assert not any(v.startswith("stage2") or v == "main" for v in avail.values())
    assert avail[1000] == ("step1000-tokens4B" if unprefixed_below else "stage1-step1000-tokens4B")
    assert len(avail) == len(range(1000, FINAL, 1000)) + 1


def test_the_old_regex_found_no_branch_as_on_the_server(refs):
    refs["names"] = _branches(0)
    with pytest.raises(RuntimeError, match="no step branches"):
        hf.list_steps(REPO, _run(), [1], branch_regex=OLD_REGEX)


@pytest.mark.parametrize("unprefixed_below", [0, 5000])
def test_olmo7b_step_fracs_pick_stage1_checkpoints(refs, unprefixed_below):
    cfg = config.get("p1_checkpoints")
    refs["names"] = _branches(unprefixed_below)
    avail = hf.list_steps(REPO, _run(), [1], branch_regex=_regex())
    wanted = p1.wanted_steps(cfg, "olmo7b", "7b", avail)
    assert cfg["suites"]["olmo7b"]["step_fracs"] == [0.002, 0.01, 0.1, 0.5, 1.0]
    assert wanted == [1857, 9286, 92865, 464323, FINAL]                   # fractions of the final stage-1 step
    chosen = hf.nearest_steps(avail, wanted)
    assert [s for s, _ in chosen] == [2000, 9000, 93000, 464000, FINAL]
    assert chosen[0][1] == ("step2000-tokens8B" if unprefixed_below else "stage1-step2000-tokens8B")
    assert all(rev.startswith("stage1-") for _, rev in chosen[1:])
    assert p1.h71_stages(cfg, "olmo7b", "7b", avail) == {}                # H7.1 only for Pythia
    for _, rev in chosen:                                                 # pins.json keys use the branch name
        assert hf.Pins.key(REPO, rev) == f"{REPO}@{rev}"


def test_other_suites_keep_their_regex():
    s = config.get("p1_checkpoints")["suites"]
    assert s["olmo"]["branch_regex"] == r"^stage1-step(\d+)-tokens"
    assert "branch_regex" not in s["pythia"] and "branch_regex" not in s["polypythia"]
    rx = _regex()
    for name, step in [("stage1-step928646-tokens3896B", 928646), ("step1000-tokens4B", 1000)]:
        assert int(re.match(rx, name).group(1)) == step
    for name in ["stage2-ingredient1-step11931-tokens50B", "stage2-ingredient2-step1000-tokens5B-soup", "main",
                 "step1000", "stage1-step1000"]:
        assert re.match(rx, name) is None
