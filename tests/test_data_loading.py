"""Dataset loading: explicit data_files for the Pile's top-level validation file, retries only for transient
errors, smoke runs that fall back only when the Hub is unreachable, and check_data.py's list of splits."""
import json
import re
from pathlib import Path

import pytest

import check_data
from common import hf


class StubRun:
    def __init__(self, offline_ok=False):
        self.offline_ok, self.require_pins = offline_ok, False
        self.notes, self.fallbacks, self.pins = [], [], {}

    def note(self, m):
        self.notes.append(m)

    def fallback(self, m):
        self.fallbacks.append(m)


def _pile_like(root: Path) -> Path:
    """monology/pile-uncopyrighted's layout: train/00.jsonl.zst, val.jsonl.zst and test.jsonl.zst at the top."""
    import zstandard
    def write(path, prefix, n):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps({"text": f"{prefix} document {i} " + "x" * 300, "meta": {}}) + "\n"
                        for i in range(n))
        path.write_bytes(zstandard.ZstdCompressor().compress(lines.encode()))
    write(root / "train" / "00.jsonl.zst", "train", 30)
    write(root / "val.jsonl.zst", "val", 20)
    write(root / "test.jsonl.zst", "test", 20)
    return root


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(hf.PINS, "resolve", lambda *a, **k: None)     # local folders have no Hub commit
    monkeypatch.setattr(hf, "HUB_RETRIES", (0, 0))


def test_builder_misses_the_top_level_validation_file_and_data_files_fix_it(tmp_path, local):
    repo = _pile_like(tmp_path / "pile")
    from datasets import load_dataset
    with pytest.raises(ValueError, match="Bad split: validation"):         # the L40S error of 2026-11-16
        next(iter(load_dataset(str(repo), split="validation", streaming=True)))
    run = StubRun()
    texts, src = hf.load_texts({"repo": str(repo), "split": "validation",
                                "data_files": {"validation": "val.jsonl.zst"}}, 5, run)
    assert src == "hub" and len(texts) == 5 and all(t.startswith("val document") for t in texts)
    train, _ = hf.load_texts({"repo": str(repo), "split": "train", "data_files": {"train": "train/*.jsonl.zst"}}, 3, run)
    assert all(t.startswith("train document") for t in train)


def test_the_full_p1_config_reads_validation_from_val_jsonl_zst():
    import config
    cfg = config.get("p1_checkpoints")
    ev, pool = hf.data_spec(cfg["eval_dataset"]), hf.data_spec(cfg["pool_dataset"])
    assert ev.split == "validation" and ev.data_files == {"validation": "val.jsonl.zst"}
    assert pool.data_files == {"train": "train/*.jsonl.zst"} and ev.revision == "main"
    with pytest.raises(hf.DataSpecError):
        hf.data_spec({"repo": "x", "split": "validation", "data_files": {"train": "a"}})


def test_only_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr(hf, "HUB_RETRIES", (0, 0, 0, 0))
    calls = []

    def bad_split():
        calls.append(1)
        raise ValueError("Bad split: validation. Available splits: ['train']")
    with pytest.raises(ValueError):
        hf.hub_retry(bad_split, StubRun())
    assert len(calls) == 1                                                   # no retry
    calls.clear()

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("Connection reset by peer")
        return "ok"
    assert hf.hub_retry(flaky, StubRun()) == "ok" and len(calls) == 3


def test_transient_classification():
    class Resp:
        def __init__(self, code):
            self.status_code = code

    class HfHubHTTPError(OSError):
        def __init__(self, code):
            super().__init__(f"{code} error")
            self.response = Resp(code)

    class ProxyError(Exception):            # httpx's ProxyError: a refusing proxy is not transient
        pass

    class OfflineModeIsEnabled(ConnectionError):
        pass

    def wrapped(outer, inner):
        try:
            raise inner
        except BaseException as e:
            try:
                raise outer from e
            except BaseException as o:
                return o
    assert not hf.is_transient(ValueError("Bad split: validation"))
    assert not hf.is_transient(KeyError("not pinned"))
    assert not hf.is_transient(hf.DataSpecError("no document"))
    assert not hf.is_transient(HfHubHTTPError(404)) and not hf.is_transient(HfHubHTTPError(401))
    assert hf.is_transient(HfHubHTTPError(503)) and hf.is_transient(HfHubHTTPError(429))
    assert hf.is_transient(ConnectionError("reset")) and hf.is_transient(TimeoutError())
    assert hf.is_transient(wrapped(FileNotFoundError("not in cache"), ConnectionError("down")))
    assert not hf.is_transient(wrapped(OSError("x"), ProxyError("403 Forbidden")))
    assert not hf.is_transient(OfflineModeIsEnabled("offline"))
    assert hf.is_transient(OSError("We couldn't connect to 'https://huggingface.co' to load the files"))
    assert not hf.is_transient(FileNotFoundError("Unable to find 'val.jsonl'"))


def test_smoke_falls_back_only_when_the_hub_is_unreachable(tmp_path, local, monkeypatch):
    repo = _pile_like(tmp_path / "pile")
    bad = {"repo": str(repo), "split": "validation"}                         # no data_files: the old spec
    run = StubRun(offline_ok=True)
    monkeypatch.setattr(hf, "_HUB_UP", True)
    with pytest.raises(ValueError, match="Bad split"):
        hf.load_texts(bad, 5, run)
    assert not run.fallbacks
    monkeypatch.setattr(hf, "_HUB_UP", False)
    texts, src = hf.load_texts(bad, 5, run)
    assert src == "fallback" and len(texts) == 5 and "Hub unreachable" in run.fallbacks[0]
    with pytest.raises(ValueError):                                          # a real run never falls back
        hf.load_texts(bad, 5, StubRun(offline_ok=False))


def test_check_data_covers_every_split_and_smoke_reads_the_same_ones():
    full = {check_data._key(s) for _, _, s in check_data.planned()}
    smoke = {check_data._key(s) for _, _, s in check_data.planned(smoke=True)}
    assert smoke == full
    names = {(k[0], k[2]) for k in full}
    assert {("monology/pile-uncopyrighted", "train"), ("monology/pile-uncopyrighted", "validation"),
            ("Salesforce/wikitext", "test"), ("Salesforce/wikitext", "train"), ("akariasai/PopQA", "test"),
            ("uoft-cs/cifar10", "train"), ("uoft-cs/cifar10", "test"), ("HuggingFaceFW/fineweb-edu", "train"),
            ("torchvision", "train"), ("torchvision", "test")} <= names
    pile_val = [k for k in full if k[0] == "monology/pile-uncopyrighted" and k[2] == "validation"]
    assert pile_val and "val.jsonl.zst" in pile_val[0][3]


def test_every_dataset_load_goes_through_the_pinned_loader():
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for f in list(root.glob("*.py")) + list((root / "common").glob("*.py")):
        if f.name == "hf.py":
            continue
        if re.search(r"\bload_dataset\(", f.read_text()):
            offenders.append(f.name)
    assert not offenders, f"load_dataset called outside common/hf.py: {offenders}"


def _wikitext_like(root: Path) -> Path:
    """WikiText's layout of rows: blank lines between headings and paragraphs; row 0 is blank."""
    rows = ["", " = Valkyria Chronicles III = \n", "", " Senjō no Valkyria 3 is a tactical role @-@ playing game .\n",
            "", " = = Gameplay = = \n", "   ", " The game began development in 2010 .\n"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "test.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in rows))
    return root


def test_empty_rows_are_skipped_by_load_texts_and_check_data(tmp_path, local):
    repo = _wikitext_like(tmp_path / "wikitext")
    spec = {"repo": str(repo), "split": "test", "data_files": {"test": "test.jsonl"}}
    texts, src = hf.load_texts(spec, 3, StubRun(), min_chars=1)            # P2's perplexity text, P4's OOD text
    assert src == "hub" and texts[0].startswith(" = Valkyria") and len(texts) == 3 and all(t for t in texts)
    item, _, i = hf.first_item(spec)
    assert i == 1 and item["text"].startswith(" = Valkyria")
    row = check_data.check_one(hf.data_spec(spec))
    assert row["status"] == "ok" and row["first_row"] == 1 and "row 1" in row["first_item"]
    (repo / "blank.jsonl").write_text("".join(json.dumps({"text": ""}) + "\n" for _ in range(5)))
    blank = check_data.check_one(hf.data_spec({"repo": str(repo), "split": "test",
                                               "data_files": {"test": "blank.jsonl"}}))
    assert blank["status"] == "FAIL" and "non-empty" in blank["error"]


def test_merge_pins_keeps_every_entry(tmp_path):
    import merge_pins
    base = {"a@main": {"commit": "1" * 40, "commit_date": "x"}, "b@main": {"commit": "2" * 40},
            "datasets/d@main": {"commit": "3" * 40}}
    local = {"a@main": {"commit": "1" * 40, "deduped_check": {"verdict": "differs"}},
             "b@main": {"commit": "9" * 40}, "c@step1": {"commit": "4" * 40}}
    merged, added, conflicts = merge_pins.merge(base, local)
    assert set(merged) == {"a@main", "b@main", "c@step1", "datasets/d@main"} and added == ["c@step1"]
    assert merged["a@main"] == {"commit": "1" * 40, "commit_date": "x", "deduped_check": {"verdict": "differs"}}
    assert conflicts == ["b@main"] and merged["b@main"]["commit"] == "9" * 40       # the local run loaded it


def test_the_pull_recipe_loses_no_local_pin(tmp_path):
    """The four steps in merge_pins.py's docstring, on a clone with local pins and an upstream commit that adds
    dataset pins (the L40S situation of 2026-11-16)."""
    import subprocess
    import merge_pins

    def git(*a, cwd):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True,
                       capture_output=True)
    up, l40s = tmp_path / "up", tmp_path / "l40s"
    up.mkdir()
    git("init", "-q", "-b", "main", cwd=up)
    write = lambda p, d: p.write_text(json.dumps(dict(sorted(d.items())), indent=1) + "\n")   # noqa: E731
    write(up / "pins.json", {f"EleutherAI/pythia-14m@step{s}": {"commit": f"{s:040d}"} for s in (1, 2)})
    git("add", "pins.json", cwd=up)
    git("commit", "-q", "-m", "pins", cwd=up)
    git("clone", "-q", str(up), str(l40s), cwd=tmp_path)
    run_pins = json.loads((l40s / "pins.json").read_text())                  # the L40S pins names on first use
    run_pins.update({"allenai/OLMo-2-1124-7B@step1000-tokens4B": {"commit": "a" * 40},
                     "EleutherAI/pythia-410m-seed3@step7000": {"commit": "b" * 40}})
    write(l40s / "pins.json", run_pins)
    up_pins = json.loads((up / "pins.json").read_text())
    up_pins["datasets/monology/pile-uncopyrighted@main"] = {"commit": "3be90335b6" + "0" * 30}
    write(up / "pins.json", up_pins)
    git("commit", "-q", "-am", "dataset pins", cwd=up)
    saved = tmp_path / "pins.local.json"
    saved.write_text((l40s / "pins.json").read_text())                       # 1
    git("checkout", "--", "pins.json", cwd=l40s)                              # 2
    git("pull", "-q", "--ff-only", cwd=l40s)                                  # 3
    assert merge_pins.main([str(saved), "--into", str(l40s / "pins.json")]) == 0   # 4
    final = json.loads((l40s / "pins.json").read_text())
    assert set(final) == set(run_pins) | set(up_pins) and len(final) == 5
