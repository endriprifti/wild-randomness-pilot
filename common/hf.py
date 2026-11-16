"""Hugging Face loading with explicit, recorded fallbacks.

Real runs load models, checkpoints (revisions) and datasets from the Hub, each at a commit pinned in
pins.json, and fail loudly. Hub calls are retried only after transient failures (network, timeouts,
HTTP 408/429/5xx); a wrong split, a missing file or repository, or any other 4xx fails at once.
Under --smoke (or --offline-ok) a failed model download is replaced by a stand-in of the same
shape (a randomly initialized GPT-NeoX of Pythia-14M shape, a byte tokenizer). A dataset is replaced
(synthetic Zipfian text or random images) only when the Hub cannot be reached at all: with the Hub up,
a smoke run reads the same datasets, splits and files as the full run, so a wrong split fails there
too. Every replacement is written to run.json under "fallbacks".
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PINS_PATH = Path(__file__).resolve().parent.parent / "pins.json"
HUB_RETRIES = (10, 30, 90, 270)   # seconds between attempts at a Hub call after a transient failure
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
# exception classes (matched by name, so that the check does not depend on which HTTP library a version of
# huggingface_hub or datasets uses) that mean the network or the server failed, not the request
_TRANSIENT_NAMES = {"ConnectionError", "TimeoutError", "TransportError", "TimeoutException", "NetworkError",
                    "RemoteProtocolError", "ChunkedEncodingError", "IncompleteRead", "ReadTimeoutError",
                    "NewConnectionError", "ProtocolError", "ClientConnectionError", "ServerDisconnectedError",
                    "ClientPayloadError", "FSTimeoutError", "RemoteDisconnected"}
# ... except these, which a retry cannot fix: offline mode, a proxy that refuses the host, a bad URL
_PERMANENT_NAMES = {"OfflineModeIsEnabled", "ProxyError", "UnsupportedProtocol", "LocalProtocolError", "InvalidURL"}
_NET_MESSAGE = re.compile(r"couldn'?t connect|connection (reset|aborted|refused|error)|timed out|read timeout|"
                          r"temporary failure|name resolution|max retries exceeded|server disconnected|"
                          r"remote end closed|incomplete read|too many requests|\b(502|503|504) server error", re.I)


class DataSpecError(ValueError):
    """A dataset request that no retry can fix (no document with the field, a split without data_files)."""


def _chain(e: BaseException, limit: int = 12) -> list[BaseException]:
    out, x = [], e
    while x is not None and len(out) < limit and all(x is not y for y in out):
        out.append(x)
        x = x.__cause__ or x.__context__
    return out


def _status(e: BaseException) -> int | None:
    for obj in (getattr(e, "response", None), e):
        st = getattr(obj, "status_code", None)
        if isinstance(st, int):
            return st
    return None


def is_transient(e: BaseException) -> bool:
    """True for failures a retry can fix: the network or the server (timeouts, dropped connections, HTTP
    408/429/5xx), looked up along the exception's cause chain (huggingface_hub, datasets and transformers wrap
    them). False for everything else: a wrong split (ValueError), a missing repository, revision or file, any
    other HTTP 4xx, an unpinned name under --require-pins (KeyError), offline mode, a refusing proxy."""
    chain = _chain(e)
    for x in chain:
        if {c.__name__ for c in type(x).__mro__} & _PERMANENT_NAMES:
            return False
        st = _status(x)
        if st is not None:
            return st in TRANSIENT_STATUS
    if any({c.__name__ for c in type(x).__mro__} & _TRANSIENT_NAMES for x in chain):
        return True
    # wrapped errors that kept only a message, e.g. transformers' OSError "We couldn't connect to ..."
    return (isinstance(e, (OSError, RuntimeError)) and not isinstance(e, (FileNotFoundError, KeyError))
            and bool(_NET_MESSAGE.search(str(e))))


_HUB_UP: bool | None = None


def hub_reachable() -> bool:
    """Whether the Hugging Face Hub answers at all (one small API call, cached for the process)."""
    global _HUB_UP
    if _HUB_UP is None:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
            _HUB_UP = False
        else:
            try:
                from huggingface_hub import HfApi
                HfApi().dataset_info("Salesforce/wikitext", timeout=20)
                _HUB_UP = True
            except Exception:  # noqa: BLE001  any failure: the Hub is not usable from here
                _HUB_UP = False
    return _HUB_UP


CODE_ERRORS = (AttributeError, TypeError, NameError, ImportError, KeyError)


def data_fallback_ok(run, e: BaseException | None = None) -> bool:
    """A synthetic stand-in for a dataset is allowed only in smoke / --offline-ok runs, and only when the Hub
    cannot be reached at all; never for an unpinned name under --require-pins (KeyError) or a bug in our own
    code or setup (AttributeError, TypeError, NameError, ImportError such as a missing zstandard)."""
    return (run is not None and bool(getattr(run, "offline_ok", False)) and not isinstance(e, CODE_ERRORS)
            and not hub_reachable())


def hub_retry(fn, run=None, what: str = "Hub call"):
    """fn() with retries after transient failures (network, rate limit, 5xx), so that one dropped connection
    in a run of hundreds of downloads does not end the job. Any other error is raised at once. Smoke and
    --offline-ok runs do not retry when the Hub is unreachable (they fall back to stand-ins instead)."""
    waits = HUB_RETRIES
    if run is not None and getattr(run, "offline_ok", False) and not hub_reachable():
        waits = ()
    for i in range(len(waits) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == len(waits) or not is_transient(e):
                raise
            msg = f"{what} failed ({type(e).__name__}: {str(e)[:120]}); retry {i + 1}/{len(waits)} in {waits[i]} s"
            if run is not None:
                run.note(msg)
            else:
                print(msg, flush=True)
            time.sleep(waits[i])


class Pins:
    """Every Hub model is loaded at a commit hash, never at a moving name.

    pins.json maps "repo@revision" (revision = branch such as step1000, or "main") to the commit the
    pilot uses, with the commit's date. The first run that meets a new name resolves it on the Hub
    and writes it here; commit pins.json, and later runs (or --require-pins) reuse exactly those
    commits. This matters for Pythia: on 24 February 2026 the pythia-14m and pythia-31m repositories were
    re-created with the standard-Pile models (the deduplicated ones moved to pythia-14m-deduped and
    pythia-31m-deduped), so a name alone does not fix the weights. Each new entry also records the LFS sha256
    of the weights files at that commit (weights_sha256), and P1 caches its name-switch content check there
    (deduped_check).
    """

    def __init__(self, path: Path = PINS_PATH):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(repo: str, revision: str | None, repo_type: str = "model") -> str:
        """'repo@revision' for models; 'datasets/repo@revision' for datasets."""
        return ("datasets/" if repo_type == "dataset" else "") + f"{repo}@{revision or 'main'}"

    def entry(self, repo: str, revision: str | None, repo_type: str = "model") -> dict | None:
        return self.data.get(self.key(repo, revision, repo_type))

    def resolve(self, repo: str, revision: str | None, run=None, repo_type: str = "model") -> str:
        k = self.key(repo, revision, repo_type)
        if re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            sha = revision                       # already a commit
        elif k in self.data:
            sha = self.data[k]["commit"]
        else:
            if run is not None and getattr(run, "require_pins", False):
                raise KeyError(f"{k} is not pinned in {self.path.name} (run once without --require-pins to pin it)")
            from huggingface_hub import HfApi
            api = HfApi()
            if repo_type == "dataset":
                info = hub_retry(lambda: api.dataset_info(repo, revision=revision), run, f"pinning {k}")
            else:
                info = hub_retry(lambda: api.model_info(repo, revision=revision, files_metadata=True), run,
                                 f"pinning {k}")
            sha = info.sha
            entry = {"commit": sha, "resolved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            if repo_type != "dataset":
                entry["weights_bytes"] = weights_bytes(info.siblings or [])
                entry["weights_sha256"] = {f: h for f, h in lfs_sha256s(info.siblings).items() if f in WEIGHTS_FILES}
            try:  # the revision's last commit; fall back to the commit list if the Hub omits it
                date = info.last_modified or api.list_repo_commits(repo, revision=sha, repo_type=repo_type)[0].created_at
                entry["commit_date"] = date.isoformat()
            except Exception:  # noqa: BLE001  the date is informative only
                pass
            self.data[k] = entry
            self.save()
        if run is not None:
            run.pins[k] = sha
        return sha

    def save(self):
        """Merge into the file on disk and write it atomically, under an exclusive lock (pins.json.lock), so two
        jobs that pin names at the same time (a run_lane.sh job next to run_all.sh) lose nothing. Entries only
        grow: a name keeps its commit, and a job adds names and fields (deduped_check). So the file on disk wins
        where both carry a field, which keeps another job's newer entries over this job's older copy (read at
        import), and this job's new names and fields are added."""
        try:
            import fcntl
        except ImportError:                       # not Linux: no lock (the cloud run and WSL are Linux)
            fcntl = None
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            on_disk = json.loads(self.path.read_text()) if self.path.exists() else {}
            for k, v in self.data.items():
                d = on_disk.get(k)
                on_disk[k] = {**v, **d} if isinstance(d, dict) and isinstance(v, dict) else (v if d is None else d)
            self.data = on_disk
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(dict(sorted(on_disk.items())), indent=1) + "\n")
            os.replace(tmp, self.path)


PINS = Pins()


def weights_bytes(siblings) -> int | None:
    """Bytes from_pretrained downloads for the weights: the safetensors files if any, else the .bin files."""
    sizes = {f.rfilename: f.size for f in siblings if getattr(f, "size", None)}
    st = sum(v for k, v in sizes.items() if k.endswith(".safetensors"))
    bn = sum(v for k, v in sizes.items() if k.endswith(".bin"))
    return (st or bn) or None


WEIGHTS_FILES = ("model.safetensors", "pytorch_model.bin")


def lfs_sha256s(siblings) -> dict[str, str]:
    """filename -> sha256 of every Git LFS file in a model_info(..., files_metadata=True) listing."""
    out = {}
    for s in siblings or []:
        lfs = getattr(s, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if sha:
            out[s.rfilename] = sha
    return out


def hub_lfs(repo: str, revision: str | None) -> tuple[str, dict[str, str]]:
    """(commit, filename -> LFS sha256) at one revision, from the Hub API's file metadata. Nothing is downloaded."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return info.sha, lfs_sha256s(info.siblings)


def compare_weights(mine: dict, theirs: dict, files=WEIGHTS_FILES) -> tuple[str | None, bool | None]:
    """The first of `files` that both listings carry, and whether its sha256 is the same on both sides
    ((None, None) if they share none of them)."""
    for f in files:
        if f in mine and f in theirs:
            return f, mine[f] == theirs[f]
    return None, None


def purge_revision(repo: str, commit: str | None, run=None) -> None:
    """Delete one revision of a model from the local Hugging Face cache (the loss-spike scan reads hundreds
    of checkpoints; keeping them would need hundreds of GB of disk)."""
    if not commit:
        return
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        hashes = [rv.commit_hash for r in cache.repos if r.repo_id == repo for rv in r.revisions
                  if rv.commit_hash == commit]
        if hashes:
            cache.delete_revisions(*hashes).execute()
    except Exception as e:  # noqa: BLE001  a failed purge costs disk, not correctness
        if run is not None:
            run.note(f"cache purge of {repo}@{commit[:10]} failed: {type(e).__name__}: {str(e)[:120]}")

PYTHIA_14M = dict(vocab_size=50304, hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
                  intermediate_size=512, max_position_embeddings=2048, rotary_pct=0.25,
                  rotary_emb_base=10000, use_parallel_residual=True, layer_norm_eps=1e-5,
                  hidden_act="gelu", tie_word_embeddings=False)

TINY_NEOX = dict(PYTHIA_14M, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=256, max_position_embeddings=512, vocab_size=512)


class ByteTokenizer:
    """Minimal stand-in tokenizer (UTF-8 bytes) used only when the real one cannot be downloaded."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1
    padding_side = "left"

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2 + (b % (self.vocab_size - 2)) for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes([max(0, int(i) - 2) % 256 for i in ids if int(i) > 1]).decode("utf-8", errors="ignore")

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512, **kw):
        if isinstance(texts, str):
            texts = [texts]
        enc = [self.encode(t)[:max_length] for t in texts]
        L = max(len(e) for e in enc) if enc else 1
        ids = torch.full((len(enc), L), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            if e:
                ids[i, L - len(e):] = torch.tensor(e)
                mask[i, L - len(e):] = 1
        return {"input_ids": ids, "attention_mask": mask}


def neox_from_config(cfg: dict, seed: int = 0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    conf = GPTNeoXConfig(**cfg)
    conf._attn_implementation = "eager"
    return GPTNeoXForCausalLM(conf)


def load_causal_lm(name: str, revision: str | None, run, dtype=torch.float32, device="cpu",
                   fallback_cfg: dict | None = None):
    """Returns (model, tokenizer, source) with source 'hub' or 'fallback'."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sha = PINS.resolve(name, revision, run)   # load the pinned commit, never the moving name
        model = hub_retry(lambda: AutoModelForCausalLM.from_pretrained(name, revision=sha, dtype=dtype,
                                                                       attn_implementation="eager"),
                          run, f"download of {name}@{revision}")
        tok = hub_retry(lambda: AutoTokenizer.from_pretrained(name, revision=sha), run, f"tokenizer of {name}@{revision}")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model.to(device).eval(), tok, "hub"
    except Exception as e:  # network, auth, missing revision
        # a stand-in for a model the Hub cannot give us (smoke-only names, no network), never for a bug in our
        # code (until 2026-11-16 an AttributeError in emmy_check's worker became a stand-in this way)
        if not run.offline_ok or isinstance(e, CODE_ERRORS):
            raise
        run.fallback(f"model {name}@{revision}: random-init GPT-NeoX stand-in ({type(e).__name__}: {str(e)[:120]})")
        cfg = dict(fallback_cfg or TINY_NEOX)
        import zlib
        seed = zlib.crc32(f"{name}@{revision}".encode()) % (2**31)
        model = neox_from_config(cfg, seed=seed).to(dtype).to(device).eval()
        return model, ByteTokenizer(cfg["vocab_size"]), "fallback"


def list_steps(repo: str, run, default_steps: Iterable[int], branch_regex: str = r"^step(\d+)") -> dict[int, str]:
    r"""step -> branch name, from the repo's branches. branch_regex picks the pretraining branches:
    Pythia 'step1000' and OLMo 2 7B 'step1000-tokens5B' (default), OLMo 2 1B 'stage1-step1000-tokens2B'
    (r'^stage1-step(\d+)-tokens'); OLMo 2 'stage2-ingredient...' soup branches never match."""
    try:
        from huggingface_hub import list_repo_refs
        refs = hub_retry(lambda: list_repo_refs(repo), run, f"branch list of {repo}")
        out = {}
        for br in refs.branches:
            m = re.match(branch_regex, br.name)
            if m:
                out[int(m.group(1))] = br.name
        if out:
            return dict(sorted(out.items()))
        raise RuntimeError("no step branches")
    except Exception as e:
        if not run.offline_ok:
            raise
        run.fallback(f"revisions of {repo}: assumed Pythia naming ({type(e).__name__})")
        return {int(s): f"step{int(s)}" for s in default_steps}


def nearest_steps(available: dict[int, str], wanted: Iterable[int]) -> list[tuple[int, str]]:
    keys = np.array(sorted(available))
    out = []
    for w in wanted:
        k = int(keys[np.argmin(np.abs(keys - w))])
        if (k, available[k]) not in out:
            out.append((k, available[k]))
    return out


def zipf_text(n_docs: int, words_per_doc: int, seed: int, vocab: int = 5000, phi: float = 1.1) -> list[str]:
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** phi
    p /= p.sum()
    words = [f"w{i}" for i in range(vocab)]
    return [" ".join(words[j] for j in rng.choice(vocab, words_per_doc, p=p)) for _ in range(n_docs)]


# ----------------------------------------------------------------------------- datasets
@dataclasses.dataclass
class DataSpec:
    """One dataset split as a job reads it. data_files maps split names to the files that make up each split
    (as in load_dataset); None leaves the split to the builder's own detection, which can miss files: in
    monology/pile-uncopyrighted it finds only the train/ folder, not val.jsonl.zst and test.jsonl.zst at the top
    level. revision is a branch or tag, loaded at the commit pins.json records for it (pinned on first use, like
    the models); field is the text column (None for datasets read row by row)."""
    repo: str
    split: str
    config: str | None = None
    data_files: dict | None = None
    revision: str = "main"
    field: str | None = "text"

    def __post_init__(self):
        if self.data_files is not None and self.split not in self.data_files:
            raise DataSpecError(f"{self.repo}: split {self.split!r} has no entry in data_files {self.data_files}")

    @property
    def label(self) -> str:
        files = f" {self.data_files[self.split]}" if self.data_files else ""
        return f"{self.repo}{'/' + self.config if self.config else ''}:{self.split}{files}@{self.revision}"


def data_spec(x, **override) -> DataSpec:
    """A DataSpec from a config entry: a dict of DataSpec fields, or the list [repo, config, split]."""
    if isinstance(x, DataSpec):
        s = x
    elif isinstance(x, dict):
        s = DataSpec(**x)
    else:
        repo, config, split = x
        s = DataSpec(repo=repo, config=config, split=split)
    return dataclasses.replace(s, **override) if override else s


def _open(spec: DataSpec, sha: str | None, streaming: bool):
    from datasets import load_dataset
    kw = {"name": spec.config, "split": spec.split, "streaming": streaming}
    if sha:
        kw["revision"] = sha
    if spec.data_files:
        kw["data_files"] = dict(spec.data_files)
    return load_dataset(spec.repo, **kw)


def pin_dataset(spec: DataSpec, run=None) -> str:
    return PINS.resolve(spec.repo, spec.revision, run, repo_type="dataset")


def is_empty(v) -> bool:
    """An empty field: missing, None, or a string of whitespace only (WikiText's blank lines)."""
    return v is None or (isinstance(v, str) and not v.strip())


def first_item(spec, run=None, max_rows: int = 1000) -> tuple[dict, str, int]:
    """The first row of a split whose field is not empty (the first row if the spec has no field), at the pinned
    commit (streaming), with the commit and the row's index. load_texts skips empty rows the same way (they are
    shorter than min_chars). Raises DataSpecError if none of the first max_rows rows has the field filled.
    Used by check_data.py."""
    spec = data_spec(spec)
    sha = pin_dataset(spec, run)

    def find():
        for i, ex in enumerate(_open(spec, sha, True)):
            if not spec.field or not is_empty(ex.get(spec.field)):
                return ex, i
            if i + 1 >= max_rows:
                break
        raise DataSpecError(f"{spec.label}: no row with a non-empty {spec.field!r} field in the first {max_rows}")
    item, i = hub_retry(find, run, f"first item of {spec.label}")
    return item, sha, i


def load_texts(spec, n: int, run, field: str | None = None, min_chars: int = 200,
               seed: int = 0) -> tuple[list[str], str]:
    """First n documents (streaming) with at least min_chars characters, at the pinned commit. A stream that
    breaks off with a transient error is restarted from the beginning (same documents)."""
    spec = data_spec(spec)
    field = field or spec.field or "text"
    try:
        sha = pin_dataset(spec, run)

        def stream():
            out = []
            for ex in _open(spec, sha, True):
                t = ex.get(field) or ""
                if len(t) >= min_chars:
                    out.append(t)
                if len(out) >= n:
                    break
            if not out:
                raise DataSpecError(f"{spec.label}: no document with a {field!r} field of {min_chars}+ characters")
            return out
        texts = hub_retry(stream, run, f"stream of {spec.label}")
        if run is not None:
            run.note(f"{spec.label} at {sha[:10] if sha else '?'}: {len(texts)} documents")
        return texts, "hub"
    except Exception as e:
        if not data_fallback_ok(run, e):
            raise
        run.fallback(f"dataset {spec.label}: synthetic Zipfian text (Hub unreachable; {type(e).__name__})")
        return zipf_text(n, 400, seed), "fallback"


CIFAR10 = {"repo": "uoft-cs/cifar10", "split": "train", "field": "img"}


def load_cifar10(run, root: str, limit_train: int | None = None, limit_test: int | None = None):
    """CIFAR-10 as uint8 images [N, 32, 32, 3] and int64 labels, plus the source.

    The Hugging Face copy (uoft-cs/cifar10) comes first: with limits (smoke) it streams only the images
    needed, without the 170 MB archive; without limits it downloads the parquet files. torchvision's
    mirror comes second; it can be very slow (130 kB/s, about 22 minutes, from the WSL2 laptop).
    Raises RuntimeError if neither works; the caller decides on a stand-in."""
    errors = []
    try:
        out = []
        for split, lim in (("train", limit_train), ("test", limit_test)):
            spec = data_spec(CIFAR10, split=split)
            sha = pin_dataset(spec, run)

            def read():
                imgs, labels = [], []
                if lim:
                    for ex in _open(spec, sha, True):
                        imgs.append(np.asarray(ex["img"].convert("RGB"), dtype=np.uint8))
                        labels.append(int(ex["label"]))
                        if len(imgs) >= lim:
                            break
                else:
                    ds = _open(spec, sha, False)
                    imgs = [np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["img"]]
                    labels = list(ds["label"])
                return imgs, labels
            imgs, labels = hub_retry(read, run, f"CIFAR-10 {split}")
            out += [np.stack(imgs), np.asarray(labels, dtype=np.int64)]
        return (*out, "hf:uoft-cs/cifar10")
    except Exception as e:  # noqa: BLE001
        if isinstance(e, KeyError):
            raise
        errors.append(f"Hugging Face: {type(e).__name__}: {str(e)[:120]}")
    try:
        import torchvision
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        Xtr, ytr = tr.data[:limit_train or None], np.asarray(tr.targets, dtype=np.int64)[:limit_train or None]
        Xte, yte = te.data[:limit_test or None], np.asarray(te.targets, dtype=np.int64)[:limit_test or None]
        return Xtr, ytr, Xte, yte, "torchvision"
    except Exception as e:  # noqa: BLE001
        errors.append(f"torchvision: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("CIFAR-10 unavailable: " + "; ".join(errors))


def load_rows(spec, run, limit: int | None = None):
    """A whole split (downloaded, not streamed) at its pinned commit, optionally its first `limit` rows; (None,
    'fallback') only when data_fallback_ok (the caller builds the stand-in)."""
    spec = data_spec(spec)
    try:
        sha = pin_dataset(spec, run)
        ds = hub_retry(lambda: _open(spec, sha, False), run, f"download of {spec.label}")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        if run is not None:
            run.note(f"{spec.label} at {sha[:10] if sha else '?'}: {len(ds)} rows")
        return ds, "hub"
    except Exception as e:
        if not data_fallback_ok(run, e):
            raise
        return None, "fallback"


def tokenize_stream(texts: list[str], tok, seq_len: int, n_seqs: int) -> torch.Tensor:
    """Concatenate documents (EOS-separated) and cut into n_seqs sequences of seq_len tokens."""
    ids: list[int] = []
    eos = getattr(tok, "eos_token_id", None) or 0
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False))
        ids.append(eos)
        if len(ids) >= seq_len * n_seqs:
            break
    n = min(n_seqs, len(ids) // seq_len)
    if n < 1:
        raise ValueError("not enough text for one sequence")
    return torch.tensor(ids[: n * seq_len], dtype=torch.int32).reshape(n, seq_len)
