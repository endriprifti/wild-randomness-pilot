"""Hugging Face loading with explicit, recorded fallbacks.

Real runs load models, checkpoints (revisions) and datasets from the Hub and fail loudly.
Under --smoke (or --offline-ok) a failed download is replaced by a stand-in of the same
shape (a randomly initialized GPT-NeoX of Pythia-14M shape, a byte tokenizer, synthetic
Zipfian text or random images); every replacement is written to run.json under "fallbacks".
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PINS_PATH = Path(__file__).resolve().parent.parent / "pins.json"


class Pins:
    """Every Hub model is loaded at a commit hash, never at a moving name.

    pins.json maps "repo@revision" (revision = branch such as step1000, or "main") to the commit the
    pilot uses, with the commit's date. The first run that meets a new name resolves it on the Hub
    and writes it here; commit pins.json, and later runs (or --require-pins) reuse exactly those
    commits. This matters for Pythia: on 27 February 2026 the pythia-14m and pythia-31m names switched
    from the deduplicated to the standard-Pile models, so a name alone does not fix the weights.
    """

    def __init__(self, path: Path = PINS_PATH):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(repo: str, revision: str | None) -> str:
        return f"{repo}@{revision or 'main'}"

    def entry(self, repo: str, revision: str | None) -> dict | None:
        return self.data.get(self.key(repo, revision))

    def resolve(self, repo: str, revision: str | None, run=None) -> str:
        k = self.key(repo, revision)
        if re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            sha = revision                       # already a commit
        elif k in self.data:
            sha = self.data[k]["commit"]
        else:
            if run is not None and getattr(run, "require_pins", False):
                raise KeyError(f"{k} is not pinned in {self.path.name} (run once without --require-pins to pin it)")
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(repo, revision=revision, files_metadata=True)
            sha = info.sha
            entry = {"commit": sha, "resolved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            entry["weights_bytes"] = weights_bytes(info.siblings or [])
            try:  # the revision's last commit; fall back to the commit list if the Hub omits it
                date = info.last_modified or api.list_repo_commits(repo, revision=sha)[0].created_at
                entry["commit_date"] = date.isoformat()
            except Exception:  # noqa: BLE001  the date is informative only
                pass
            self.data[k] = entry
            self.save()
        if run is not None:
            run.pins[k] = sha
        return sha

    def save(self):
        on_disk = json.loads(self.path.read_text()) if self.path.exists() else {}
        on_disk.update(self.data)                # merge: another job may have pinned other names
        self.data = on_disk
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(sorted(on_disk.items())), indent=1) + "\n")
        os.replace(tmp, self.path)


PINS = Pins()


def weights_bytes(siblings) -> int | None:
    """Bytes from_pretrained downloads for the weights: the safetensors files if any, else the .bin files."""
    sizes = {f.rfilename: f.size for f in siblings if getattr(f, "size", None)}
    st = sum(v for k, v in sizes.items() if k.endswith(".safetensors"))
    bn = sum(v for k, v in sizes.items() if k.endswith(".bin"))
    return (st or bn) or None


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
        model = AutoModelForCausalLM.from_pretrained(name, revision=sha, dtype=dtype,
                                                     attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(name, revision=sha)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model.to(device).eval(), tok, "hub"
    except Exception as e:  # network, auth, missing revision
        if not run.offline_ok or isinstance(e, KeyError):
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
        refs = list_repo_refs(repo)
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


def load_texts(dataset: str, config: str | None, split: str, n: int, run, field: str = "text",
               min_chars: int = 200, seed: int = 0) -> tuple[list[str], str]:
    """First n documents (streaming) with at least min_chars characters."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, config, split=split, streaming=True)
        out = []
        for ex in ds:
            t = ex.get(field) or ""
            if len(t) >= min_chars:
                out.append(t)
            if len(out) >= n:
                break
        if not out:
            raise RuntimeError("empty stream")
        return out, "hub"
    except Exception as e:
        if not run.offline_ok:
            raise
        run.fallback(f"dataset {dataset}/{config}:{split}: synthetic Zipfian text ({type(e).__name__})")
        return zipf_text(n, 400, seed), "fallback"


def load_rows(dataset: str, config: str | None, split: str, run, limit: int | None = None):
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset, config, split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return ds, "hub"
    except Exception:
        if not run.offline_ok:
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
