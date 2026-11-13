"""Hugging Face loading with explicit, recorded fallbacks.

Real runs load models, checkpoints (revisions) and datasets from the Hub and fail loudly.
Under --smoke (or --offline-ok) a failed download is replaced by a stand-in of the same
shape (a randomly initialized GPT-NeoX of Pythia-14M shape, a byte tokenizer, synthetic
Zipfian text or random images); every replacement is written to run.json under "fallbacks".
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch

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
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                                     attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model.to(device).eval(), tok, "hub"
    except Exception as e:  # network, auth, missing revision
        if not run.offline_ok:
            raise
        run.fallback(f"model {name}@{revision}: random-init GPT-NeoX stand-in ({type(e).__name__}: {str(e)[:120]})")
        cfg = dict(fallback_cfg or TINY_NEOX)
        import zlib
        seed = zlib.crc32(f"{name}@{revision}".encode()) % (2**31)
        model = neox_from_config(cfg, seed=seed).to(dtype).to(device).eval()
        return model, ByteTokenizer(cfg["vocab_size"]), "fallback"


def list_steps(repo: str, run, default_steps: Iterable[int]) -> dict[int, str]:
    """step -> revision name, from the repo's branches (Pythia 'step1000', OLMo 'step1000-tokens4B')."""
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo)
        out = {}
        for br in refs.branches:
            m = re.match(r"step(\d+)", br.name)
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
