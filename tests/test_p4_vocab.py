"""P4's vocabulary: added tokens above tokenizer.vocab_size must have rows in the model (laptop bug, 2026-11-13)."""
import pytest
import torch

import config
import p4_clipping as p4
from common import hf


class AddedTokensTokenizer:
    """Like Pythia's tokenizer: vocab_size counts the base vocabulary only, len() also counts added tokens,
    and the added tokens occur in the text (here every 'ws' word maps to an added ID)."""

    vocab_size = 100
    eos_token_id = 0

    def __len__(self):
        return 110

    def encode(self, text, add_special_tokens=False):
        return [100 + (i % 10) if w == "ws" else 1 + (hash(w) % 99) for i, w in enumerate(text.split())]


def test_tokenizer_vocab_counts_added_tokens():
    assert p4.tokenizer_vocab(AddedTokensTokenizer()) == 110
    assert p4.tokenizer_vocab(hf.ByteTokenizer(512)) == 512


def test_added_token_ids_fit_the_model_in_smoke_and_full_settings():
    tok = AddedTokensTokenizer()
    ids = hf.tokenize_stream(["a b ws c ws ws d " * 40] * 4, tok, 32, 8)
    assert int(ids.max()) >= tok.vocab_size                          # the added IDs really occur
    data = {"train": ids[:6], "test": ids[6:], "ood": ids[:2], "vocab": p4.tokenizer_vocab(tok)}
    for smoke in (True, False):
        cfg = config.get("p4_clipping", smoke)
        cfg["transformer"]["neox"] = "TINY_NEOX"                     # keep the full-settings model small here
        model = p4.build_transformer(cfg, data["vocab"], seed=0)
        assert model.config.vocab_size >= 110
        p4.check_token_ids(data, model.config.vocab_size)           # passes
        with torch.no_grad():
            out = model(input_ids=data["train"][:2].long())          # no index error
        assert out.logits.shape[-1] == model.config.vocab_size


def test_check_token_ids_fails_readably():
    data = {"train": torch.tensor([[1, 2, 105]]), "vocab": 110}
    with pytest.raises(ValueError, match="token ID 105 in the train split is outside the model's vocabulary of 100"):
        p4.check_token_ids(data, 100)


def test_full_runs_keep_pythia_slots():
    cfg = config.get("p4_clipping")
    assert cfg["transformer"]["neox"] == "PYTHIA_14M"
    model = p4.build_transformer(cfg, 50277, seed=0)                 # len() of the Pythia tokenizer
    assert model.config.vocab_size == 50304                          # unchanged: max(50,304, 50,277)
