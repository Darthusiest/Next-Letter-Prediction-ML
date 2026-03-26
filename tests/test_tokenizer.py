"""Tests for the BPE tokenizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import BPETokenizer


def test_bpe_encode_empty_string():
    text = "the quick brown fox " * 50
    tok = BPETokenizer.train(text, vocab_size=100)
    ids = tok.encode("")
    assert isinstance(ids, list)
    assert len(ids) == 0


def test_bpe_min_frequency_respected():
    text = "aaaa bbbb cccc dddd " * 20 + "zzzz_unique_token "
    tok = BPETokenizer.train(text, vocab_size=200, min_frequency=5)
    vocab = tok.char2id
    assert "zzzz_unique_token" not in vocab, \
        "a token appearing only once should not be merged at min_frequency=5"
