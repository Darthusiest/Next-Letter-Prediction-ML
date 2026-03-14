"""Tests for character vocabulary."""

import sys
from pathlib import Path

# Allow importing src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vocab import build_vocab_from_text, CharVocab, build_vocab


def test_build_vocab_from_text():
    text = "hello world"
    vocab = build_vocab_from_text(text, allowed_chars=None)
    assert vocab.vocab_size == len(set(text))
    assert vocab.encode("hello") == [vocab.char2id[c] for c in "hello"]
    assert vocab.decode(vocab.encode("hello")) == "hello"


def test_build_vocab_with_allowed_chars():
    from src.config import DEFAULT_ALLOWED_CHARS
    text = "abc"
    vocab = build_vocab_from_text(text, allowed_chars=DEFAULT_ALLOWED_CHARS)
    # Vocab includes all allowed chars that appear, plus we use sorted(allowed_chars) in plan
    # Actually we build from sorted(allowed_chars) so vocab has many chars
    assert "a" in vocab.char2id
    assert " " in vocab.char2id
    enc = vocab.encode("a b")
    assert len(enc) == 3
    assert vocab.decode(enc) == "a b"


def test_encode_decode_roundtrip():
    text = "the cat sat on the mat."
    vocab = build_vocab_from_text(text)
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)
    assert decoded == text


def test_save_load():
    import tempfile
    text = "abcdef"
    vocab = build_vocab_from_text(text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        vocab.save(path)
        loaded = CharVocab.load(path)
        assert loaded.char2id == vocab.char2id
        assert loaded.decode(loaded.encode(text)) == text
