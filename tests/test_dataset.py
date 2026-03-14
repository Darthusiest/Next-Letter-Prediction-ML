"""Tests for sliding-window dataset and splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vocab import build_vocab_from_text
from src.dataset import CharSequenceDataset, get_splits
from src.config import DEFAULT_ALLOWED_CHARS, CONTEXT_LENGTH


def test_dataset_length():
    text = "a" * (CONTEXT_LENGTH + 100)
    vocab = build_vocab_from_text(text, allowed_chars=DEFAULT_ALLOWED_CHARS)
    ds = CharSequenceDataset(text, vocab, context_length=CONTEXT_LENGTH)
    assert len(ds) == 100


def test_dataset_item():
    L = 4
    text = "abcdefghij"
    vocab = build_vocab_from_text(text)
    ds = CharSequenceDataset(text, vocab, context_length=L)
    context, target = ds[0]
    assert context.shape == (L,)
    assert context[0].item() == vocab.char2id["a"]
    assert context[-1].item() == vocab.char2id["d"]
    assert target == vocab.char2id["e"]


def test_get_splits():
    text = "a" * 1000
    vocab = build_vocab_from_text(text)
    train_ds, val_ds, test_ds = get_splits(text, vocab, context_length=8)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    # Contiguous: no overlap
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total <= 1000 - 8
