"""Tests for train.py helper functions."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import _amp_device_for_training, _load_vocab_from_checkpoint
from src.vocab import CharVocab


def test_amp_device_cuda():
    device = torch.device("cuda")
    assert _amp_device_for_training(device) == "cuda"


def test_amp_device_cpu_returns_none():
    device = torch.device("cpu")
    assert _amp_device_for_training(device) is None


def test_load_vocab_from_checkpoint_char():
    vocab = CharVocab({"a": 0, "b": 1, "c": 2})
    ckpt = {"tokenizer_type": "char", "vocab": vocab}
    loaded = _load_vocab_from_checkpoint(ckpt)
    assert isinstance(loaded, CharVocab)
    assert loaded.vocab_size == 3


def test_load_vocab_from_checkpoint_char_from_dict():
    ckpt = {"tokenizer_type": "char", "vocab": {"a": 0, "b": 1}}
    loaded = _load_vocab_from_checkpoint(ckpt)
    assert isinstance(loaded, CharVocab)
    assert loaded.vocab_size == 2


def test_load_vocab_from_checkpoint_bpe_missing_path_raises():
    ckpt = {"tokenizer_type": "bpe"}
    with pytest.raises(FileNotFoundError, match="tokenizer_path"):
        _load_vocab_from_checkpoint(ckpt)
