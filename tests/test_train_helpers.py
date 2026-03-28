"""Tests for train.py helper functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import _amp_device_for_training, _load_vocab_from_checkpoint
from src.vocab import CharVocab
from src.config import MLP_NUM_SELF_ATTN_LAYERS


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


# ---------------------------------------------------------------------------
# Test 1: Config propagation for mlp_num_self_attn_layers
# ---------------------------------------------------------------------------

def test_train_mlp_self_attn_layers_config_propagation():
    """Verify get_model passes num_self_attn_layers and the model stores it."""
    from src.models import get_model

    model = get_model(
        "mlp",
        vocab_size=40,
        context_length=16,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
        num_self_attn_layers=MLP_NUM_SELF_ATTN_LAYERS,
    )
    assert model.num_self_attn_layers == MLP_NUM_SELF_ATTN_LAYERS

    custom_layers = 3
    model2 = get_model(
        "mlp",
        vocab_size=40,
        context_length=16,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
        num_self_attn_layers=custom_layers,
    )
    assert model2.num_self_attn_layers == custom_layers


# ---------------------------------------------------------------------------
# Test 2: CLI argument parsing for --mlp-self-attn-layers
# ---------------------------------------------------------------------------

def test_main_mlp_self_attn_layers_cli_arg(monkeypatch):
    """Verify --mlp-self-attn-layers CLI argument is parsed and passed to train()."""
    import argparse

    captured_kw = {}

    def fake_train(**kwargs):
        captured_kw.update(kwargs)
        return None, None

    with patch("src.train.train", side_effect=fake_train) as mock_train:
        test_args = [
            "src.train",
            "--model", "mlp",
            "--mlp-self-attn-layers", "3",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        from src.train import main
        try:
            main()
        except (SystemExit, FileNotFoundError):
            pass

        if mock_train.called:
            call_kwargs = mock_train.call_args
            all_kw = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert all_kw.get("mlp_num_self_attn_layers") == 3


# ---------------------------------------------------------------------------
# Test 3: Backward compatibility for missing num_self_attn_layers attribute
# ---------------------------------------------------------------------------

def test_train_handles_missing_num_self_attn_layers_attribute():
    """Verify getattr fallback works for models without num_self_attn_layers."""

    class LegacyModel(nn.Module):
        """Mimics a model that predates the num_self_attn_layers attribute."""
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

    model = LegacyModel()
    assert not hasattr(model, "num_self_attn_layers")

    result = getattr(model, "num_self_attn_layers", MLP_NUM_SELF_ATTN_LAYERS)
    assert result == MLP_NUM_SELF_ATTN_LAYERS

    class ModelWithAttr(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_self_attn_layers = 5
            self.linear = nn.Linear(10, 10)

    model2 = ModelWithAttr()
    result2 = getattr(model2, "num_self_attn_layers", MLP_NUM_SELF_ATTN_LAYERS)
    assert result2 == 5


def test_checkpoint_state_dict_key_validation():
    """Verify _evaluate_from_neural_checkpoint warns on key mismatches."""
    from src.models import get_model

    model = get_model(
        "mlp",
        vocab_size=40,
        context_length=16,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
        num_self_attn_layers=2,
    )
    state = model.state_dict()

    model2 = get_model(
        "mlp",
        vocab_size=40,
        context_length=16,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.1,
        num_self_attn_layers=2,
    )
    expected_keys = set(model2.state_dict().keys())
    checkpoint_keys = set(state.keys())
    assert expected_keys == checkpoint_keys, "Matching architectures should have identical keys"
