"""Tests for n-gram, MLP, CNN, and RNN models."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline_ngram import NGramModel
from src.models.mlp import MLPCharModel
from src.models.cnn import CNNCharModel
from src.models.rnn import RNNCharModel
from src.tokenizer import BPETokenizer


def test_ngram_fit_forward():
    model = NGramModel(order=3, vocab_size=5, smoothing=0.01)
    ids = [0, 1, 2, 1, 2, 3, 2, 3, 4]
    model.fit(ids)
    ctx = torch.tensor([[1, 2]], dtype=torch.long)
    out = model(ctx)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(1), atol=0.01)


def test_mlp_forward():
    model = MLPCharModel(
        vocab_size=10,
        context_length=8,
        embed_dim=4,
        hidden_dim=16,
        dropout=0.0,
        num_hidden_layers=1,
    )
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mlp_multilayer_forward():
    model = MLPCharModel(
        vocab_size=10,
        context_length=8,
        embed_dim=4,
        hidden_dim=16,
        dropout=0.0,
        num_hidden_layers=3,
    )
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
    assert len(model.extra) == 2


def test_mlp_embed_dropout():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.5, num_hidden_layers=1,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.train()
    out_a = model(x)
    out_b = model(x)
    assert not torch.equal(out_a, out_b), "dropout should cause stochastic outputs in train mode"
    model.eval()
    out_c = model(x)
    out_d = model(x)
    assert torch.equal(out_c, out_d), "eval mode should be deterministic"


def test_mlp_positional_embed():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1,
    )
    assert hasattr(model, "pos_embed")
    assert model.pos_embed.shape == (1, 8, 4)


def test_mlp_attention_pooling():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1,
    )
    assert hasattr(model, "attn_score")
    assert hasattr(model, "proj")
    x = torch.randint(0, 10, (2, 8))
    model.eval()
    with torch.no_grad():
        emb = model.embed(x) + model.pos_embed
        weights = torch.softmax(model.attn_score(emb).squeeze(-1), dim=1)
        assert weights.shape == (2, 8)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), "attention weights must sum to 1"


def test_cnn_embed_dropout():
    model = CNNCharModel(
        vocab_size=10, context_length=8, embed_dim=4, num_channels=8,
        kernel_sizes=(3,), dropout=0.5,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.eval()
    out_a = model(x)
    out_b = model(x)
    assert torch.equal(out_a, out_b)


def test_rnn_embed_dropout():
    model = RNNCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        num_layers=1, dropout=0.5,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.eval()
    out_a = model(x)
    out_b = model(x)
    assert torch.equal(out_a, out_b)


def test_bpe_tokenizer_train_encode_decode():
    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = BPETokenizer.train(text, vocab_size=100)
    assert tok.vocab_size <= 100
    ids = tok.encode("the quick")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert "the" in decoded and "quick" in decoded


def test_bpe_tokenizer_save_load(tmp_path):
    text = "hello world hello world hello world " * 30
    tok = BPETokenizer.train(text, vocab_size=50)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    ids_orig = tok.encode("hello world")
    ids_loaded = loaded.encode("hello world")
    assert ids_orig == ids_loaded


def test_mlp_with_bpe_vocab():
    model = MLPCharModel(
        vocab_size=200, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1,
    )
    x = torch.randint(0, 200, (2, 8))
    out = model(x)
    assert out.shape == (2, 200)
    assert torch.isfinite(out).all()
