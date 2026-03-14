"""Tests for n-gram and MLP models."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline_ngram import NGramModel
from src.models.mlp import MLPCharModel


def test_ngram_fit_forward():
    model = NGramModel(order=3, vocab_size=5, smoothing=0.01)
    ids = [0, 1, 2, 1, 2, 3, 2, 3, 4]
    model.fit(ids)
    ctx = torch.tensor([[1, 2]], dtype=torch.long)
    out = model(ctx)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
    # Log probs should sum to ~0 in log space (sum log p = log 1 = 0)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(1), atol=0.01)


def test_mlp_forward():
    model = MLPCharModel(
        vocab_size=10,
        context_length=8,
        embed_dim=4,
        hidden_dim=16,
        dropout=0.0,
    )
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
