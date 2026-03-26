"""Tests for evaluation: loss, accuracy, and perplexity."""

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate, perplexity
from src.models.mlp import MLPCharModel
from src.models.baseline_ngram import NGramModel


def _make_loader(context, target, batch_size=4):
    ds = TensorDataset(context, target)
    return DataLoader(ds, batch_size=batch_size)


def test_evaluate_neural_loss_and_accuracy():
    model = MLPCharModel(
        vocab_size=5, context_length=4, embed_dim=4, hidden_dim=8,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    model.eval()
    ctx = torch.randint(0, 5, (8, 4))
    tgt = torch.randint(0, 5, (8,))
    loader = _make_loader(ctx, tgt)
    loss, acc = evaluate(model, loader, torch.device("cpu"), is_ngram=False)
    assert isinstance(loss, float) and loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate_ngram_uses_nll_loss():
    model = NGramModel(order=2, vocab_size=5, smoothing=0.01)
    model.fit([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    ctx = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    tgt = torch.tensor([2, 4], dtype=torch.long)
    loader = _make_loader(ctx, tgt, batch_size=2)
    loss, acc = evaluate(model, loader, torch.device("cpu"), is_ngram=True)
    assert isinstance(loss, float) and loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate_empty_dataloader_returns_zero():
    model = MLPCharModel(
        vocab_size=5, context_length=4, embed_dim=4, hidden_dim=8,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 4, dtype=torch.long),
                                            torch.empty(0, dtype=torch.long)))
    loss, acc = evaluate(model, empty_loader, torch.device("cpu"), is_ngram=False)
    assert loss == 0.0
    assert acc == 0.0


def test_perplexity_matches_exp_loss():
    for loss_val in [0.0, 0.5, 1.0, 2.3, 5.0]:
        assert abs(perplexity(loss_val) - math.exp(loss_val)) < 1e-6
