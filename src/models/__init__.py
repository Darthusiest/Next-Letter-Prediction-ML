"""
Character-level models for next-letter prediction.
"""

from .baseline_ngram import NGramModel
from .mlp import MLPCharModel
from .rnn import RNNCharModel

__all__ = ["NGramModel", "MLPCharModel", "RNNCharModel"]


def get_model(name: str, vocab_size: int, context_length: int, **kwargs):
    """Factory: get model by name with given vocab_size and context_length."""
    if name == "ngram":
        return NGramModel(order=kwargs.get("order", 4), vocab_size=vocab_size)
    if name == "mlp":
        return MLPCharModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=kwargs.get("embed_dim", 32),
            hidden_dim=kwargs.get("hidden_dim", 128),
            dropout=kwargs.get("dropout", 0.2),
        )
    if name == "rnn":
        return RNNCharModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=kwargs.get("embed_dim", 64),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
        )
    raise ValueError(f"Unknown model: {name}")

