"""
Character-level models for next-letter prediction.
"""

from ..config import RNN_NUM_LAYERS, MLP_NUM_HIDDEN_LAYERS, MLP_NUM_ATTN_HEADS, MLP_NUM_SELF_ATTN_LAYERS

from .baseline_ngram import NGramModel
from .mlp import MLPCharModel
from .rnn import RNNCharModel
from .cnn import CNNCharModel

__all__ = ["NGramModel", "MLPCharModel", "RNNCharModel", "CNNCharModel"]


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
            num_hidden_layers=kwargs.get("num_hidden_layers", MLP_NUM_HIDDEN_LAYERS),
            num_attn_heads=kwargs.get("num_attn_heads", MLP_NUM_ATTN_HEADS),
            num_self_attn_layers=kwargs.get("num_self_attn_layers", MLP_NUM_SELF_ATTN_LAYERS),
        )
    if name == "rnn":
        return RNNCharModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=kwargs.get("embed_dim", 64),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", RNN_NUM_LAYERS),
            dropout=kwargs.get("dropout", 0.2),
        )
    if name == "cnn":
        return CNNCharModel(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=kwargs.get("embed_dim", 64),
            num_channels=kwargs.get("num_channels", 128),
            kernel_sizes=kwargs.get("kernel_sizes", (3, 5, 7)),
            dropout=kwargs.get("dropout", 0.2),
        )
    raise ValueError(f"Unknown model: {name}")


