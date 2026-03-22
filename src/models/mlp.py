"""
MLP over fixed-length character context for next-letter prediction.
Input: (batch, context_length) character ids.
Embed -> flatten -> stack of (Linear -> GELU -> LayerNorm -> Dropout) -> logits.
Extra hidden blocks use residual connections for stable gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCharModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_hidden_layers: int = 1,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        flat_size = context_length * embed_dim
        self.fc1 = nn.Linear(flat_size, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # H→H blocks with residual connections
        self.extra = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.extra_ln = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.embed(context)
        x = x.view(x.size(0), -1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.dropout(x)
        for lin, ln in zip(self.extra, self.extra_ln):
            x = x + self.dropout(ln(F.gelu(lin(x))))
        return self.fc2(x)
