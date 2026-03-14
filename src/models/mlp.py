"""
MLP over fixed-length character context for next-letter prediction.
Input: (batch, context_length) character ids.
Embed -> flatten -> hidden -> logits (batch, vocab_size).
"""

import torch
import torch.nn as nn


class MLPCharModel(nn.Module):
    """
    Embedding -> flatten -> one or two hidden layers (ReLU, dropout) -> logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        flat_size = context_length * embed_dim
        self.fc1 = nn.Linear(flat_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (batch, context_length) long tensor of character ids.
        Returns: (batch, vocab_size) logits for next character.
        """
        # (B, L) -> (B, L, E)
        x = self.embed(context)
        # (B, L*E)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
