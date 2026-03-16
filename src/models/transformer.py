"""
Transformer-based character model for next-letter prediction.
Small causal decoder-style transformer over character embeddings.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerCharModel(nn.Module):
    """
    Causal transformer over character embeddings.
    Uses the representation at the final position to predict the next character.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=context_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # Mask out future positions (upper triangular)
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (batch, context_length) long tensor of character ids.
        Returns: (batch, vocab_size) logits.
        """
        x = self.embed(context)              # (B, L, D)
        x = self.pos_encoding(x)             # add positional information
        L = x.size(1)
        mask = self._causal_mask(L, x.device)
        encoded = self.encoder(x, mask)      # (B, L, D)
        last = encoded[:, -1, :]             # final position representation
        last = self.dropout(last)
        logits = self.fc(last)               # (B, V)
        return logits

