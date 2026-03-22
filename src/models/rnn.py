"""
RNN-based (LSTM) character model for next-letter prediction.
Takes a fixed-length context of character ids and predicts the next character.
"""

import torch
import torch.nn as nn


class RNNCharModel(nn.Module):
    """
    LSTM over character embeddings.
    - Input: (batch, context_length) of character ids.
    - Output: (batch, vocab_size) logits for the next character.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (batch, context_length) long tensor of character ids.
        Returns: (batch, vocab_size) logits.
        """
        # (B, L) -> (B, L, E)
        x = self.embed(context)
        # LSTM must walk all L timesteps (inherent cost vs parallel MLP/CNN).
        # Use final-layer final hidden state instead of full output to avoid
        # allocating (B, L, H), which cuts memory traffic on long contexts.
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, H)
        last_hidden = self.ln(last_hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (B, V)
        return logits

