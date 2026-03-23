"""
MLP with positional embeddings and attention pooling for next-letter prediction.
Input: (batch, context_length) character ids.
Embed + positional embed -> attention-weighted pool -> hidden blocks with residual -> logits.

The attention pooling learns which context positions matter most for prediction,
replacing the flat concatenation that treated all positions equally.
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
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        # Attention pooling: learn per-position importance scores
        self.attn_score = nn.Linear(embed_dim, 1)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # H->H blocks with residual connections
        self.extra = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.extra_ln = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.embed(context) + self.pos_embed          # (B, L, E)
        x = self.embed_drop(x)
        attn_weights = self.attn_score(x).squeeze(-1)     # (B, L)
        attn_weights = F.softmax(attn_weights, dim=1)     # (B, L)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)   # (B, E)
        x = self.ln1(F.gelu(self.proj(x)))                # (B, H)
        x = self.dropout(x)
        for lin, ln in zip(self.extra, self.extra_ln):
            x = x + self.dropout(ln(F.gelu(lin(x))))
        return self.fc2(x)
