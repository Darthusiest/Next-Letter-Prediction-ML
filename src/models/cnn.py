"""
CNN-based character model for next-letter prediction.
Uses 1D convolutions over embeddings to capture local spelling patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCharModel(nn.Module):
    """
    Conv1d over character embeddings with multiple kernel sizes.
    Global max-pooling aggregates over the context, then LayerNorm +
    dropout + linear projection to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 64,
        num_channels: int = 128,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.embed = nn.Embedding(vocab_size, embed_dim)

        convs = []
        for k in kernel_sizes:
            convs.append(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
        self.convs = nn.ModuleList(convs)
        total_channels = num_channels * len(kernel_sizes)
        self.ln = nn.LayerNorm(total_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_channels, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.embed(context)          # (B, L, E)
        x = x.transpose(1, 2)            # (B, E, L)
        feats = []
        for conv in self.convs:
            h = F.gelu(conv(x))          # (B, C, L)
            h, _ = torch.max(h, dim=2)   # global max-pool → (B, C)
            feats.append(h)
        h_cat = torch.cat(feats, dim=1)  # (B, C * K)
        h_cat = self.ln(h_cat)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)            # (B, V)

