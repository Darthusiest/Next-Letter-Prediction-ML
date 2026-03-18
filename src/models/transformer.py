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


class _EncoderLayerWithAttn(nn.Module):
    """
    Minimal Transformer encoder layer that exposes per-head attention weights.
    Implemented so post-training analysis can visualize attention heatmaps.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_weights = None
        if return_attn:
            attn_out, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,  # (B, heads, L, L)
            )
        else:
            attn_out, _ = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                need_weights=False,
            )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x, attn_weights


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
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=context_length)

        self.layers = nn.ModuleList(
            [
                _EncoderLayerWithAttn(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.attn_weights: list[torch.Tensor] = []

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # Mask out future positions (upper triangular)
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, context: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """
        context: (batch, context_length) long tensor of character ids.
        Returns: (batch, vocab_size) logits.
        """
        x = self.embed(context)              # (B, L, D)
        x = self.pos_encoding(x)             # add positional information
        L = x.size(1)
        mask = self._causal_mask(L, x.device)
        if return_attn:
            self.attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=mask, return_attn=return_attn)
            if return_attn and attn is not None:
                self.attn_weights.append(attn.detach().cpu())
        encoded = x
        last = encoded[:, -1, :]             # final position representation
        last = self.dropout(last)
        logits = self.fc(last)               # (B, V)
        return logits

