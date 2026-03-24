"""
MLP with self-attention, multi-head attention pooling, and SwiGLU residual blocks
for next-letter prediction.

Architecture:
  embed(context) + pos_embed → embed_drop
  → single-layer multi-head self-attention (positions interact)
  → multi-head attention pooling → (B, num_heads * embed_dim)
  → proj → GELU → dropout
  → SwiGLU residual blocks with pre-LayerNorm
  → final LayerNorm → [tie_proj] → logits

Modern LLM techniques applied:
  1. Multi-head attention pooling (replaces single-head; removes bottleneck)
  2. SwiGLU activation in residual blocks (replaces GELU)
  3. Pre-LayerNorm (LN on sublayer input, not output; stabler gradients)
  4. Self-attention layer before pooling (positions interact before aggregation)
  5. Weight tying (embed weights shared with output projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUBlock(nn.Module):
    """Pre-LayerNorm residual block with SwiGLU: x + drop(SiLU(W_gate(LN(x))) * W_up(LN(x)))."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, dim)
        self.w_up = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        return x + self.drop(F.silu(self.w_gate(h)) * self.w_up(h))


class MultiHeadSelfAttention(nn.Module):
    """Pre-LayerNorm multi-head self-attention using scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self._dropout_p = dropout

        self.ln = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, E = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self._dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, E)
        return x + self.resid_drop(self.out_proj(out))


class MLPCharModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_hidden_layers: int = 1,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if embed_dim % num_attn_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_attn_heads ({num_attn_heads})"
            )
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        self.self_attn = MultiHeadSelfAttention(embed_dim, num_attn_heads, dropout)

        # Batched attention pooling: one Linear scores all N heads at once.
        # Each head learns independent position-importance scores, softmax over
        # L, then produces an embed_dim-sized weighted sum.  Output: (B, N*E).
        self.attn_pool = nn.Linear(embed_dim, num_attn_heads)
        pool_dim = num_attn_heads * embed_dim

        self.proj = nn.Linear(pool_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SwiGLUBlock(hidden_dim, dropout) for _ in range(num_hidden_layers - 1)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Weight tying: if hidden_dim != embed_dim, project back first
        if hidden_dim != embed_dim:
            self.tie_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.tie_proj = None
        self.fc2 = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc2.weight = self.embed.weight

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.embed(context) + self.pos_embed          # (B, L, E)
        x = self.embed_drop(x)
        x = self.self_attn(x)                             # (B, L, E)

        # Batched multi-head attention pooling: single matmul, no Python loop
        scores = self.attn_pool(x)                         # (B, L, N)
        weights = F.softmax(scores, dim=1)                 # (B, L, N)
        x = weights.transpose(1, 2) @ x                   # (B, N, E)
        x = x.reshape(x.size(0), -1)                      # (B, N*E)

        x = self.dropout(F.gelu(self.proj(x)))             # (B, H)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)                              # (B, H)

        if self.tie_proj is not None:
            x = self.tie_proj(x)                          # (B, E)
        return self.fc2(x)                                # (B, V)
