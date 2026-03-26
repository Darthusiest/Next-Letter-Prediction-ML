"""
MLP with stacked causal self-attention (RoPE), multi-head attention pooling,
and SwiGLU residual blocks for next-letter prediction.

Architecture:
  embed(context) [no pos_embed -- RoPE applied in attention]
  → embed_drop
  → N causal self-attention layers with RoPE (positions interact left-to-right)
  → multi-head attention pooling → (B, num_heads * embed_dim)
  → proj → GELU → dropout
  → SwiGLU residual blocks with pre-LayerNorm
  → final LayerNorm → [tie_proj] → logits

Modern LLM techniques applied:
  1. Rotary Position Embeddings (RoPE) for relative position encoding
  2. Causal masking in self-attention (left-to-right only)
  3. Multi-head attention pooling (removes bottleneck)
  4. SwiGLU activation in residual blocks
  5. Pre-LayerNorm (LN on sublayer input; stabler gradients)
  6. Stacked self-attention layers (configurable depth)
  7. KV caching for efficient autoregressive generation
  8. Weight tying (embed weights shared with output projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionEmbedding(nn.Module):
    """Precompute and apply rotary position embeddings to Q and K tensors."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings. q, k shape: (B, heads, L, head_dim)."""
        seq_len = q.shape[2]
        self._build_cache(offset + seq_len)
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


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
    """Pre-LayerNorm causal multi-head self-attention with RoPE and optional KV cache."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self._dropout_p = dropout

        self.ln = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, E = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv.unbind(0)

        if rope is not None:
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            q, k = rope(q, k, offset=offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None and L > 1),
            dropout_p=self._dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, E)
        return x + self.resid_drop(self.out_proj(out)), new_cache


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
        num_self_attn_layers: int = 1,
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
        self.num_self_attn_layers = num_self_attn_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        head_dim = embed_dim // num_attn_heads
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len=context_length * 4)

        self.self_attn_layers = nn.ModuleList(
            [MultiHeadSelfAttention(embed_dim, num_attn_heads, dropout)
             for _ in range(num_self_attn_layers)]
        )

        self.attn_pool = nn.Linear(embed_dim, num_attn_heads)
        pool_dim = num_attn_heads * embed_dim

        self.proj = nn.Linear(pool_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SwiGLUBlock(hidden_dim, dropout) for _ in range(num_hidden_layers - 1)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        if hidden_dim != embed_dim:
            self.tie_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.tie_proj = None
        self.fc2 = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc2.weight = self.embed.weight

    def forward(
        self,
        context: torch.Tensor,
        kv_cache: Optional[List] = None,
        use_cache: bool = False,
    ):
        x = self.embed(context)                            # (B, L, E)
        x = self.embed_drop(x)

        new_kv_caches: list = []
        for i, layer in enumerate(self.self_attn_layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = layer(
                x, rope=self.rope, kv_cache=layer_cache, use_cache=use_cache,
            )
            new_kv_caches.append(new_cache)

        # For incremental decoding, concatenate cached hidden states with new
        pool_input = x
        if kv_cache is not None and len(kv_cache) > len(self.self_attn_layers):
            cached_hidden = kv_cache[len(self.self_attn_layers)]
            pool_input = torch.cat([cached_hidden, x], dim=1)

        if use_cache:
            new_kv_caches.append(pool_input.detach())

        # Batched multi-head attention pooling
        scores = self.attn_pool(pool_input)                # (B, L', N)
        weights = F.softmax(scores, dim=1)                 # (B, L', N)
        x = weights.transpose(1, 2) @ pool_input           # (B, N, E)
        x = x.reshape(x.size(0), -1)                      # (B, N*E)

        x = self.dropout(F.gelu(self.proj(x)))             # (B, H)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)                              # (B, H)

        if self.tie_proj is not None:
            x = self.tie_proj(x)                          # (B, E)
        logits = self.fc2(x)                              # (B, V)

        if use_cache:
            return logits, new_kv_caches
        return logits

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Handle legacy checkpoint formats (pos_embed, single self_attn)."""
        state_dict = dict(state_dict)
        if "self_attn.qkv.weight" in state_dict:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("self_attn."):
                    remapped["self_attn_layers.0." + k[len("self_attn."):]] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        state_dict.pop("pos_embed", None)
        return super().load_state_dict(state_dict, *args, **kwargs)
