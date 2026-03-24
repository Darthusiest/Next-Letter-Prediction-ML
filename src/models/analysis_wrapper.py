"""
ModelAnalysisWrapper: a thin, model-agnostic API for post-training analysis.

This wrapper exposes:
- predict_next_distribution(context: str)
- get_char_embeddings()
- get_context_representation(context: str)
- get_attention_weights(context: str)  # transformer only

so that analysis code can work across MLP / RNN / CNN / Transformer models
without depending on their internal implementation details.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..vocab import CharVocab


class ModelAnalysisWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        vocab: CharVocab,
        device: torch.device,
        model_type: str,
        context_length: int,
    ) -> None:
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model_type = model_type
        self.context_length = context_length
        self.model.eval()

    # ------------------------------------------------------------------ helpers
    def _prepare_context_tensor(self, context: str) -> torch.Tensor:
        """
        Encode context string into a tensor of shape (1, L) with padding/truncation.
        """
        ids = self.vocab.encode(context)
        if len(ids) >= self.context_length:
            ids = ids[-self.context_length :]
        else:
            pad_id = self.vocab.encode(" ")[0] if self.vocab.encode(" ") else 0
            pad_len = self.context_length - len(ids)
            ids = [pad_id] * pad_len + ids
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return x

    # ---------------------------------------------------------------- prediction
    def predict_next_distribution(self, context: str) -> Dict[str, float]:
        """
        Return {char: prob} for P(next_char | context).
        Works for all neural models that map (B, L) ids -> logits over vocab.
        """
        x = self._prepare_context_tensor(context)
        with torch.no_grad():
            logits = self.model(x)  # (1, V)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
        return {self.vocab.id2char[i]: float(p) for i, p in enumerate(probs)}

    # ------------------------------------------------------------ embeddings API
    def get_char_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """
        If the model has an embedding layer called 'embed', return {char: vector}.
        Otherwise return None.
        """
        embed = getattr(self.model, "embed", None)
        if embed is None or not hasattr(embed, "weight"):
            return None
        weight = embed.weight.detach().cpu().numpy()
        return {self.vocab.id2char[i]: weight[i] for i in range(weight.shape[0])}

    # --------------------------------------------------------- context features
    def get_context_representation(self, context: str) -> Optional[np.ndarray]:
        """
        Return a single vector representing `context`.
        Implementation is deliberately simple:
        - For MLP: hidden layer before logits.
        - For RNN: final hidden state.
        - For CNN: pooled feature vector before final linear layer.
        - For Transformer: final token state.
        Returns None if not implemented.
        """
        x = self._prepare_context_tensor(context)
        with torch.no_grad():
            if self.model_type == "rnn" and hasattr(self.model, "lstm"):
                emb = self.model.embed(x)
                output, (h_n, _) = self.model.lstm(emb)
                # h_n: (num_layers, B, H) -> take last layer
                vec = h_n[-1, 0].cpu().numpy()
                return vec
            if self.model_type == "cnn" and hasattr(self.model, "convs"):
                emb = self.model.embed(x).transpose(1, 2)
                feats = []
                for conv in self.model.convs:
                    h = torch.relu(conv(emb))
                    h, _ = torch.max(h, dim=2)
                    feats.append(h)
                h_cat = torch.cat(feats, dim=1)[0]
                return h_cat.cpu().numpy()
            # MLP with self-attention + batched multi-head attention pooling
            if hasattr(self.model, "attn_pool") and hasattr(self.model, "self_attn"):
                emb = self.model.embed(x) + self.model.pos_embed
                emb = self.model.self_attn(emb)
                scores = self.model.attn_pool(emb)
                weights = F.softmax(scores, dim=1)
                h = (weights.transpose(1, 2) @ emb).reshape(1, -1)
                h = F.gelu(self.model.proj(h))
                for block in self.model.blocks:
                    h = block(h)
                h = self.model.final_ln(h)
                return h[0].cpu().numpy()
            # MLP with self-attention + ModuleList attention heads (legacy)
            if hasattr(self.model, "attn_heads") and hasattr(self.model, "self_attn"):
                emb = self.model.embed(x) + self.model.pos_embed
                emb = self.model.self_attn(emb)
                pooled = []
                for head in self.model.attn_heads:
                    w = F.softmax(head(emb).squeeze(-1), dim=1)
                    pooled.append((emb * w.unsqueeze(-1)).sum(dim=1))
                h = torch.cat(pooled, dim=-1)
                h = F.gelu(self.model.proj(h))
                for block in self.model.blocks:
                    h = block(h)
                h = self.model.final_ln(h)
                return h[0].cpu().numpy()
            # Old MLP with single-head attention pooling
            if hasattr(self.model, "attn_score") and hasattr(self.model, "proj"):
                emb = self.model.embed(x) + self.model.pos_embed
                attn_w = F.softmax(self.model.attn_score(emb).squeeze(-1), dim=1)
                pooled = (emb * attn_w.unsqueeze(-1)).sum(dim=1)
                h = F.gelu(self.model.proj(pooled))[0]
                return h.cpu().numpy()
            # Legacy MLP with flatten + fc1
            if hasattr(self.model, "embed") and hasattr(self.model, "fc1"):
                emb = self.model.embed(x)
                flat = emb.view(emb.size(0), -1)
                h = torch.relu(self.model.fc1(flat))[0]
                return h.cpu().numpy()
        return None

    # -------------------------------------------------------------- attention API
    def get_attention_weights(self, context: str) -> Optional[np.ndarray]:
        """
        For transformers only.
        If the transformer model supports attention recording, return a numpy array
        with shape (num_layers, num_heads, L, L) for the given context.
        """
        if self.model_type != "transformer":
            return None
        if not hasattr(self.model, "attn_weights"):
            return None
        x = self._prepare_context_tensor(context)
        with torch.no_grad():
            # Forward pass in attention-recording mode. The model is expected
            # to populate self.model.attn_weights as a list of tensors.
            try:
                _ = self.model(x, return_attn=True)
            except TypeError:
                return None
        attn_list = getattr(self.model, "attn_weights", None)
        if not attn_list:
            return None
        # Each element: (B, heads, L, L). Use batch 0.
        stacked = torch.stack([a[0] for a in attn_list], dim=0)  # (layers, heads, L, L)
        return stacked.numpy()


