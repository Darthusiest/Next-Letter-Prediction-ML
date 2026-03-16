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
            # left-pad with the id of space if available, else first char id
            pad_id = self.vocab.char2id.get(" ", next(iter(self.vocab.char2id.values())))
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
            # Fallback: last hidden before final linear if attribute 'fc1' exists
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
        Current models do not expose attention weights, so this returns None.
        The transformer implementation can later be extended to record attention.
        """
        return None


