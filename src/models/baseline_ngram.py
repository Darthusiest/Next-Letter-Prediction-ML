"""
N-gram baseline for next-character prediction.
Count-based with Laplace smoothing; no gradient training.
Use .fit(ids) to count from a sequence, then .forward(context) returns log probs.
"""

from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn


class NGramModel(nn.Module):
    """
    (order-1)-gram context -> next character distribution.
    order=4 means we use the last 3 characters to predict the next.
    """

    def __init__(self, order: int = 4, vocab_size: int = 37, smoothing: float = 0.01):
        super().__init__()
        self.order = order
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # counts[context_tuple] = list of length vocab_size (counts for each next char)
        self._counts: dict = defaultdict(lambda: [0.0] * vocab_size)

    def fit(self, ids: List[int]) -> None:
        """Count n-grams from a single sequence of character ids."""
        n = self.order
        for i in range(len(ids) - n):
            context = tuple(ids[i : i + n - 1])
            next_id = ids[i + n - 1]
            if 0 <= next_id < self.vocab_size:
                self._counts[context][next_id] += 1.0

    def get_log_probs(self, context_tuple: Tuple[int, ...]) -> torch.Tensor:
        """Get log P(next | context) over vocab. context has length (order-1)."""
        counts = self._counts[context_tuple]
        # Laplace: add smoothing to each count
        smoothed = [c + self.smoothing for c in counts]
        total = sum(smoothed)
        probs = [x / total for x in smoothed]
        return torch.log(torch.tensor(probs, dtype=torch.float32))

    def forward(
        self, context: torch.Tensor
    ) -> torch.Tensor:
        """
        context: (batch, L) long tensor. We use the last (order-1) positions.
        Returns: (batch, vocab_size) log probabilities.
        """
        device = context.device
        B, L = context.shape
        n = self.order
        if L < n - 1:
            # Not enough context: return uniform log probs
            u = torch.log(torch.ones(B, self.vocab_size, device=device) / self.vocab_size)
            return u
        # Use last (n-1) characters
        ctx = context[:, -(n - 1) :]
        log_probs_list = []
        for b in range(B):
            key = tuple(ctx[b].tolist())
            log_probs_list.append(self.get_log_probs(key))
        out = torch.stack(log_probs_list, dim=0).to(device)
        return out
