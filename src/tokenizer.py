"""
BPE subword tokenizer using HuggingFace tokenizers.
Exposes the same encode/decode/vocab_size interface as CharVocab
so the rest of the pipeline works without branching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained on a corpus."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self._vocab_size = self._tok.get_vocab_size()
        self._build_id2token()

    def _build_id2token(self) -> None:
        vocab = self._tok.get_vocab()
        self.id2char = {i: tok for tok, i in vocab.items()}
        self.char2id = vocab

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 2000,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Train a BPE tokenizer from scratch on the given text."""
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[],
            show_progress=False,
        )
        tok.train_from_iterator([text], trainer=trainer)
        return cls(tok)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, s: str) -> List[int]:
        """Encode string to list of token ids."""
        return self._tok.encode(s).ids

    def decode(self, ids: List[int]) -> str:
        """Decode list of token ids back to string."""
        return self._tok.decode(ids)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> BPETokenizer:
        """Load tokenizer from a JSON file."""
        tok = Tokenizer.from_file(str(path))
        return cls(tok)
