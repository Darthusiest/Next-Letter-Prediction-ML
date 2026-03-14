"""
Character vocabulary: build from text, encode/decode, save/load.
Maps each character to a unique id in [0, V-1].
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


# Special token id for unknown (if we ever add OOV handling)
UNK_ID = 0


def build_vocab(text: str, allowed_chars: Optional[str] = None) -> Dict[str, int]:
    """
    Build char -> id mapping from character set or from characters in text.
    If allowed_chars is given, vocab is built in that order (deterministic).
    Otherwise we use sorted(set(text)); id 0 is reserved for UNK.
    """
    if allowed_chars is not None:
        chars = [c for c in allowed_chars if c in text or True]
        # Build from allowed set so all expected chars have an id
        seen = set(text)
        chars = list(allowed_chars) if isinstance(allowed_chars, (list, set)) else list(allowed_chars)
        # Only include chars that appear or we explicitly want
        char_set = set(chars)
        for c in sorted(seen):
            if c not in char_set:
                char_set.add(c)
        char_list = sorted(char_set)
    else:
        char_list = sorted(set(text))

    return {c: i for i, c in enumerate(char_list)}


def build_vocab_from_charset(charset: Union[str, set]) -> Dict[str, int]:
    """
    Build vocab from a fixed character set (e.g. a-z + space + punctuation).
    Deterministic order: sorted.
    """
    char_list = sorted(set(charset))
    return {c: i for i, c in enumerate(char_list)}


class CharVocab:
    """Character vocabulary with encode/decode and save/load."""

    def __init__(self, char2id: Dict[str, int]):
        self.char2id = char2id
        self.id2char = {i: c for c, i in char2id.items()}
        self.vocab_size = len(char2id)

    def encode(self, s: str) -> List[int]:
        """Encode string to list of ids. All characters in s must be in vocab (use cleaned text with same charset)."""
        return [self.char2id[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        """Decode list of ids to string. Unknown ids use '?' if not in id2char."""
        return "".join(self.id2char.get(i, "?") for i in ids)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharVocab":
        with open(path, encoding="utf-8") as f:
            char2id = json.load(f)
        return cls(char2id)


def build_vocab_from_text(text: str, allowed_chars: Optional[set] = None) -> CharVocab:
    """
    Build CharVocab from text. If allowed_chars is set, vocab is built from
    that set (sorted) so cleaned text using the same charset is always encodable.
    """
    if allowed_chars is not None:
        char_list = sorted(allowed_chars)
        char2id = {c: i for i, c in enumerate(char_list)}
    else:
        char2id = build_vocab(text)
    return CharVocab(char2id)
