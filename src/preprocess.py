"""
Load raw text and clean/normalize for character-level modeling.
Predict any char: letters (upper/lower), digits, punctuation.
- Optionally lowercase / remove digits (default: keep both).
- Collapse runs of space/tab to single space; newline kept.
- Only keep allowed_chars (drop stray unicode etc.).
"""

import re
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_ALLOWED_CHARS, MAX_CHARS


def load_text(paths: List[Path], max_chars: Optional[int] = MAX_CHARS) -> str:
    """
    Load and concatenate text from one or more files.
    If max_chars is set, truncate total length (for fast iteration).
    """
    parts: List[str] = []
    total = 0
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None:
            remaining = max_chars - total
            if remaining <= 0:
                break
            text = text[:remaining]
            total += len(text)
        parts.append(text)
        if max_chars is not None and total >= max_chars:
            break
    return "".join(parts)


def clean_text(
    text: str,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    remove_digits: bool = False,
    allowed_chars: Optional[set] = None,
) -> str:
    """
    Normalize text for next-character prediction (any char: letter, digit, punctuation, case).
    - lowercase: if True, fold to lowercase (default False to keep capitalization).
    - collapse_whitespace: replace runs of space/tab with single space; newline kept.
    - remove_digits: if True, strip digits (default False to predict numbers).
    - allowed_chars: if set, only keep these; else use DEFAULT_ALLOWED_CHARS.
    """
    if allowed_chars is None:
        allowed_chars = DEFAULT_ALLOWED_CHARS

    if lowercase:
        text = text.lower()

    if remove_digits:
        text = re.sub(r"\d+", " ", text)

    if collapse_whitespace:
        # Collapse space and tab to single space; keep newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = text.strip()

    # Keep only allowed characters; drop others (e.g. stray unicode)
    result = "".join(c for c in text if c in allowed_chars)

    if collapse_whitespace:
        result = re.sub(r" +", " ", result)

    return result
