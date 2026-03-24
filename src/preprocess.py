"""
Load raw text and clean/normalize for character-level modeling.
Predict any char: letters (upper/lower), digits, punctuation.
- Optionally lowercase / remove digits (default: keep both).
- Collapse runs of space/tab to single space; newline kept.
- Only keep allowed_chars (drop stray unicode etc.).
- Filter OCR garbage, boilerplate, and whitespace-dense noise.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_ALLOWED_CHARS, MAX_CHARS

logger = logging.getLogger(__name__)

# Patterns matched against raw lines (before allowed_chars filtering)
_PAGE_REF_RE = re.compile(r"\d+:\d+\s+(AM|PM)\s+Page\b", re.IGNORECASE)
_TOC_DOTS_RE = re.compile(r"\.{4,}")
_GUTENBERG_BOUNDARY_RE = re.compile(
    r"\*\*\*\s*(START|END)\s+OF\s+(THE|THIS)\s+PROJECT\s+GUTENBERG",
    re.IGNORECASE,
)


def load_text(
    paths: List[Path],
    max_chars: Optional[int] = MAX_CHARS,
    strip_gutenberg: bool = True,
) -> str:
    """
    Load and concatenate text from one or more files.
    If max_chars is set, truncate total length (for fast iteration).
    If strip_gutenberg is True, strip Project Gutenberg header/footer
    boilerplate from each file individually before concatenation.
    """
    parts: List[str] = []
    total = 0
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if strip_gutenberg:
            text = _strip_gutenberg_boilerplate(text)
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


def _is_noisy_line(line: str) -> bool:
    """
    Return True if a line is OCR garbage, boilerplate, or structural noise
    that would hurt character-level model training.
    """
    stripped = line.strip()
    if not stripped:
        return False  # blank lines handled separately

    non_ws = "".join(c for c in stripped if not c.isspace())
    non_ws_len = len(non_ws)

    # Very short fragments (single/double chars like "f", "CK", "3")
    if non_ws_len < 3:
        return True

    alpha_count = sum(1 for c in non_ws if c.isalpha())
    alpha_ratio = alpha_count / non_ws_len

    # Mostly non-alphabetic content (catches "4 ; y", "~ :", "_ 3", "22 155615")
    if alpha_ratio < 0.4:
        return True

    words = stripped.split()
    num_words = len(words)
    avg_word_len = sum(len(w) for w in words) / num_words if num_words else 0

    # Short 1-2 word lines under 8 chars are almost always OCR fragments
    # ("pesite", "iter:", "ere ie", "Tica", "Sy IFES")
    # but not chapter headings like "CHAPTER I" (9 chars).
    if num_words <= 2 and len(stripped) < 8:
        return True

    if num_words >= 3:
        # OCR fragments: many tiny words ("ere ie St", "De ei ue", "i eh et See er")
        if avg_word_len < 2.5:
            return True

        # Short lines with small words are fragment clusters, not sentences
        if avg_word_len < 3.0 and len(stripped) < 30:
            return True

        # High ratio of tiny words (<=2 chars) signals OCR fragmentation.
        # Normal prose rarely has >40% of words this short.
        tiny_words = sum(1 for w in words if len(w) <= 2)
        if tiny_words / num_words > 0.4 and num_words >= 5:
            return True

    # Table-of-contents lines ("THE MONSTER'S WIFE ............ 108")
    if _TOC_DOTS_RE.search(stripped):
        return True

    # PDF extraction page references ("01_559168 ffirs.qxd  11/30/04  11:38 PM  Page i")
    if _PAGE_REF_RE.search(stripped):
        return True

    return False


def _strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header/footer boilerplate.  Handles
    concatenated multi-book text by finding all START/END boundary
    pairs and keeping only the content between each pair, plus any
    text that falls outside of any boundary markers.
    """
    lines = text.split("\n")
    boundaries = []
    for i, line in enumerate(lines):
        m = _GUTENBERG_BOUNDARY_RE.search(line)
        if m:
            kind = "start" if m.group(1).upper() == "START" else "end"
            boundaries.append((i, kind))

    if not boundaries:
        return text

    kept: list[str] = []

    # Pair up boundaries into (START, END) pairs, keeping book content
    # and discarding license headers / donation footers.
    bi = 0
    while bi < len(boundaries):
        idx, kind = boundaries[bi]
        if kind == "start":
            if bi + 1 < len(boundaries) and boundaries[bi + 1][1] == "end":
                end_idx = boundaries[bi + 1][0]
                kept.extend(lines[idx + 1 : end_idx])
                bi += 2
            else:
                kept.extend(lines[idx + 1 :])
                return "\n".join(kept)
        else:
            bi += 1

    if not kept:
        return text

    return "\n".join(kept)


def filter_noisy_lines(text: str) -> str:
    """
    Remove OCR garbage, boilerplate, and whitespace-dense lines.
    Operates on raw text before allowed_chars filtering so that
    non-ASCII indicators (curly braces, angle brackets, etc.) are
    still visible for heuristic checks.

    NOTE: Gutenberg header/footer stripping is done per-file in
    load_text(), not here, to avoid boundary mismatches when
    multiple books are concatenated.
    """
    lines = text.split("\n")
    clean = []
    dropped = 0
    for line in lines:
        if _is_noisy_line(line):
            dropped += 1
            continue
        clean.append(line)

    # Collapse runs of 3+ blank lines to a single blank line
    result = "\n".join(clean)
    result = re.sub(r"\n{3,}", "\n\n", result)

    if dropped:
        logger.info("Noise filter: removed %d noisy lines", dropped)

    return result


def clean_text(
    text: str,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    remove_digits: bool = False,
    allowed_chars: Optional[set] = None,
    filter_noise: bool = True,
) -> str:
    """
    Normalize text for next-character prediction (any char: letter, digit, punctuation, case).
    - lowercase: if True, fold to lowercase (default False to keep capitalization).
    - collapse_whitespace: replace runs of space/tab with single space; newline kept.
    - remove_digits: if True, strip digits (default False to predict numbers).
    - allowed_chars: if set, only keep these; else use DEFAULT_ALLOWED_CHARS.
    - filter_noise: if True, remove OCR garbage and boilerplate before other cleaning.
    """
    if allowed_chars is None:
        allowed_chars = DEFAULT_ALLOWED_CHARS

    if filter_noise:
        text = filter_noisy_lines(text)

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
