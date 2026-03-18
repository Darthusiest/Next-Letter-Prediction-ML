"""
Corpus-level plots: character frequency histogram and bigram heatmap.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import new_figure, save_figure, display_char
from ..vocab import CharVocab


def plot_char_frequency(text: str, vocab: CharVocab, output_dir: Path, show: bool = False) -> None:
    counts = Counter(c for c in text if c in vocab.char2id)
    # Show top-N most frequent characters for readability.
    top_n = 40
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_items = items[:top_n]
    other_total = sum(v for _, v in items[top_n:])
    chars = [c for c, _ in top_items] + (["<OTHER>"] if other_total > 0 else [])
    freqs = [v for _, v in top_items] + ([other_total] if other_total > 0 else [])

    plots_dir = output_dir / "plots"
    fig, ax = new_figure(figsize=(11, 4.5))
    ax.bar(range(len(chars)), freqs)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(
        [display_char(c) if c != "<OTHER>" else "<OTHER>" for c in chars], rotation=0
    )
    ax.set_ylabel("Count")
    ax.set_title(f"Character Frequency in Training Corpus (top {top_n})")
    save_figure(fig, plots_dir / "char_frequency.png", show=show)


def plot_bigram_heatmap(text: str, vocab: CharVocab, output_dir: Path, show: bool = False) -> None:
    # Build bigram counts for characters in vocab
    bigram_counts: Dict[tuple, int] = defaultdict(int)
    filtered = [c for c in text if c in vocab.char2id]
    for c1, c2 in zip(filtered, filtered[1:]):
        bigram_counts[(c1, c2)] += 1

    # Restrict heatmap to top-N characters by unigram frequency for readability.
    top_n = 40
    uni_counts = Counter(filtered)
    top_chars = [c for c, _ in uni_counts.most_common(top_n)]
    idx = {c: i for i, c in enumerate(top_chars)}
    n = len(top_chars)
    mat = np.zeros((n, n), dtype=float)
    for (c1, c2), cnt in bigram_counts.items():
        if c1 not in idx or c2 not in idx:
            continue
        i = idx[c1]
        j = idx[c2]
        mat[i, j] += cnt

    # Normalize rows to probabilities where possible
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    plots_dir = output_dir / "plots"
    fig, ax = new_figure(figsize=(10, 8))
    im = ax.imshow(mat, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [display_char(c) for c in top_chars]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next character")
    ax.set_ylabel("Current character")
    ax.set_title(f"Character Bigram Probabilities in Corpus (top {top_n})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, plots_dir / "bigram_heatmap.png", show=show)


