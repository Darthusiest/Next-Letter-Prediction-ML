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
    chars = [c for c in vocab.char2id.keys()]
    freqs = [counts.get(c, 0) for c in chars]

    plots_dir = output_dir / "plots"
    fig, ax = new_figure(figsize=(8, 4))
    ax.bar(range(len(chars)), freqs)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels([display_char(c) for c in chars], rotation=90)
    ax.set_ylabel("Count")
    ax.set_title("Character Frequency in Training Corpus")
    save_figure(fig, plots_dir / "char_frequency.png", show=show)


def plot_bigram_heatmap(text: str, vocab: CharVocab, output_dir: Path, show: bool = False) -> None:
    # Build bigram counts for characters in vocab
    bigram_counts: Dict[tuple, int] = defaultdict(int)
    filtered = [c for c in text if c in vocab.char2id]
    for c1, c2 in zip(filtered, filtered[1:]):
        bigram_counts[(c1, c2)] += 1

    chars = list(vocab.char2id.keys())
    n = len(chars)
    mat = np.zeros((n, n), dtype=float)
    for (c1, c2), cnt in bigram_counts.items():
        i = vocab.char2id[c1]
        j = vocab.char2id[c2]
        mat[i, j] = cnt

    # Normalize rows to probabilities where possible
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    plots_dir = output_dir / "plots"
    fig, ax = new_figure(figsize=(8, 6))
    im = ax.imshow(mat, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [display_char(c) for c in chars]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next character")
    ax.set_ylabel("Current character")
    ax.set_title("Character Bigram Probabilities in Corpus")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, plots_dir / "bigram_heatmap.png", show=show)


