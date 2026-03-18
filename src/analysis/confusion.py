"""
Confusion matrix and error-by-character plots for next-letter prediction.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils.plotting import new_figure, save_figure, display_char
from ..vocab import CharVocab
from ..models.analysis_wrapper import ModelAnalysisWrapper


def compute_confusion(
    model_wrapper: ModelAnalysisWrapper,
    dataloader: torch.utils.data.DataLoader,
    vocab: CharVocab,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Compute confusion matrix over characters on a dataset.
    Returns:
      - confusion matrix M where M[i, j] is count(true=i, pred=j)
      - per-id (correct, total) dict
    """
    device = model_wrapper.device
    num_classes = len(vocab.char2id)
    mat = np.zeros((num_classes, num_classes), dtype=int)
    per_id = defaultdict(lambda: [0, 0])  # id -> [correct, total]

    model = model_wrapper.model
    model.eval()
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            logits = model(context)
            pred = logits.argmax(dim=1)
            for t, p in zip(target.tolist(), pred.tolist()):
                mat[t, p] += 1
                per_id[t][1] += 1
                if t == p:
                    per_id[t][0] += 1

    return mat, per_id


def plot_confusion_matrix(mat: np.ndarray, vocab: CharVocab, output_dir: Path, show: bool = False) -> None:
    # Restrict to top-N most frequent true characters for readability.
    top_n = 40
    true_totals = mat.sum(axis=1)
    idxs = np.argsort(true_totals)[::-1][:top_n]
    sub = mat[np.ix_(idxs, idxs)].astype(float)
    # Normalize rows (per-true-char) so patterns are visible.
    row_sums = sub.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        sub = np.divide(sub, row_sums, out=np.zeros_like(sub), where=row_sums != 0)

    chars_full = list(vocab.char2id.keys())
    labels = [display_char(chars_full[i]) for i in idxs]
    n = len(labels)

    fig, ax = new_figure(figsize=(10, 8))
    im = ax.imshow(sub, interpolation="nearest", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Next-character Confusion Matrix (top {top_n}, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(pred | true)")
    save_figure(fig, output_dir / "plots" / "confusion_matrix.png", show=show)


def plot_error_by_character(
    per_id: Dict[int, Tuple[int, int]],
    vocab: CharVocab,
    output_dir: Path,
    show: bool = False,
) -> None:
    # Compute error rates and show top-N most frequent characters.
    rows = []
    for ch, idx in vocab.char2id.items():
        correct, total = per_id.get(idx, (0, 0))
        if total == 0:
            continue
        err_rate = 1.0 - (correct / total)
        rows.append((ch, err_rate, total))
    rows.sort(key=lambda t: t[2], reverse=True)
    top_n = 40
    rows = rows[:top_n]
    chars = [display_char(ch) for ch, _, _ in rows]
    errors = [er for _, er, _ in rows]

    fig, ax = new_figure(figsize=(11, 4.5))
    ax.bar(range(len(chars)), errors)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, rotation=0)
    ax.set_ylabel("Error rate (1 - accuracy)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Error Rate by Character on Test Set (top {top_n} by frequency)")
    save_figure(fig, output_dir / "plots" / "error_by_character.png", show=show)

