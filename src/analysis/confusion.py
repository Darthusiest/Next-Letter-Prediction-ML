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
    chars = list(vocab.char2id.keys())
    n = len(chars)
    fig, ax = new_figure(figsize=(8, 6))
    im = ax.imshow(mat, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [display_char(c) for c in chars]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Next-character Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_dir / "plots" / "confusion_matrix.png", show=show)


def plot_error_by_character(
    per_id: Dict[int, Tuple[int, int]],
    vocab: CharVocab,
    output_dir: Path,
    show: bool = False,
) -> None:
    chars = []
    errors = []
    for ch, idx in vocab.char2id.items():
        correct, total = per_id.get(idx, (0, 0))
        if total == 0:
            continue
        err_rate = 1.0 - (correct / total)
        chars.append(ch)
        errors.append(err_rate)

    fig, ax = new_figure(figsize=(8, 4))
    ax.bar(range(len(chars)), errors)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels([display_char(c) for c in chars], rotation=90)
    ax.set_ylabel("Error rate")
    ax.set_title("Error Rate by Character on Test Set")
    save_figure(fig, output_dir / "plots" / "error_by_character.png", show=show)

