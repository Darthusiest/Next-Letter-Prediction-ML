"""
Entropy and confidence analysis for next-character prediction.
"""

from __future__ import annotations

from math import log
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..utils.plotting import new_figure, save_figure


def compute_entropy_distribution(
    model_wrapper: ModelAnalysisWrapper,
    dataloader: torch.utils.data.DataLoader,
    max_samples: int,
) -> List[Tuple[str, float, str, float]]:
    """
    Sample contexts from dataloader and compute entropy of next-char distribution.
    Returns list of (context_str, entropy, top_char, top_prob).
    """
    device = model_wrapper.device
    vocab = model_wrapper.vocab
    results: List[Tuple[str, float, str, float]] = []

    model = model_wrapper.model
    model.eval()
    with torch.no_grad():
        for context_ids, _ in dataloader:
            context_ids = context_ids.to(device)
            # use raw ids to build context strings
            for row in context_ids:
                if len(results) >= max_samples:
                    return results
                ids = row.tolist()
                chars = [vocab.id2char[i] for i in ids]
                ctx_str = "".join(chars).rstrip()
                dist = model_wrapper.predict_next_distribution(ctx_str)
                probs = np.array(list(dist.values()), dtype=float)
                # avoid log(0)
                mask = probs > 0
                entropy = float(-np.sum(probs[mask] * np.log(probs[mask])))
                # most likely char
                top_char, top_prob = max(dist.items(), key=lambda kv: kv[1])
                results.append((ctx_str, entropy, top_char, top_prob))
    return results


def plot_entropy_hist(entropies: Iterable[float], output_dir: Path, show: bool = False) -> Path:
    entropies = list(entropies)
    if not entropies:
        return output_dir / "plots" / "entropy_histogram.png"
    plots_dir = output_dir / "plots"
    fig, ax = new_figure()
    ax.hist(entropies, bins=30)
    ax.set_xlabel("Entropy of next-character distribution")
    ax.set_ylabel("Count")
    ax.set_title("Entropy distribution over sampled contexts")
    out_path = plots_dir / "entropy_histogram.png"
    save_figure(fig, out_path, show=show)
    return out_path


def write_confidence_summary(
    samples: List[Tuple[str, float, str, float]],
    reports_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Write confidence_summary.txt describing most/least confident contexts.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "confidence_summary.txt"

    if not samples:
        path.write_text("No samples available for entropy/confidence analysis.\n", encoding="utf-8")
        return path

    # sort by entropy
    sorted_by_entropy = sorted(samples, key=lambda x: x[1])
    most_confident = sorted_by_entropy[:top_n]
    least_confident = list(reversed(sorted_by_entropy))[:top_n]

    with path.open("w", encoding="utf-8") as f:
        f.write("Entropy / confidence summary for next-character prediction\n")
        f.write("=========================================================\n\n")

        f.write(f"Total sampled contexts: {len(samples)}\n\n")

        f.write("Most confident contexts (lowest entropy):\n")
        for ctx, ent, top_ch, top_p in most_confident:
            f.write(f"  ctx={repr(ctx)}  entropy={ent:.4f}  top='{top_ch}'  P(top)={top_p:.4f}\n")

        f.write("\nLeast confident contexts (highest entropy):\n")
        for ctx, ent, top_ch, top_p in least_confident:
            f.write(f"  ctx={repr(ctx)}  entropy={ent:.4f}  top='{top_ch}'  P(top)={top_p:.4f}\n")

    return path


