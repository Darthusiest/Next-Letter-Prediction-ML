"""
Character embedding visualizations (PCA and t-SNE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..utils.plotting import new_figure, save_figure


def _plot_2d(points: np.ndarray, labels, title: str, path: Path, show: bool) -> None:
    fig, ax = new_figure(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=30)
    for (x, y), label in zip(points, labels):
        ax.text(x, y, label, fontsize=8, ha="center", va="center")
    ax.set_title(title)
    save_figure(fig, path, show=show)


def plot_char_embeddings(
    embeddings: Dict[str, np.ndarray],
    output_dir: Path,
    show: bool = False,
) -> None:
    if not embeddings:
        return
    chars = list(embeddings.keys())
    mat = np.stack([embeddings[c] for c in chars], axis=0)

    plots_dir = output_dir / "plots"

    # PCA 2D
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(mat)
    _plot_2d(
        pca_2d,
        chars,
        "Character Embeddings (PCA 2D)",
        plots_dir / "embeddings_pca_2d.png",
        show,
    )

    # t-SNE 2D (can be slow for large vocab; here vocab is small)
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=5)
    tsne_2d = tsne.fit_transform(mat)
    _plot_2d(
        tsne_2d,
        chars,
        "Character Embeddings (t-SNE 2D)",
        plots_dir / "embeddings_tsne_2d.png",
        show,
    )

