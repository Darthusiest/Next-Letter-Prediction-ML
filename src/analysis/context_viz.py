"""
Context representation similarity visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..utils.plotting import new_figure, save_figure


def get_context_vectors(
    model_wrapper: ModelAnalysisWrapper,
    contexts,
) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    for ctx in contexts:
        vec = model_wrapper.get_context_representation(ctx)
        if vec is not None:
            vectors[ctx] = vec
    return vectors


def plot_context_similarity_heatmap(
    vectors: Dict[str, np.ndarray],
    output_dir: Path,
    show: bool = False,
) -> Path:
    labels = list(vectors.keys())
    if not labels:
        return output_dir / "plots" / "context_similarity_heatmap.png"
    mat = np.stack([vectors[c] for c in labels], axis=0)
    sim = cosine_similarity(mat)

    plots_dir = output_dir / "plots"
    fig, ax = new_figure(figsize=(6, 5))
    im = ax.imshow(sim, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("Context representation cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = plots_dir / "context_similarity_heatmap.png"
    save_figure(fig, out_path, show=show)
    return out_path


def plot_context_pca(
    vectors: Dict[str, np.ndarray],
    output_dir: Path,
    enable_3d: bool = False,
    show: bool = False,
) -> Path:
    labels = list(vectors.keys())
    if not labels:
        return output_dir / "plots" / "context_pca_2d.png"
    mat = np.stack([vectors[c] for c in labels], axis=0)

    plots_dir = output_dir / "plots"

    # 2D PCA
    pca = PCA(n_components=2)
    pts2 = pca.fit_transform(mat)
    fig, ax = new_figure(figsize=(6, 6))
    ax.scatter(pts2[:, 0], pts2[:, 1])
    for (x, y), label in zip(pts2, labels):
        ax.text(x, y, label, fontsize=8, ha="center", va="center")
    ax.set_title("Context representations (PCA 2D)")
    out_2d = plots_dir / "context_pca_2d.png"
    save_figure(fig, out_2d, show=show)

    if enable_3d:
        pca3 = PCA(n_components=3)
        pts3 = pca3.fit_transform(mat)
        fig = plt.figure(figsize=(6, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2])
        for (x, y, z), label in zip(pts3, labels):
            ax3.text(x, y, z, label, fontsize=7)
        ax3.set_title("Context representations (PCA 3D)")
        out_3d = plots_dir / "context_pca_3d.png"
        save_figure(fig, out_3d, show=show)

    return out_2d


