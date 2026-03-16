"""
Nearest-neighbor similarity reports for characters and contexts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.analysis_wrapper import ModelAnalysisWrapper


def _nearest_neighbors(
    labels: List[str],
    vectors: np.ndarray,
    target_indices: Iterable[int],
    top_k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    sims = cosine_similarity(vectors)
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for idx in target_indices:
        row = sims[idx]
        # exclude self, sort others
        order = np.argsort(row)[::-1]
        result: List[Tuple[str, float]] = []
        for j in order:
            if j == idx:
                continue
            result.append((labels[j], float(row[j])))
            if len(result) >= top_k:
                break
        neighbors[labels[idx]] = result
    return neighbors


def run_similarity_report(
    model_wrapper: ModelAnalysisWrapper,
    chars: List[str],
    contexts: List[str],
    reports_dir: Path,
    top_k: int = 10,
) -> Path | None:
    """
    Generate similarity_report.txt for characters and contexts, if vectors exist.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "similarity_report.txt"

    # Character embeddings
    char_embs = model_wrapper.get_char_embeddings()
    have_char_section = char_embs is not None

    # Context representations
    ctx_vectors: Dict[str, np.ndarray] = {}
    for ctx in contexts:
        vec = model_wrapper.get_context_representation(ctx)
        if vec is not None:
            ctx_vectors[ctx] = vec
    have_ctx_section = bool(ctx_vectors)

    if not have_char_section and not have_ctx_section:
        # Nothing to write; signal to caller that we skipped
        return None

    with path.open("w", encoding="utf-8") as f:
        f.write("Nearest-neighbor similarity report\n")
        f.write("=================================\n\n")

        if have_char_section:
            labels = []
            vecs = []
            for ch, vec in char_embs.items():
                labels.append(ch)
                vecs.append(vec)
            mat = np.stack(vecs, axis=0)
            target_indices = [labels.index(ch) for ch in chars if ch in labels]
            f.write("Character neighbors (cosine similarity):\n")
            if not target_indices:
                f.write("  (No requested characters found in vocabulary.)\n\n")
            else:
                nn = _nearest_neighbors(labels, mat, target_indices, top_k)
                for ch in chars:
                    if ch not in nn:
                        continue
                    f.write(f"  Target '{ch}':\n")
                    for other, sim in nn[ch]:
                        f.write(f"    {other!r}: {sim:.4f}\n")
                f.write("\n")

        if have_ctx_section:
            labels = list(ctx_vectors.keys())
            mat = np.stack([ctx_vectors[c] for c in labels], axis=0)
            target_indices = list(range(len(labels)))
            f.write("Context neighbors (cosine similarity):\n")
            nn = _nearest_neighbors(labels, mat, target_indices, top_k)
            for ctx in labels:
                f.write(f"  Target {ctx!r}:\n")
                for other, sim in nn[ctx]:
                    f.write(f"    {other!r}: {sim:.4f}\n")
            f.write("\n")

    return path


