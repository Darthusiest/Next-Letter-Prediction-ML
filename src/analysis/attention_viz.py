"""
Transformer attention heatmaps (optional).

Note: This relies on ModelAnalysisWrapper.get_attention_weights returning
attention tensors for transformer models. If that is not yet implemented
for a given checkpoint, the caller should treat a None return as "skipped".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..utils.plotting import new_figure, save_figure


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "ctx"


def run_attention_viz(
    model_wrapper: ModelAnalysisWrapper,
    contexts: Iterable[str],
    output_dir: Path,
    layers: List[int] | None = None,
    heads: List[int] | None = None,
    show: bool = False,
) -> List[Path] | None:
    """
    Generate attention heatmaps for selected contexts.

    Returns list of created paths, or None if attention is unavailable.
    """
    plots: List[Path] = []
    plots_root = output_dir / "plots" / "attention"
    plots_root.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        attn = model_wrapper.get_attention_weights(ctx)
        if attn is None:
            # If any context returns None we treat the whole feature as unavailable.
            if not plots:
                return None
            break

        # Expected shape: (num_layers, num_heads, L, L)
        num_layers, num_heads, L, _ = attn.shape
        layer_indices = layers if layers is not None else [0, num_layers - 1]
        head_indices = heads if heads is not None else [0]
        layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
        head_indices = [h for h in head_indices if 0 <= h < num_heads]

        chars = list(ctx[-L:])

        ctx_slug = _slugify(ctx)
        for li in layer_indices:
            for hi in head_indices:
                weights = attn[li, hi]  # (L, L)
                fig, ax = new_figure(figsize=(5, 4))
                im = ax.imshow(weights, interpolation="nearest")
                ax.set_xticks(range(L))
                ax.set_yticks(range(L))
                ax.set_xticklabels(chars, rotation=90)
                ax.set_yticklabels(chars)
                ax.set_xlabel("Key positions")
                ax.set_ylabel("Query positions")
                ax.set_title(f"Attention - ctx={ctx!r}, layer={li}, head={hi}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                out_path = plots_root / f"{ctx_slug}_layer{li}_head{hi}.png"
                save_figure(fig, out_path, show=show)
                plots.append(out_path)

    return plots


