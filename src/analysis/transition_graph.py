"""
Letter transition network graph based on bigram statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..vocab import CharVocab
from ..utils.plotting import save_figure


def _compute_bigram_probs(text: str, vocab: CharVocab) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], int] = {}
    total_out: dict[str, int] = {}
    for a, b in zip(text[:-1], text[1:]):
        if a not in vocab.char2id or b not in vocab.char2id:
            continue
        key = (a, b)
        counts[key] = counts.get(key, 0) + 1
        total_out[a] = total_out.get(a, 0) + 1
    probs: dict[tuple[str, str], float] = {}
    for (a, b), c in counts.items():
        denom = total_out.get(a, 1)
        probs[(a, b)] = c / denom
    return probs


def run_transition_graph(
    text: str,
    vocab: CharVocab,
    output_dir: Path,
    threshold: float = 0.05,
    max_out_edges: int = 5,
    show: bool = False,
) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    probs = _compute_bigram_probs(text, vocab)

    G = nx.DiGraph()

    # add nodes for all characters seen
    for ch in vocab.id2char:
        G.add_node(ch)

    # for each source char, keep top-K outgoing edges above threshold
    outgoing: dict[str, list[tuple[str, float]]] = {}
    for (a, b), p in probs.items():
        if p < threshold:
            continue
        outgoing.setdefault(a, []).append((b, p))

    for a, edges in outgoing.items():
        edges.sort(key=lambda x: x[1], reverse=True)
        for b, p in edges[:max_out_edges]:
            G.add_edge(a, b, weight=p)

    # filter out isolated nodes with no edges to keep plot readable
    H = G.edge_subgraph(G.edges()).copy()
    if not H.nodes:
        path = plots_dir / "letter_transition_graph.png"
        fig = plt.figure()
        fig.text(0.5, 0.5, "No transitions above threshold.", ha="center", va="center")
        save_figure(fig, path, show=show)
        return path

    pos = nx.spring_layout(H, seed=42)
    weights = [H[u][v]["weight"] for u, v in H.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 4.0 * (w / max_w) for w in weights]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    nx.draw_networkx_nodes(H, pos, node_size=500, ax=ax)
    nx.draw_networkx_edges(H, pos, width=widths, arrows=True, arrowstyle="->", ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
    ax.set_title("Letter transition graph (corpus bigrams)")
    ax.axis("off")

    path = plots_dir / "letter_transition_graph.png"
    save_figure(fig, path, show=show)
    return path


