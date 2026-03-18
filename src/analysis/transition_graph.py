"""
Letter transition network graph based on bigram statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..vocab import CharVocab
from ..utils.plotting import save_figure, display_char


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
    max_out_edges: int = 2,
    show: bool = False,
) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    probs = _compute_bigram_probs(text, vocab)
    # Unigram counts for node sizing and top-N filtering
    unigram = {}
    for ch in text:
        if ch in vocab.char2id:
            unigram[ch] = unigram.get(ch, 0) + 1
    # Keep fewer nodes so the graph stays readable (matches tighter threshold).
    top_nodes = [
        c for c, _ in sorted(unigram.items(), key=lambda kv: kv[1], reverse=True)[:25]
    ]

    G = nx.DiGraph()

    for ch in top_nodes:
        G.add_node(ch)

    # for each source char, keep top-K outgoing edges above threshold
    outgoing: dict[str, list[tuple[str, float]]] = {}
    for (a, b), p in probs.items():
        if a not in G.nodes or b not in G.nodes:
            continue
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

    # Circular layout is much easier to read than spring layout for dense graphs.
    pos = nx.circular_layout(H)
    weights = [H[u][v]["weight"] for u, v in H.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 4.0 * (w / max_w) for w in weights]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    node_sizes = []
    for n in H.nodes:
        c = unigram.get(n, 1)
        node_sizes.append(200 + 1800 * (c / max(unigram.values())))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(
        H, pos, width=widths, arrows=True, arrowstyle="->", ax=ax, alpha=0.35
    )
    nx.draw_networkx_labels(
        H, pos, labels={n: display_char(n) for n in H.nodes}, font_size=9, ax=ax
    )
    ax.set_title("Letter transition graph (top 35 chars; corpus bigrams)")
    ax.axis("off")

    path = plots_dir / "letter_transition_graph.png"
    save_figure(fig, path, show=show)
    return path


