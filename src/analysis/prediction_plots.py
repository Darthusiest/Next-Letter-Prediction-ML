"""
Prediction probability bar charts for sample contexts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..utils.plotting import new_figure, save_figure


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower() or "empty"


def plot_prediction_bars(
    model_wrapper: ModelAnalysisWrapper,
    contexts: Iterable[str],
    top_k: int,
    output_dir: Path,
    show: bool = False,
) -> None:
    plots_root = output_dir / "plots" / "predictions"
    plots_root.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        dist = model_wrapper.predict_next_distribution(ctx)
        # sort by prob descending and take top_k
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        chars, probs = zip(*items) if items else ([], [])

        fig, ax = new_figure()
        ax.bar(range(len(chars)), probs)
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars)
        ax.set_ylabel("Probability")
        ax.set_title(f"Next-character distribution for context {repr(ctx)}")

        out_path = plots_root / f"{_slugify(ctx)}_probs.png"
        save_figure(fig, out_path, show=show)

