"""
Training and validation loss (and optional perplexity) plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.plotting import new_figure, save_figure


def plot_loss_curves(loss_history: Dict, output_dir: Path, show: bool = False) -> None:
    epochs = loss_history.get("epochs")
    train_loss = loss_history.get("train_loss")
    val_loss = loss_history.get("val_loss")
    if not epochs or train_loss is None or val_loss is None:
        return

    plots_dir = output_dir / "plots"

    # Training loss
    fig, ax = new_figure()
    ax.plot(epochs, train_loss, label="train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.legend()
    save_figure(fig, plots_dir / "training_loss.png", show=show)

    # Validation loss
    fig, ax = new_figure()
    ax.plot(epochs, val_loss, label="val", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss Over Epochs")
    ax.legend()
    save_figure(fig, plots_dir / "validation_loss.png", show=show)


