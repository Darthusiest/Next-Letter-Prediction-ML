"""
Common matplotlib setup for analysis plots.
Keeps figure creation consistent across training/data/embedding visualizations.
"""

from __future__ import annotations

import os
from pathlib import Path

_mpl_config_dir = Path("outputs") / ".mplconfig"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir.resolve()))

import matplotlib

# Use a non-interactive backend so analysis can run headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def display_char(ch: str) -> str:
    """
    Render characters safely for plot labels (make whitespace visible).
    This avoids matplotlib/font warnings for control characters like tab.
    """
    if ch == "\t":
        return "TAB"
    if ch == "\n":
        return "NL"
    if ch == " ":
        return "SPACE"
    return ch


def new_figure(figsize=(6, 4)):
    """Create a new matplotlib figure and axes with a default size."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, path: Path, show: bool = False) -> None:
    """Save a figure to disk and optionally display it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


