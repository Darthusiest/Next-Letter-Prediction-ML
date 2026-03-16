"""
Common matplotlib setup for analysis plots.
Keeps figure creation consistent across training/data/embedding visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


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


