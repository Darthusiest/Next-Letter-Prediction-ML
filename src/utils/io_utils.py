"""
IO helpers for the character-level next-letter project.
Used by training and post-training analysis to manage run directories and JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..config import PROJECT_ROOT


def get_runs_root() -> Path:
    """Root directory for per-run artifacts."""
    return PROJECT_ROOT / "outputs" / "runs"


def ensure_run_dir(run_id: str) -> Path:
    """
    Create (if needed) and return the directory for a given run_id:
    outputs/runs/<run_id>/
    """
    run_dir = get_runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    return run_dir


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file into a dict."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


