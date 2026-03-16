"""
Utilities package: device selection, random seed, logging, and helpers.

This replaces the earlier single-module `src/utils.py` so that submodules
like `utils.io_utils` and `utils.plotting` can be imported cleanly.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import torch

from ..config import SEED, PROJECT_ROOT

logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return torch device: cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for training/eval scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_run(run_info: Dict[str, Any], logs_dir: Path | None = None) -> Path:
    """
    Write a single run's config and metrics to logs/runs/<run_id>.json.
    """
    if logs_dir is None:
        logs_dir = PROJECT_ROOT / "logs" / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_info.get("run_id")
    if not run_id:
        import time

        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_info["run_id"] = run_id
    path = logs_dir / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    logger.info("Wrote run log to %s", path)
    return path


__all__ = ["set_seed", "get_device", "setup_logging", "log_run"]

