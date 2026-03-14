"""
Utilities: device selection, random seed, logging.
"""

import random
import torch
import logging

from .config import SEED

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
