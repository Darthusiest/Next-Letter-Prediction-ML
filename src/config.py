"""
Configuration defaults for the character-level next-letter prediction pipeline.
Single place for context length, batch size, paths, and model hyperparameters.
"""

import os
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEAN_TEXT_DIR = PROCESSED_DATA_DIR / "clean_texts"

# Data
CONTEXT_LENGTH = 64
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_CHARS = None  # None = use all; set to e.g. 500_000 for fast iteration

# Vocabulary: any character the model can predict (letters upper/lower, digits, punctuation)
DEFAULT_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n\t"
    ".,!?;:'\"-()"
)

# Dataloader
BATCH_SIZE = 64
# >0 overlaps batch prep with GPU/MPS work; use 0 only if debugging worker issues.
NUM_WORKERS = min(4, os.cpu_count() or 1)

# MLP defaults (slightly larger to reach ~20% val/test with full vocab)
EMBED_DIM = 64
HIDDEN_DIM = 256
DROPOUT = 0.25

# Training (enough to reach ~20% val/test accuracy with full char set)
LEARNING_RATE = 1e-3
EPOCHS = 30
# Validation can be expensive on CPU; evaluate less frequently by default.
EVAL_EVERY_N_STEPS = 5000
EARLY_STOPPING_PATIENCE = 5  # epochs without val improvement
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# N-gram baseline
NGRAM_ORDER = 4
NGRAM_SMOOTHING = 0.01  # Laplace-style smoothing

# Generation
DEFAULT_TEMPERATURE = 1.0
DEFAULT_GENERATION_LENGTH = 100

# Reproducibility
SEED = 42
