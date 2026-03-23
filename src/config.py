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
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
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
BATCH_SIZE = 128
# >0 overlaps batch prep with GPU/MPS work; use 0 only if debugging worker issues.
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Model architecture
EMBED_DIM = 128
HIDDEN_DIM = 512
DROPOUT = 0.3
# MLP: first block is pool→HIDDEN_DIM; extras are SwiGLU residual blocks.
MLP_NUM_HIDDEN_LAYERS = 5
MLP_NUM_ATTN_HEADS = 4

# RNN: extra LSTM layers multiply sequential work; 1 layer is the default for speed.
RNN_NUM_LAYERS = 1

# Training
LEARNING_RATE = 3e-4
EPOCHS = 50
EVAL_EVERY_N_STEPS = 5000
# Early stopping uses **end-of-epoch** validation only (not eval_every mid-epoch).
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# AdamW weight decay (decoupled from gradient, unlike Adam L2).
WEIGHT_DECAY = 1e-2
# Soft targets for cross-entropy (neural models only); 0 disables.
LABEL_SMOOTHING = 0.1
# ReduceLROnPlateau on epoch-end validation loss (on top of cosine annealing).
USE_PLATEAU_LR = True
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
# Max gradient L2 norm; prevents spikes during training. 0 disables clipping.
GRAD_CLIP_NORM = 1.0
# Linear warmup: ramp LR from near-zero to LEARNING_RATE over this many steps. 0 disables.
WARMUP_STEPS = 1000

# Tokenization: "char" for character-level, "bpe" for byte-pair encoding subwords.
TOKENIZER_TYPE = "char"
BPE_VOCAB_SIZE = 2000

# N-gram baseline
NGRAM_ORDER = 4
NGRAM_SMOOTHING = 0.01  # Laplace-style smoothing

# Generation
DEFAULT_TEMPERATURE = 1.0
DEFAULT_GENERATION_LENGTH = 100

# Reproducibility
SEED = 42
