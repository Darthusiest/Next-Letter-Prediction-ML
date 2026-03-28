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

# ---------------------------------------------------------------------------
# Ablation notes (hyperparameter changes applied together — see docs/design.md
# §5 "Performance" for the combined result of ~20% → 56% val accuracy):
#
#   DROPOUT  0.15 → 0.13 : model was underfitting; lower dropout lets the
#                           network use more capacity now that causal masking
#                           prevents information leakage.
#   LABEL_SMOOTHING 0.03 → 0.05 : softer targets for generalization (see LABEL_SMOOTHING).
#   DEFAULT_TEMPERATURE 1.0 → 0.8 : sharper sampling matches stronger model.
#   MLP_NUM_SELF_ATTN_LAYERS (new, =2) : deeper attention captures word-fragment
#                           patterns; second-largest contributor after causal mask.
#   COSINE_ETA_MIN_FACTOR 0.05 → 0.10 : higher LR floor late in training.
#
# These were applied simultaneously.  For future tuning, isolate each change
# in a separate commit or use CLI overrides (e.g. --dropout, --label-smoothing)
# to ablate one variable at a time against the baseline.
# ---------------------------------------------------------------------------

# Model architecture
EMBED_DIM = 128
HIDDEN_DIM = 512
# Reduced from 0.15 → 0.13: prior model was underfitting (train > val loss);
# lower dropout lets the network use more capacity with causal masking in place.
DROPOUT = 0.130
# MLP: first block is pool→HIDDEN_DIM; extras are SwiGLU residual blocks.
MLP_NUM_HIDDEN_LAYERS = 5
MLP_NUM_ATTN_HEADS = 4
# Stacked causal self-attention layers (RoPE + SDPA) before attention pooling.
# Added for the MLP architecture: deeper attention captures richer inter-position
# dependencies (e.g. composing digraphs into word fragments like "tion", "ment").
MLP_NUM_SELF_ATTN_LAYERS = 2

# RNN: extra LSTM layers multiply sequential work; 1 layer is the default for speed.
RNN_NUM_LAYERS = 1

# Training
LEARNING_RATE = 3e-4
EPOCHS = 75
EVAL_EVERY_N_STEPS = 5000
# Early stopping uses **end-of-epoch** validation only (not eval_every mid-epoch).
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# AdamW weight decay (decoupled from gradient, unlike Adam L2).
WEIGHT_DECAY = 5e-3
# Soft targets for cross-entropy (neural models only); 0 disables.
# Bumped from 0.03 → 0.05: softer targets improve generalization as model capacity grows.
LABEL_SMOOTHING = 0.050
# ReduceLROnPlateau on epoch-end validation loss (on top of cosine annealing).
USE_PLATEAU_LR = True
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
# Max gradient L2 norm; prevents spikes during training. 0 disables clipping.
GRAD_CLIP_NORM = 1.0
# Cosine annealing minimum LR as a fraction of LEARNING_RATE. 10% floor keeps
# gradient updates meaningful through the end of training.
COSINE_ETA_MIN_FACTOR = 0.10
# Cosine warm restarts (after warmup): restart period T_0 and multiplier T_mult.
# If COSINE_T0 is None, T_0 is set at train time to max(1, cosine_steps // 3).
COSINE_T0 = None  # None → T_0 = max(1, cosine_steps // 3) at runtime
COSINE_T_MULT = 1
# Use CosineAnnealingWarmRestarts instead of a single CosineAnnealingLR decay.
USE_COSINE_WARM_RESTARTS = True
# Linear warmup: ramp LR from near-zero to LEARNING_RATE over this many steps. 0 disables.
WARMUP_STEPS = 1000

# Tokenization: "char" for character-level, "bpe" for byte-pair encoding subwords.
TOKENIZER_TYPE = "char"
BPE_VOCAB_SIZE = 2000

# N-gram baseline
NGRAM_ORDER = 4
NGRAM_SMOOTHING = 0.01  # Laplace-style smoothing

# Generation
# Reduced from 1.0 → 0.8: with the stronger MLP (causal + RoPE), the model's
# top-1 predictions are more reliable; a lower temperature sharpens sampling
# toward likely characters without collapsing diversity entirely.
DEFAULT_TEMPERATURE = 0.800
DEFAULT_GENERATION_LENGTH = 100

# Reproducibility
SEED = 42


def validate_config():
    """Validate hyperparameter ranges to catch configuration errors early."""
    assert 0 <= DROPOUT < 1, f"DROPOUT must be in [0, 1), got {DROPOUT}"
    assert 0 <= LABEL_SMOOTHING <= 1, f"LABEL_SMOOTHING must be in [0, 1], got {LABEL_SMOOTHING}"
    assert 0 < DEFAULT_TEMPERATURE, f"DEFAULT_TEMPERATURE must be > 0, got {DEFAULT_TEMPERATURE}"
    assert MLP_NUM_SELF_ATTN_LAYERS >= 1, f"MLP_NUM_SELF_ATTN_LAYERS must be >= 1, got {MLP_NUM_SELF_ATTN_LAYERS}"
    assert COSINE_T_MULT >= 1, f"COSINE_T_MULT must be >= 1, got {COSINE_T_MULT}"
    if COSINE_T0 is not None:
        assert COSINE_T0 >= 1, f"COSINE_T0 must be >= 1 when set, got {COSINE_T0}"
