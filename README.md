# Character-Level Next-Letter Prediction

Predict the next character in English text from prior context. The model predicts **any character**: letters (upper- and lowercase), digits, and punctuation. The MLP architecture achieves **~56% validation accuracy** (up from ~20% with prior architecture). Implements n-gram, MLP, RNN (LSTM), and CNN models.

## Setup

```bash
cd /path/to/NLP-ML
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

### Recommended: plain text

Place one or more `.txt` files (books, articles, etc.) in either:

- `data/raw/`
- `data/raw/nlp-ebooks/`

The training script loads and concatenates **all** `*.txt` files found in those two locations. The pipeline will:

- Load and concatenate them (stripping Project Gutenberg header/footer boilerplate per-file)
- **Filter noise**: remove OCR garbage, short fragments, whitespace-dense lines, table-of-contents entries, and PDF page references
- Clean: collapse runs of space/tab; keep case, digits, and punctuation (a-z, A-Z, 0-9, space, newline, tab, `. , ! ? ; : ' " - ( )`)
- Build a character vocabulary and sliding-window (context -> next char) examples
- Split contiguously into train / val / test (80 / 10 / 10). Target: ~20% val/test accuracy.

### Optional: PDF / EPUB → TXT

You can also start from `.pdf` or `.epub` sources in `data/raw/nlp-ebooks/` and convert them to `.txt` first:

- **PDF**: install `pypdf` and run  
  ```bash
  pip install pypdf
  python tools/convert_to_txt.py book.pdf data/raw/book_from_pdf.txt
  ```
- **Batch PDF conversion** (recommended if you have many PDFs in `data/raw/nlp-ebooks/`):
  ```bash
  python tools/batch_convert_pdfs.py
  ```
- **EPUB**: install `ebooklib` and `beautifulsoup4` and run  
  ```bash
  pip install ebooklib beautifulsoup4
  python tools/convert_to_txt.py book.epub data/raw/book_from_epub.txt
  ```

After conversion, the resulting `.txt` files in `data/raw/` are handled exactly like any other text input.

### Recommended: clean noisy OCR/boilerplate (strong mode)

If your converted `.txt` files include OCR artifacts, page numbers, library stamps, or front-matter (common with scanned PDFs),
run the strong cleaning script to produce a cleaner training corpus:

```bash
python tools/clean_corpus.py
```

This writes cleaned texts to `data/processed/clean_texts/` plus:

- `data/processed/clean_texts/merged_corpus.txt` (single merged cleaned corpus)
- `data/processed/cleaning_report.json` (what was dropped/skipped and why)

The strong cleaner also includes a conservative heuristic to drop “layout-like”
OCR lines that are dominated by whitespace plus many very short tokens
(often seen in scanned book boilerplate), which tends to reduce the
“least confident” uncertainty that clusters on non-prose artifacts.

To train on the cleaned corpus:

```bash
python -m src.train --model mlp --data-source cleaned
```

For a quick run you can limit corpus size in `src/config.py` by setting `MAX_CHARS = 500_000`.

## Run training

From the project root:

```bash
python -m src.train --model mlp   # compare: --model rnn or --model cnn
```

This trains the selected model (default **MLP**) and writes per-run artifacts under `outputs/runs/<run_id>/`, including:

- `best.pt` — weights at **best validation loss** (includes `model_name`, `model_kwargs`, `vocab`, `context_length`)
- `checkpoint.pt` — weights after the **last training step** (same metadata shape; use to study overfitting vs `best.pt`)
- `loss_history.json`
- `metrics.json` — **`test_loss` / `test_accuracy` are computed on `best.pt`** when it exists; `test_*_final_epoch` use last-step weights (see `metrics_checkpoint_policy` in the file)
- `train_corpus.txt` (cleaned corpus snapshot)
- `plots/` and `reports/` (post-training analysis outputs; includes `reports/checkpoint_comparison.txt` when both checkpoints exist)

At the end of training, the script automatically runs the full post-training analysis pipeline.

### Checkpoints, metrics, and analysis

- **Early stopping** uses **end-of-epoch** validation loss only. If `eval_every` runs mid-epoch, it can still update `best.pt` and `best_val_loss`, but it does **not** increment early-stopping patience (patience resets only when epoch-end val improves). Tune `EARLY_STOPPING_PATIENCE` in [`src/config.py`](src/config.py) (default **5** epochs without epoch-end improvement).
- **`run_post_training_analysis`** loads **`best.pt` first**, then `checkpoint.pt`, so prediction plots and most model-based analyses match **best validation** weights by default.

To train the **n-gram** baseline instead (no GPU, fast):

```bash
python -m src.train --model ngram
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--model` / `-m` | `ngram`, `mlp`, `rnn`, or `cnn` (default: `mlp`). |
| `--data-source` | `raw` (default) or `cleaned` (`data/processed/clean_texts/`). |
| `--max-chars` | Truncate corpus length after load (overrides `MAX_CHARS` in config). |
| `--eval-every` | Run validation every N training steps (default: `EVAL_EVERY_N_STEPS` in `src/config.py`). |
| `--compile` | Enable `torch.compile` on the model (sometimes helps RNN/CNN on GPU/MPS). |
| `--rnn-layers` | LSTM layer count for `--model rnn` (default: `RNN_NUM_LAYERS` in `src/config.py`, currently 1). |
| `--weight-decay` | AdamW weight decay (default: `WEIGHT_DECAY`, currently `5e-3`; `0.0` disables). |
| `--label-smoothing` | Neural cross-entropy label smoothing (default: `LABEL_SMOOTHING`, currently `0.05`; `0.0` disables). |
| `--plateau-lr` / `--no-plateau-lr` | `ReduceLROnPlateau` on epoch-end val loss (default follows `USE_PLATEAU_LR`). Note: cosine annealing steps per-batch and overwrites plateau reductions; plateau serves as a diagnostic signal. |
| `--grad-clip` | Max gradient L2 norm (default: `GRAD_CLIP_NORM` in config, typically `1.0`; `0` disables). |
| `--warmup-steps` | Linear LR warmup from near-zero to `LEARNING_RATE` (default: `WARMUP_STEPS`, typically `1000`; `0` disables). |
| `--cosine-restarts` / `--no-cosine-restarts` | Cosine LR schedule with warm restarts vs single cosine decay (default: `USE_COSINE_WARM_RESTARTS` in config, `True`). |
| `--mlp-hidden-layers` | MLP depth: number of hidden Linear blocks (`proj` + SwiGLU residual blocks), MLP only (default: `MLP_NUM_HIDDEN_LAYERS` in config). |
| `--mlp-attn-heads` | Number of attention heads for MLP self-attention and pooling (default: `MLP_NUM_ATTN_HEADS` in config, 4). |
| `--mlp-self-attn-layers` | Stacked causal self-attention layers with RoPE (default: `MLP_NUM_SELF_ATTN_LAYERS` in config, 2). |
| `--dropout` | Dropout rate for all layers including embedding (default: `DROPOUT` in config). |
| `--embed-dim` | Embedding dimension (default: `EMBED_DIM` in config). |
| `--hidden-dim` | Hidden layer dimension (default: `HIDDEN_DIM` in config). |
| `--tokenizer` | `char` (default) or `bpe` for byte-pair subword tokenization. |
| `--bpe-vocab-size` | BPE vocabulary size (default: `BPE_VOCAB_SIZE` in config, 2000). |

Other overrides live on `train()` in `src/train.py`, e.g. `num_workers`, `epochs`, `batch_size`.

**Config validation:** `validate_config()` in `src/config.py` is called at the start of `train()` and asserts that hyperparameters are within valid ranges (e.g. `0 <= DROPOUT < 1`, `LABEL_SMOOTHING` in [0, 1], `DEFAULT_TEMPERATURE > 0`, `MLP_NUM_SELF_ATTN_LAYERS >= 1`). Edit the function to add project-specific constraints.

#### Regularization and architecture ablations

Use the same `--max-chars`, `--data-source`, and seed (set `SEED` in config) when comparing runs. Examples:

```bash
# Baseline (defaults from config)
python -m src.train --model mlp --data-source cleaned

# Higher dropout + smaller capacity
python -m src.train --model mlp --data-source cleaned --dropout 0.4 --hidden-dim 256

# Deeper network
python -m src.train --model mlp --data-source cleaned --mlp-hidden-layers 7

# More attention heads
python -m src.train --model mlp --data-source cleaned --mlp-attn-heads 8

# BPE subword tokenization
python -m src.train --model mlp --data-source cleaned --tokenizer bpe --bpe-vocab-size 2000
```

Report **`val_loss_best`** from `metrics.json` and **`test_loss` / `test_accuracy`** (these are **test@best**).

#### RNN vs MLP speed (expect a large gap)

An LSTM walks the context **one timestep at a time** per layer (`CONTEXT_LENGTH` × `RNN_NUM_LAYERS`), while an MLP over the same window is mostly **parallel matmuls**. On Apple MPS especially, RNNs are often several times slower per step than MLPs—that is normal, not a misconfiguration.

To make RNN experiments tolerable: use **`--compile`**, keep **`RNN_NUM_LAYERS = 1`** (default), try a larger **`BATCH_SIZE`** if memory allows, shorten **`CONTEXT_LENGTH`** for exploratory runs, or cap **`--max-chars`**. If you only need a strong cheap baseline, **MLP or CNN** is the better default; keep RNN when you explicitly care about recurrent order effects.

### Training speed and devices

Training picks a device in this order: **CUDA → Apple MPS → CPU** (`get_device()` in `src/utils/__init__.py`). On CUDA, cuDNN benchmark, TF32, and mixed precision (`GradScaler` + autocast) are enabled. On MPS (PyTorch 2+), training uses float16 autocast without a scaler.

The default `DataLoader` uses up to **four worker processes** and prefetch (`src/config.py` `NUM_WORKERS`, `src/dataset.py`) so batching overlaps with GPU/MPS work. Set `NUM_WORKERS = 0` in config or call `train(..., num_workers=0)` if workers cause issues on your platform.

`CharVocab.encode()` uses a fast **ordinal lookup table** for typical ASCII-heavy text. The optimizer is Adam with **`foreach=True`** when supported. Optional **`torch.compile`**: CLI **`--compile`** or `train(..., compile_model=True)`.

Evaluation in `src/evaluate.py` uses `torch.inference_mode()`.

## Evaluate

Load a checkpoint from a finished run and evaluate on the test set (replace `RUN_ID` with a folder under `outputs/runs/`). Prefer **`best.pt`** to match **`metrics.json`** (`test_loss` / `test_accuracy`). Use **`checkpoint.pt`** for **last-epoch** weights. Both include `model_kwargs` on new runs; older `best.pt` files may rely on shape inference.

```python
from pathlib import Path
import torch
from src.config import CONTEXT_LENGTH
from src.dataset import get_splits, get_dataloaders
from src.preprocess import load_text, clean_text
from src.vocab import build_vocab_from_text
from src.config import DEFAULT_ALLOWED_CHARS, RAW_DATA_DIR
from src.models import get_model
from src.evaluate import evaluate, perplexity
from src.utils import get_device

run_dir = Path("outputs/runs/RUN_ID")
ckpt_path = run_dir / "best.pt"
if not ckpt_path.exists():
    ckpt_path = run_dir / "checkpoint.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

paths = list(RAW_DATA_DIR.glob("*.txt"))
text = clean_text(load_text(paths))
vocab = build_vocab_from_text(text, allowed_chars=DEFAULT_ALLOWED_CHARS)
_, _, test_ds = get_splits(text, vocab, context_length=CONTEXT_LENGTH)
test_loader = get_dataloaders(_, _, test_ds)[2]

model_name = ckpt.get("model_name", "mlp")
mkw = dict(ckpt.get("model_kwargs") or {})
model = get_model(
    model_name,
    vocab_size=ckpt["vocab"].vocab_size,
    context_length=ckpt["context_length"],
    **mkw,
)
model.load_state_dict(ckpt["model_state"])
device = get_device()
loss, acc = evaluate(model, test_loader, device, is_ngram=False)
print("Test loss:", loss, "perplexity:", perplexity(loss), "accuracy:", acc)
```

## Generate

Sample text from the trained model:

```python
from pathlib import Path
import torch
from src.generate import generate
from src.models import get_model
from src.utils import get_device

run_dir = Path("outputs/runs/RUN_ID")
ckpt_path = run_dir / "best.pt"
if not ckpt_path.exists():
    ckpt_path = run_dir / "checkpoint.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
vocab = ckpt["vocab"]
model_name = ckpt.get("model_name", "mlp")
mkw = dict(ckpt.get("model_kwargs") or {})
model = get_model(
    model_name,
    vocab_size=vocab.vocab_size,
    context_length=ckpt["context_length"],
    **mkw,
)
model.load_state_dict(ckpt["model_state"])
device = get_device()
model.to(device)
out = generate(model, vocab, seed="the ", length=100, temperature=0.9)
print(out)
```

## Project layout

- `data/raw/` — input `.txt` files
- `data/raw/nlp-ebooks/` — optional ebook sources (`.pdf`, `.epub`, etc.) and/or converted `.txt`
- `data/processed/` — optional serialized vocab/indices
- `src/config.py` — defaults (context length, batch size, paths, etc.)
- `src/preprocess.py` — load text (with Gutenberg stripping), filter OCR/boilerplate noise, and clean
- `src/vocab.py` — character vocabulary (build, encode, decode, save/load)
- `src/dataset.py` — sliding-window dataset and train/val/test split
- `src/models/baseline_ngram.py` — n-gram baseline
- `src/models/mlp.py` — MLP with stacked causal self-attention (RoPE), multi-head attention pooling, SwiGLU, KV cache, and weight tying
- `src/models/rnn.py` — LSTM over context
- `src/models/cnn.py` — Conv1d over character embeddings
- `src/tokenizer.py` — BPE subword tokenizer (HuggingFace `tokenizers`)
- `src/train.py` — training loop and CLI (`python -m src.train`)
- `src/evaluate.py` — perplexity and accuracy
- `src/generate.py` — autoregressive sampling with temperature
- `src/analysis/` — post-training analysis pipeline (plots + reports)
- `src/utils/` — utilities (seed/device/logging + io/plotting helpers)
- `notebooks/` — exploratory notebooks
- `tests/` — unit tests (`test_models.py`, `test_preprocess.py`, `test_evaluate.py`, `test_train_helpers.py`, `test_tokenizer.py`, `test_dataset.py`, `test_vocab.py`)

## Post-training analysis outputs

Each run creates plots/reports under `outputs/runs/<run_id>/`:

- `plots/`:
  - `training_loss.png`, `validation_loss.png`
  - `char_frequency.png`, `bigram_heatmap.png`
  - `confusion_matrix.png`, `error_by_character.png`
  - `embeddings_pca_2d.png`, `embeddings_tsne_2d.png` (if embeddings exist)
  - `entropy_histogram.png` (if enabled)
  - `entropy_by_context_category.png`, `error_by_context_category.png` (if enabled)
  - `context_similarity_heatmap.png`, `context_pca_2d.png` (and `context_pca_3d.png` if enabled)
  - `letter_transition_graph.png` (if enabled)
- `reports/`:
  - `summary_report.txt`
  - `checkpoint_comparison.txt` (best.pt vs last-epoch entropy/top-k on sample contexts, when both checkpoints exist)
  - `confidence_summary.txt` (if enabled)
  - `residual_noise_diagnostics.txt` (if entropy analysis is enabled)
  - `similarity_report.txt` (if enabled)

## Default hyperparameters

| Parameter   | Value |
|------------|--------|
| Context length | 64 |
| Batch size    | 128 |
| Embed dim     | 128 |
| Hidden dim    | 512 |
| MLP hidden layers | 5 (`proj` + 4 SwiGLU residual blocks with pre-LayerNorm) |
| MLP attention heads | 4 (self-attention layers + multi-head attention pooling) |
| MLP self-attention layers | 2 (stacked causal self-attention with RoPE) |
| MLP architecture | RoPE (no learned pos_embed) → N causal self-attention layers → multi-head attention pooling → LayerNorm(pool) → SwiGLU blocks → weight-tied output; embeddings scaled by √d |
| Position encoding | Rotary Position Embeddings (RoPE) applied to Q/K in each self-attention layer |
| Activation    | SwiGLU (MLP residual blocks); GELU (MLP projection, CNN); LSTM gates (RNN) |
| Normalization | Pre-LayerNorm in MLP residual blocks + self-attention; LayerNorm on pooled attention features before `proj`; post-LN in RNN, CNN |
| Dropout       | 0.13 (embedding + hidden layers; all models have embedding dropout) |
| Learning rate | 3e-4 (linear warmup 1000 steps → cosine schedule with warm restarts, `eta_min` = 10% of peak) |
| Epochs        | 75 (early stopping) |
| Early stop patience | 5 epochs without **epoch-end** val improvement |
| Optimizer     | AdamW (`WEIGHT_DECAY` 5e-3, `LABEL_SMOOTHING` 0.05, `GRAD_CLIP_NORM` 1.0) |
| LR schedule   | Warmup → cosine with warm restarts (`COSINE_T0` auto = cosine steps ÷ 3 when unset, `COSINE_T_MULT` 1; `eta_min` = 10% of peak LR). Use `--no-cosine-restarts` for a single cosine decay to `eta_min`. |
| Tokenization  | Character-level (default) or BPE subword (`--tokenizer bpe`) |
| KV cache      | Supported for MLP during generation (incremental decoding) |
| DataLoader workers | Up to 4 (`NUM_WORKERS` in `src/config.py`) |

## Performance: what drove the ~20% → 56% accuracy jump

The MLP's validation accuracy improved from ~20% to **56%** (within the first training epoch). The key changes, ranked by impact:

1. **Causal masking** — the largest single factor. Prior bidirectional self-attention let each position see future characters during training, creating a train/generation mismatch. Causal masking (`is_causal=True`) forces left-to-right attention, aligning training with autoregressive generation.
2. **Stacked self-attention (2 layers)** — a second attention layer lets the model compose local character pairs into higher-level word fragment patterns (`tion`, `ment`, `ing`).
3. **RoPE** — rotary position embeddings encode relative position directly into the attention computation, replacing learned positional embeddings that needed training from scratch.
4. **Reduced regularization** — the prior model was underfitting (train loss > val loss). Dropout was reduced from 0.3 → 0.15 → 0.13, and weight decay from 1e-2 → 5e-3, letting the model use its capacity. Label smoothing was tuned over time (e.g. 0.05 → 0.03 when fixing causal masking; current default is **0.05** for generalization).
5. **Noise filtering** — removing OCR garbage and boilerplate from training data freed capacity for actual prose patterns.
6. **LR schedule** — cosine **warm restarts** (`CosineAnnealingWarmRestarts`) periodically raise the LR again to escape plateaus; the cosine floor (`COSINE_ETA_MIN_FACTOR`) was raised from 1% → 5% → **10%** of peak to keep updates meaningful late in training.

## Next steps (roadmap)

1. N-gram baseline — done
2. MLP baseline — done (evolved: stacked causal self-attention with RoPE + multi-head pooling + SwiGLU + pre-LN + weight tying + KV cache)
3. LSTM/GRU — done (`rnn`)
4. CNN over characters — done (`cnn`)
5. BPE subword tokenization — done (`--tokenizer bpe`)
6. OCR/boilerplate noise filtering — done (heuristic line filter in preprocessing)
7. Regularization tuning — done (reduced dropout/smoothing/decay; extended schedule)
8. Analysis: vowel/consonant accuracy, confusion, context length, temperature

Train a model from the repo root (use the project venv):

```bash
source .venv/bin/activate
python -m src.train --model mlp    # or: rnn, cnn, ngram
python -m src.train --model rnn --data-source cleaned
python -m src.train -h               # all CLI flags
```
