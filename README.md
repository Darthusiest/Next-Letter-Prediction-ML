# Character-Level Next-Letter Prediction

Predict the next character in English text from prior context. The model predicts **any character**: letters (upper- and lowercase), digits, and punctuation. Target performance is ~20% validation and test accuracy. Implements n-gram, MLP, RNN (LSTM), and CNN baselines.

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

- Load and concatenate them
- Clean: collapse runs of space/tab; keep case, digits, and punctuation (a–z, A–Z, 0–9, space, newline, tab, `. , ! ? ; : ' " - ( )`)
- Build a character vocabulary and sliding-window (context → next char) examples
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

- **Early stopping** uses **end-of-epoch** validation loss only. If `eval_every` runs mid-epoch, it can still update `best.pt` and `best_val_loss`, but it does **not** increment early-stopping patience (patience resets only when epoch-end val improves). Tune `EARLY_STOPPING_PATIENCE` in [`src/config.py`](src/config.py) (default **3** epochs without epoch-end improvement).
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
| `--weight-decay` | Adam L2 penalty (default: `WEIGHT_DECAY` in config; `0.0` preserves old behavior). |
| `--label-smoothing` | Neural cross-entropy label smoothing (default: `LABEL_SMOOTHING`, usually `0.0`). |
| `--plateau-lr` | Enable `ReduceLROnPlateau` on epoch-end val loss (neural only; off by default). |

Other overrides live on `train()` in `src/train.py`, e.g. `num_workers`, `epochs`, `batch_size`.

#### Regularization ablations (fair comparison)

Use the same `--max-chars`, `--data-source`, and seed (set `SEED` in config) when comparing runs. Examples:

```bash
# Baseline (defaults)
python -m src.train --model mlp --data-source cleaned --max-chars 500000

# Stronger L2
python -m src.train --model mlp --data-source cleaned --max-chars 500000 --weight-decay 1e-4

# Higher dropout: set DROPOUT = 0.4 in src/config.py (or add a CLI later)
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
- `src/preprocess.py` — load and clean text
- `src/vocab.py` — character vocabulary (build, encode, decode, save/load)
- `src/dataset.py` — sliding-window dataset and train/val/test split
- `src/models/baseline_ngram.py` — n-gram baseline
- `src/models/mlp.py` — MLP over fixed context
- `src/models/rnn.py` — LSTM over context
- `src/models/cnn.py` — Conv1d over character embeddings
- `src/train.py` — training loop and CLI (`python -m src.train`)
- `src/evaluate.py` — perplexity and accuracy
- `src/generate.py` — autoregressive sampling with temperature
- `src/analysis/` — post-training analysis pipeline (plots + reports)
- `src/utils/` — utilities (seed/device/logging + io/plotting helpers)
- `notebooks/` — exploratory notebooks
- `tests/` — unit tests

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

## Default hyperparameters (first run)

Tuned for full character set (letters, digits, punctuation, case) and ~20% val/test accuracy:

| Parameter   | Value |
|------------|--------|
| Context length | 64 |
| Batch size    | 64 |
| Embed dim     | 64 |
| Hidden dim    | 256 |
| Dropout       | 0.25 |
| Learning rate | 1e-3 |
| Epochs        | 30 (early stopping) |
| Early stop patience | 3 epochs without **epoch-end** val improvement |
| Optimizer     | Adam (foreach multi-tensor path when supported; optional `WEIGHT_DECAY`) |
| DataLoader workers | Up to 4 (`NUM_WORKERS` in `src/config.py`) |

## Next steps (roadmap)

1. N-gram baseline — done
2. MLP baseline — done
3. LSTM/GRU — done (`rnn`)
4. CNN over characters — done (`cnn`)
5. Analysis: vowel/consonant accuracy, confusion, context length, temperature

Train a model from the repo root (use the project venv):

```bash
source .venv/bin/activate
python -m src.train --model mlp    # or: rnn, cnn, ngram
python -m src.train --model rnn --data-source cleaned
python -m src.train -h               # all CLI flags
```
