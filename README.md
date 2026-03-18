# Character-Level Next-Letter Prediction

Predict the next character in English text from prior context. The model predicts **any character**: letters (upper- and lowercase), digits, and punctuation. Target performance is ~20% validation and test accuracy. Implements a clean baseline (n-gram and MLP) and is structured to scale to RNN, CNN, and transformer models.

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

To train on the cleaned corpus:

```bash
python -c "from src.train import train; train(model_name='mlp', data_source='cleaned')"
```

For a quick run you can limit corpus size in `src/config.py` by setting `MAX_CHARS = 500_000`.

## Run training

From the project root:

```bash
python -m src.train
```

This trains the **MLP** baseline by default and writes per-run artifacts under `outputs/runs/<run_id>/`, including:

- `checkpoint.pt` (model weights + vocab + config needed for analysis)
- `loss_history.json`
- `metrics.json`
- `train_corpus.txt` (cleaned corpus snapshot)
- `plots/` and `reports/` (post-training analysis outputs)

At the end of training, the script automatically runs the full post-training analysis pipeline.

To train the **n-gram** baseline instead (no GPU, fast):

Edit `src/train.py` and call `train(model_name="ngram")` in `main()`, or from a script:

```python
from src.train import train
train(model_name="ngram")
```

## Evaluate

Load the checkpoint and run evaluation on the test set (example snippet):

```python
import torch
from src.config import CONTEXT_LENGTH
from src.dataset import get_splits, get_dataloaders
from src.preprocess import load_text, clean_text
from src.vocab import build_vocab_from_text
from src.config import DEFAULT_ALLOWED_CHARS, RAW_DATA_DIR
from src.models.mlp import MLPCharModel
from src.evaluate import evaluate, perplexity

paths = list(RAW_DATA_DIR.glob("*.txt"))
text = clean_text(load_text(paths))
vocab = build_vocab_from_text(text, allowed_chars=DEFAULT_ALLOWED_CHARS)
_, _, test_ds = get_splits(text, vocab, context_length=CONTEXT_LENGTH)
test_loader = get_dataloaders(_, _, test_ds)[2]

ckpt = torch.load("checkpoints/best.pt", weights_only=False)
model = MLPCharModel(
    vocab_size=ckpt["vocab"].vocab_size,
    context_length=ckpt["context_length"],
)
model.load_state_dict(ckpt["model_state"])
loss, acc = evaluate(model, test_loader, torch.device("cpu"), is_ngram=False)
print("Test loss:", loss, "perplexity:", perplexity(loss), "accuracy:", acc)
```

## Generate

Sample text from the trained model:

```python
from src.generate import generate
from src.utils import get_device
import torch

ckpt = torch.load("checkpoints/best.pt", weights_only=False)
model = ckpt.get("model")  # for n-gram; for MLP load state into MLPCharModel
vocab = ckpt["vocab"]
# If MLP: build model, load_state_dict(ckpt["model_state"])
# Then:
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
- `src/train.py` — training loop
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
  - `context_similarity_heatmap.png`, `context_pca_2d.png` (and `context_pca_3d.png` if enabled)
  - `letter_transition_graph.png` (if enabled)
- `reports/`:
  - `summary_report.txt`
  - `confidence_summary.txt` (if enabled)
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
| Optimizer     | Adam |

## Next steps (roadmap)

1. N-gram baseline — done
2. MLP baseline — done
3. LSTM/GRU
4. CNN over characters
5. Causal transformer
6. Analysis: vowel/consonant accuracy, confusion, context length, temperature
