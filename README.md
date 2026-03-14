# Character-Level Next-Letter Prediction

Predict the next character in English text from prior context. This project implements a clean baseline (n-gram and MLP) and is structured to scale to RNN, CNN, and transformer models.

## Setup

```bash
cd /path/to/NLP-ML
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Place one or more `.txt` files (books, articles, etc.) in `data/raw/`. The pipeline will:

- Load and concatenate them
- Clean: lowercase, collapse whitespace, remove digits, keep letters + space + minimal punctuation (`. , ! ? ; : ' " - ( )` and newline)
- Build a character vocabulary and sliding-window (context → next char) examples
- Split contiguously into train / val / test (80 / 10 / 10)

For a quick run you can limit corpus size in `src/config.py` by setting `MAX_CHARS = 500_000`.

## Run training

From the project root:

```bash
PYTHONPATH=. python src/train.py
```

Or run as a module:

```bash
python -m src.train
```

This trains the **MLP** baseline by default, saves the best checkpoint to `checkpoints/best.pt`, and logs train/val loss and accuracy.

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
- `src/analysis.py` — (placeholder for vowel/consonant, confusion, etc.)
- `src/utils.py` — seed, device, logging
- `notebooks/` — exploratory notebooks
- `tests/` — unit tests

## Default hyperparameters (first run)

| Parameter   | Value |
|------------|--------|
| Context length | 64 |
| Batch size    | 64 |
| Embed dim     | 32 |
| Hidden dim    | 128 |
| Dropout       | 0.2 |
| Learning rate | 1e-3 |
| Optimizer     | Adam |

## Next steps (roadmap)

1. N-gram baseline — done
2. MLP baseline — done
3. LSTM/GRU
4. CNN over characters
5. Causal transformer
6. Analysis: vowel/consonant accuracy, confusion, context length, temperature
