# Stronger Models, Analysis, and Experiment Logging

This document records the design for extending the character‑level next‑letter
project beyond the initial n‑gram and MLP baselines. It is intended as
documentation for future writing (e.g. a paper or report) and for guiding
implementation work.

---

## Implementation status (summary)

As of the current tree:

- **Models in `get_model` / training:** `ngram`, `mlp`, `rnn`, `cnn`. There is
  **no** transformer module in the repository.
- **Training entrypoint:** `python -m src.train` with argparse (`--model`,
  `--data-source`, `--max-chars`, `--eval-every`); programmatic `train()` in
  `src/train.py` supports further knobs (e.g. `num_workers`, `compile_model`).
- **Device selection:** CUDA if available, else Apple **MPS**, else CPU
  (`src/utils/__init__.py`). Mixed precision: CUDA uses `GradScaler` + autocast;
  MPS uses float16 autocast (PyTorch 2+). See **README** for dataloader workers,
  fast `CharVocab.encode`, Adam `foreach`, and `torch.inference_mode` in eval.
- **Artifacts per run:** `outputs/runs/<run_id>/` — `train_corpus.txt`,
  `checkpoint.pt` (includes `model_kwargs`), `best.pt`, `metrics.json`,
  `loss_history.json`, plus analysis under `plots/` and `reports/`.
- **Logging:** `logs/runs/<run_id>.json` via `log_run()`.
- **Analysis:** `src/analysis/run_analysis.py` (`run_post_training_analysis`);
  optional attention plots only apply if a model exposes attention (no
  transformer shipped).

---

## 1. Goals and scope

- **Stronger sequence models**
  - Add neural sequence models that can capture longer and more structured
    dependencies than the MLP and n‑gram baselines:
    - `RNNCharModel` (LSTM / GRU) in `src/models/rnn.py` — **implemented**.
    - `CNNCharModel` in `src/models/cnn.py` — **implemented**.
- **Linguistics‑informed analysis**
  - Extend `src/analysis.py` with tools to evaluate the model in terms that
    matter for English spelling and morphology (vowels vs consonants, word
    boundaries, confusions, etc.).
- **Experiment logging**
  - Introduce simple but structured logging (configs + metrics per run) so that
    experiments are reproducible and results can be compared and written up.

Constraints:

- Reuse the existing data pipeline (preprocess → vocab → dataset).
- Keep models sized to run on a single CPU‑only machine.
- Avoid heavy dependencies and over‑engineering; prefer clear, readable PyTorch.

---

## 2. Architecture overview

High‑level flow, extending the current baseline:

- **Data path**
  - Raw text (`data/raw/*.txt` or converted from PDF/EPUB) →
  - `preprocess.py` (cleaning, normalization) →
  - `vocab.py` (char↔id mapping) →
  - `dataset.py` (`CharSequenceDataset`, sliding window context→next‑char) →
  - `DataLoader` (train/val/test splits).

- **Models**
  - Baselines:
    - `NGramModel` (local spelling patterns only).
    - `MLPCharModel` (fixed window, position‑specific but no recurrence).
  - Sequence / local pattern models (plug into the same training loop via `models.get_model`):
    - `RNNCharModel` — LSTM over embeddings; last hidden state → logits.
    - `CNNCharModel` — Conv1d over embeddings with multiple kernel sizes +
      global max pooling → logits.

- **Evaluation & analysis**
  - `train.py` trains and validates any model selected by name.
  - `evaluate.py` computes loss, accuracy, and perplexity.
  - `analysis.py` consumes models, vocab, datasets, and logged metrics to
    produce linguistically meaningful analyses.
  - `logs/` stores JSON/CSV summaries of runs.

---

## 3. Stronger models

### 3.1 RNN character model (`src/models/rnn.py`)

- **Purpose**
  - Serve as the first sequence‑aware neural baseline beyond MLP and n‑gram.
  - Capture variable‑length dependencies within a fixed context window, such as
    morphological patterns (`un-`, `re-`, `-ing`, `-ed`, `-tion`) and word‑level
    effects.

- **Interface**
  - `RNNCharModel(vocab_size, context_length, embed_dim, hidden_dim, num_layers, dropout)`.
  - `forward(context: LongTensor[B, L]) -> logits[B, V]`:
    - Embed: `(B, L) → (B, L, E)`.
    - LSTM: `(B, L, E) → (B, L, H)`.
    - Use **last time step** `(B, H)` as summary of the context.
    - Apply dropout and a linear projection to obtain logits `(B, V)`.

- **Design rationale**
  - RNNs/LSTMs respect **order** and can accumulate history via the hidden
    state, which is crucial for:
    - Word‑initial vs mid‑word vs word‑final positions.
    - Common affixes and longer local patterns than a small n‑gram can handle.

### 3.2 CNN character model (`src/models/cnn.py`)

- **Purpose**
  - Specialize in **local spelling patterns** (digraphs, trigraphs, doubled
    consonants) using convolutional filters over the character sequence.

- **Interface**
  - `CNNCharModel(vocab_size, context_length, embed_dim, num_channels, kernel_sizes, dropout)`.
  - `forward(context: LongTensor[B, L]) -> logits[B, V]`:
    - Embed: `(B, L, E)`.
    - Permute to `(B, E, L)` and apply several `Conv1d` layers with kernel
      sizes like 3, 5, 7.
    - Global max‑pool each feature map over the sequence → `(B, C_total)`.
    - Dropout + linear projection to logits `(B, V)`.

- **Design rationale**
  - Convolutional filters act as learned detectors for local patterns like
    `th`, `ing`, `ion`, consonant clusters, and doubled letters.
  - Highly parallelizable and efficient; ideal for studying **orthographic
    neighborhoods** and local spelling statistics.
  - Less suited to long‑range dependencies (e.g. agreement across multiple
    words) unless made deeper.

### 3.3 Integration via `get_model` and `train.py`

- `src/models/__init__.py::get_model(name, vocab_size, context_length, **kwargs)`
  acts as a single factory entry point for all models:
  - `name == "ngram"`, `"mlp"`, `"rnn"`, `"cnn"`.
  - Passes through model‑specific hyperparameters (embedding size, hidden size,
    number of layers, etc.).
- `src/train.py` constructs the model via `get_model` using configuration from
  `src/config.py` and trains it using the same training loop (cross‑entropy
  loss, Adam optimizer, validation, early stopping).
- CLI: `python -m src.train --model mlp` (or `rnn`, `cnn`, `ngram`).

---

## 4. Analysis tools (`src/analysis.py`)

Analysis should go beyond scalar loss/accuracy and connect to **linguistic
structure** in English:

### 4.1 Vowel vs consonant performance

- Define vowels (`a, e, i, o, u` plus uppercase) and treat other letters as
  consonants; track spaces and punctuation separately.
- Compute:
  - Per‑character accuracy.
  - Aggregated accuracy for vowels vs consonants vs spaces vs punctuation.
- Purpose: check whether the model behaves differently on vowels (which often
  carry syllabic information) vs consonants (which often occur in clusters and
  define the skeleton of words).

### 4.2 Confusion matrices

- Build a confusion matrix over characters using validation or test sets:
  - Tally `(true_char, predicted_char)` pairs.
  - Normalize rows to get probabilities.
- Inspect which letters the model most often confuses (e.g. `c` vs `k`, `m` vs
  `n`, `i` vs `l`), and where rare letters (q, z, j, x) fail.

### 4.3 Position‑aware accuracy

- Analyze accuracy as a function of **position relative to spaces**:
  - Word‑initial (previous character is space).
  - Mid‑word.
  - Word‑final (next character is space).
- Purpose: understand how well the model captures **word boundaries**, and
  whether it learns typical word‑initial and word‑final distributions.

### 4.4 Context length and embedding dimension

- Provide helpers that run small sweeps over:
  - `CONTEXT_LENGTH` (e.g. 16, 32, 64, 128).
  - Embedding size (e.g. 16, 32, 64, 128).
- For each setting, call the existing `train()` function, record validation
  loss and accuracy, and write results to logs.
- Purpose: quantify how much longer context and higher embedding capacity
  actually help on **next‑character prediction** in this corpus.

### 4.5 Temperature and generation behavior

- Generate samples at different temperatures (e.g. 0.7, 1.0, 1.3) using
  `generate.py`.
- Optionally compute simple statistics:
  - Character frequency distributions.
  - Repetition rates and average word length.
- Purpose: relate **subjective generation quality** (coherent spelling vs
  diversity) to measurable statistics.

---

## 5. Experiment logging

### 5.1 Logging format and storage

- Create a `logs/` directory with:
  - `logs/runs/` — one JSON file per training run.
  - Optional `logs/summary.csv` — a table aggregating run metadata and final
    metrics.

- Example run JSON fields:

  - `run_id` (timestamp + model name).
  - `timestamp`.
  - `model` (ngram/mlp/rnn/cnn).
  - `config` (context length, embedding size, hidden size, dropout, epochs,
    etc.).
  - `data` (number of characters used, vocab size, which files from
    `data/raw/`).
  - `metrics` (best validation loss/accuracy, test loss/accuracy).

### 5.2 Integration with training

- After training finishes (or after the best checkpoint is found), `train.py`
  will:
  - Gather the configuration values used for the run (from `config.py` and any
    overrides).
  - Call `evaluate()` on the test set to obtain final metrics.
  - Write a JSON run file to `logs/runs/` via a small helper (e.g.
    `log_run(config, data_info, metrics)`).
- Optionally append a row to `logs/summary.csv` for quick comparisons without
  parsing JSON.

### 5.3 Reproducibility

- Each run log should also record:
  - Random seed.
  - Git commit hash (if available).
  - List of dataset files from `data/raw/` used to build the corpus.
- This supports reproducible experiments and clear provenance for future
  publications.

---

## 6. Implementation order (for reference)

1. ~~Implement `RNNCharModel` and integrate it into `models.get_model`.~~ **Done.**
2. ~~Implement `CNNCharModel` and integrate it into `models.get_model`; add
   tests for basic forward behavior.~~ **Done** (see `tests/`).
3. ~~Extend `train.py` to select between all model names~~ — **Done** via CLI
   and `get_model`; `generate.py` works with any neural model loaded from a
   checkpoint.
4. ~~Add experiment logging helpers and wire them into `train.py`.~~ **Done**
   (`log_run`, `metrics.json`, run directories).
5. Flesh out analysis with vowel/consonant accuracy, confusion matrices,
   positional analysis, and context/embedding/temperature experiments — **partially
   done** (`src/analysis/` modules; some items remain aspirational).
6. Keep `README.md` and `docs/design.md` aligned with shipped models and training
   behavior — **ongoing**.

This document mirrors the "Stronger Models, Analysis, and Experiment Logging"
plan and serves as a stable reference for why these additions are being made
and how they connect to the overall research goals of the project.
