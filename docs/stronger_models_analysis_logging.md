# Stronger Models, Analysis, and Experiment Logging

This document records the design for extending the character‑level next‑letter
project beyond the initial n‑gram and MLP baselines. It is intended as
documentation for future writing (e.g. a paper or report) and for guiding
implementation work.

---

## 1. Goals and scope

- **Stronger sequence models**
  - Add neural sequence models that can capture longer and more structured
    dependencies than the MLP and n‑gram baselines:
    - `RNNCharModel` (LSTM / GRU) in `src/models/rnn.py`.
    - `CNNCharModel` in `src/models/cnn.py`.
    - `TransformerCharModel` in `src/models/transformer.py`.
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
  - New models (plug into the same training loop via `models.get_model`):
    - `RNNCharModel` — LSTM over embeddings; last hidden state → logits.
    - `CNNCharModel` — Conv1d over embeddings with multiple kernel sizes +
      global max pooling → logits.
    - `TransformerCharModel` — causal, decoder‑style transformer with
      positional encodings; last position state → logits.

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
  - Cheaper and simpler than a transformer; a good “middle” reference model.

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

### 3.3 Transformer character model (`src/models/transformer.py`)

- **Purpose**
  - Provide a modern, attention‑based sequence model that can make use of
    **longer‑range context** within the character window and model interactions
    between any pair of positions.

- **Interface**
  - `TransformerCharModel(vocab_size, context_length, embed_dim, num_heads, num_layers, ff_dim, dropout)`.
  - Uses learned or sinusoidal **positional encodings** to represent order.
  - Applies a small causal transformer encoder/decoder stack with self‑attention.
  - `forward(context: LongTensor[B, L]) -> logits[B, V]`:
    - Embed + positional encoding → `(B, L, D)`.
    - Causal mask ensures each position attends only to **previous** positions.
    - Use the final position’s representation `(B, D)` for next‑character
      prediction, project to logits `(B, V)`.

- **Design rationale**
  - Attention layers can focus on the **most relevant characters** anywhere in
    the context (e.g. beginnings of words or previous words in a phrase).
  - Better suited than RNNs for capturing interactions across many positions
    and for longer contexts, at the cost of higher computational overhead.

### 3.4 Integration via `get_model` and `train.py`

- `src/models/__init__.py::get_model(name, vocab_size, context_length, **kwargs)`
  acts as a single factory entry point for all models:
  - `name == "ngram"`, `"mlp"`, `"rnn"`, `"cnn"`, `"transformer"`.
  - Passes through model‑specific hyperparameters (embedding size, hidden size,
    number of layers, etc.).
- `src/train.py` constructs the model via `get_model` using configuration from
  `src/config.py` and trains it using the same training loop (cross‑entropy
  loss, Adam optimizer, validation, early stopping).\n+
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
  - Normalize rows to get probabilities.\n+- Inspect which letters the model most often confuses (e.g. `c` vs `k`, `m` vs
  `n`, `i` vs `l`), and where rare letters (q, z, j, x) fail.\n+
### 4.3 Position‑aware accuracy\n+\n+- Analyze accuracy as a function of **position relative to spaces**:\n+  - Word‑initial (previous character is space).\n+  - Mid‑word.\n+  - Word‑final (next character is space).\n+- Purpose: understand how well the model captures **word boundaries**, and\n+  whether it learns typical word‑initial and word‑final distributions.\n+\n+### 4.4 Context length and embedding dimension\n+\n+- Provide helpers that run small sweeps over:\n+  - `CONTEXT_LENGTH` (e.g. 16, 32, 64, 128).\n+  - Embedding size (e.g. 16, 32, 64, 128).\n+- For each setting, call the existing `train()` function, record validation\n+  loss and accuracy, and write results to logs.\n+- Purpose: quantify how much longer context and higher embedding capacity\n+  actually help on **next‑character prediction** in this corpus.\n+\n+### 4.5 Temperature and generation behavior\n+\n+- Generate samples at different temperatures (e.g. 0.7, 1.0, 1.3) using\n+  `generate.py`.\n+- Optionally compute simple statistics:\n+  - Character frequency distributions.\n+  - Repetition rates and average word length.\n+- Purpose: relate **subjective generation quality** (coherent spelling vs\n+  diversity) to measurable statistics.\n+\n+---\n+\n+## 5. Experiment logging\n+\n+### 5.1 Logging format and storage\n+\n+- Create a `logs/` directory with:\n+  - `logs/runs/` — one JSON file per training run.\n+  - Optional `logs/summary.csv` — a table aggregating run metadata and final\n+    metrics.\n+\n+- Example run JSON fields:\n+\n+  - `run_id` (timestamp + model name).\n+  - `timestamp`.\n+  - `model` (ngram/mlp/rnn/cnn/transformer).\n+  - `config` (context length, embedding size, hidden size, dropout, epochs,\n+    etc.).\n+  - `data` (number of characters used, vocab size, which files from\n+    `data/raw/`).\n+  - `metrics` (best validation loss/accuracy, test loss/accuracy).\n+\n+### 5.2 Integration with training\n+\n+- After training finishes (or after the best checkpoint is found), `train.py`\n+  will:\n+  - Gather the configuration values used for the run (from `config.py` and any\n+    overrides).\n+  - Call `evaluate()` on the test set to obtain final metrics.\n+  - Write a JSON run file to `logs/runs/` via a small helper (e.g.\n+    `log_run(config, data_info, metrics)`).\n+- Optionally append a row to `logs/summary.csv` for quick comparisons without\n+  parsing JSON.\n+\n+### 5.3 Reproducibility\n+\n+- Each run log should also record:\n+  - Random seed.\n+  - Git commit hash (if available).\n+  - List of dataset files from `data/raw/` used to build the corpus.\n+- This supports reproducible experiments and clear provenance for future\n+  publications.\n+\n+---\n+\n+## 6. Implementation order (for reference)\n+\n+1. Implement `RNNCharModel` and integrate it into `models.get_model`.\n+2. Implement `CNNCharModel` and integrate it into `models.get_model`; add\n+   tests for basic forward behavior.\n+3. Implement `TransformerCharModel` with causal masking and positional\n+   encodings; integrate and test.\n+4. Extend `train.py` to select between all model names using config‑driven\n+   hyperparameters; ensure `generate.py` works with the new models.\n+5. Add experiment logging helpers and wire them into `train.py`.\n+6. Flesh out `analysis.py` with vowel/consonant accuracy, confusion matrices,\n+   positional analysis, and context/embedding/temperature experiments.\n+7. Update top‑level docs (`README.md`, `docs/design.md`) to describe the new\n+   models and analysis capabilities.\n+\n+This document mirrors the \"Stronger Models, Analysis, and Experiment Logging\"\n+plan and serves as a stable reference for why these additions are being made\n+and how they connect to the overall research goals of the project.\n+
