# Stronger Models, Analysis, and Experiment Logging

This document records the design for extending the character-level next-letter
project beyond the initial n-gram and MLP baselines. It is intended as
documentation for future writing (e.g. a paper or report) and for guiding
implementation work.

---

## Implementation status (summary)

As of the current tree:

- **Models in `get_model` / training:** `ngram`, `mlp`, `rnn`, `cnn`. There is
  **no** transformer module in the repository.
- **MLP architecture:** the MLP has been evolved with five techniques from modern
  LLMs -- self-attention, multi-head attention pooling, SwiGLU activation,
  pre-LayerNorm, and weight tying (see section 3.3 below). It is _not_ a full
  transformer; there is one self-attention layer before pooling, not full
  autoregressive causal attention.
- **Training entrypoint:** `python -m src.train` with argparse (`--model`,
  `--data-source`, `--max-chars`, `--eval-every`, `--compile`, `--rnn-layers`,
  `--weight-decay`, `--label-smoothing`, `--plateau-lr`, `--grad-clip`,
  `--warmup-steps`, `--mlp-hidden-layers`, `--mlp-attn-heads`, `--dropout`,
  `--embed-dim`, `--hidden-dim`, `--tokenizer`, `--bpe-vocab-size`);
  programmatic `train()` in `src/train.py` supports further knobs (e.g.
  `num_workers`, `compile_model`).
- **Device selection:** CUDA if available, else Apple **MPS**, else CPU
  (`src/utils/__init__.py`). Mixed precision: CUDA uses `GradScaler` + autocast;
  MPS uses float16 autocast (PyTorch 2+). See **README** for dataloader workers,
  fast `CharVocab.encode`, Adam `foreach`, and `torch.inference_mode` in eval.
- **Artifacts per run:** `outputs/runs/<run_id>/` -- `train_corpus.txt`,
  `best.pt` (best **validation** weights; includes `model_kwargs`),
  `checkpoint.pt` (**last training step** weights + `model_kwargs`),
  `metrics.json`, `loss_history.json`, plus analysis under `plots/` and `reports/`.
- **Metrics policy:** `metrics.json` fields **`test_loss` / `test_accuracy`** are
  computed from **`best.pt`** when present; **`test_*_final_epoch`** use the
  last-step weights. Field **`metrics_checkpoint_policy`** explains this in-file.
- **Analysis checkpoint order:** `run_post_training_analysis` loads **`best.pt`
  first**, then `checkpoint.pt`, so prediction plots and model-based analyses
  align with best validation by default.
- **Calibration report:** when both checkpoints exist,
  `reports/checkpoint_comparison.txt` compares entropy and top-k next-char
  probabilities on sample contexts (`pre`, `th`, `ing`) for best vs last epoch
  (`src/analysis/calibration_compare.py`).
- **Early stopping:** patience counts **epochs without improvement on
  end-of-epoch** validation only (`EARLY_STOPPING_PATIENCE` in `config.py`;
  default 5). Mid-epoch `eval_every` can still refresh `best.pt` but does not
  advance patience (see README / `docs/design.md`).
- **Regularization (neural):** config `WEIGHT_DECAY` (default `1e-2`),
  `LABEL_SMOOTHING` (default `0.1`), `GRAD_CLIP_NORM` (default `1.0`),
  `DROPOUT` (default `0.3`, applied to embeddings and hidden layers),
  and `USE_PLATEAU_LR` + `ReduceLROnPlateau` on epoch-end val loss
  (on by default; CLI `--no-plateau-lr` to disable).
- **Tokenization:** Character-level (default) or BPE subword via
  `--tokenizer bpe --bpe-vocab-size N`. Config `TOKENIZER_TYPE` and
  `BPE_VOCAB_SIZE` (default 2000). BPE tokenizer trained on corpus via
  HuggingFace `tokenizers` library (`src/tokenizer.py`).
- **Data quality pipeline:** `load_text()` strips Project Gutenberg
  header/footer boilerplate per-file. `clean_text()` applies a noise line filter
  (`filter_noisy_lines`) that removes OCR garbage, short fragments,
  whitespace-dense lines, table-of-contents entries, and PDF page references
  using six heuristics. Both can be disabled. The corpus expanded from 3 raw
  text files (~1.4M chars) to 11 classic literature books (~5.4M clean chars)
  via Calibre ebook conversion of epub/mobi sources in `data/raw/nlp-ebooks/`.
- **Logging:** `logs/runs/<run_id>.json` via `log_run()`.
- **Analysis:** `src/analysis/run_analysis.py` (`run_post_training_analysis`);
  optional attention plots only apply if a model exposes attention (no
  transformer shipped).

---

## 1. Goals and scope

- **Stronger sequence models**
  - Add neural sequence models that can capture longer and more structured
    dependencies than the n-gram baseline:
    - `RNNCharModel` (LSTM / GRU) in `src/models/rnn.py` -- **implemented**.
    - `CNNCharModel` in `src/models/cnn.py` -- **implemented**.
  - Evolve the MLP baseline with modern LLM techniques to improve capacity
    without switching to a full transformer:
    - Self-attention, multi-head pooling, SwiGLU, pre-LN, weight tying --
      **implemented** in `src/models/mlp.py`.
- **Linguistics-informed analysis**
  - Extend `src/analysis.py` with tools to evaluate the model in terms that
    matter for English spelling and morphology (vowels vs consonants, word
    boundaries, confusions, etc.).
- **Experiment logging**
  - Introduce simple but structured logging (configs + metrics per run) so that
    experiments are reproducible and results can be compared and written up.

Constraints:

- Reuse the existing data pipeline (preprocess -> vocab -> dataset).
- Keep models sized to run on a single CPU-only machine.
- Avoid heavy dependencies and over-engineering; prefer clear, readable PyTorch.

---

## 2. Architecture overview

High-level flow, extending the current baseline:

- **Data path**
  - Raw text (`data/raw/*.txt` or converted from PDF/EPUB) ->
  - `preprocess.py` (cleaning, normalization) ->
  - `vocab.py` (char<->id mapping) ->
  - `dataset.py` (`CharSequenceDataset`, sliding window context->next-char) ->
  - `DataLoader` (train/val/test splits).

- **Models**
  - Baselines:
    - `NGramModel` (local spelling patterns only).
    - `MLPCharModel` (positional embeddings -> single-layer multi-head
      self-attention -> multi-head attention pooling -> SwiGLU residual blocks
      with pre-LayerNorm -> weight-tied output; depth via `num_hidden_layers` /
      `MLP_NUM_HIDDEN_LAYERS`, default 5; heads via `num_attn_heads` /
      `MLP_NUM_ATTN_HEADS`, default 4;
      CLI `--mlp-hidden-layers`, `--mlp-attn-heads`, `--dropout`, `--embed-dim`,
      `--hidden-dim`).
  - Sequence / local pattern models (plug into the same training loop via `models.get_model`):
    - `RNNCharModel` -- LSTM over embeddings; last hidden state -> logits.
    - `CNNCharModel` -- Conv1d over embeddings with multiple kernel sizes +
      global max pooling -> logits.

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
  - Serve as the first sequence-aware neural baseline beyond MLP and n-gram.
  - Capture variable-length dependencies within a fixed context window, such as
    morphological patterns (`un-`, `re-`, `-ing`, `-ed`, `-tion`) and word-level
    effects.

- **Interface**
  - `RNNCharModel(vocab_size, context_length, embed_dim, hidden_dim, num_layers, dropout)`.
  - `forward(context: LongTensor[B, L]) -> logits[B, V]`:
    - Embed: `(B, L) -> (B, L, E)`.
    - LSTM: `(B, L, E) -> (B, L, H)`.
    - Use **last time step** `(B, H)` as summary of the context.
    - Apply dropout and a linear projection to obtain logits `(B, V)`.

- **Design rationale**
  - RNNs/LSTMs respect **order** and can accumulate history via the hidden
    state, which is crucial for:
    - Word-initial vs mid-word vs word-final positions.
    - Common affixes and longer local patterns than a small n-gram can handle.

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
    - Global max-pool each feature map over the sequence -> `(B, C_total)`.
    - Dropout + linear projection to logits `(B, V)`.

- **Design rationale**
  - Convolutional filters act as learned detectors for local patterns like
    `th`, `ing`, `ion`, consonant clusters, and doubled letters.
  - Highly parallelizable and efficient; ideal for studying **orthographic
    neighborhoods** and local spelling statistics.
  - Less suited to long-range dependencies (e.g. agreement across multiple
    words) unless made deeper.

### 3.3 MLP architecture evolution (`src/models/mlp.py`)

The MLP was evolved from a naive flatten-based architecture through several
iterations to its current form, which incorporates five techniques from modern
LLMs:

- **Interface**
  - `MLPCharModel(vocab_size, context_length, embed_dim, hidden_dim, dropout, num_hidden_layers, num_attn_heads)`.
  - `forward(context: LongTensor[B, L]) -> logits[B, V]`.

- **Architecture (current)**

  ```
  embed(context) + pos_embed -> embed_drop           (B, L, E)
  -> MultiHeadSelfAttention (pre-LN, N heads)         (B, L, E)
  -> multi-head attention pooling (N heads)            (B, N*E)
  -> proj -> GELU -> dropout                           (B, H)
  -> SwiGLU residual blocks x (num_hidden_layers - 1)  (B, H)
  -> final LayerNorm                                   (B, H)
  -> [tie_proj if H != E]                              (B, E)
  -> fc2 (weight-tied with embed)                      (B, V)
  ```

- **Five modern LLM techniques applied**

  1. **Multi-head attention pooling** (replaces single-head `attn_score`):
     N independent Linear(E, 1) heads each learn separate position-importance
     scores, softmax over L, and produce an E-dim weighted sum. Concatenated
     to N*E before projection. This eliminates the E-dim bottleneck that
     limited the single-head version.

  2. **SwiGLU activation** (replaces GELU in residual blocks):
     Each residual block uses `SiLU(W_gate(x)) * W_up(x)` -- two parallel
     linear projections where one gates the other. For this small model,
     both projections use `hidden_dim` (no intermediate expansion).

  3. **Pre-LayerNorm** (instead of post-LayerNorm):
     Changed from `x + drop(LN(act(linear(x))))` to `x + drop(act(linear(LN(x))))`.
     LayerNorm on the input to each sublayer rather than the output, with a
     `final_ln` after all blocks. More stable gradients in deeper networks.

  4. **Self-attention layer before pooling**:
     One layer of multi-head self-attention (Q=K=V=x, pre-LN, same number of
     heads as pooling) between embedding and attention pooling. Lets positions
     interact before aggregation -- e.g. "q" at position 50 can see "u" at
     position 51. O(L^2) but L=64 is negligible.

  5. **Weight tying** (embed <-> output):
     `fc2.weight` is shared with `embed.weight`. When `hidden_dim != embed_dim`
     (the typical case: 512 vs 128), a learned `tie_proj = Linear(H, E, bias=False)`
     bridges the gap. Reduces parameters and acts as a regularizer.

- **Architecture history**

  | Version | Architecture | Key change |
  |---------|-------------|------------|
  | MLP-1 | flatten, 3 layers, no embed drop | Initial baseline |
  | MLP-2 | flatten, 3 layers, embed drop | +embedding dropout |
  | 5-layer | flatten, 5 layers, embed drop | +depth |
  | Attn pool | single-head attn pool, 5L | +positional embed, attention pooling |
  | Current | self-attn + multi-head pool + SwiGLU + pre-LN + weight tying | Modern LLM techniques |

- **Design rationale**
  - The single-head attention pooling created a 128-dim bottleneck that limited
    capacity. Multi-head pooling (4 heads x 128 = 512) feeds the full hidden_dim
    into the projection, matching the residual block width.
  - SwiGLU and pre-LN are standard in modern LLMs (LLaMA, PaLM) and improve
    training stability and expressiveness at negligible cost.
  - Self-attention lets the model capture local interactions (digraphs, common
    pairs) at the embedding level before pooling collapses the sequence.
  - Weight tying is a free regularizer that also reduces the parameter count.

### 3.4 Integration via `get_model` and `train.py`

- `src/models/__init__.py::get_model(name, vocab_size, context_length, **kwargs)`
  acts as a single factory entry point for all models:
  - `name == "ngram"`, `"mlp"`, `"rnn"`, `"cnn"`.
  - Passes through model-specific hyperparameters (embedding size, hidden size,
    number of layers, number of attention heads, etc.).
- `src/train.py` constructs the model via `get_model` using configuration from
  `src/config.py` and trains it using the same training loop (cross-entropy
  loss, Adam optimizer, validation, early stopping).
- CLI: `python -m src.train --model mlp` (optional `--mlp-hidden-layers N`,
  `--mlp-attn-heads N`), or `rnn`, `cnn`, `ngram`.

---

## 4. Analysis tools (`src/analysis.py`)

Analysis should go beyond scalar loss/accuracy and connect to **linguistic
structure** in English:

### 4.1 Vowel vs consonant performance

- Define vowels (`a, e, i, o, u` plus uppercase) and treat other letters as
  consonants; track spaces and punctuation separately.
- Compute:
  - Per-character accuracy.
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

### 4.3 Position-aware accuracy

- Analyze accuracy as a function of **position relative to spaces**:
  - Word-initial (previous character is space).
  - Mid-word.
  - Word-final (next character is space).
- Purpose: understand how well the model captures **word boundaries**, and
  whether it learns typical word-initial and word-final distributions.

### 4.4 Context length and embedding dimension

- Provide helpers that run small sweeps over:
  - `CONTEXT_LENGTH` (e.g. 16, 32, 64, 128).
  - Embedding size (e.g. 16, 32, 64, 128).
- For each setting, call the existing `train()` function, record validation
  loss and accuracy, and write results to logs.
- Purpose: quantify how much longer context and higher embedding capacity
  actually help on **next-character prediction** in this corpus.

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
  - `logs/runs/` -- one JSON file per training run.
  - Optional `logs/summary.csv` -- a table aggregating run metadata and final
    metrics.

- Example run JSON fields:

  - `run_id` (timestamp + model name).
  - `timestamp`.
  - `model` (ngram/mlp/rnn/cnn).
  - `config` (context length, embedding size, hidden size, dropout, MLP hidden
    layer count, attention heads, epochs, etc.).
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
3. ~~Extend `train.py` to select between all model names~~ -- **Done** via CLI
   and `get_model`; `generate.py` works with any neural model loaded from a
   checkpoint.
4. ~~Add experiment logging helpers and wire them into `train.py`.~~ **Done**
   (`log_run`, `metrics.json`, run directories).
5. ~~Evolve MLP with modern LLM techniques~~ -- **Done**: self-attention,
   multi-head attention pooling, SwiGLU, pre-LayerNorm, weight tying.
   Config `MLP_NUM_ATTN_HEADS`, CLI `--mlp-attn-heads`.
6. Flesh out analysis with vowel/consonant accuracy, confusion matrices,
   positional analysis, and context/embedding/temperature experiments -- **partially
   done** (`src/analysis/` modules; some items remain aspirational).
7. Keep `README.md` and `docs/design.md` aligned with shipped models and training
   behavior -- **ongoing**.

This document mirrors the "Stronger Models, Analysis, and Experiment Logging"
plan and serves as a stable reference for why these additions are being made
and how they connect to the overall research goals of the project.
