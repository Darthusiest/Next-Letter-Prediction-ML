# Character-Level Next-Letter Prediction — Design Brief

This document captures the **original problem statement and design requirements**
for the project, to be used later when writing a paper or longer report.

---

## 1. Core goal

- Build a **character-level language model** whose task is to **predict the next
  character** given a sequence of previous characters.
- Input text comes from **books, novels, articles, and other English corpora**.
- The system should start with a **clean, understandable baseline** and support
  **stronger character-level models** without unnecessary complexity.

**Current codebase:** n-gram, MLP, LSTM (`RNNCharModel`), and CNN (`CNNCharModel`)
are implemented and selectable via `python -m src.train --model ...`.
A standalone causal **transformer** module is not included; instead, the MLP
incorporates key transformer techniques: stacked **causal** self-attention with
**Rotary Position Embeddings (RoPE)**, multi-head attention pooling, SwiGLU,
pre-LayerNorm, weight tying, and **KV caching** for efficient generation.

---

## 2. Problem formulation

**Formal task:** given a sequence of characters \(c_1, \dots, c_L\), predict a
distribution over the next character \(c_{L+1}\), trained by **maximum
likelihood / cross-entropy**.

### Vocabulary design questions

We considered several options for what the model predicts:

- **Letters only (a-z)**
  - Pros: tiny vocab; focuses purely on spelling.
  - Cons: cannot model spaces or punctuation; unrealistic for natural text.

- **Letters + space (a-z + " ")**
  - Pros: adds **word boundaries**; still small.
  - Cons: no punctuation; sentences run together.

- **Letters + space + punctuation**
  - Pros: more realistic; captures **sentence boundaries** and contractions
    (e.g. `n't`, `'s`).
  - Cons: vocab a bit larger; punctuation can be noisy for very small datasets.

- **Letters + capitalization**
  - Pros: distinguishes sentence-initial caps, proper nouns, pronoun "I".
  - Cons: 2-3x vocab size; needs more data; often redundant for pure next-letter
    prediction.

**Initial design choice (plan):**

- Start with **letters (a-z) + space + minimal punctuation**, merge case, and
  optionally remove digits. This yields ~37-40 characters and focuses on:
  - **Word boundaries** (via space),
  - **Basic sentence structure** (via punctuation),
  - **Spelling patterns** (letters).

**Current implementation note:** we later generalized the implementation to
allow **any character** (letters upper/lower, digits, punctuation) while keeping
the same modeling pipeline.

---

## 3. Pipeline requirements

The learning pipeline should:

1. **Ingest raw text** from multiple files (e.g. in `data/raw/`).
2. **Clean and normalize** text: case handling, whitespace, digit handling,
   punctuation filtering.
3. **Build a character vocabulary** (`char -> id`, `id -> char`).
4. **Create training examples** via a **sliding window**:
   - For each position \(i\), context = `text[i-L:i]`, target = `text[i]`.
5. **Split into train / validation / test**:
   - Prefer **contiguous** or **by document** splits to avoid leakage.
6. **Wrap as datasets + dataloaders** for PyTorch:
   - `CharSequenceDataset`, `DataLoader` with shuffled train and fixed val/test.

This pipeline is implemented in:

- `src/preprocess.py` -- load text (with per-file Gutenberg boilerplate stripping),
  filter OCR garbage / noise lines, and normalize text.
- `src/vocab.py` -- build and persist vocabulary.
- `src/dataset.py` -- sliding-window dataset and contiguous splits.

### Data quality pipeline

Raw text from scanned PDFs and ebook conversions often contains OCR artifacts,
library stamps, page references, and whitespace-dense fragments that the model
would waste capacity learning (100% error rate on noise categories in analysis).
The preprocessing pipeline addresses this with two stages:

1. **Gutenberg boilerplate stripping** (`_strip_gutenberg_boilerplate`): applied
   per-file during `load_text()` to remove Project Gutenberg license headers and
   donation footers, keyed on `*** START/END OF ... PROJECT GUTENBERG ***` markers.
2. **Noise line filtering** (`filter_noisy_lines`): applied on the concatenated
   text inside `clean_text()` with six heuristics:
   - Very short fragments (< 3 non-whitespace chars)
   - Low alphabetic ratio (< 40% of non-whitespace chars are letters)
   - Short 1-2 word lines (< 8 chars total)
   - OCR fragment clusters (avg word length < 2.5, or < 3.0 with line < 30 chars)
   - High tiny-word ratio (> 40% of words are <= 2 chars in 5+ word lines)
   - Table-of-contents and PDF page reference patterns

Both stages can be disabled (`strip_gutenberg=False`, `filter_noise=False`).

---

## 4. Linguistics-informed design

The project explicitly integrates **linguistic intuitions** into modeling and
analysis (not as hand-crafted rules, but as guiding concepts):

- **Letter frequency** (E, T, A, ...):
  - Expect higher accuracy on frequent letters; plan to analyze per-character
    accuracy and confusion for rare letters (Q, Z, J).

- **Vowels vs consonants:**
  - Vowels and consonants behave differently (e.g. alternating patterns,
    consonant clusters like `str`, `ght`).
  - Plan analysis: separate accuracy and error patterns for vowels vs
    consonants.

- **Common digraphs/trigraphs** (`th`, `he`, `ing`, `tion`, etc.):
  - Justify **local context** (3-5 chars) and **CNN/n-gram baselines** for
    local spelling patterns.

- **Word boundaries (space):**
  - Space must be part of the vocab.
  - Planned metrics: accuracy immediately after a space vs mid-word; word-initial
    vs word-final errors.

- **Affixes and morphology** (`un-`, `re-`, `pre-`, `-ing`, `-ed`, `-tion`...):
  - Motivate **larger context windows (64-128)** and **sequence models**
    (RNN/LSTM) to capture morphological patterns.

- **Doubled letters** (`tt`, `ee`, `oo`, etc.):
  - Highlight importance of **short-range dependencies** (n-grams, small CNN
    kernels).

- **Orthography vs phonetics:**
  - We model written English; the system learns **orthographic** patterns (e.g.
    `gh`, `tion`) from data rather than phonetic rules.

These considerations inform:

- **Preprocessing**: do not remove spaces or all punctuation; be careful with
  case-folding and digit stripping.
- **Model selection**: local vs longer-range architectures.
- **Error analysis**: structured breakdowns (by vowel/consonant, position in
  word, affix patterns, etc.).

---

## 5. Modeling options and roles

Several model families are in scope, each with a clear role. **Status** is noted
where the repo already implements the family.

- **N-gram baseline (order 4-5):** -- **implemented** (`ngram`)
  - Captures **local letter combinations** (bigrams/trigrams).
  - Very fast and interpretable.
  - No long-range dependencies; no parameter sharing across positions.
  - Serves as the **simplest meaningful baseline** and a ceiling for purely
    local models.

- **MLP over fixed window (evolved neural baseline):** -- **implemented** (`mlp`)
  - Input: fixed context of length \(L\) (e.g. 64) -> char embed (no learned
    positional embeddings; RoPE applied in attention) -> N stacked causal
    self-attention layers -> multi-head attention pooling -> projection ->
    SwiGLU residual blocks -> weight-tied output logits.
  - Incorporates eight techniques from modern LLMs:
    1. **Rotary Position Embeddings (RoPE)** -- relative position encoding
       applied to Q and K in each self-attention layer, replacing learned
       `pos_embed`.  Better generalization and no per-sequence-length parameter.
    2. **Causal masking** -- `is_causal=True` in `scaled_dot_product_attention`
       enforces left-to-right attention (each position only sees earlier
       positions), matching autoregressive generation.
    3. **Stacked self-attention** -- configurable number of causal self-attention
       layers (`num_self_attn_layers`, default 2) before attention pooling.
       Deeper attention captures richer inter-position dependencies.
    4. **Multi-head attention pooling** -- N independent heads each score
       positions and produce an `embed_dim`-sized weighted sum; concatenated
       to `N * embed_dim` before projection.
    5. **SwiGLU activation** -- residual blocks use `SiLU(W_gate(x)) * W_up(x)`
       (gated linear unit) instead of plain GELU.
    6. **Pre-LayerNorm** -- LayerNorm on sublayer input (`x + sublayer(LN(x))`),
       with a final LayerNorm after all blocks.  Stabler gradients in depth.
    7. **Weight tying** -- `fc2.weight` shared with `embed.weight`; a learned
       projection bridges `hidden_dim -> embed_dim` when they differ.
    8. **KV caching** -- optional key/value cache in self-attention for efficient
       incremental generation (only compute new token's Q/K/V, reuse cached
       context).
  - Configurable via `--mlp-hidden-layers`, `--mlp-attn-heads`,
    `--mlp-self-attn-layers`, `--embed-dim`, `--hidden-dim`, `--dropout`.
  - Still a fixed-window model -- does not model variable-length history beyond
    the context length (KV cache extends the effective window during generation).

- **RNN / LSTM / GRU:** -- **implemented** (`rnn`, LSTM in code)
  - Processes characters sequentially with a hidden state.
  - Better for **variable-length dependencies**, word-level and short-range
    morphology.
  - Slower; long-range dependencies still limited.

- **CNN over characters:** -- **implemented** (`cnn`)
  - 1D convolutions over embeddings with small kernels (3-5).
  - Excellent for **local spelling**, digraphs, trigraphs, doubled letters.
  - Needs depth for long-range effects; mostly a "local pattern" expert.

- **Causal transformer (standalone):** -- **not a separate module**
  - A full autoregressive transformer with dedicated decoder blocks is not
    included as a separate model.  Instead, the MLP now incorporates causal
    self-attention with RoPE (see above), bridging the gap without requiring
    a separate transformer implementation.

**Build order (original plan) vs current:**

1. N-gram baseline -- done.
2. MLP baseline -- done; evolved with self-attention, multi-head pooling, SwiGLU,
   pre-LayerNorm, and weight tying (see above).
3. LSTM -- done (`RNNCharModel`).
4. CNN -- done (`CNNCharModel`).
5. Rich analysis tools -- partially done (`src/analysis/`, post-training pipeline).
6. Transformer -- deferred / out of repo (the MLP now borrows key transformer
   techniques without full autoregressive self-attention).

---

## 6. Key concepts (project-specific interpretations)

- **Context window / sequence length (L):** number of previous characters the
  model sees. For next-letter prediction:
  - \(L \approx 32\): word-level + short phrases.
  - \(L \approx 64\text{--}128\): adds clause-level and morphological patterns.
  - Default chosen: **64** (balance of information and CPU cost).

- **Embedding size:** dimensionality of each character vector.
  - Small vocab -> moderate dims (32-128) are sufficient to encode patterns like
    "vowel-like vs consonant-like", punctuation roles, etc. Default: **128**.
  - Must be divisible by `num_attn_heads` for the MLP self-attention layer.

- **Hidden dimension:** size of model's internal representations (MLP hidden,
  RNN hidden, CNN channels).
  - 128-512 depending on corpus size and compute budget; default **512**.

- **Attention heads (MLP):** number of heads for both the self-attention layers
  and the multi-head attention pooling. Default: **4**. Configurable via
  `MLP_NUM_ATTN_HEADS` in config or `--mlp-attn-heads` on the CLI.

- **Self-attention depth (MLP):** number of stacked causal self-attention layers
  with RoPE, applied before attention pooling. Default: **2**. Configurable via
  `MLP_NUM_SELF_ATTN_LAYERS` in config or `--mlp-self-attn-layers` on the CLI.

- **Projection layer & logits:** final linear layer maps hidden states to a
  vector of size `vocab_size`, producing **logits** (unnormalized scores for
  each character). Softmax over logits gives the **next-character
  distribution**. With weight tying, the output projection shares weights
  with the embedding layer; a `tie_proj` linear maps `hidden_dim -> embed_dim`
  when they differ.

- **Loss:** **cross-entropy** between predicted distribution and true next
  character, averaged over batch.

- **Generation temperature:** scaling logits by \(1/T\) before softmax to
  control randomness during sampling:
  - \(T < 1\): sharper, more deterministic (good for correct spelling).
  - \(T > 1\): more random (good for diversity).
  - Planned range: ~0.7-1.2.

---

## 7. Roadmap and phases

The project is organized into explicit phases:

1. **Phase 1 -- Design and planning** (this document and the plan file):
   - Clarify task, vocabulary, pipeline, and model options.
   - Define folder structure and first milestone.

2. **Phase 2 -- Baseline implementation:**
   - Implement text ingest, preprocessing, vocabulary, dataset.
   - Implement **n-gram** and **MLP** baselines.
   - Training, evaluation (loss, accuracy), and simple generation.
   - **Training note:** early stopping patience counts **epochs without improvement on validation loss measured at the end of each epoch**; optional mid-epoch `eval_every` checks can still refresh `best.pt` but do not advance patience. Reported **test** metrics and default analysis use **best validation** weights (`best.pt`) when available.

3. **Phase 3 -- Stronger models:**
   - **Done in repo:** RNN (LSTM) and CNN via `get_model` / `python -m src.train`.
   - **Not in repo:** transformer character model (see section 5).
   - MLP evolved with modern LLM techniques (self-attention, multi-head pooling,
     SwiGLU, pre-LN, weight tying) to close the gap without full transformer
     complexity.
   - Compare runs on validation metrics and qualitative generation (`outputs/runs/<run_id>/`).

4. **Phase 4 -- Analysis:**
   - Tools for: most common letters, vowel vs consonant performance, confusion
     matrices, typical failure modes, effect of context length, embedding
     dimension, and generation temperature.

This mirrors the user's initial specification and is intended to support a
future **paper-style write-up** of the system's behavior and design choices.
