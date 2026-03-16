# Character-Level Next-Letter Prediction — Design Brief

This document captures the **original problem statement and design requirements**
for the project, to be used later when writing a paper or longer report.

---

## 1. Core goal

- Build a **character-level language model** whose task is to **predict the next
  character** given a sequence of previous characters.
- Input text comes from **books, novels, articles, and other English corpora**.
- The system should start with a **clean, understandable baseline** and be
  structured so it can scale to **more advanced architectures** (RNNs, CNNs,
  transformers).

---

## 2. Problem formulation

**Formal task:** given a sequence of characters \(c_1, \dots, c_L\), predict a
distribution over the next character \(c_{L+1}\), trained by **maximum
likelihood / cross-entropy**.

### Vocabulary design questions

We considered several options for what the model predicts:

- **Letters only (a–z)**  
  - Pros: tiny vocab; focuses purely on spelling.  
  - Cons: cannot model spaces or punctuation; unrealistic for natural text.

- **Letters + space (a–z + “ ”)**  
  - Pros: adds **word boundaries**; still small.  
  - Cons: no punctuation; sentences run together.

- **Letters + space + punctuation**  
  - Pros: more realistic; captures **sentence boundaries** and contractions
    (e.g. `n't`, `'s`).  
  - Cons: vocab a bit larger; punctuation can be noisy for very small datasets.

- **Letters + capitalization**  
  - Pros: distinguishes sentence-initial caps, proper nouns, pronoun “I”.  
  - Cons: 2–3× vocab size; needs more data; often redundant for pure next‑letter
    prediction.

**Initial design choice (plan):**

- Start with **letters (a–z) + space + minimal punctuation**, merge case, and
  optionally remove digits. This yields ~37–40 characters and focuses on:
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
3. **Build a character vocabulary** (`char → id`, `id → char`).
4. **Create training examples** via a **sliding window**:
   - For each position \(i\), context = `text[i-L:i]`, target = `text[i]`.
5. **Split into train / validation / test**:
   - Prefer **contiguous** or **by document** splits to avoid leakage.
6. **Wrap as datasets + dataloaders** for PyTorch:
   - `CharSequenceDataset`, `DataLoader` with shuffled train and fixed val/test.

This pipeline is implemented in:

- `src/preprocess.py` — load and clean text.  
- `src/vocab.py` — build and persist vocabulary.  
- `src/dataset.py` — sliding-window dataset and contiguous splits.

---

## 4. Linguistics-informed design

The project explicitly integrates **linguistic intuitions** into modeling and
analysis (not as hand‑crafted rules, but as guiding concepts):

- **Letter frequency** (E, T, A, …):  
  - Expect higher accuracy on frequent letters; plan to analyze per‑character
    accuracy and confusion for rare letters (Q, Z, J).

- **Vowels vs consonants:**  
  - Vowels and consonants behave differently (e.g. alternating patterns,
    consonant clusters like `str`, `ght`).  
  - Plan analysis: separate accuracy and error patterns for vowels vs
    consonants.

- **Common digraphs/trigraphs** (`th`, `he`, `ing`, `tion`, etc.):  
  - Justify **local context** (3–5 chars) and **CNN/n‑gram baselines** for
    local spelling patterns.

- **Word boundaries (space):**  
  - Space must be part of the vocab.  
  - Planned metrics: accuracy immediately after a space vs mid‑word; word‑initial
    vs word‑final errors.

- **Affixes and morphology** (`un-`, `re-`, `pre-`, `-ing`, `-ed`, `-tion`…):  
  - Motivate **larger context windows (64–128)** and **sequence models**
    (RNN/transformer) to capture morphological patterns.

- **Doubled letters** (`tt`, `ee`, `oo`, etc.):  
  - Highlight importance of **short‑range dependencies** (n‑grams, small CNN
    kernels).

- **Orthography vs phonetics:**  
  - We model written English; the system learns **orthographic** patterns (e.g.
    `gh`, `tion`) from data rather than phonetic rules.

These considerations inform:

- **Preprocessing**: do not remove spaces or all punctuation; be careful with
  case‑folding and digit stripping.
- **Model selection**: local vs longer‑range architectures.  
- **Error analysis**: structured breakdowns (by vowel/consonant, position in
  word, affix patterns, etc.).

---

## 5. Modeling options and roles

The project plans to explore several model families, each with a clear role:

- **N‑gram baseline (order 4–5):**
  - Captures **local letter combinations** (bigrams/trigrams).  
  - Very fast and interpretable.  
  - No long‑range dependencies; no parameter sharing across positions.  
  - Serves as the **simplest meaningful baseline** and a ceiling for purely
    local models.

- **MLP over fixed window (first neural baseline):**
  - Input: fixed context of length \(L\) (e.g. 64) → embeddings → flatten →
    hidden layers → logits.  
  - Captures **within‑window spelling patterns**; simple to implement.  
  - Does not model variable‑length history beyond the window.

- **RNN / LSTM / GRU:**  
  - Processes characters sequentially with a hidden state.  
  - Better for **variable‑length dependencies**, word‑level and short‑range
    morphology.  
  - Slower; long‑range dependencies still limited.

- **CNN over characters:**  
  - 1D convolutions over embeddings with small kernels (3–5).  
  - Excellent for **local spelling**, digraphs, trigraphs, doubled letters.  
  - Needs depth for long‑range effects; mostly a “local pattern” expert.

- **Transformer:**  
  - Causal self‑attention over previous characters only; uses positional
    encodings.  
  - Strong on **long‑range dependencies**, phrase‑ and sentence‑level
    structure.  
  - More data‑hungry and heavier; reserved for later stages.

Planned **build order**:

1. N‑gram baseline.  
2. Tiny MLP.  
3. Full MLP baseline.  
4. LSTM/GRU.  
5. CNN.  
6. Transformer.  
7. Rich analysis tools.

---

## 6. Key concepts (project‑specific interpretations)

- **Context window / sequence length (L):** number of previous characters the
  model sees. For next‑letter prediction:
  - \(L \approx 32\): word‑level + short phrases.  
  - \(L \approx 64\text{–}128\): adds clause‑level and morphological patterns.  
  - Default chosen: **64** (balance of information and CPU cost).

- **Embedding size:** dimensionality of each character vector.  
  - Small vocab → moderate dims (32–64) are sufficient to encode patterns like
    “vowel‑like vs consonant‑like”, punctuation roles, etc.

- **Hidden dimension:** size of model’s internal representations (MLP hidden,
  RNN hidden, transformer width).  
  - 128–256 is a good starting range; larger can overfit small corpora.

- **Projection layer & logits:** final linear layer maps hidden states to a
  vector of size `vocab_size`, producing **logits** (unnormalized scores for
  each character). Softmax over logits gives the **next‑character
  distribution**.

- **Loss:** **cross‑entropy** between predicted distribution and true next
  character, averaged over batch.

- **Generation temperature:** scaling logits by \(1/T\) before softmax to
  control randomness during sampling:
  - \(T < 1\): sharper, more deterministic (good for correct spelling).  
  - \(T > 1\): more random (good for diversity).  
  - Planned range: ~0.7–1.2.

---

## 7. Roadmap and phases

The project is organized into explicit phases:

1. **Phase 1 – Design and planning** (this document and the plan file):  
   - Clarify task, vocabulary, pipeline, and model options.  
   - Define folder structure and first milestone.

2. **Phase 2 – Baseline implementation:**  
   - Implement text ingest, preprocessing, vocabulary, dataset.  
   - Implement **n‑gram** and **MLP** baselines.  
   - Training, evaluation (loss, accuracy), and simple generation.

3. **Phase 3 – Stronger models:**  
   - Add RNN/LSTM/GRU, CNN, and transformer character models.  
   - Compare on validation metrics and qualitative generation.

4. **Phase 4 – Analysis:**  
   - Tools for: most common letters, vowel vs consonant performance, confusion
     matrices, typical failure modes, effect of context length, embedding
     dimension, and generation temperature.

This mirrors the user’s initial specification and is intended to support a
future **paper‑style write‑up** of the system’s behavior and design choices.

