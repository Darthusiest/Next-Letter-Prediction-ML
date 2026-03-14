"""
Training loop: load data, build model, train with cross-entropy (or nll_loss for n-gram),
validation every N steps, checkpoint best model.
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import (
    CONTEXT_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    EVAL_EVERY_N_STEPS,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR,
    RAW_DATA_DIR,
    MAX_CHARS,
    EMBED_DIM,
    HIDDEN_DIM,
    DROPOUT,
    NGRAM_ORDER,
    NGRAM_SMOOTHING,
    DEFAULT_ALLOWED_CHARS,
    SEED,
)
from .preprocess import load_text, clean_text
from .vocab import build_vocab_from_text
from .dataset import get_splits, get_dataloaders
from .models import get_model
from .models.baseline_ngram import NGramModel
from .evaluate import evaluate
from .utils import set_seed, get_device, setup_logging

logger = logging.getLogger(__name__)


def train(
    data_paths: list = None,
    model_name: str = "mlp",
    max_chars: int = None,
    context_length: int = CONTEXT_LENGTH,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    eval_every: int = EVAL_EVERY_N_STEPS,
    early_stop_patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: Path = None,
    seed: int = SEED,
):
    set_seed(seed)
    device = get_device()
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if data_paths is None:
        data_paths = list(Path(RAW_DATA_DIR).glob("*.txt"))
        if not data_paths:
            raise FileNotFoundError(
                f"No .txt files in {RAW_DATA_DIR}. Put text files in data/raw/."
            )

    max_chars = max_chars or MAX_CHARS
    raw = load_text(data_paths, max_chars=max_chars)
    text = clean_text(raw)
    logger.info("Loaded and cleaned text: %d characters", len(text))

    vocab = build_vocab_from_text(text, allowed_chars=DEFAULT_ALLOWED_CHARS)
    logger.info("Vocabulary size: %d", vocab.vocab_size)

    train_ds, val_ds, test_ds = get_splits(
        text, vocab, context_length=context_length
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    if model_name == "ngram":
        model = get_model(
            "ngram",
            vocab_size=vocab.vocab_size,
            context_length=context_length,
            order=NGRAM_ORDER,
            smoothing=NGRAM_SMOOTHING,
        )
        # Fit n-gram on full train segment (ids)
        model.fit(train_ds.ids)
        model.to(device)
        # No optimizer; just evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, is_ngram=True)
        logger.info("N-gram val loss=%.4f acc=%.4f", val_loss, val_acc)
        torch.save(
            {"model": model, "vocab": vocab, "config": {"model": "ngram"}},
            checkpoint_dir / "ngram_baseline.pt",
        )
        return model, vocab
    else:
        model = get_model(
            "mlp",
            vocab_size=vocab.vocab_size,
            context_length=context_length,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        step = 0
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(context)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            step += 1
            if step % eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, is_ngram=False)
                logger.info(
                    "Epoch %d step %d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, step, total_loss / num_batches, val_loss, val_acc,
                )
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "vocab": vocab,
                            "context_length": context_length,
                            "model_name": model_name,
                        },
                        checkpoint_dir / "best.pt",
                    )
                else:
                    patience_counter += 1

        epoch_train_loss = total_loss / num_batches
        val_loss, val_acc = evaluate(model, val_loader, device, is_ngram=False)
        logger.info(
            "Epoch %d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, epoch_train_loss, val_loss, val_acc,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "context_length": context_length,
                    "model_name": model_name,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience:
            logger.info("Early stopping after %d epochs", epoch + 1)
            break

    return model, vocab


def main():
    """Entry point: run training with config defaults."""
    setup_logging()
    train(model_name="mlp")


if __name__ == "__main__":
    main()
