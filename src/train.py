"""
Training loop: load data, build model, train with cross-entropy (or nll_loss for n-gram),
validation every N steps, checkpoint best model.
"""

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script: `python src/train.py`
    # Preferred is `python -m src.train`, but this keeps both working.
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import (
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
from src.preprocess import load_text, clean_text
from src.vocab import build_vocab_from_text
from src.dataset import get_splits, get_dataloaders
from src.models import get_model
from src.models.baseline_ngram import NGramModel
from src.evaluate import evaluate
from src.utils import set_seed, get_device, setup_logging, log_run
from src.utils.io_utils import ensure_run_dir, save_json
from src.analysis.config import AnalysisConfig
from src.analysis.run_analysis import run_post_training_analysis

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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Choose run_id and run directory
    import time

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_run_dir(run_id)
    checkpoint_dir = run_dir  # for best.pt / checkpoint.pt

    if data_paths is None:
        # Collect .txt files from data/raw/ and data/raw/nlp-ebooks/
        raw_dir = Path(RAW_DATA_DIR)
        ebook_dir = raw_dir / "nlp-ebooks"
        data_paths = list(raw_dir.glob("*.txt"))
        if ebook_dir.exists():
            data_paths.extend(ebook_dir.glob("*.txt"))
        if not data_paths:
            raise FileNotFoundError(
                f"No .txt files in {RAW_DATA_DIR} or {RAW_DATA_DIR / 'nlp-ebooks'}. "
                "Put text files in data/raw/ or data/raw/nlp-ebooks/."
            )

    max_chars = max_chars or MAX_CHARS
    raw = load_text(data_paths, max_chars=max_chars)
    text = clean_text(raw)
    logger.info("Loaded and cleaned text: %d characters", len(text))
    # Save cleaned corpus snapshot for analysis
    (run_dir / "train_corpus.txt").write_text(text, encoding="utf-8")

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
        # For neural models, delegate to get_model with the requested name.
        model = get_model(
            model_name,
            vocab_size=vocab.vocab_size,
            context_length=context_length,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    epochs_list = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        step = 0
        for context, target in train_loader:
            non_blocking = device.type == "cuda"
            context = context.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(context)
                loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "vocab": vocab,
                            "context_length": context_length,
                            "model_name": model_name,
                        },
                        checkpoint_dir / "best.pt",
                    )

        epoch_train_loss = total_loss / num_batches
        epochs_list.append(epoch + 1)
        train_loss_history.append(epoch_train_loss)
        val_loss, val_acc = evaluate(model, val_loader, device, is_ngram=False)
        val_loss_history.append(val_loss)
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

    # After training, evaluate on test set for logging
    test_loss, test_acc = evaluate(model, test_loader, device, is_ngram=False)

    # Save loss history for analysis
    loss_history = {
        "epochs": epochs_list,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }
    save_json(loss_history, run_dir / "loss_history.json")

    # Save checkpoint compatible with analysis
    model_kwargs = {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
    }
    torch.save(
        {
            "model_name": model_name,
            "model_state": model.state_dict(),
            "vocab": vocab,
            "context_length": context_length,
            "model_kwargs": model_kwargs,
        },
        run_dir / "checkpoint.pt",
    )

    # Save metrics summary
    metrics = {
        "run_id": run_id,
        "model": model_name,
        "train_loss_final": train_loss_history[-1] if train_loss_history else None,
        "val_loss_best": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "num_chars": len(text),
        "vocab_size": vocab.vocab_size,
    }
    save_json(metrics, run_dir / "metrics.json")

    # Also log run summary to logs/runs for quick overview
    run_info = {
        "run_id": run_id,
        "model": model_name,
        "config": {
            "context_length": context_length,
            "batch_size": batch_size,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": lr,
            "epochs": epochs,
        },
        "data": {
            "num_chars": len(text),
            "vocab_size": vocab.vocab_size,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
        },
        "metrics": {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    }
    log_run(run_info)

    # Run post-training analysis (including extended diagnostics)
    analysis_cfg = AnalysisConfig(
        run_id=run_id,
        generate_entropy_plots=True,
        generate_context_viz=True,
        generate_similarity_report=True,
        generate_transition_graph=True,
    )
    run_post_training_analysis(analysis_cfg)

    return model, vocab


def main():
    """Entry point: run training with config defaults."""
    setup_logging()
    train(model_name="mlp")


if __name__ == "__main__":
    main()
