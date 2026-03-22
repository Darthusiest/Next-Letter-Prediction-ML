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

import contextlib
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
    CLEAN_TEXT_DIR,
    MAX_CHARS,
    NUM_WORKERS,
    EMBED_DIM,
    HIDDEN_DIM,
    DROPOUT,
    NGRAM_ORDER,
    NGRAM_SMOOTHING,
    DEFAULT_ALLOWED_CHARS,
    SEED,
    RNN_NUM_LAYERS,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    USE_PLATEAU_LR,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
)
from src.preprocess import load_text, clean_text
from src.vocab import build_vocab_from_text, CharVocab
from src.dataset import get_splits, get_dataloaders
from src.models import get_model
from src.models.baseline_ngram import NGramModel
from src.evaluate import evaluate
from src.utils import set_seed, get_device, setup_logging, log_run
from src.utils.io_utils import ensure_run_dir, ensure_run_dir_path, save_json
from src.analysis.config import AnalysisConfig
from src.analysis.run_analysis import (
    _infer_model_kwargs_from_state,
    run_post_training_analysis,
)

logger = logging.getLogger(__name__)


def _amp_device_for_training(device: torch.device) -> str | None:
    """Return autocast device_type string if mixed precision helps on this device."""
    if device.type == "cuda":
        return "cuda"
    if device.type != "mps":
        return None
    v = torch.__version__.split("+")[0].split(".")
    try:
        major, minor = int(v[0]), int(v[1])
    except (ValueError, IndexError):
        return None
    if (major, minor) >= (2, 0):
        return "mps"
    return None


def _evaluate_from_neural_checkpoint(
    ckpt: dict,
    test_loader,
    device: torch.device,
) -> tuple[float, float]:
    """Load neural weights from a checkpoint dict and run test evaluate."""
    from src.models import get_model

    model_name = ckpt.get("model_name", "mlp")
    vocab_obj = ckpt["vocab"]
    vocab = vocab_obj if isinstance(vocab_obj, CharVocab) else CharVocab(vocab_obj)
    context_length = ckpt.get("context_length", CONTEXT_LENGTH)
    state = ckpt.get("model_state") or {}
    model_kwargs = ckpt.get("model_kwargs") or _infer_model_kwargs_from_state(
        model_name, state
    )
    eval_model = get_model(
        model_name,
        vocab_size=vocab.vocab_size,
        context_length=context_length,
        **model_kwargs,
    )
    eval_model.load_state_dict(state)
    eval_model.to(device)
    return evaluate(eval_model, test_loader, device, is_ngram=False)


def train(
    data_paths: list = None,
    model_name: str = "mlp",
    data_source: str = "raw",
    max_chars: int = None,
    context_length: int = CONTEXT_LENGTH,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    eval_every: int = EVAL_EVERY_N_STEPS,
    log_every: int = 200,
    early_stop_patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: Path = None,
    seed: int = SEED,
    num_workers: int | None = None,
    compile_model: bool = False,
    rnn_num_layers: int | None = None,
    weight_decay: float | None = None,
    label_smoothing: float | None = None,
    use_plateau_lr: bool | None = None,
):
    wd = WEIGHT_DECAY if weight_decay is None else weight_decay
    ls = LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    plateau = USE_PLATEAU_LR if use_plateau_lr is None else use_plateau_lr

    set_seed(seed)
    device = get_device()
    logger.info("Using device: %s", device)
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

    def _save_checkpoint(path: Path, payload: dict) -> None:
        ensure_run_dir_path(path.parent)
        torch.save(payload, path)

    if data_paths is None:
        if data_source == "cleaned":
            merged = Path(CLEAN_TEXT_DIR) / "merged_corpus.txt"
            if merged.exists():
                data_paths = [merged]
            else:
                data_paths = list(Path(CLEAN_TEXT_DIR).glob("*.txt"))
            if not data_paths:
                raise FileNotFoundError(
                    f"No cleaned .txt files found in {CLEAN_TEXT_DIR}. "
                    "Run: python tools/clean_corpus.py"
                )
        else:
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
    nw = NUM_WORKERS if num_workers is None else num_workers
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=nw
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
        _save_checkpoint(
            checkpoint_dir / "ngram_baseline.pt",
            {"model": model, "vocab": vocab, "config": {"model": "ngram"}},
        )
        return model, vocab
    else:
        # For neural models, delegate to get_model with the requested name.
        rnn_kw = {}
        if model_name == "rnn":
            rnn_kw["num_layers"] = (
                RNN_NUM_LAYERS if rnn_num_layers is None else rnn_num_layers
            )
        model = get_model(
            model_name,
            vocab_size=vocab.vocab_size,
            context_length=context_length,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            **rnn_kw,
        )
        model.to(device)
        if compile_model:
            try:
                model = torch.compile(model)  # type: ignore[assignment]
                logger.info("torch.compile enabled for model forward")
            except Exception as e:
                logger.warning("torch.compile skipped: %s", e)

        neural_model_kwargs: dict = {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
        }
        if model_name == "rnn":
            neural_model_kwargs["num_layers"] = model.lstm.num_layers

        def _best_pt_payload() -> dict:
            return {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "context_length": context_length,
                "model_name": model_name,
                "model_kwargs": dict(neural_model_kwargs),
            }

        try:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, foreach=True, weight_decay=wd
            )
        except TypeError:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None
        if plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=LR_PLATEAU_FACTOR,
                patience=LR_PLATEAU_PATIENCE,
            )
            logger.info("ReduceLROnPlateau enabled on epoch-end val loss")
        amp_device = _amp_device_for_training(device)
        use_amp = amp_device is not None
        use_scaler = amp_device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

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
            if use_amp and amp_device == "cuda":
                autocast_cm = torch.amp.autocast("cuda", enabled=True)
            elif use_amp and amp_device == "mps":
                autocast_cm = torch.amp.autocast("mps", dtype=torch.float16)
            else:
                autocast_cm = contextlib.nullcontext()
            with autocast_cm:
                logits = model(context)
                loss = F.cross_entropy(
                    logits, target, label_smoothing=float(ls) if ls > 0 else 0.0
                )
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            step += 1
            if log_every and step % log_every == 0:
                logger.info(
                    "Epoch %d step %d train_loss=%.4f",
                    epoch + 1,
                    step,
                    total_loss / num_batches,
                )
            if step % eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, is_ngram=False)
                logger.info(
                    "Epoch %d step %d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, step, total_loss / num_batches, val_loss, val_acc,
                )
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(
                        checkpoint_dir / "best.pt",
                        _best_pt_payload(),
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
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                _best_pt_payload(),
            )
        else:
            patience_counter += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        if patience_counter >= early_stop_patience:
            logger.info("Early stopping after %d epochs", epoch + 1)
            break

    # Test set: last-epoch weights vs best validation checkpoint
    test_loss_final_epoch, test_acc_final_epoch = evaluate(
        model, test_loader, device, is_ngram=False
    )
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        try:
            best_ckpt = torch.load(
                best_path, map_location=device, weights_only=False
            )
            test_loss, test_acc = _evaluate_from_neural_checkpoint(
                best_ckpt, test_loader, device
            )
            logger.info(
                "Test metrics use best.pt (best val). "
                "final_epoch test_loss=%.4f acc=%.4f | best.pt test_loss=%.4f acc=%.4f",
                test_loss_final_epoch,
                test_acc_final_epoch,
                test_loss,
                test_acc,
            )
        except Exception as e:
            logger.warning(
                "Could not evaluate best.pt; using final-epoch test metrics: %s", e
            )
            test_loss, test_acc = test_loss_final_epoch, test_acc_final_epoch
    else:
        logger.warning("best.pt missing; test metrics use final-epoch weights")
        test_loss, test_acc = test_loss_final_epoch, test_acc_final_epoch

    # Save loss history for analysis
    loss_history = {
        "epochs": epochs_list,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }
    save_json(loss_history, run_dir / "loss_history.json")

    # Last-epoch checkpoint (for debugging / calibration compare vs best.pt)
    _save_checkpoint(
        run_dir / "checkpoint.pt",
        {
            "model_name": model_name,
            "model_state": model.state_dict(),
            "vocab": vocab,
            "context_length": context_length,
            "model_kwargs": dict(neural_model_kwargs),
        },
    )

    # Save metrics summary (primary test_* = best validation weights)
    metrics = {
        "run_id": run_id,
        "model": model_name,
        "train_loss_final": train_loss_history[-1] if train_loss_history else None,
        "val_loss_best": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_loss_final_epoch": test_loss_final_epoch,
        "test_accuracy_final_epoch": test_acc_final_epoch,
        "metrics_checkpoint_policy": (
            "test_loss and test_accuracy are from best.pt when available; "
            "test_*_final_epoch are from last training step. "
            "checkpoint.pt stores last-epoch weights; analysis prefers best.pt first."
        ),
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
            "weight_decay": wd,
            "label_smoothing": ls,
            "use_plateau_lr": plateau,
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
            "test_loss_final_epoch": test_loss_final_epoch,
            "test_accuracy_final_epoch": test_acc_final_epoch,
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
    """Entry point: argparse CLI over `train()`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train character-level next-character models (ngram, mlp, rnn, cnn)."
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlp",
        choices=["ngram", "mlp", "rnn", "cnn"],
        help="Model architecture (default: mlp).",
    )
    parser.add_argument(
        "--data-source",
        default="raw",
        choices=["raw", "cleaned"],
        help='Corpus: "raw" from data/raw, or "cleaned" from data/processed/clean_texts.',
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate corpus to this many characters (default: config MAX_CHARS).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Validation every N train steps (default: EVAL_EVERY_N_STEPS in config).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on the model (can help RNN/CNN on some backends).",
    )
    parser.add_argument(
        "--rnn-layers",
        type=int,
        default=None,
        metavar="N",
        help="LSTM depth for --model rnn (default: RNN_NUM_LAYERS in config).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Adam weight decay (default: WEIGHT_DECAY in config).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Cross-entropy label smoothing, neural only (default: LABEL_SMOOTHING).",
    )
    parser.add_argument(
        "--plateau-lr",
        action="store_true",
        help="Enable ReduceLROnPlateau on epoch-end val loss (neural only).",
    )
    args = parser.parse_args()
    setup_logging()
    kw = {}
    if args.eval_every is not None:
        kw["eval_every"] = args.eval_every
    if args.compile:
        kw["compile_model"] = True
    if args.rnn_layers is not None:
        kw["rnn_num_layers"] = args.rnn_layers
    if args.weight_decay is not None:
        kw["weight_decay"] = args.weight_decay
    if args.label_smoothing is not None:
        kw["label_smoothing"] = args.label_smoothing
    if args.plateau_lr:
        kw["use_plateau_lr"] = True
    train(
        model_name=args.model,
        data_source=args.data_source,
        max_chars=args.max_chars,
        **kw,
    )


if __name__ == "__main__":
    main()
