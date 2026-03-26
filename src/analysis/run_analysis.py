"""
Entry point for running post-training analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)

from .config import AnalysisConfig
from . import (
    training_plots,
    data_plots,
    prediction_plots,
    confusion,
    embedding_viz,
    reporting,
    entropy_confidence,
    context_viz,
    similarity,
    transition_graph,
    attention_viz,
)
from ..utils.io_utils import ensure_run_dir, load_json
from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..vocab import CharVocab
from ..dataset import CharSequenceDataset, get_dataloaders
from ..config import CONTEXT_LENGTH, BATCH_SIZE
from ..models import get_model
from ..utils import get_device


def _infer_model_kwargs_from_state(model_name: str, state: dict) -> dict:
    """
    Best-effort inference of model hyperparameters from state_dict shapes.
    This is mainly to support older checkpoints that didn't store model kwargs.
    """
    if not state:
        return {}

    if model_name == "mlp":
        out = {}
        if "embed.weight" in state:
            out["embed_dim"] = int(state["embed.weight"].shape[1])
        if "proj.weight" in state:
            out["hidden_dim"] = int(state["proj.weight"].shape[0])
        elif "fc1.weight" in state:
            out["hidden_dim"] = int(state["fc1.weight"].shape[0])
        # New architecture: SwiGLU blocks
        if "blocks.0.ln.weight" in state:
            num_blocks = 0
            while f"blocks.{num_blocks}.ln.weight" in state:
                num_blocks += 1
            out["num_hidden_layers"] = 1 + num_blocks
        else:
            extra = 0
            while f"extra.{extra}.weight" in state:
                extra += 1
            out["num_hidden_layers"] = 1 + extra
        # Batched attention pooling (new) or ModuleList heads (legacy)
        if "attn_pool.weight" in state:
            out["num_attn_heads"] = int(state["attn_pool.weight"].shape[0])
        elif "attn_heads.0.weight" in state:
            num_heads = 0
            while f"attn_heads.{num_heads}.weight" in state:
                num_heads += 1
            out["num_attn_heads"] = num_heads
        # Stacked self-attention layers (RoPE) or legacy single self_attn
        if "self_attn_layers.0.qkv.weight" in state:
            n = 0
            while f"self_attn_layers.{n}.qkv.weight" in state:
                n += 1
            out["num_self_attn_layers"] = n
        elif "self_attn.qkv.weight" in state:
            out["num_self_attn_layers"] = 1
        return out

    if model_name == "rnn":
        out = {}
        if "embed.weight" in state:
            out["embed_dim"] = int(state["embed.weight"].shape[1])
        # LSTM weight_hh_l0 is (4H, H)
        if "lstm.weight_hh_l0" in state:
            out["hidden_dim"] = int(state["lstm.weight_hh_l0"].shape[1])
        # Count layers by checking weight_ih_l{k}
        num_layers = 0
        while f"lstm.weight_ih_l{num_layers}" in state:
            num_layers += 1
        if num_layers:
            out["num_layers"] = num_layers
        return out

    if model_name == "cnn":
        out = {}
        if "embed.weight" in state:
            out["embed_dim"] = int(state["embed.weight"].shape[1])
        # convs.<i>.weight has shape (C_out, C_in, K)
        kernel_sizes = []
        num_channels = None
        i = 0
        while f"convs.{i}.weight" in state:
            w = state[f"convs.{i}.weight"]
            if num_channels is None:
                num_channels = int(w.shape[0])
            kernel_sizes.append(int(w.shape[2]))
            i += 1
        if num_channels is not None:
            out["num_channels"] = num_channels
        if kernel_sizes:
            out["kernel_sizes"] = tuple(kernel_sizes)
        return out

    return {}


def run_post_training_analysis(config: AnalysisConfig) -> None:
    run_dir = ensure_run_dir(config.run_id)
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"

    checkpoint_path = run_dir / "checkpoint.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"
    loss_history_path = run_dir / "loss_history.json"
    train_corpus_path = run_dir / "train_corpus.txt"

    generated: List[Path] = []
    skipped: List[str] = []

    device = get_device()

    # Load metrics and loss history if present
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    loss_history = load_json(loss_history_path) if loss_history_path.exists() else {}

    # Load checkpoint and construct wrapper.
    #
    # Prefer best.pt (best validation weights) for plots/metrics alignment;
    # fall back to checkpoint.pt (last epoch, full metadata).
    checkpoint_candidates = [best_path, checkpoint_path]
    ckpt = None
    checkpoint_loaded_from: Path | None = None
    load_errors: list[str] = []
    for candidate in checkpoint_candidates:
        if not candidate.exists():
            continue
        try:
            # Checkpoint was saved by this project and is trusted, so we allow
            # loading full pickled objects (weights_only=False). This avoids
            # PyTorch 2.6's stricter default which blocks custom classes.
            ckpt = torch.load(candidate, map_location=device, weights_only=False)
            checkpoint_loaded_from = candidate
            break
        except Exception as e:
            load_errors.append(f"{candidate.name}: {repr(e)}")

    if ckpt is None:
        skipped.append(
            "No valid checkpoint/best checkpoint found; skipping model-based analyses."
            + (f" Load errors: {load_errors}" if load_errors else "")
        )
        model_wrapper = None
    else:
        model_name = ckpt.get("model_name", "mlp")
        tok_type = ckpt.get("tokenizer_type", "char")
        if tok_type == "bpe":
            from ..tokenizer import BPETokenizer
            tok_path = ckpt.get("tokenizer_path")
            if tok_path:
                vocab = BPETokenizer.load(tok_path)
            else:
                skipped.append("BPE checkpoint missing tokenizer_path; skipping.")
                model_wrapper = None
                vocab = None
        else:
            vocab_obj = ckpt.get("vocab")
            if not isinstance(vocab_obj, CharVocab):
                vocab = CharVocab(vocab_obj)
            else:
                vocab = vocab_obj
        context_length = ckpt.get("context_length", CONTEXT_LENGTH)
        state = ckpt.get("model_state") or {}
        model_kwargs = ckpt.get("model_kwargs") or _infer_model_kwargs_from_state(
            model_name, state
        )
        model = get_model(
            model_name,
            vocab_size=vocab.vocab_size,
            context_length=context_length,
            **model_kwargs,
        )
        if state is not None:
            model.load_state_dict(state)
        model_wrapper = ModelAnalysisWrapper(
            model=model,
            vocab=vocab,
            device=device,
            model_type=model_name,
            context_length=context_length,
        )

    from .calibration_compare import write_checkpoint_comparison_report

    cmp_report = write_checkpoint_comparison_report(run_dir, device)
    if cmp_report is not None:
        generated.append(cmp_report)
        logger.info("Wrote checkpoint comparison report to %s", cmp_report)

    # 1) Training plots
    if config.generate_training_plots and loss_history:
        training_plots.plot_loss_curves(loss_history, run_dir, show=config.show_plots)
        generated.append(plots_dir / "training_loss.png")
        generated.append(plots_dir / "validation_loss.png")

    # 2) Data plots (char freq, bigrams)
    if config.generate_data_plots:
        if train_corpus_path.exists() and model_wrapper is not None:
            text = train_corpus_path.read_text(encoding="utf-8")
            data_plots.plot_char_frequency(
                text, model_wrapper.vocab, run_dir, show=config.show_plots
            )
            data_plots.plot_bigram_heatmap(
                text, model_wrapper.vocab, run_dir, show=config.show_plots
            )
            generated.append(plots_dir / "char_frequency.png")
            generated.append(plots_dir / "bigram_heatmap.png")
        else:
            skipped.append(
                "train_corpus.txt missing or model unavailable; skipping data_plots."
            )

    # 3) Prediction probability plots
    if config.generate_prediction_plots and model_wrapper is not None:
        prediction_plots.plot_prediction_bars(
            model_wrapper,
            config.sample_contexts,
            config.top_k,
            run_dir,
            show=config.show_plots,
        )
        # We don't enumerate all filenames here; folder is enough.
        generated.append(plots_dir / "predictions")
    elif config.generate_prediction_plots:
        skipped.append("Model unavailable; skipping prediction_plots.")

    # 4) Confusion matrix + error-by-character (requires test dataset)
    if (
        config.generate_confusion_plots
        and model_wrapper is not None
        and train_corpus_path.exists()
    ):
        # For v1, reuse the same corpus to build a simple dataset; a more
        # precise version would reload the actual train/val/test splits.
        text = train_corpus_path.read_text(encoding="utf-8")
        ds = CharSequenceDataset(text, model_wrapper.vocab, model_wrapper.context_length)
        _, _, test_loader = get_dataloaders(ds, ds, ds, batch_size=BATCH_SIZE)
        mat, per_id = confusion.compute_confusion(
            model_wrapper, test_loader, model_wrapper.vocab
        )
        confusion.plot_confusion_matrix(
            mat, model_wrapper.vocab, run_dir, show=config.show_plots
        )
        confusion.plot_error_by_character(
            per_id, model_wrapper.vocab, run_dir, show=config.show_plots
        )
        generated.append(plots_dir / "confusion_matrix.png")
        generated.append(plots_dir / "error_by_character.png")
    elif config.generate_confusion_plots:
        skipped.append("Model or corpus unavailable; skipping confusion plots.")

    # 5) Embedding visualizations
    if config.generate_embedding_plots and model_wrapper is not None:
        embeddings = model_wrapper.get_char_embeddings()
        if embeddings is None:
            skipped.append("Model has no embeddings; skipping embedding_viz.")
        else:
            embedding_viz.plot_char_embeddings(
                embeddings, run_dir, show=config.show_plots
            )
            generated.append(plots_dir / "embeddings_pca_2d.png")
            generated.append(plots_dir / "embeddings_tsne_2d.png")
    elif config.generate_embedding_plots:
        skipped.append("Model unavailable; skipping embedding_viz.")

    # 6) Entropy / confidence analysis
    if (
        config.generate_entropy_plots
        and model_wrapper is not None
        and train_corpus_path.exists()
    ):
        text = train_corpus_path.read_text(encoding="utf-8")
        ds = CharSequenceDataset(text, model_wrapper.vocab, model_wrapper.context_length)
        _, _, loader = get_dataloaders(ds, ds, ds, batch_size=BATCH_SIZE)
        samples = entropy_confidence.compute_entropy_distribution(
            model_wrapper, loader, max_samples=config.max_entropy_samples
        )
        entropies = [s.entropy for s in samples]
        hist_path = entropy_confidence.plot_entropy_hist(
            entropies, run_dir, show=config.show_plots
        )
        summary_path = entropy_confidence.write_confidence_summary(
            samples, reports_dir
        )
        cat_entropy_path, cat_error_path = entropy_confidence.plot_entropy_and_error_by_category(
            samples, run_dir, show=config.show_plots
        )
        residual_noise_path = entropy_confidence.write_residual_noise_report(
            samples, reports_dir
        )
        generated.append(hist_path)
        generated.append(summary_path)
        generated.append(cat_entropy_path)
        generated.append(cat_error_path)
        generated.append(residual_noise_path)
    elif config.generate_entropy_plots:
        skipped.append("Model or corpus unavailable; skipping entropy/confidence plots.")

    # 7) Context representation similarity
    if config.generate_context_viz and model_wrapper is not None:
        vectors = context_viz.get_context_vectors(
            model_wrapper, config.context_viz_contexts
        )
        if not vectors:
            skipped.append(
                "No context representations available; skipping context_viz."
            )
        else:
            heat_path = context_viz.plot_context_similarity_heatmap(
                vectors, run_dir, show=config.show_plots
            )
            pca_path = context_viz.plot_context_pca(
                vectors,
                run_dir,
                enable_3d=config.enable_3d,
                show=config.show_plots,
            )
            generated.append(heat_path)
            generated.append(pca_path)
    elif config.generate_context_viz:
        skipped.append("Model unavailable; skipping context_viz.")

    # 8) Similarity report (characters + contexts)
    if config.generate_similarity_report and model_wrapper is not None:
        sim_path = similarity.run_similarity_report(
            model_wrapper,
            chars=config.similarity_chars,
            contexts=config.similarity_contexts,
            reports_dir=reports_dir,
        )
        if sim_path is None:
            skipped.append(
                "No embeddings or context representations; skipping similarity report."
            )
        else:
            generated.append(sim_path)
    elif config.generate_similarity_report:
        skipped.append("Model unavailable; skipping similarity report.")

    # 9) Letter transition graph
    if (
        config.generate_transition_graph
        and train_corpus_path.exists()
        and model_wrapper is not None
    ):
        text = train_corpus_path.read_text(encoding="utf-8")
        tg_path = transition_graph.run_transition_graph(
            text,
            model_wrapper.vocab,
            run_dir,
            show=config.show_plots,
        )
        generated.append(tg_path)
    elif config.generate_transition_graph:
        skipped.append("Model or corpus unavailable; skipping transition graph.")

    # 10) Transformer attention heatmaps (optional)
    if config.enable_attention_plots and model_wrapper is not None:
        attn_paths = attention_viz.run_attention_viz(
            model_wrapper,
            contexts=config.attention_contexts,
            output_dir=run_dir,
            show=config.show_plots,
        )
        if attn_paths is None:
            skipped.append(
                "Model does not expose attention weights; skipping attention_viz."
            )
        else:
            generated.extend(attn_paths)

    # 11) Summary report
    reporting.write_summary_report(
        metrics,
        generated_files=generated,
        skipped_reasons=skipped,
        reports_dir=reports_dir,
    )

    print(f"Post-training analysis summary (run_id={config.run_id}):")
    print(f"  Plots dir: {plots_dir}")
    print(f"  Reports dir: {reports_dir}")
    if skipped:
        print("  Skipped:")
        for reason in skipped:
            print(f"    - {reason}")

