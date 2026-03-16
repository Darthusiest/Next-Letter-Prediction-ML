"""
Entry point for running post-training analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

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


def run_post_training_analysis(config: AnalysisConfig) -> None:
    run_dir = ensure_run_dir(config.run_id)
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"

    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.json"
    loss_history_path = run_dir / "loss_history.json"
    train_corpus_path = run_dir / "train_corpus.txt"

    generated: List[Path] = []
    skipped: List[str] = []

    device = get_device()

    # Load metrics and loss history if present
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    loss_history = load_json(loss_history_path) if loss_history_path.exists() else {}

    # Load checkpoint and construct wrapper
    if not checkpoint_path.exists():
        skipped.append("No checkpoint.pt found; skipping model-based analyses.")
        model_wrapper = None
    else:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_name = ckpt.get("model_name", "mlp")
        vocab_obj = ckpt.get("vocab")
        if not isinstance(vocab_obj, CharVocab):
            vocab = CharVocab(vocab_obj)
        else:
            vocab = vocab_obj
        context_length = ckpt.get("context_length", CONTEXT_LENGTH)
        model = get_model(
            model_name,
            vocab_size=vocab.vocab_size,
            context_length=context_length,
        )
        state = ckpt.get("model_state")
        if state is not None:
            model.load_state_dict(state)
        model_wrapper = ModelAnalysisWrapper(
            model=model,
            vocab=vocab,
            device=device,
            model_type=model_name,
            context_length=context_length,
        )

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
        entropies = [e for _, e, _, _ in samples]
        hist_path = entropy_confidence.plot_entropy_hist(
            entropies, run_dir, show=config.show_plots
        )
        summary_path = entropy_confidence.write_confidence_summary(
            samples, reports_dir
        )
        generated.append(hist_path)
        generated.append(summary_path)
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

