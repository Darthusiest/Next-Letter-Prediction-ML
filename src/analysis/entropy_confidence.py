"""
Entropy and confidence analysis for next-character prediction.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..utils.plotting import display_char, new_figure, save_figure


@dataclass(frozen=True)
class EntropyConfidenceSample:
    context: str
    entropy: float
    top_char: str
    top_prob: float
    target_char: str
    pred_char: str
    correct: bool
    category: str


def compute_entropy_distribution(
    model_wrapper: ModelAnalysisWrapper,
    dataloader: torch.utils.data.DataLoader,
    max_samples: int,
) -> List[EntropyConfidenceSample]:
    """
    Sample contexts from dataloader and compute entropy of next-char distribution.
    Returns list of samples with entropy + top prediction + correctness.
    """
    device = model_wrapper.device
    vocab = model_wrapper.vocab
    results: List[EntropyConfidenceSample] = []

    model = model_wrapper.model
    model.eval()
    with torch.no_grad():
        for context_ids, target_ids in dataloader:
            context_ids = context_ids.to(device)
            target_ids = target_ids.to(device)
            # use raw ids to build context strings (keeps whitespace alignment)
            for row, target_id in zip(context_ids, target_ids):
                if len(results) >= max_samples:
                    return results
                ids = row.tolist()
                chars = [vocab.id2char[i] for i in ids]
                ctx_str = "".join(chars)
                dist = model_wrapper.predict_next_distribution(ctx_str)
                probs = np.array(list(dist.values()), dtype=float)
                # avoid log(0)
                mask = probs > 0
                entropy = float(-np.sum(probs[mask] * np.log(probs[mask])))
                # most likely char
                top_char, top_prob = max(dist.items(), key=lambda kv: kv[1])
                target_char = vocab.id2char[int(target_id.item())]
                pred_char = top_char
                correct = pred_char == target_char
                category = _classify_context_heuristic(ctx_str)
                results.append(
                    EntropyConfidenceSample(
                        context=ctx_str,
                        entropy=entropy,
                        top_char=top_char,
                        top_prob=top_prob,
                        target_char=target_char,
                        pred_char=pred_char,
                        correct=correct,
                        category=category,
                    )
                )
    return results


def plot_entropy_hist(entropies: Iterable[float], output_dir: Path, show: bool = False) -> Path:
    entropies = list(entropies)
    if not entropies:
        return output_dir / "plots" / "entropy_histogram.png"
    plots_dir = output_dir / "plots"
    fig, ax = new_figure()
    ax.hist(entropies, bins=30)
    ax.set_xlabel("Entropy of next-character distribution")
    ax.set_ylabel("Count")
    ax.set_title("Entropy distribution over sampled contexts")
    out_path = plots_dir / "entropy_histogram.png"
    save_figure(fig, out_path, show=show)
    return out_path


def _classify_context_heuristic(ctx: str) -> str:
    """
    Light-weight heuristics to label sampled contexts by “non-prose-ness”.
    Goal: validate the hypothesis that uncertainty clusters on formatting/OCR artifacts.

    Note: this is intentionally simple and not a training signal—analysis-only.
    """
    if not ctx:
        return "empty"

    lower = ctx.lower()
    boilerplate_keywords = [
        "copyright",
        "all rights reserved",
        "printed in",
        "page",
        "chapter",
        "contents",
        "isbn",
        "illustrations",
        "introduction",
        "publisher",
        "edition",
        "www.",
        "http://",
        "https://",
    ]
    if any(k in lower for k in boilerplate_keywords):
        return "boilerplate_keywords"

    L = len(ctx)
    digits = sum(ch.isdigit() for ch in ctx)
    letters = sum(ch.isalpha() for ch in ctx)
    whitespace = sum(ch in {" ", "\n", "\t"} for ch in ctx)
    punctuation = sum((not ch.isalnum()) and (ch not in {" ", "\n", "\t"}) for ch in ctx)

    digit_ratio = digits / L
    whitespace_ratio = whitespace / L
    punct_ratio = punctuation / L

    if digit_ratio >= 0.15:
        return "digits"

    # “Prose formatting” patterns: common after sentences / clauses
    if ctx.endswith(" "):
        if len(ctx) >= 2 and ctx[-2] in {".", "!", "?", ";", ":"}:
            return "post_punct_space"
        if whitespace_ratio >= 0.25:
            return "whitespace_dense"
    if ctx.endswith("\n"):
        if whitespace_ratio >= 0.25:
            return "newline_whitespace_dense"

    if punct_ratio >= 0.2:
        return "punctuation_dense"

    # When few special chars exist, treat as “letters/prose”
    if letters / L >= 0.5:
        return "letters_prose"

    return "other_nonprose"


def write_confidence_summary(
    samples: List[EntropyConfidenceSample],
    reports_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Write confidence_summary.txt describing most/least confident contexts.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "confidence_summary.txt"

    if not samples:
        path.write_text("No samples available for entropy/confidence analysis.\n", encoding="utf-8")
        return path

    # sort by entropy
    sorted_by_entropy = sorted(samples, key=lambda s: s.entropy)
    most_confident = sorted_by_entropy[:top_n]
    least_confident = list(reversed(sorted_by_entropy))[:top_n]

    with path.open("w", encoding="utf-8") as f:
        f.write("Entropy / confidence summary for next-character prediction\n")
        f.write("=========================================================\n\n")

        f.write(f"Total sampled contexts: {len(samples)}\n\n")

        f.write("Most confident contexts (lowest entropy):\n")
        for s in most_confident:
            f.write(
                "  "
                f"ctx={repr(s.context)}  "
                f"entropy={s.entropy:.4f}  "
                f"target='{s.target_char}'  pred='{s.pred_char}'  correct={s.correct}  "
                f"top='{s.top_char}'  P(top)={s.top_prob:.4f}  "
                f"category={s.category}\n"
            )

        f.write("\nLeast confident contexts (highest entropy):\n")
        for s in least_confident:
            f.write(
                "  "
                f"ctx={repr(s.context)}  "
                f"entropy={s.entropy:.4f}  "
                f"target='{s.target_char}'  pred='{s.pred_char}'  correct={s.correct}  "
                f"top='{s.top_char}'  P(top)={s.top_prob:.4f}  "
                f"category={s.category}\n"
            )

    return path


def plot_entropy_and_error_by_category(
    samples: List[EntropyConfidenceSample],
    output_dir: Path,
    show: bool = False,
) -> tuple[Path, Path]:
    """
    Plot average entropy and error-rate grouped by heuristic context category.
    This directly implements the inference: quantify whether uncertainty/error clusters
    on formatting/whitespace/boilerplate-like contexts.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_cat: DefaultDict[str, List[EntropyConfidenceSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    if not by_cat:
        out_entropy = plots_dir / "entropy_by_context_category.png"
        out_error = plots_dir / "error_by_context_category.png"
        return out_entropy, out_error

    cats = sorted(by_cat.keys(), key=lambda c: np.mean([x.entropy for x in by_cat[c]]), reverse=True)
    avg_entropies = [float(np.mean([x.entropy for x in by_cat[c]])) for c in cats]
    error_rates = [float(1.0 - np.mean([1.0 if x.correct else 0.0 for x in by_cat[c]])) for c in cats]

    # --- entropy by category
    fig, ax = new_figure(figsize=(11, 4.5))
    ax.bar(range(len(cats)), avg_entropies)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Average entropy")
    ax.set_title("Uncertainty (entropy) by heuristic context category")
    out_entropy = plots_dir / "entropy_by_context_category.png"
    save_figure(fig, out_entropy, show=show)

    # --- error by category
    fig, ax = new_figure(figsize=(11, 4.5))
    ax.bar(range(len(cats)), error_rates)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Error rate (1 - accuracy)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Error rate by heuristic context category")
    out_error = plots_dir / "error_by_context_category.png"
    save_figure(fig, out_error, show=show)

    return out_entropy, out_error


def _format_context_for_report(ctx: str, max_len_chars: int = 64) -> str:
    """
    Human-readable context for debugging whitespace/punctuation artifacts.
    """
    tail = ctx[-max_len_chars:]
    return "".join(display_char(ch) for ch in tail)


def write_residual_noise_report(
    samples: List[EntropyConfidenceSample],
    reports_dir: Path,
    top_n_per_category: int = 8,
) -> Path:
    """
    Write a text report showing which categories contribute most to high entropy / errors.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "residual_noise_diagnostics.txt"

    if not samples:
        path.write_text("No samples available.\n", encoding="utf-8")
        return path

    by_cat: DefaultDict[str, List[EntropyConfidenceSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    ranked_cats = sorted(
        by_cat.keys(),
        key=lambda c: np.mean([x.entropy for x in by_cat[c]]),
        reverse=True,
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("Residual noise diagnostics (analysis-only heuristics)\n")
        f.write("=====================================================\n\n")
        f.write(f"Total sampled contexts: {len(samples)}\n\n")

        f.write("Category summary (heuristic):\n")
        for cat in ranked_cats:
            xs = by_cat[cat]
            avg_entropy = float(np.mean([x.entropy for x in xs]))
            err_rate = float(1.0 - np.mean([1.0 if x.correct else 0.0 for x in xs]))
            acc = 1.0 - err_rate
            f.write(f"  {cat}: n={len(xs)}  avg_entropy={avg_entropy:.4f}  acc={acc:.3f}  err={err_rate:.3f}\n")
        f.write("\n")

        f.write("Top least-confident examples per category (highest entropy):\n")
        for cat in ranked_cats:
            xs = sorted(by_cat[cat], key=lambda x: x.entropy, reverse=True)[:top_n_per_category]
            f.write(f"\n[{cat}]\n")
            for s in xs:
                f.write(
                    "  "
                    f"entropy={s.entropy:.4f}  "
                    f"target='{s.target_char}'  pred='{s.pred_char}'  correct={s.correct}  "
                    f"top='{s.top_char}'  P(top)={s.top_prob:.4f}\n"
                )
                f.write(f"    ctx_tail={repr(s.context)}\n")
                f.write(f"    ctx_disp={_format_context_for_report(s.context)}\n")

    return path


