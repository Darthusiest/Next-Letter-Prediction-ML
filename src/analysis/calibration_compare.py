"""
Compare next-char calibration (entropy, top-k) for best.pt vs last-epoch checkpoint.pt.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..models import get_model
from ..models.analysis_wrapper import ModelAnalysisWrapper
from ..vocab import CharVocab
from .run_analysis import _infer_model_kwargs_from_state


def _entropy(dist: Dict[str, float]) -> float:
    return float(
        -sum(p * math.log(p + 1e-12) for p in dist.values() if p > 0)
    )


def _topk(dist: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _wrapper_from_pt(path: Path, device: torch.device) -> ModelAnalysisWrapper | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None
    model_name = ckpt.get("model_name", "mlp")
    vocab_obj = ckpt.get("vocab")
    if vocab_obj is None:
        return None
    vocab = vocab_obj if isinstance(vocab_obj, CharVocab) else CharVocab(vocab_obj)
    context_length = ckpt.get("context_length", 64)
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
    model.load_state_dict(state)
    return ModelAnalysisWrapper(
        model=model,
        vocab=vocab,
        device=device,
        model_type=model_name,
        context_length=context_length,
    )


def write_checkpoint_comparison_report(
    run_dir: Path,
    device: torch.device,
    contexts: List[str] | None = None,
    top_k: int = 5,
) -> Path | None:
    """
    If both best.pt and checkpoint.pt exist, write reports/checkpoint_comparison.txt
    with entropy and top-k probs per context for each checkpoint.
    """
    contexts = contexts or ["pre", "th", "ing"]
    best_path = run_dir / "best.pt"
    last_path = run_dir / "checkpoint.pt"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "checkpoint_comparison.txt"

    w_best = _wrapper_from_pt(best_path, device)
    w_last = _wrapper_from_pt(last_path, device)
    if w_best is None or w_last is None:
        return None

    lines = [
        "Next-character calibration: best.pt (best val) vs checkpoint.pt (last epoch)",
        f"Contexts: {contexts!r}",
        "",
    ]
    for ctx in contexts:
        lines.append(f"=== context {ctx!r} ===")
        for label, w in [("best.pt", w_best), ("checkpoint.pt (last)", w_last)]:
            dist = w.predict_next_distribution(ctx)
            ent = _entropy(dist)
            top = _topk(dist, top_k)
            top_s = ", ".join(f"{c!r}:{p:.4f}" for c, p in top)
            lines.append(f"  {label}: entropy={ent:.4f}  top-{top_k}: {top_s}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
