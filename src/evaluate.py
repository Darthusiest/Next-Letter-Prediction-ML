"""
Evaluation: perplexity and accuracy (and optionally top-k) on a dataset.
"""

import torch
import torch.nn.functional as F


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    is_ngram: bool = False,
    max_batches: int | None = None,
) -> tuple:
    """
    Compute average cross-entropy loss (nll_loss for n-gram) and accuracy.
    Returns (loss, accuracy). Loss is per-token; perplexity = exp(loss).

    If *max_batches* is set, evaluation stops after that many batches
    (useful for fast mid-epoch validation on a subsample).
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_tokens = 0
    non_blocking = device.type in ("cuda", "mps")
    with torch.inference_mode():
        for batch_idx, (context, target) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            context = context.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            out = model(context)
            if is_ngram:
                loss = F.nll_loss(out, target, reduction="sum")
            else:
                loss = F.cross_entropy(out, target, reduction="sum")
            total_loss += loss
            total_correct += (out.argmax(dim=1) == target).sum()
            total_tokens += target.size(0)
    n = total_tokens
    avg_loss = total_loss.item() / n if n else 0.0
    accuracy = total_correct.item() / n if n else 0.0
    return avg_loss, accuracy


def perplexity(loss: float) -> float:
    """Perplexity = exp(loss)."""
    return float(__import__("math").exp(loss))
