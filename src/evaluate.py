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
) -> tuple:
    """
    Compute average cross-entropy loss (nll_loss for n-gram) and accuracy.
    Returns (loss, accuracy). Loss is per-token; perplexity = exp(loss).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for context, target in dataloader:
            context = context.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            out = model(context)
            if is_ngram:
                loss = F.nll_loss(out, target, reduction="sum")
            else:
                loss = F.cross_entropy(out, target, reduction="sum")
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_tokens += target.size(0)
    n = total_tokens
    avg_loss = total_loss / n if n else 0.0
    accuracy = total_correct / n if n else 0.0
    return avg_loss, accuracy


def perplexity(loss: float) -> float:
    """Perplexity = exp(loss)."""
    return float(__import__("math").exp(loss))
