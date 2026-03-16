"""
Analysis tools for next-letter prediction.

Functions here are linguistically informed: they measure performance on vowels
vs consonants, confusion patterns, and position relative to word boundaries.
They can be used with any trained model and dataloader.
"""

from collections import Counter, defaultdict
from typing import Dict, Tuple

import torch

from .vocab import CharVocab


VOWELS = set("aeiouAEIOU")


def per_char_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    vocab: CharVocab,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute accuracy for each character in the vocabulary on a given dataloader.
    """
    model.eval()
    total = Counter()
    correct = Counter()
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            logits = model(context)
            pred = logits.argmax(dim=1)
            for t, p in zip(target.tolist(), pred.tolist()):
                ch = vocab.id2char[t]
                total[ch] += 1
                if t == p:
                    correct[ch] += 1
    acc = {
        ch: (correct[ch] / total[ch]) if total[ch] > 0 else 0.0
        for ch in vocab.char2id.keys()
    }
    return acc


def vowel_consonant_summary(per_char_acc: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregate per-character accuracies into vowels, consonants, spaces, punctuation.
    """
    vowel_vals = []
    consonant_vals = []
    space_vals = []
    punct_vals = []
    for ch, a in per_char_acc.items():
        if ch in VOWELS:
            vowel_vals.append(a)
        elif ch.isalpha():
            consonant_vals.append(a)
        elif ch.isspace():
            space_vals.append(a)
        else:
            punct_vals.append(a)
    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    return {
        "vowel_acc": avg(vowel_vals),
        "consonant_acc": avg(consonant_vals),
        "space_acc": avg(space_vals),
        "punct_acc": avg(punct_vals),
    }


def confusion_counts(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    vocab: CharVocab,
    device: torch.device,
) -> Dict[Tuple[str, str], int]:
    """
    Return raw confusion counts (true_char, pred_char) over a dataloader.
    """
    model.eval()
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)
            logits = model(context)
            pred = logits.argmax(dim=1)
            for t, p in zip(target.tolist(), pred.tolist()):
                true_ch = vocab.id2char[t]
                pred_ch = vocab.id2char[p]
                counts[(true_ch, pred_ch)] += 1
    return counts


def confusion_matrix_normalized(
    counts: Dict[Tuple[str, str], int]
) -> Dict[Tuple[str, str], float]:
    """
    Normalize confusion counts to probabilities per true character.
    """
    row_totals: Dict[str, int] = defaultdict(int)
    for (true_ch, _), c in counts.items():
        row_totals[true_ch] += c
    probs: Dict[Tuple[str, str], float] = {}
    for (true_ch, pred_ch), c in counts.items():
        if row_totals[true_ch] > 0:
            probs[(true_ch, pred_ch)] = c / row_totals[true_ch]
    return probs

