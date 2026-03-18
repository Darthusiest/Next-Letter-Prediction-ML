"""
Sliding-window dataset for next-character prediction.
Each example is (context of length L, target next character).
Train/val/test split is contiguous to avoid leakage.
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import CharVocab
from .config import (
    CONTEXT_LENGTH,
    BATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    NUM_WORKERS,
)


class CharSequenceDataset(Dataset):
    """
    Dataset of (context, target) for next-character prediction.
    context: tensor of shape (L,) of character ids
    target: int (id of next character)
    """

    def __init__(self, text: str, vocab: CharVocab, context_length: int = CONTEXT_LENGTH):
        self.vocab = vocab
        self.context_length = context_length
        # Store ids in two forms:
        # - Python list for compatibility with n-gram baseline and misc usage
        # - Torch tensor for fast slicing in __getitem__ (avoid per-sample allocation)
        self.ids = vocab.encode(text)
        self.ids_tensor = torch.tensor(self.ids, dtype=torch.long)
        # Valid indices: we need context_length chars before, so start at context_length
        self.valid_length = len(self.ids) - context_length
        if self.valid_length <= 0:
            raise ValueError(
                f"Text too short: {len(self.ids)} chars, need at least {context_length + 1}"
            )

    def __len__(self) -> int:
        return self.valid_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        start = idx
        end = start + self.context_length
        context = self.ids_tensor[start:end]
        target = int(self.ids_tensor[end].item())
        return context, target


def get_splits(
    text: str,
    vocab: CharVocab,
    context_length: int = CONTEXT_LENGTH,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[CharSequenceDataset, CharSequenceDataset, CharSequenceDataset]:
    """
    Split text contiguously into train / val / test, then create datasets.
    No shuffling of positions so adjacent characters stay in same split (no leakage).
    """
    n = len(text)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train_text = text[:t1]
    val_text = text[t1:t2]
    test_text = text[t2:]
    train_ds = CharSequenceDataset(train_text, vocab, context_length)
    val_ds = CharSequenceDataset(val_text, vocab, context_length)
    test_ds = CharSequenceDataset(test_text, vocab, context_length)
    return train_ds, val_ds, test_ds


def get_dataloaders(
    train_ds: CharSequenceDataset,
    val_ds: CharSequenceDataset,
    test_ds: CharSequenceDataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for train, val, test. Shuffle only train."""
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader
