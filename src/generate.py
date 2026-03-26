"""
Sample text from the model given a seed string.
Autoregressive: repeatedly predict next character with temperature, append, repeat.
Supports KV-cached incremental decoding for MLPCharModel.
"""

import torch
import torch.nn.functional as F

from .vocab import CharVocab
from .config import DEFAULT_TEMPERATURE, DEFAULT_GENERATION_LENGTH, CONTEXT_LENGTH


def generate(
    model: torch.nn.Module,
    vocab: CharVocab,
    seed: str,
    length: int = DEFAULT_GENERATION_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE,
    context_length: int = CONTEXT_LENGTH,
    device: torch.device = None,
) -> str:
    """
    Generate `length` new characters after the seed.
    Seed is padded or truncated to context_length for the first step.
    Uses KV caching when the model supports it (MLPCharModel with self_attn_layers).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if len(seed) == 0:
        seed = " "  # avoid empty context
    # For char vocab, filter to known chars; BPE handles arbitrary input
    if hasattr(vocab, "char2id") and isinstance(vocab.char2id, dict):
        seed = "".join(c for c in seed if c in vocab.char2id)
    ids = vocab.encode(seed)
    # Pad or truncate token sequence to context_length
    if len(ids) > context_length:
        ids = ids[-context_length:]
    elif len(ids) < context_length:
        pad_id = vocab.encode(" ")[0] if vocab.encode(" ") else 0
        ids = [pad_id] * (context_length - len(ids)) + ids
    generated = list(ids)
    is_ngram = hasattr(model, "_counts")
    supports_cache = hasattr(model, "self_attn_layers") and not is_ngram

    with torch.no_grad():
        if supports_cache:
            context = torch.tensor(
                [generated[-context_length:]], dtype=torch.long, device=device
            )
            logits, kv_cache = model(context, use_cache=True)
            probs = F.softmax(logits.squeeze(0) / temperature, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)

            for _ in range(length - 1):
                token = torch.tensor(
                    [[next_id]], dtype=torch.long, device=device
                )
                logits, kv_cache = model(token, kv_cache=kv_cache, use_cache=True)
                probs = F.softmax(logits.squeeze(0) / temperature, dim=0)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(next_id)
        else:
            for _ in range(length):
                context_ids = generated[-context_length:]
                context = torch.tensor(
                    [context_ids], dtype=torch.long, device=device
                )
                out = model(context)
                if is_ngram:
                    probs = torch.exp(out).squeeze(0)
                else:
                    logits = out.squeeze(0)
                    probs = F.softmax(logits / temperature, dim=0)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(next_id)

    # Return only the newly generated part (after seed)
    new_ids = generated[len(ids):]
    return vocab.decode(new_ids)
