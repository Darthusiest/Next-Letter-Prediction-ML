"""Tests for n-gram, MLP, CNN, and RNN models."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline_ngram import NGramModel
import pytest

from src.models.mlp import MLPCharModel, RotaryPositionEmbedding, SwiGLUBlock, MultiHeadSelfAttention
from src.models.cnn import CNNCharModel
from src.models.rnn import RNNCharModel
from src.tokenizer import BPETokenizer


def test_ngram_fit_forward():
    model = NGramModel(order=3, vocab_size=5, smoothing=0.01)
    ids = [0, 1, 2, 1, 2, 3, 2, 3, 4]
    model.fit(ids)
    ctx = torch.tensor([[1, 2]], dtype=torch.long)
    out = model(ctx)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(1), atol=0.01)


def test_mlp_forward():
    model = MLPCharModel(
        vocab_size=10,
        context_length=8,
        embed_dim=4,
        hidden_dim=16,
        dropout=0.0,
        num_hidden_layers=1,
        num_attn_heads=2,
        num_self_attn_layers=1,
    )
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mlp_multilayer_forward():
    model = MLPCharModel(
        vocab_size=10,
        context_length=8,
        embed_dim=4,
        hidden_dim=16,
        dropout=0.0,
        num_hidden_layers=3,
        num_attn_heads=2,
        num_self_attn_layers=2,
    )
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
    assert len(model.blocks) == 2


def test_mlp_embed_dropout():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.5, num_hidden_layers=1, num_attn_heads=2,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.train()
    out_a = model(x)
    out_b = model(x)
    assert not torch.equal(out_a, out_b), "dropout should cause stochastic outputs in train mode"
    model.eval()
    out_c = model(x)
    out_d = model(x)
    assert torch.equal(out_c, out_d), "eval mode should be deterministic"


def test_mlp_rope():
    """Verify RoPE replaces pos_embed and is applied in self-attention."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    assert hasattr(model, "rope"), "model should have RoPE"
    assert isinstance(model.rope, RotaryPositionEmbedding)
    assert not hasattr(model, "pos_embed"), "pos_embed should be removed"
    # RoPE should have precomputed sin/cos tables
    assert hasattr(model.rope, "cos_cached")
    assert hasattr(model.rope, "sin_cached")
    x = torch.randint(0, 10, (2, 8))
    model.eval()
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mlp_multihead_attention_pooling():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    assert hasattr(model, "attn_pool")
    assert model.attn_pool.out_features == 2
    x = torch.randint(0, 10, (2, 8))
    model.eval()
    with torch.no_grad():
        emb = model.embed(x)
        for layer in model.self_attn_layers:
            emb, _ = layer(emb, rope=model.rope)
        scores = model.attn_pool(emb)
        weights = torch.softmax(scores, dim=1)
        assert weights.shape == (2, 8, 2)
        for n in range(2):
            sums = weights[:, :, n].sum(dim=1)
            assert torch.allclose(sums, torch.ones(2), atol=1e-5), \
                "attention weights must sum to 1 per head"


def test_mlp_self_attention():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=2,
    )
    assert hasattr(model, "self_attn_layers")
    assert len(model.self_attn_layers) == 2
    assert model.self_attn_layers[0].num_heads == 2
    assert model.self_attn_layers[0].head_dim == 2
    x = torch.randint(0, 10, (2, 8))
    model.eval()
    with torch.no_grad():
        emb = model.embed(x)
        for layer in model.self_attn_layers:
            emb, _ = layer(emb, rope=model.rope)
        assert emb.shape == (2, 8, 4), "self-attention should preserve shape"
        assert torch.isfinite(emb).all()


def test_mlp_swiglu_blocks():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=3, num_attn_heads=2,
    )
    assert len(model.blocks) == 2
    for block in model.blocks:
        assert hasattr(block, "w_gate")
        assert hasattr(block, "w_up")
        assert hasattr(block, "ln")
    x = torch.randint(0, 10, (2, 8))
    model.eval()
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mlp_weight_tying():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    assert model.fc2.weight is model.embed.weight, \
        "fc2.weight should be tied to embed.weight"
    assert model.tie_proj is not None, \
        "tie_proj needed when hidden_dim != embed_dim"

def test_mlp_weight_tying_same_dim():
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=16, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=4,
    )
    assert model.fc2.weight is model.embed.weight
    assert model.tie_proj is None, \
        "no tie_proj needed when hidden_dim == embed_dim"

def test_mlp_pre_layernorm():
    """Verify residual blocks use pre-LN (LN on input, not output)."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=3, num_attn_heads=2,
    )
    assert hasattr(model, "final_ln"), "should have a final LayerNorm"
    for block in model.blocks:
        assert hasattr(block, "ln"), "each block should have pre-LN"


def test_cnn_embed_dropout():
    model = CNNCharModel(
        vocab_size=10, context_length=8, embed_dim=4, num_channels=8,
        kernel_sizes=(3,), dropout=0.5,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.eval()
    out_a = model(x)
    out_b = model(x)
    assert torch.equal(out_a, out_b)


def test_rnn_embed_dropout():
    model = RNNCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        num_layers=1, dropout=0.5,
    )
    assert hasattr(model, "embed_drop")
    x = torch.randint(0, 10, (4, 8))
    model.eval()
    out_a = model(x)
    out_b = model(x)
    assert torch.equal(out_a, out_b)


def test_bpe_tokenizer_train_encode_decode():
    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = BPETokenizer.train(text, vocab_size=100)
    assert tok.vocab_size <= 100
    ids = tok.encode("the quick")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert "the" in decoded and "quick" in decoded


def test_bpe_tokenizer_save_load(tmp_path):
    text = "hello world hello world hello world " * 30
    tok = BPETokenizer.train(text, vocab_size=50)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    ids_orig = tok.encode("hello world")
    ids_loaded = loaded.encode("hello world")
    assert ids_orig == ids_loaded


def test_mlp_with_bpe_vocab():
    model = MLPCharModel(
        vocab_size=200, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
    )
    x = torch.randint(0, 200, (2, 8))
    out = model(x)
    assert out.shape == (2, 200)
    assert torch.isfinite(out).all()


def test_mlp_default_num_attn_heads():
    """Default num_attn_heads=4 works with embed_dim divisible by 4."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=8, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1,
    )
    assert model.num_attn_heads == 4
    assert model.attn_pool.out_features == 4
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mlp_gradient_flow():
    """Verify gradients flow through the full architecture."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=3, num_attn_heads=2, num_self_attn_layers=2,
    )
    x = torch.randint(0, 10, (2, 8))
    target = torch.randint(0, 10, (2,))
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_mlp_causal_masking():
    """Changing a future token must not affect self-attention outputs for earlier positions."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=2,
    )
    model.eval()
    x = torch.randint(0, 10, (1, 8))
    x_mod = x.clone()
    x_mod[0, 7] = (x[0, 7] + 1) % 10

    with torch.no_grad():
        emb1 = model.embed(x)
        for layer in model.self_attn_layers:
            emb1, _ = layer(emb1, rope=model.rope)

        emb2 = model.embed(x_mod)
        for layer in model.self_attn_layers:
            emb2, _ = layer(emb2, rope=model.rope)

    assert torch.allclose(emb1[0, :7], emb2[0, :7], atol=1e-5), \
        "causal masking: future token change must not affect earlier positions"
    assert not torch.allclose(emb1[0, 7:], emb2[0, 7:], atol=1e-5), \
        "changed token should produce different representation"


def test_mlp_multi_self_attn_layers():
    """Stacking 3 self-attention layers should work and gradients flow."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=2, num_attn_heads=2, num_self_attn_layers=3,
    )
    assert len(model.self_attn_layers) == 3
    x = torch.randint(0, 10, (2, 8))
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()

    target = torch.randint(0, 10, (2,))
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_mlp_kv_cache_consistency():
    """KV-cached incremental decoding must produce identical output to full forward."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=2,
    )
    model.eval()
    x = torch.randint(0, 10, (1, 8))

    with torch.no_grad():
        out_full = model(x)

        # Incremental: first 7 tokens, then the 8th
        _, cache = model(x[:, :7], use_cache=True)
        out_cached, _ = model(x[:, 7:8], kv_cache=cache, use_cache=True)

    assert torch.allclose(out_full, out_cached, atol=1e-4), \
        "KV-cached output must match full forward pass"


def test_mlp_legacy_state_dict_loading():
    """Old checkpoints with pos_embed and single self_attn should load correctly."""
    model_new = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=1,
    )
    state = model_new.state_dict()

    # Simulate old checkpoint: rename self_attn_layers.0.* → self_attn.*
    # and add pos_embed
    old_state = {}
    for k, v in state.items():
        if k.startswith("self_attn_layers.0."):
            old_state["self_attn." + k[len("self_attn_layers.0."):]] = v
        else:
            old_state[k] = v
    old_state["pos_embed"] = torch.zeros(1, 8, 4)

    model_load = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=1,
    )
    model_load.load_state_dict(old_state)

    model_new.eval()
    model_load.eval()
    x = torch.randint(0, 10, (1, 8))
    with torch.no_grad():
        out_new = model_new(x)
        out_load = model_load(x)
    assert torch.allclose(out_new, out_load, atol=1e-5), \
        "legacy state dict should produce same output after remapping"


def test_mlp_num_self_attn_layers_attribute_exists():
    """MLPCharModel must expose num_self_attn_layers for train.py logging."""
    model = MLPCharModel(
        vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
        dropout=0.0, num_hidden_layers=1, num_attn_heads=2, num_self_attn_layers=3,
    )
    assert hasattr(model, "num_self_attn_layers")
    assert model.num_self_attn_layers == 3


def test_swiglu_block_residual_shape():
    """SwiGLUBlock output shape must match input shape (residual connection)."""
    block = SwiGLUBlock(dim=16, dropout=0.0)
    x = torch.randn(2, 16)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_multihead_self_attention_shape_preserved():
    """MultiHeadSelfAttention output shape must equal input shape."""
    attn = MultiHeadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0)
    x = torch.randn(2, 6, 8)
    out, cache = attn(x)
    assert out.shape == x.shape
    assert cache is None


def test_mlp_invalid_embed_dim_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        MLPCharModel(
            vocab_size=10, context_length=8, embed_dim=5, hidden_dim=16,
            dropout=0.0, num_hidden_layers=1, num_attn_heads=2,
        )


def test_mlp_num_hidden_layers_zero_raises():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        MLPCharModel(
            vocab_size=10, context_length=8, embed_dim=4, hidden_dim=16,
            dropout=0.0, num_hidden_layers=0, num_attn_heads=2,
        )


def test_clean_text_filters_noise():
    """Noise filter removes OCR garbage and short fragments."""
    from src.preprocess import clean_text
    text = "Hello world.\nf\nCK\n22 155615\nThis is a sentence.\n"
    cleaned = clean_text(text, filter_noise=True)
    assert "Hello world" in cleaned
    assert "This is a sentence" in cleaned
    assert "22 155615" not in cleaned
