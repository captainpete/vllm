# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CompressedTensorsKVCacheMethod K_CACHE/Q_ATTN dispatch.

Covers _has_kq_attn_transform: config reading, per-layer targeting via
is_match, and validation errors raised at model load time.

These tests do not call the hadacore_transform kernel and therefore
run on all platforms.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsKVCacheMethod,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transform_config(
    location: str,
    scheme_type: str = "hadamard",
    targets: list[str] | None = None,
    head_dim: int | None = None,
):
    """Build a minimal TransformConfig with a single apply entry."""
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
    )

    return TransformConfig(
        config_groups={
            "r3": TransformScheme(
                type=scheme_type,
                head_dim=head_dim,
                apply=[
                    TransformArgs(
                        targets=targets if targets is not None else ["re:.*"],
                        location=location,
                    )
                ],
            )
        }
    )


def _make_quant_config(transform_config=None):
    """Minimal CompressedTensorsConfig stub with transform_config."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.transform_config = transform_config
    cfg.kv_cache_scheme = None
    return cfg


def _make_layer(layer_name: str = "model.layers.0.self_attn", head_size: int = 128):
    """A minimal layer stub with the attributes create_weights reads."""
    layer = torch.nn.Module()
    layer.num_kv_heads = 8
    layer.layer_name = layer_name
    layer.head_size = head_size
    return layer


# ---------------------------------------------------------------------------
# Enabled cases
# ---------------------------------------------------------------------------


def test_kq_transform_set_for_k_cache():
    """K_CACHE location targeting this layer → True."""
    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._kq_attn_transform is True


def test_kq_transform_set_for_q_attn():
    """Q_ATTN location targeting this layer → True."""
    quant_config = _make_quant_config(_make_transform_config("q_attn"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._kq_attn_transform is True


# ---------------------------------------------------------------------------
# Disabled cases
# ---------------------------------------------------------------------------


def test_kq_transform_none_without_transform_config():
    """No transform_config → _kq_attn_transform is False."""
    quant_config = _make_quant_config(transform_config=None)
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._kq_attn_transform is False


def test_kq_transform_none_for_non_attention_locations():
    """INPUT/OUTPUT locations should not trigger KV rotation."""
    quant_config = _make_quant_config(_make_transform_config("input"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._kq_attn_transform is False


# ---------------------------------------------------------------------------
# Per-layer targeting
# ---------------------------------------------------------------------------


def test_kq_transform_per_layer_targeting():
    """Scheme targeting only self_attn layers must not fire on other layers.

    Real R3 checkpoints target by class name (e.g. LlamaAttention) or regex.
    This test uses a regex that matches *self_attn suffixes only.
    """
    quant_config = _make_quant_config(
        _make_transform_config("k_cache", targets=["re:.*self_attn"])
    )
    method = CompressedTensorsKVCacheMethod(quant_config)

    attn_layer = _make_layer("model.layers.0.self_attn", head_size=128)
    other_layer = _make_layer("model.layers.0.mlp", head_size=128)

    method.create_weights(attn_layer)
    method.create_weights(other_layer)

    assert attn_layer._kq_attn_transform is True, (
        "self_attn layer should have rotation enabled"
    )
    assert other_layer._kq_attn_transform is False, (
        "mlp layer should not have rotation enabled"
    )


# ---------------------------------------------------------------------------
# Error cases (all raised at model load, not at runtime)
# ---------------------------------------------------------------------------


def test_random_hadamard_raises_not_implemented():
    """random-hadamard type must raise NotImplementedError at model load.

    Silently skipping would apply no rotation at serving time, corrupting
    attention for checkpoints that require the random rotation.
    """
    quant_config = _make_quant_config(
        _make_transform_config("k_cache", scheme_type="random-hadamard")
    )
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    with pytest.raises(NotImplementedError, match="random-hadamard"):
        method.create_weights(layer)


def test_missing_head_size_raises():
    """Layer without head_size attribute must raise ValueError at model load."""
    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    del layer.head_size
    with pytest.raises(ValueError, match="head_size"):
        method.create_weights(layer)


def test_scheme_head_dim_mismatch_raises():
    """scheme.head_dim != layer.head_size must raise ValueError at model load.

    K_CACHE/Q_ATTN rotation operates at head granularity. A mismatch means the
    checkpoint was calibrated with a different block size than the layer uses.
    """
    quant_config = _make_quant_config(_make_transform_config("k_cache", head_dim=64))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer(head_size=128)  # scheme says 64, layer says 128
    with pytest.raises(ValueError, match="head_dim"):
        method.create_weights(layer)


def test_non_power_of_two_head_dim_raises():
    """Non-power-of-two head_dim must raise ValueError at model load.

    The hadacore kernel enforces this constraint. Failing at model load gives
    a clear error instead of a CUDA assertion during the first forward pass.
    """
    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer(head_size=96)  # 96 is not a power of two
    with pytest.raises(ValueError, match="power of two"):
        method.create_weights(layer)


def test_head_dim_exceeds_kernel_limit_raises():
    """head_dim > 2^15 must raise ValueError at model load."""
    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer(head_size=2**16)
    with pytest.raises(ValueError, match="2\\^15"):
        method.create_weights(layer)
