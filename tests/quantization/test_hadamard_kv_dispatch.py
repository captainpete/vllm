# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CompressedTensorsKVCacheMethod K_CACHE/Q_ATTN dispatch.

Covers _has_kq_attn_transform: config reading, per-layer targeting via
is_match, and validation errors raised at model load time.

These tests do not call the hadacore_transform kernel and therefore
run on all platforms.
"""

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsKVCacheMethod,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transform_config(
    location: str, scheme_type: str = "hadamard", targets: list[str] | None = None
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
