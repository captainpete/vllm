# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CompressedTensorsKVCacheMethod K_CACHE/Q_ATTN dispatch.

Covers _resolve_kv_transform: config reading, per-layer targeting via
is_match, and validation errors raised at model load time.

Also covers apply_kv_cache: transform dispatch on scheme.type and the
no-op path when _ct_kv_transform is None.

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
    randomize: bool = False,
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
                randomize=randomize,
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
# _resolve_kv_transform: enabled cases
# ---------------------------------------------------------------------------


def test_kv_transform_set_for_k_cache():
    """K_CACHE location targeting this layer → TransformScheme stored."""
    from compressed_tensors.transform import TransformScheme

    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert isinstance(layer._ct_kv_transform, TransformScheme)


def test_kv_transform_set_for_q_attn():
    """Q_ATTN location targeting this layer → TransformScheme stored."""
    from compressed_tensors.transform import TransformScheme

    quant_config = _make_quant_config(_make_transform_config("q_attn"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert isinstance(layer._ct_kv_transform, TransformScheme)


# ---------------------------------------------------------------------------
# _resolve_kv_transform: disabled cases
# ---------------------------------------------------------------------------


def test_kv_transform_none_without_transform_config():
    """No transform_config → _ct_kv_transform is None."""
    quant_config = _make_quant_config(transform_config=None)
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._ct_kv_transform is None


def test_kv_transform_none_for_non_attention_locations():
    """INPUT/OUTPUT locations should not trigger KV rotation."""
    quant_config = _make_quant_config(_make_transform_config("input"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)

    assert layer._ct_kv_transform is None


# ---------------------------------------------------------------------------
# _resolve_kv_transform: per-layer targeting
# ---------------------------------------------------------------------------


def test_kv_transform_per_layer_targeting():
    """Scheme targeting only self_attn layers must not fire on other layers.

    Real R3 checkpoints target by class name (e.g. LlamaAttention) or regex.
    This test uses a regex that matches *self_attn suffixes only.
    """
    from compressed_tensors.transform import TransformScheme

    quant_config = _make_quant_config(
        _make_transform_config("k_cache", targets=["re:.*self_attn"])
    )
    method = CompressedTensorsKVCacheMethod(quant_config)

    attn_layer = _make_layer("model.layers.0.self_attn", head_size=128)
    other_layer = _make_layer("model.layers.0.mlp", head_size=128)

    method.create_weights(attn_layer)
    method.create_weights(other_layer)

    assert isinstance(attn_layer._ct_kv_transform, TransformScheme), (
        "self_attn layer should have a resolved TransformScheme"
    )
    assert other_layer._ct_kv_transform is None, (
        "mlp layer should not have a resolved TransformScheme"
    )


# ---------------------------------------------------------------------------
# apply_kv_cache: transform dispatch
# ---------------------------------------------------------------------------


def test_apply_kv_cache_calls_hadamard_transform():
    """apply_kv_cache with a hadamard scheme must rotate query and key."""
    from unittest.mock import MagicMock, patch

    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)
    layer.calculate_kv_scales = False
    layer.impl = MagicMock()

    query = torch.randn(4, 8, 128)
    key = torch.randn(4, 8, 128)
    value = torch.randn(4, 8, 128)
    kv_cache = torch.zeros(2, 4, 8, 128)
    slot_mapping = torch.arange(4)

    rotated_query = torch.randn(4, 8, 128)
    rotated_key = torch.randn(4, 8, 128)
    rotate_results = iter([rotated_query, rotated_key])

    with patch(
        "vllm.model_executor.layers.quantization.compressed_tensors"
        ".compressed_tensors.ops.hadacore_transform",
        side_effect=lambda x: next(rotate_results),
    ) as mock_rotate:
        out_query, out_key = method.apply_kv_cache(
            layer, query, key, value, kv_cache, slot_mapping
        )

    assert mock_rotate.call_count == 2, (
        "hadacore_transform must be called for query and key"
    )
    assert out_query is rotated_query
    assert out_key is rotated_key
    layer.impl.do_kv_cache_update.assert_called_once_with(
        layer, rotated_key, value, kv_cache, slot_mapping
    )


def test_apply_kv_cache_no_transform_when_scheme_is_none():
    """apply_kv_cache with no resolved scheme must not rotate query or key."""
    from unittest.mock import MagicMock, patch

    quant_config = _make_quant_config(transform_config=None)
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)
    layer.calculate_kv_scales = False
    layer.impl = MagicMock()

    query = torch.randn(4, 8, 128)
    key = torch.randn(4, 8, 128)
    value = torch.randn(4, 8, 128)
    kv_cache = torch.zeros(2, 4, 8, 128)
    slot_mapping = torch.arange(4)

    with patch(
        "vllm.model_executor.layers.quantization.compressed_tensors"
        ".compressed_tensors.ops.hadacore_transform",
    ) as mock_rotate:
        out_query, out_key = method.apply_kv_cache(
            layer, query, key, value, kv_cache, slot_mapping
        )

    mock_rotate.assert_not_called()
    assert out_query is query
    assert out_key is key


def test_apply_kv_cache_raises_if_calculate_kv_scales():
    """apply_kv_cache must raise when scheme is set and calculate_kv_scales=True."""
    from unittest.mock import MagicMock

    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    method.create_weights(layer)
    layer.calculate_kv_scales = True
    layer.impl = MagicMock()

    query = torch.randn(4, 8, 128)
    key = torch.randn(4, 8, 128)
    value = torch.randn(4, 8, 128)
    kv_cache = torch.zeros(2, 4, 8, 128)
    slot_mapping = torch.arange(4)

    with pytest.raises(ValueError, match="calculate_kv_scales"):
        method.apply_kv_cache(layer, query, key, value, kv_cache, slot_mapping)


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


def test_randomize_true_raises_not_implemented():
    """hadamard with randomize=True must raise NotImplementedError at model load.

    randomize=True means a unique per-layer matrix is stored in the checkpoint.
    Silently applying the deterministic FWHT instead would corrupt attention.
    """
    quant_config = _make_quant_config(_make_transform_config("k_cache", randomize=True))
    method = CompressedTensorsKVCacheMethod(quant_config)

    layer = _make_layer()
    with pytest.raises(NotImplementedError, match="randomize"):
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


def test_rocm_raises_not_implemented():
    """K_CACHE/Q_ATTN on ROCm must raise NotImplementedError at model load.

    hadacore_transform is a CUDA-only kernel. Failing at model load is better
    than a cryptic op error during the first forward pass.
    """
    from unittest.mock import patch

    quant_config = _make_quant_config(_make_transform_config("k_cache"))
    method = CompressedTensorsKVCacheMethod(quant_config)
    layer = _make_layer()

    with (
        patch(
            "vllm.model_executor.layers.quantization.compressed_tensors"
            ".compressed_tensors.current_platform.is_rocm",
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="ROCm"),
    ):
        method.create_weights(layer)
