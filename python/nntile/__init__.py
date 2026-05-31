# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file python/nntile/__init__.py
# NNTile graph-first Python package.
#
# @version 1.1.0

"""NNTile Python bindings for the libnntile graph API."""

from .nntile import (  # type: ignore[attr-defined]
    ActivationType,
    AdamW,
    CausalLmBatch,
    CausalLmBatchConfig,
    CausalLmBatchIterator,
    Context,
    DataType,
    Gpt2Causal,
    Gpt2Config,
    GraphRuntime,
    Linear,
    Mlp,
    Module,
    NNGraph,
    Runtime,
    TensorGraph,
    TensorNode,
    TileGraph,
    TokenMemoryMap,
    apply_gpt2_tiling_json,
    init_random_parameter_hints,
    load_gpt2_config_json,
    make_tiny_gpt2_config,
    sync_param_hint_from_runtime,
)

__all__ = [
    'ActivationType',
    'AdamW',
    'CausalLmBatch',
    'CausalLmBatchConfig',
    'CausalLmBatchIterator',
    'Context',
    'DataType',
    'Gpt2Causal',
    'Gpt2Config',
    'GraphRuntime',
    'Linear',
    'Mlp',
    'Module',
    'NNGraph',
    'Runtime',
    'TensorGraph',
    'TensorNode',
    'TileGraph',
    'TokenMemoryMap',
    'apply_gpt2_tiling_json',
    'init_random_parameter_hints',
    'load_gpt2_config_json',
    'make_tiny_gpt2_config',
    'nn',
    'sync_param_hint_from_runtime',
]

# Submodule populated by pybind11
from . import nntile as _ext  # type: ignore[attr-defined]

nn = _ext.nn
