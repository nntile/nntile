# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/cache_utils.py
# key-value cache implementations
#
# @version 1.1.0

import nntile.utils.constructors as nntc
from nntile.tensor import copy_intersection_async


class KVCache:
    def __init__(self, max_cache_size, seq_size_dim):
        self.max_cache_size = max_cache_size
        self.seq_size_dim = seq_size_dim

        self.k = None
        self.v = None

        self.k_cache_size = 0
        self.v_cache_size = 0

    def _init_from_tensor(self, tensor):
        cached_shape = tensor.shape
        cached_shape[self.seq_size_dim] = self.max_cache_size

        cached_basetile_shape = tensor.basetile_shape
        cached_basetile_shape[self.seq_size_dim] = self.max_cache_size

        init_cache_tensor = nntc.zeros(
            cached_shape,
            dtype=type(tensor),
            basetile_shape=cached_basetile_shape
        )
        return init_cache_tensor

    def append(self, k_partial, v_partial):
        assert k_partial.shape[self.seq_size_dim] == v_partial.shape[self.seq_size_dim]  # noqa: E501

        if not self.k:
            self.k = self._init_from_tensor(k_partial)

        copy_intersection_async(
            k_partial, [0, self.k_cache_size, 0, 0], self.k, [0, 0, 0, 0]
        )
        self.k_cache_size += k_partial.shape[self.seq_size_dim]

        if not self.v:
            self.v = self._init_from_tensor(v_partial)

        copy_intersection_async(
            v_partial, [0, self.v_cache_size, 0, 0], self.v, [0, 0, 0, 0]
        )
        self.v_cache_size += v_partial.shape[self.seq_size_dim]

    @property
    def k_partial(self):
        # For correct softmax we should next use only currently cached seq_size
        # So copy here
        cached_shape = self.k.shape
        cached_shape[self.seq_size_dim] = self.k_cache_size

        cached_basetile_shape = self.k.basetile_shape
        cached_basetile_shape[self.seq_size_dim] = self.k_cache_size

        k_partial = nntc.empty(
            cached_shape,
            dtype=type(self.k),
            basetile_shape=cached_basetile_shape,
        )
        copy_intersection_async(
            self.k, [0, 0, 0, 0], k_partial, [0, 0, 0, 0]
        )
        return k_partial

    @property
    def v_partial(self):
        # For correct softmax we should next use only currently cached seq_size
        # So copy here
        cached_shape = self.v.shape
        cached_shape[self.seq_size_dim] = self.v_cache_size

        cached_basetile_shape = self.v.basetile_shape
        cached_basetile_shape[self.seq_size_dim] = self.v_cache_size

        v_partial = nntc.empty(
            cached_shape,
            dtype=type(self.v),
            basetile_shape=cached_basetile_shape,
        )
        copy_intersection_async(
            self.v, [0, 0, 0, 0], v_partial, [0, 0, 0, 0]
        )
        return v_partial

    def clear(self):
        self.k = None
        self.v = None
        self.k_cache_size = 0
        self.v_cache_size = 0

    def __len__(self):
        return self.k_cache_size
