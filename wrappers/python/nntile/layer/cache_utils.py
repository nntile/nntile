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


class KVCacheStorage:
    def __init__(self):
        self.kv_caches = None
        self.is_initialized = False

    def is_initialized(self):
        return self.is_initialized

    def init(self, num_layers, max_cache_size, seq_size_dim=1):
        self.kv_caches = [
            KVCache(max_cache_size, seq_size_dim) for _ in range(num_layers)
        ]
        self.is_initialized = True

    def get_cache(self):
        assert self.kv_caches
        return self.kv_caches


class ParallelSamplingCacheStorage(KVCacheStorage):
    def __init__(self, num_beams):
        self.num_beams = num_beams
        self.num_layers = 0
        super().__init__()

    def init(self, num_layers, max_cache_size, seq_size_dim=1):
        self.num_layers = num_layers
        self.kv_caches = [
            ParallelSamplingKVCache(
                num_beams=self.num_beams,
                max_cache_size=max_cache_size,
                seq_size_dim=seq_size_dim,
            )
            for _ in range(self.num_layers)
        ]

    def get_cache(self):
        return self.get_prefill()

    def get_prefill(self):
        assert self.kv_caches

        prefill_kv_caches = [
            self.kv_caches[i].get_base() for i in range(self.num_layers)
        ]
        return prefill_kv_caches

    def get_beam(self, beam):
        assert self.kv_caches

        beam_kv_caches = [
            self.kv_caches[i].get_beam(beam) for i in range(self.num_layers)
        ]
        return beam_kv_caches

    def reduce(self, beams_ids):
        assert self.kv_caches

        for i in range(self.num_layers):
            self.kv_caches[i].reduce(beams_ids)


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
            basetile_shape=cached_basetile_shape,
        )
        return init_cache_tensor

    def append(self, k_partial, v_partial):
        assert (
            k_partial.shape[self.seq_size_dim]
            == v_partial.shape[self.seq_size_dim]
        )  # noqa: E501

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
        copy_intersection_async(self.k, [0, 0, 0, 0], k_partial, [0, 0, 0, 0])
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
        copy_intersection_async(self.v, [0, 0, 0, 0], v_partial, [0, 0, 0, 0])
        return v_partial

    def clear(self):
        self.k = None
        self.v = None
        self.k_cache_size = 0
        self.v_cache_size = 0

    def __len__(self):
        return self.k_cache_size


class DynamicKVCacheWithBase:
    def __init__(self, base, head):
        assert base.seq_size_dim == head.seq_size_dim
        self.base = base
        self.head = head

    def __len__(self):
        return len(self.base) + len(self.head)

    def append(self, k_partial, v_partial):
        self.head.append(k_partial, v_partial)

    @property
    def k_partial(self):
        k_partial_shape = self.base.k.shape
        k_partial_basetile_shape = self.base.k.basetile_shape
        k_partial_shape[self.base.seq_size_dim] = len(self.base) + len(
            self.head
        )
        k_partial_basetile_shape[self.base.seq_size_dim] = len(
            self.base
        ) + len(self.head)

        k_partial = nntc.zeros(
            k_partial_shape,
            basetile_shape=k_partial_basetile_shape,
            dtype=type(self.base.k),
        )

        copy_intersection_async(
            self.base.k, [0, 0, 0, 0], k_partial, [0, 0, 0, 0]
        )

        offset = len(self.base)
        for block in self.head.iter_k():
            copy_intersection_async(
                block, [0, offset, 0, 0], k_partial, [0, 0, 0, 0]
            )
            offset += block.shape[self.head.seq_size_dim]

        return k_partial

    @property
    def v_partial(self):
        v_partial_shape = self.base.v.shape
        v_partial_basetile_shape = self.base.v.basetile_shape
        v_partial_shape[self.base.seq_size_dim] = len(self.base) + len(
            self.head
        )
        v_partial_basetile_shape[self.base.seq_size_dim] = len(
            self.base
        ) + len(self.head)

        v_partial = nntc.zeros(
            v_partial_shape,
            basetile_shape=v_partial_basetile_shape,
            dtype=type(self.base.v),
        )

        copy_intersection_async(
            self.base.v, [0, 0, 0, 0], v_partial, [0, 0, 0, 0]
        )

        offset = len(self.base)
        for block in self.head.iter_v():
            copy_intersection_async(
                block, [0, offset, 0, 0], v_partial, [0, 0, 0, 0]
            )
            offset += block.shape[self.head.seq_size_dim]

        return v_partial


class DynamicKVCache:
    def __init__(self, max_cache_size, seq_size_dim, clone_input=False):
        self.max_cache_size = max_cache_size
        self.seq_size_dim = seq_size_dim

        self.k_buff = []
        self.v_buff = []

        self.clone_input = clone_input

        self.k_cache_size = 0
        self.v_cache_size = 0

    def shallow_copy(self):
        cl = DynamicKVCache(
            self.max_cache_size, self.seq_size_dim, self.clone_input
        )
        # only copy buffers, tensors stays same
        cl.k_buff = self.k_buff.copy()
        cl.v_buff = self.v_buff.copy()
        cl.k_cache_size = self.k_cache_size
        cl.v_cache_size = self.v_cache_size
        return cl

    def append(self, k_partial, v_partial):
        assert (
            k_partial.shape[self.seq_size_dim]
            == v_partial.shape[self.seq_size_dim]
        )

        if self.clone_input:
            self.k_buff.append(nntc.clone(k_partial))
            self.v_buff.append(nntc.clone(v_partial))
        else:
            self.k_buff.append(k_partial)
            self.v_buff.append(v_partial)

        self.k_cache_size += k_partial.shape[self.seq_size_dim]
        self.v_cache_size += v_partial.shape[self.seq_size_dim]

    def iter_k(self):
        return self.k_buff

    def iter_v(self):
        return self.v_buff

    @property
    def k_partial(self):
        assert self.k_cache_size != 0

        k_partial_shape = self.k_buff[0].shape
        k_partial_basetile_shape = self.k_buff[0].basetile_shape
        k_partial_shape[self.seq_size_dim] = self.k_cache_size
        k_partial_basetile_shape[self.seq_size_dim] = self.k_cache_size

        k_partial = nntc.zeros(
            k_partial_shape,
            basetile_shape=k_partial_basetile_shape,
            dtype=type(self.k_buff[0]),
        )

        offset = 0
        for block in self.iter_k():
            copy_intersection_async(
                block, [0, offset, 0, 0], k_partial, [0, 0, 0, 0]
            )
            offset += block.shape[self.seq_size_dim]

        return k_partial

    @property
    def v_partial(self):
        assert self.v_cache_size != 0

        v_partial_shape = self.v_buff[0].shape
        v_partial_basetile_shape = self.v_buff[0].basetile_shape
        v_partial_shape[self.seq_size_dim] = self.v_cache_size
        v_partial_basetile_shape[self.seq_size_dim] = self.v_cache_size

        v_partial = nntc.zeros(
            v_partial_shape,
            basetile_shape=v_partial_basetile_shape,
            dtype=type(self.v_buff[0]),
        )

        offset = 0
        for block in self.iter_v():
            copy_intersection_async(
                block, [0, offset, 0, 0], v_partial, [0, 0, 0, 0]
            )
            offset += block.shape[self.seq_size_dim]

        return v_partial

    def clear(self):
        self.k_buff = []
        self.v_buff = []
        self.k_cache_size = 0
        self.v_cache_size = 0

    def __len__(self):
        return self.k_cache_size


class ParallelSamplingKVCache:
    def __init__(
        self, num_beams, max_cache_size, seq_size_dim, clone_input=False
    ):
        self.base = KVCache(max_cache_size, seq_size_dim)
        self.num_beams = num_beams
        self.beams = [
            DynamicKVCacheWithBase(
                self.base,
                DynamicKVCache(
                    max_cache_size, seq_size_dim, clone_input=clone_input
                ),
            )
            for _ in range(num_beams)
        ]

    def get_base(self):
        return self.base

    def get_beam(self, beam):
        return self.beams[beam]

    def reduce(self, indexes):
        assert len(indexes) == self.num_beams
        new_beams = [
            DynamicKVCacheWithBase(
                self.base, self.beams[index].head.shallow_copy()
            )
            for index in indexes
        ]
        self.beams = new_beams
