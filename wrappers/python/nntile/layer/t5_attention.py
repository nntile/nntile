# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/t5_attention.py
# T5 Attention layer of NNTile Python package
#
# @version 1.1.0
# ruff: noqa: E501

from typing import Optional

import numpy as np
import torch
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigTorch
from transformers.models.t5.modeling_t5 import T5Attention as T5AttentionTorch

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.layer.cache_utils import KVCache
from nntile.model.t5_config import T5ConfigNNTile
from nntile.tensor import (
    Tensor, Tensor_bool, TensorMoments, TensorTraits, add_async,
    add_fiber_inplace_async, add_slice_inplace_async, clear_async, copy_async,
    copy_intersection_async, embedding_async, embedding_backward_async,
    gemm_async, mask_scalar_async, maxsumexp_async, notrans,
    prod_inplace_async, softmax_inplace_async, sum_fiber_async,
    sumprod_slice_async, trans, transpose_async)


def relative_position_bucket_numpy(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: a numpy array
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a numpy array with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(np.int32) * num_buckets
        relative_position = np.abs(relative_position)
    else:
        relative_position = -np.minimum(
            relative_position, np.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        np.log(relative_position.astype(np.float32) / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int32)
    relative_position_if_large = np.minimum(
        relative_position_if_large,
        np.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += np.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


# Multi-head attention
# Inputs:
#  x_q: (n_emb, n_seq, n_batch) tensor
#  x_k: (n_emb_k, n_seq, n_batch) tensor
#  x_v: (n_emb_v, n_seq, n_batch) tensor
# Output:
#  y: (n_emb, n_seq, n_batch) tensor
class T5Attention(BaseLayer):
    x_q: TensorMoments
    x_k: TensorMoments
    x_v: TensorMoments
    y: TensorMoments
    w_q: TensorMoments
    w_k: TensorMoments
    w_v: TensorMoments
    w: TensorMoments
    q_transposed: TensorMoments
    q: TensorMoments
    k_transposed: TensorMoments
    k: TensorMoments
    v_transposed: TensorMoments
    v: TensorMoments
    a: TensorMoments
    a_maxsumexp: Tensor
    a_sumprod_slice: Tensor
    b: TensorMoments
    b_transposed: TensorMoments
    n_head: int
    head_size: int

    # Construct attention layer with all the provided data
    def __init__(
        self,
        x_q: TensorMoments,
        x_k: TensorMoments,
        x_v: TensorMoments,
        y: TensorMoments,
        w_q: TensorMoments,
        w_k: TensorMoments,
        w_v: TensorMoments,
        w: TensorMoments,
        q_transposed: TensorMoments,
        q: TensorMoments,
        k_transposed: TensorMoments,
        k: TensorMoments,
        v_transposed: TensorMoments,
        v: TensorMoments,
        a: TensorMoments,
        a_maxsumexp: Tensor,
        a_sumprod_slice: Tensor,
        b: TensorMoments,
        b_transposed: TensorMoments,
        in_proj_bias_q: TensorMoments,
        in_proj_bias_k: TensorMoments,
        in_proj_bias_v: TensorMoments,
        out_proj_bias: TensorMoments,
        mask=None,
        scale_attn: bool = False,
        has_relative_bias: bool = False,
        relative_bias_position_buckets: TensorMoments = None,
        relative_bias_embedding: TensorMoments = None,
        relative_bias: TensorMoments = None,
        redux: bool = False,
        is_decoder: bool = False,
    ):
        qkv_bias_list = []
        if in_proj_bias_q:
            qkv_bias_list.append(in_proj_bias_q)
            in_proj_bias_q.grad.set_reduction_add()
        if in_proj_bias_k:
            qkv_bias_list.append(in_proj_bias_k)
            in_proj_bias_k.grad.set_reduction_add()
        if in_proj_bias_v:
            qkv_bias_list.append(in_proj_bias_v)
            in_proj_bias_v.grad.set_reduction_add()
        if out_proj_bias:
            bias_list_out_proj = [out_proj_bias]
            out_proj_bias.grad.set_reduction_add()
        else:
            bias_list_out_proj = []

        if has_relative_bias:
            # assert relative_bias_position_buckets is not None, "relative_bias_position_buckets must be provided if has_relative_bias is True"
            assert (
                relative_bias is not None
            ), "relative_bias must be provided if has_relative_bias is True"
            assert (
                relative_bias_embedding is not None
            ), "relative_bias_embedding must be provided if has_relative_bias is True"
            relative_bias_embedding.grad.set_reduction_add()

        # Redirect to BaseClass initialization
        super().__init__(
            [x_q, x_k, x_v],
            [y],
            (
                [w_q, w_k, w_v] + qkv_bias_list + [w] + [relative_bias_embedding]
                if has_relative_bias
                else [] + bias_list_out_proj
            ),
            (
                [q_transposed, q, k_transposed, k, v_transposed, v, a]
                + [
                    # relative_bias_position_buckets,
                    relative_bias,
                ]
                if has_relative_bias
                else []
                + [
                    a_maxsumexp,
                    a_sumprod_slice,
                    b,
                    b_transposed,
                ]
            ),
        )
        self.x_q = x_q
        self.x_q.grad.set_reduction_add()
        self.x_k = x_k
        self.x_k.grad.set_reduction_add()
        self.x_v = x_v
        self.x_v.grad.set_reduction_add()
        self.y = y
        self.y.value.set_reduction_add()
        self.w_q = w_q
        self.w_q.grad.set_reduction_add()
        self.w_k = w_k
        self.w_k.grad.set_reduction_add()
        self.w_v = w_v
        self.w_v.grad.set_reduction_add()
        self.w = w
        self.w.grad.set_reduction_add()
        self.q_transposed = q_transposed
        self.q_transposed.value.set_reduction_add()
        self.q = q
        self.q.grad.set_reduction_add()
        self.k_transposed = k_transposed
        self.k_transposed.value.set_reduction_add()
        self.k = k
        self.k.grad.set_reduction_add()
        self.v_transposed = v_transposed
        self.v_transposed.value.set_reduction_add()
        self.v = v
        self.v.grad.set_reduction_add()
        self.a = a
        self.a.value.set_reduction_add()
        self.a.grad.set_reduction_add()
        self.a_maxsumexp = a_maxsumexp
        self.a_maxsumexp.set_reduction_maxsumexp()
        self.a_sumprod_slice = a_sumprod_slice
        self.a_sumprod_slice.set_reduction_add()
        self.b = b
        self.b.value.set_reduction_add()
        self.b_transposed = b_transposed
        self.b_transposed.grad.set_reduction_add()
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        self.out_proj_bias = out_proj_bias
        self.n_head = w_q.value.shape[0]
        self.n_head_tile = w_q.value.basetile_shape[0]
        self.n_emb = x_q.value.shape[0]
        self.n_emb_tile = x_q.value.basetile_shape[0]

        self.has_relative_bias = has_relative_bias
        self.relative_bias_position_buckets = relative_bias_position_buckets
        self.relative_bias = relative_bias
        self.relative_bias_embedding = relative_bias_embedding

        # inner dim is the same for all the tensors, but can be different from n_emb
        self.head_size = self.w_q.value.shape[1]
        self.mask = mask
        if mask:
            self.val = -np.float32(np.inf)
        if redux:
            self.redux = 1
        else:
            self.redux = 0
        self.scale_attn = scale_attn
        self.is_decoder = is_decoder
        # need to fill with valid values for dynamic api usage
        clear_async(self.q.value)
        clear_async(self.k.value)
        clear_async(self.v.value)

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple(
        x_q: TensorMoments,
        x_k: TensorMoments,
        x_v: TensorMoments,
        n_head: int,
        n_head_tile: int,
        next_tag: int,
        inner_dim: int = None,
        inner_dim_tile: int = None,
        bias=False,
        mask=None,
        scale_attn: bool = False,
        has_relative_bias: bool = False,
        relative_vocab_size: int = 0,
        redux: bool = False,
        is_decoder: bool = False,
    ):
        # Get sizes
        n_batch_tile = x_q.value.basetile_shape[2]
        n_emb, n_seq, n_batch = x_q.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x_q.value.basetile_shape

        inner_dim = n_emb if inner_dim is None else inner_dim
        inner_dim_tile = n_emb_tile if inner_dim_tile is None else inner_dim_tile

        head_size = inner_dim // n_head
        # Stupid check, that is not necessary, as the code shall work
        if inner_dim != head_size * n_head:
            raise RuntimeError
        n_emb_k = x_k.value.shape[0]
        n_emb_k_tile = x_k.value.basetile_shape[0]
        # if [n_seq, n_batch] != x_k.value.shape[1:]:
        #     raise ValueError(f"Invalid shape of x_k: {x_k.value.shape[1:]} != {[n_seq, n_batch]}")
        # if [n_seq_tile, n_batch_tile] != x_k.value.basetile_shape[1:]:
        #     raise ValueError(f"Invalid basetile shape of x_k: {x_k.value.basetile_shape[1:]} != {[n_seq_tile, n_batch_tile]}")
        n_emb_v = x_v.value.shape[0]
        n_emb_v_tile = x_v.value.basetile_shape[0]
        # if [n_seq, n_batch] != x_v.value.shape[1:]:
        #     raise ValueError(f"Invalid shape of x_v: {x_v.value.shape[1:]} != {[n_seq, n_batch]}")
        # if [n_seq_tile, n_batch_tile] != x_v.value.basetile_shape[1:]:
        #     raise ValueError(f"Invalid basetile shape of x_v: {x_v.value.basetile_shape[1:]} != {[n_seq_tile, n_batch_tile]}")
        # Fixed for now
        head_size_tile = head_size
        # Define shape of each tensor
        w_q_shape = [n_head, head_size, n_emb]
        w_k_shape = [n_head, head_size, n_emb_k]
        w_v_shape = [n_head, head_size, n_emb_v]
        w_shape = [n_emb, n_head, head_size]
        q_transposed_shape = [n_head, head_size, n_seq, n_batch]
        q_shape = [head_size, n_seq, n_batch, n_head]
        k_transposed_shape = [n_head, head_size, n_seq, n_batch]
        k_shape = [head_size, n_seq, n_batch, n_head]
        v_transposed_shape = [n_head, head_size, n_seq, n_batch]
        v_shape = [head_size, n_seq, n_batch, n_head]
        a_shape = [n_seq, n_seq, n_batch, n_head]
        a_maxsumexp_shape = [2, n_seq, n_batch, n_head]
        a_sumprod_slice_shape = [n_seq, n_batch, n_head]
        b_shape = [head_size, n_seq, n_batch, n_head]
        b_transposed_shape = [n_head, head_size, n_seq, n_batch]
        if has_relative_bias:
            relative_bias_shape = [n_seq, n_seq, n_batch, n_head]
            relative_bias_basetile = [n_seq_tile, n_seq_tile, n_batch_tile, n_head_tile]

            assert relative_vocab_size > 0, "relative_vocab_size must be greater than 0"
            relative_bias_embedding_shape = [n_head, relative_vocab_size]
            relative_bias_embedding_basetile = [n_head_tile, relative_vocab_size]

        # Define tile shapes of each tensor
        w_q_basetile = [n_head_tile, head_size_tile, n_emb_tile]
        w_k_basetile = [n_head_tile, head_size_tile, n_emb_k_tile]
        w_v_basetile = [n_head_tile, head_size_tile, n_emb_v_tile]
        w_basetile = [n_emb_tile, n_head_tile, head_size_tile]
        q_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        q_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        k_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        k_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        v_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        v_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile, n_head_tile]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile, n_head_tile]
        a_sumprod_slice_basetile = [n_seq_tile, n_batch_tile, n_head_tile]
        b_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
        b_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        w_traits = TensorTraits(w_shape, w_basetile)
        q_transposed_traits = TensorTraits(q_transposed_shape, q_transposed_basetile)
        q_traits = TensorTraits(q_shape, q_basetile)
        k_transposed_traits = TensorTraits(k_transposed_shape, k_transposed_basetile)
        k_traits = TensorTraits(k_shape, k_basetile)
        v_transposed_traits = TensorTraits(v_transposed_shape, v_transposed_basetile)
        v_traits = TensorTraits(v_shape, v_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(a_maxsumexp_shape, a_maxsumexp_basetile)
        a_sumprod_slice_traits = TensorTraits(
            a_sumprod_slice_shape, a_sumprod_slice_basetile
        )
        b_traits = TensorTraits(b_shape, b_basetile)
        b_transposed_traits = TensorTraits(b_transposed_shape, b_transposed_basetile)
        if has_relative_bias:
            relative_bias_traits = TensorTraits(
                relative_bias_shape, relative_bias_basetile
            )
            relative_bias_distr = [0] * relative_bias_traits.grid.nelems

            relative_bias_embedding_traits = TensorTraits(
                relative_bias_embedding_shape, relative_bias_embedding_basetile
            )
            relative_bias_embedding_distr = [
                0
            ] * relative_bias_embedding_traits.grid.nelems

        # TODO change distribution
        w_q_distr = [0] * w_q_traits.grid.nelems
        w_k_distr = [0] * w_k_traits.grid.nelems
        w_v_distr = [0] * w_v_traits.grid.nelems
        w_distr = [0] * w_traits.grid.nelems
        q_transposed_distr = [0] * q_transposed_traits.grid.nelems
        q_distr = [0] * q_traits.grid.nelems
        k_transposed_distr = [0] * k_transposed_traits.grid.nelems
        k_distr = [0] * k_traits.grid.nelems
        v_transposed_distr = [0] * v_transposed_traits.grid.nelems
        v_distr = [0] * v_traits.grid.nelems
        a_distr = [0] * a_traits.grid.nelems
        a_maxsumexp_distr = [0] * a_maxsumexp_traits.grid.nelems
        a_sumprod_slice_distr = [0] * a_sumprod_slice_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        b_transposed_distr = [0] * b_transposed_traits.grid.nelems
        if bias:
            in_proj_bias_qkv_traits = TensorTraits(
                [head_size, n_head], [head_size_tile, n_head_tile]
            )
            in_proj_bias_qkv_distr = [0] * in_proj_bias_qkv_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_value.next_tag
        w_q_grad = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_grad.next_tag
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if bias:
            in_proj_bias_q_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_q_value.next_tag
            in_proj_bias_q_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_q_grad.next_tag
            bias_inproj_q = TensorMoments(
                in_proj_bias_q_value, in_proj_bias_q_grad, True
            )
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_value.next_tag
        w_k_grad = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_grad.next_tag
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if bias:
            in_proj_bias_k_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_k_value.next_tag
            in_proj_bias_k_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_k_grad.next_tag
            bias_inproj_k = TensorMoments(
                in_proj_bias_k_value, in_proj_bias_k_grad, True
            )
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_value.next_tag
        w_v_grad = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_grad.next_tag
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if bias:
            in_proj_bias_v_value = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_v_value.next_tag
            in_proj_bias_v_grad = type(x_q.value)(
                in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, next_tag
            )
            next_tag = in_proj_bias_v_grad.next_tag
            bias_inproj_v = TensorMoments(
                in_proj_bias_v_value, in_proj_bias_v_grad, True
            )
        else:
            bias_inproj_v = None
        # w
        w_value = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        w_grad = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        w = TensorMoments(w_value, w_grad, True)
        # q_transposed
        q_transposed_value = type(x_q.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_value.next_tag
        q_transposed_grad = type(x_q.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_grad.next_tag
        q_transposed = TensorMoments(q_transposed_value, q_transposed_grad, True)
        # q
        q_value = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_value.next_tag
        q_grad = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_grad.next_tag
        q = TensorMoments(q_value, q_grad, True)
        # k_transposed
        k_transposed_value = type(x_q.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_value.next_tag
        k_transposed_grad = type(x_q.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_grad.next_tag
        k_transposed = TensorMoments(k_transposed_value, k_transposed_grad, True)
        # k
        k_value = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_value.next_tag
        k_grad = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_grad.next_tag
        k = TensorMoments(k_value, k_grad, True)
        # v_transposed
        v_transposed_value = type(x_q.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_value.next_tag
        v_transposed_grad = type(x_q.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_grad.next_tag
        v_transposed = TensorMoments(v_transposed_value, v_transposed_grad, True)
        # v
        v_value = type(x_q.value)(v_traits, v_distr, next_tag)
        next_tag = v_value.next_tag
        v_grad = type(x_q.value)(v_traits, v_distr, next_tag)
        next_tag = v_grad.next_tag
        v = TensorMoments(v_value, v_grad, True)
        # a
        a_value = type(x_q.value)(a_traits, a_distr, next_tag)
        next_tag = a_value.next_tag
        a_grad = type(x_q.value)(a_traits, a_distr, next_tag)
        next_tag = a_grad.next_tag
        a = TensorMoments(a_value, a_grad, True)
        # a_maxsumexp
        a_maxsumexp = type(x_q.value)(a_maxsumexp_traits, a_maxsumexp_distr, next_tag)
        next_tag = a_maxsumexp.next_tag
        # a_sumprod_slice
        a_sumprod_slice = type(x_q.value)(
            a_sumprod_slice_traits, a_sumprod_slice_distr, next_tag
        )
        next_tag = a_sumprod_slice.next_tag
        # b
        b_value = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_value.next_tag
        b_grad = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_grad.next_tag
        b = TensorMoments(b_value, b_grad, True)
        # b_transposed
        b_transposed_value = type(x_q.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_value.next_tag
        b_transposed_grad = type(x_q.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_grad.next_tag
        b_transposed = TensorMoments(b_transposed_value, b_transposed_grad, True)

        relative_bias = None
        relative_bias_embedding = None
        if has_relative_bias:
            relative_bias_value = type(x_q.value)(
                relative_bias_traits, relative_bias_distr, next_tag
            )
            relative_bias_grad = type(x_q.value)(
                relative_bias_traits, relative_bias_distr, next_tag
            )
            next_tag = relative_bias_value.next_tag
            relative_bias = TensorMoments(relative_bias_value, relative_bias_grad, True)

            relative_bias_embedding_value = type(x_q.value)(
                relative_bias_embedding_traits, relative_bias_embedding_distr, next_tag
            )
            next_tag = relative_bias_embedding_value.next_tag
            relative_bias_embedding_grad = type(x_q.value)(
                relative_bias_embedding_traits, relative_bias_embedding_distr, next_tag
            )
            next_tag = relative_bias_embedding_grad.next_tag
            relative_bias_embedding = TensorMoments(
                relative_bias_embedding_value, relative_bias_embedding_grad, True
            )

        # Allocate tensors for bias for q, k, v and output projection
        if bias:
            out_proj_bias_traits = TensorTraits([n_emb], [n_emb_tile])
            out_proj_bias_distr = [0] * out_proj_bias_traits.grid.nelems
            out_proj_bias_value = type(x_q.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_value.next_tag
            out_proj_bias_grad = type(x_q.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_grad.next_tag
            out_proj_bias = TensorMoments(out_proj_bias_value, out_proj_bias_grad, True)
        else:
            out_proj_bias = None
        # Allocate tensor for output y
        y_traits = TensorTraits(x_q.value.shape, x_q.value.basetile_shape)
        y_value = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        # Create attention layer with all the provided data
        layer = T5Attention(
            x_q,
            x_k,
            x_v,
            y,
            w_q,
            w_k,
            w_v,
            w,
            q_transposed,
            q,
            k_transposed,
            k,
            v_transposed,
            v,
            a,
            a_maxsumexp,
            a_sumprod_slice,
            b,
            b_transposed,
            bias_inproj_q,
            bias_inproj_k,
            bias_inproj_v,
            out_proj_bias,
            mask,
            scale_attn=scale_attn,
            has_relative_bias=has_relative_bias,
            relative_bias=relative_bias,
            relative_bias_embedding=relative_bias_embedding,
            redux=redux,
            is_decoder=is_decoder,
        )
        # Return layer and next tag to be used
        return (layer, next_tag)

    @classmethod
    def from_torch(
        cls,
        torch_layer: T5AttentionTorch,
        x: TensorMoments,
        mask: Tensor,
        config: T5ConfigNNTile,
        next_tag: int,
        encoder_output: TensorMoments = None,
    ):
        attn, next_tag = T5Attention.generate_simple(
            x_q=x,
            x_k=encoder_output or x,  # for cross-attention
            x_v=encoder_output or x,
            n_head=config.n_head,
            n_head_tile=config.n_head_tile,
            next_tag=next_tag,
            inner_dim=config.d_kv * config.n_head,
            inner_dim_tile=config.d_kv_tile,
            bias=False,
            mask=mask,
            scale_attn=False,
            has_relative_bias=torch_layer.has_relative_attention_bias,
            relative_vocab_size=(
                torch_layer.relative_attention_bias.weight.shape[0]
                if torch_layer.has_relative_attention_bias
                else 0
            ),
            redux=config.redux,
            is_decoder=config.is_decoder,
        )

        attn.w_q.value.from_array(
            torch_layer.q.weight.view((torch_layer.n_heads, -1, config.d_model))
            .cpu()
            .detach()
            .numpy()
        )
        attn.w_k.value.from_array(
            torch_layer.k.weight.view((torch_layer.n_heads, -1, config.d_model))
            .cpu()
            .detach()
            .numpy()
        )
        attn.w_v.value.from_array(
            torch_layer.v.weight.view(torch_layer.n_heads, -1, config.d_model)
            .cpu()
            .detach()
            .numpy()
        )

        attn.w.value.from_array(
            torch_layer.o.weight.view(config.d_model, torch_layer.n_heads, -1)
            .cpu()
            .detach()
            .numpy()
        )

        if attn.has_relative_bias:
            attn.relative_bias_embedding.value.from_array(
                torch_layer.relative_attention_bias.weight.T.cpu().detach().numpy()
            )

        return attn, next_tag

    def to_torch(self):
        """Convert NNTile T5Attention to PyTorch T5Attention"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.n_emb,
            d_ff=None,  # self.n_emb * 4,  # Default value
            num_heads=self.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=None,  # 1e-6,  # Default value
            is_gated_act=True,
            is_encoder_decoder=True,
        )

        # Create PyTorch attention layer
        torch_attention = T5AttentionTorch(
            torch_config, has_relative_attention_bias=self.has_relative_bias
        )
        from nntile.tensor import to_numpy

        # Convert weights
        # Reshape and convert weights from NNTile format to PyTorch format
        q_weight = torch.tensor(to_numpy(self.w_q.value)).reshape(-1, self.n_emb)
        k_weight = torch.tensor(to_numpy(self.w_k.value)).reshape(-1, self.n_emb)
        v_weight = torch.tensor(to_numpy(self.w_v.value)).reshape(-1, self.n_emb)
        o_weight = torch.tensor(to_numpy(self.w.value)).reshape(self.n_emb, -1)

        torch_attention.q.weight.data = q_weight
        torch_attention.k.weight.data = k_weight
        torch_attention.v.weight.data = v_weight
        torch_attention.o.weight.data = o_weight

        # Convert relative attention bias if present
        if self.has_relative_bias:
            relative_bias = torch.tensor(to_numpy(self.relative_bias_embedding.value)).T
            torch_attention.relative_attention_bias.weight.data = relative_bias
            torch_attention.has_relative_attention_bias = True

        return torch_attention

    def _forward_mlp_q_async(self):
        # Q_transposed = einsum('jkl,lmn->jkmn', W_Q, X_Q)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_q.value,
            notrans,
            self.x_q.value,
            0.0,
            self.q_transposed.value,
            1,
            0,
            redux=self.redux,
        )
        # Rotate axes into (head_size, n_seq, n_batch, n_head)
        transpose_async(1.0, self.q_transposed.value, self.q.value, 1)
        # X_Q, W_Q and Q_transposed can be offloaded from GPU
        self.x_q.value.wont_use()
        # self.q_transposed.value.wont_use()
        self.q_transposed.value.invalidate_submit()
        self.w_q.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_q is not None:
            # batched add_fiber_inplace (head_size, batch=n_head) into
            # (head_size, n_seq, n_batch, batch=n_head)
            add_fiber_inplace_async(1, self.in_proj_bias_q.value, 1, self.q.value, 0, 1)
            self.in_proj_bias_q.value.wont_use()

    def _get_tmp_tr_for_cache(self, x):
        partial_tr_shape = (self.n_head, self.head_size) + tuple(x.shape[1:])
        partial_tr_basetile_shape = (self.n_head_tile, self.head_size) + tuple(
            x.shape[1:]
        )
        return nntc.empty(
            partial_tr_shape,
            dtype=type(x),
            basetile_shape=partial_tr_basetile_shape,
        )

    def _get_tmp_for_cache(self, x):
        partial_shape = (self.head_size,) + tuple(x.shape[1:]) + (self.n_head,)
        partial_basetile_shape = (
            (self.head_size,) + tuple(x.shape[1:]) + (self.n_head_tile,)
        )
        return nntc.empty(
            partial_shape, dtype=type(x), basetile_shape=partial_basetile_shape
        )

    def _forward_mlp_q_dynamic(self, x: Tensor):
        q_partial_tr = self._get_tmp_tr_for_cache(x)
        q_partial = self._get_tmp_for_cache(x)

        gemm_async(
            1.0,
            notrans,
            self.w_q.value,
            notrans,
            x,
            0.0,
            q_partial_tr,
            1,
            0,
            redux=self.redux,
        )

        transpose_async(1.0, q_partial_tr, q_partial, 1)

        if self.in_proj_bias_q is not None:
            add_fiber_inplace_async(1, self.in_proj_bias_q.value, 1, q_partial, 0, 1)

        return q_partial

    def _forward_mlp_k_async(self):
        # K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_k.value,
            notrans,
            self.x_k.value,
            0.0,
            self.k_transposed.value,
            1,
            0,
            redux=self.redux,
        )
        # Rotate axes into (head_size, n_seq, n_batch, n_head)
        transpose_async(1.0, self.k_transposed.value, self.k.value, 1)
        # X_K, W_K and K_transposed can be offloaded from GPU
        self.x_k.value.wont_use()
        # self.k_transposed.value.wont_use()
        self.k_transposed.value.invalidate_submit()
        self.w_k.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_k is not None:
            # batched add_fiber_inplace (head_size, batch=n_head) into
            # (head_size, n_seq, n_batch, batch=n_head)
            add_fiber_inplace_async(1, self.in_proj_bias_k.value, 1, self.k.value, 0, 1)
            self.in_proj_bias_k.value.wont_use()

    def _forward_mlp_k_dynamic(self, x: Tensor):
        k_partial_tr = self._get_tmp_tr_for_cache(x)
        k_partial = self._get_tmp_for_cache(x)

        gemm_async(
            1.0,
            notrans,
            self.w_k.value,
            notrans,
            x,
            0.0,
            k_partial_tr,
            1,
            0,
            redux=self.redux,
        )

        transpose_async(1.0, k_partial_tr, k_partial, 1)

        if self.in_proj_bias_k is not None:
            add_fiber_inplace_async(1, self.in_proj_bias_k.value, 1, k_partial, 0, 1)

        return k_partial

    def _forward_mlp_v_async(self):
        # V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_v.value,
            notrans,
            self.x_v.value,
            0.0,
            self.v_transposed.value,
            1,
            0,
            redux=self.redux,
        )
        # Rotate axes into (head_size, n_seq, n_batch, n_head)
        transpose_async(1.0, self.v_transposed.value, self.v.value, 1)
        # X_V, W_V and V_transposed can be offloaded from GPU
        self.x_v.value.wont_use()
        # self.v_transposed.value.wont_use()
        self.v_transposed.value.invalidate_submit()
        self.w_v.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_v is not None:
            # batched add_fiber_inplace (head_size, batch=n_head) into
            # (head_size, n_seq, n_batch, batch=n_head)
            add_fiber_inplace_async(1, self.in_proj_bias_v.value, 1, self.v.value, 0, 1)
            self.in_proj_bias_v.value.wont_use()

    def _forward_mlp_v_dynamic(self, x: Tensor):
        v_partial_tr = self._get_tmp_tr_for_cache(x)
        v_partial = self._get_tmp_for_cache(x)

        gemm_async(
            1.0,
            notrans,
            self.w_v.value,
            notrans,
            x,
            0.0,
            v_partial_tr,
            1,
            0,
            redux=self.redux,
        )

        transpose_async(1.0, v_partial_tr, v_partial, 1)

        if self.in_proj_bias_v is not None:
            add_fiber_inplace_async(1, self.in_proj_bias_v.value, 1, v_partial, 0, 1)

        return v_partial

    def _add_positional_bias_async(self):
        query_length, key_length = self.a.value.shape[:2]
        context_position = np.arange(query_length, dtype=np.int32)[:, None]
        memory_position = np.arange(key_length, dtype=np.int32)[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket_np_value = relative_position_bucket_numpy(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=32,
            max_distance=128,
        )
        relative_position_bucket_np_value = relative_position_bucket_np_value[None, ...]
        if self.a.value.shape[2] > 1:
            relative_position_bucket_np_value = (
                relative_position_bucket_np_value.repeat(self.a.value.shape[2], axis=0)
            )

        self.relative_position_bucket_nnt = nntc.from_array(
            relative_position_bucket_np_value.T,
            self.relative_bias.value.basetile_shape[:3],
        )

        embedding_async(
            self.relative_position_bucket_nnt,
            self.relative_bias_embedding.value,
            self.relative_bias.value,
            axis=3,  # so result will batch self.a.shape
        )

        # print("INTERNAL TYPE: ", type(self.relative_bias.value), nntc.to_numpy(self.relative_bias.value))

        add_async(1.0, self.a.value, 1.0, self.relative_bias.value, self.a.value)

    def _add_positional_bias_backward(self):
        copy_async(self.a.grad, self.relative_bias.grad)

        embedding_backward_async(
            self.relative_position_bucket_nnt,
            self.relative_bias.grad,
            self.relative_bias_embedding.grad,
            axis=3,
        )

    def _forward_attn_async(self):
        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K, Q)
        # single batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (head_size, n_seq, batch=n_batch, batch=n_head) into
        # (n_seq, n_seq, batch=n_batch, batch=n_head)
        gemm_async(
            1.0 / self.head_size**0.5 if self.scale_attn else 1.0,
            trans,
            self.k.value,
            notrans,
            self.q.value,
            0.0,
            self.a.value,
            1,
            2,
            redux=self.redux,
        )
        if self.has_relative_bias:
            self._add_positional_bias_async()

        clear_async(self.a_maxsumexp)
        # Q and K can be offloaded from GPU
        self.q.value.wont_use()
        self.k.value.wont_use()
        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.mask:
            mask_scalar_async(self.mask, self.val, self.a.value, 2)
            # self.mask.wont_use()
            # add_async(-1e30, self.mask, 1.0, self.a.value, self.a.value)
        # Calculate max and sumexp along axis
        maxsumexp_async(self.a.value, self.a_maxsumexp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(self.a_maxsumexp, 1.0, self.a.value, 0)
        # A_maxsumexp can be deleted
        # self.a_maxsumexp.wont_use()
        self.a_maxsumexp.invalidate_submit()
        # Apply value tensor
        # B = einsum('jklb,kmlb->jmlb', V, A)
        # batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (n_seq, n_seq, batch=n_batch, batch=n_head) into
        # (head_size, n_seq, batch=n_batch, batch=n_head)
        gemm_async(
            1.0,
            notrans,
            self.v.value,
            notrans,
            self.a.value,
            0.0,
            self.b.value,
            1,
            2,
            redux=self.redux,
        )
        # V and A can be offloaded from GPU
        self.v.value.wont_use()
        self.a.value.wont_use()
        # Accumulate result from all the heads
        # rotate axes (head_size, n_seq, n_batch, n_head) into
        # (n_head, head_size, n_seq, n_batch) and then
        transpose_async(1.0, self.b.value, self.b_transposed.value, 3)
        # Y = einsum('jkl,klmn->jmn', W, B_transposed)
        # gemm (n_emb, n_head, head_size) by
        # (n_head, head_size, n_seq, n_batch) into (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w.value,
            notrans,
            self.b_transposed.value,
            0.0,
            self.y.value,
            2,
            0,
            redux=self.redux,
        )
        # W, B and B_transposed can be offloaded from GPU
        self.w.value.wont_use()
        # self.b.value.wont_use()
        self.b.value.invalidate_submit()
        self.b_transposed.value.wont_use()
        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_inplace_async(
                1.0, self.out_proj_bias.value, 1.0, self.y.value, 0, 0
            )
            self.out_proj_bias.value.wont_use()
        self.y.value.wont_use()

    def _forward_attn_dynamic(self, q, k, v):
        a_tmp = nntc.empty(
            (k.shape[1],) + (q.shape[1],) + tuple(k.shape[2:]),
            dtype=type(q),
            basetile_shape=(k.shape[1],)
            + (q.shape[1],)
            + (k.shape[2],)
            + (self.n_head_tile,),
        )  # (n_seq, n_seq, batch=n_batch, batch=n_head)
        a_maxsumexp_tmp = nntc.empty(
            (2,) + tuple(a_tmp.shape[1:]),
            dtype=type(q),
            basetile_shape=(2,) + tuple(a_tmp.shape[1:-1]) + (self.n_head_tile,),
        )
        b_tmp = nntc.empty(
            q.shape,
            dtype=type(q),
            basetile_shape=tuple(q.shape[:-1]) + (self.n_head_tile,),
        )  # (head_size, n_seq, n_batch, n_head)
        b_tr_tmp = nntc.empty(
            (self.n_head, self.head_size) + tuple(q.shape[1:3]),
            dtype=type(q),
            basetile_shape=(self.n_head_tile, self.head_size) + tuple(q.shape[1:3]),
        )  # (n_head, head_size, n_seq, n_batch)
        self.y_tensor = nntc.empty(
            (self.n_emb,) + tuple(q.shape[1:3]),
            dtype=type(q),
            basetile_shape=(self.n_emb_tile,) + tuple(q.shape[1:3]),
        )  # (n_emb, n_seq, n_batch)
        y_tensor = self.y_tensor
        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K, Q)
        # single batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (head_size, n_seq, batch=n_batch, batch=n_head) into
        # (n_seq, n_seq, batch=n_batch, batch=n_head)
        gemm_async(
            1.0 / self.head_size**0.5,
            trans,
            k,
            notrans,
            q,
            0.0,
            a_tmp,
            1,
            2,
            redux=self.redux,
        )

        clear_async(a_maxsumexp_tmp)
        # Q and K can be offloaded from GPU
        q.wont_use()
        k.wont_use()

        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.mask:
            mask_tmp = nntc.empty(
                a_tmp.shape[:2],
                basetile_shape=a_tmp.basetile_shape[:2],
                dtype=Tensor_bool,
            )
            copy_intersection_async(
                self.mask, [0, 0], mask_tmp, [0, k.shape[1] - q.shape[1]]
            )
            mask_scalar_async(mask_tmp, self.val, a_tmp, 2)

        # Calculate max and sumexp along axis
        maxsumexp_async(a_tmp, a_maxsumexp_tmp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(a_maxsumexp_tmp, 1.0, a_tmp, 0)
        # A_maxsumexp can be deleted
        # a_maxsumexp_tmp.invalidate_submit()

        # Apply value tensor
        # B = einsum('jklb,kmlb->jmlb', V, A)
        # batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (n_seq, n_seq, batch=n_batch, batch=n_head) into
        # (head_size, n_seq, batch=n_batch, batch=n_head)
        gemm_async(1.0, notrans, v, notrans, a_tmp, 0.0, b_tmp, 1, 2, redux=self.redux)
        # V and A can be offloaded from GPU
        v.wont_use()
        a_tmp.wont_use()

        # Accumulate result from all the heads
        # rotate axes (head_size, n_seq, n_batch, n_head) into
        # (n_head, head_size, n_seq, n_batch) and then
        transpose_async(1.0, b_tmp, b_tr_tmp, 3)
        # Y = einsum('jkl,klmn->jmn', W, B_transposed)
        # gemm (n_emb, n_head, head_size) by
        # (n_head, head_size, n_seq, n_batch) into (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w.value,
            notrans,
            b_tr_tmp,
            0.0,
            y_tensor,
            2,
            0,
            redux=self.redux,
        )
        # W, B and B_transposed can be offloaded from GPU
        self.w.value.wont_use()
        # self.b.value.wont_use()
        # b_tmp.invalidate_submit()
        b_tr_tmp.wont_use()

        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_inplace_async(1.0, self.out_proj_bias.value, 1.0, y_tensor, 0, 0)
            self.out_proj_bias.value.wont_use()
        return y_tensor

    # Forward propagation of the attention layer
    def forward_async(self, effective_size=None):
        # Compute query, key and value tensors
        self._forward_mlp_q_async()
        self._forward_mlp_k_async()
        self._forward_mlp_v_async()

        # compute attention and weight result
        self._forward_attn_async()

    def forward_dynamic(self, x: TensorMoments, kv_cache: Optional[KVCache] = None):
        if (kv_cache is not None) and (
            x.value.shape[1] + len(kv_cache) > self.x_v.value.shape[1]
        ):  # noqa: E501
            raise Exception(
                "Overload internal state: "
                f"try add {x.value.shape[1]} "
                f"to {len(kv_cache)}, max: {self.x_v.value.shape[1]}. "
            )

        # Compute query, key and value tensors
        q_partial = self._forward_mlp_q_dynamic(x.value)
        k_partial = self._forward_mlp_k_dynamic(x.value)
        v_partial = self._forward_mlp_v_dynamic(x.value)

        if kv_cache is not None:
            kv_cache.append(k_partial, v_partial)
            k = kv_cache.k_partial
            v = kv_cache.v_partial
        else:
            k = k_partial
            v = v_partial

        # compute attention and weight result
        y_tensor = self._forward_attn_dynamic(q_partial, k, v)
        return TensorMoments(y_tensor, None, False), kv_cache

    # Backward propagation of the linear layer
    def backward_async(self):
        # Apply backward of bias if needed
        if self.out_proj_bias is not None:
            if self.out_proj_bias.grad_required:
                sum_fiber_async(
                    1.0,
                    self.y.grad,
                    1.0,
                    self.out_proj_bias.grad,
                    0,
                    0,
                    redux=self.redux,
                )
                self.out_proj_bias.grad.wont_use()
        # Backward for Y = einsum('jkl,klmn->jmn', W, B_transposed)
        if self.w.grad_required:
            # dW += einsum('jmn,klmn->jkl', dY, B_transposed)
            gemm_async(
                1.0,
                notrans,
                self.y.grad,
                trans,
                self.b_transposed.value,
                1.0,
                self.w.grad,
                2,
                0,
                redux=self.redux,
            )
        # B_transposed can be deleted
        # self.b_transposed.value.wont_use()
        self.b_transposed.value.invalidate_submit()
        self.w.grad.wont_use()
        if self.b_transposed.grad_required:
            # dB_transposed = einsum('jkl,jmn->klmn', W, dY)
            gemm_async(
                1.0,
                trans,
                self.w.value,
                notrans,
                self.y.grad,
                0.0,
                self.b_transposed.grad,
                1,
                0,
                redux=self.redux,
            )
        # W can be offloaded from GPU
        self.w.value.wont_use()
        # dY can be offloaded from GPU
        self.y.grad.wont_use()
        # Backward for axes rotation
        if self.b.grad_required:
            # rotate axes (n_head, head_size, n_seq, n_batch) into
            # (head_size, n_seq, n_batch, n_head) and then
            transpose_async(1.0, self.b_transposed.grad, self.b.grad, 1)
        # self.b_transposed.grad.wont_use()
        self.b_transposed.grad.invalidate_submit()
        # Backward for B = einsum('jklb,kmlb->jmlb', V, A)
        if self.a.grad_required:
            # dA = einsum('jklb,jmlb->kmlb', V, dB)
            gemm_async(
                1.0,
                trans,
                self.v.value,
                notrans,
                self.b.grad,
                0.0,
                self.a.grad,
                1,
                2,
                redux=self.redux,
            )
        # V can be deleted
        # self.v.value.wont_use()
        self.v.value.invalidate_submit()
        if self.v.grad_required:
            # dV = einsum('jmlb,kmlb->jklb', dB, A)
            gemm_async(
                1.0,
                notrans,
                self.b.grad,
                trans,
                self.a.value,
                0.0,
                self.v.grad,
                1,
                2,
                redux=self.redux,
            )
        # dB can be deleted
        # self.b.grad.wont_use()
        self.b.grad.invalidate_submit()
        # Backward for A = softmax(A, axis=0)
        if self.a.grad_required:
            # A_sumprod_slice = einsum('kmlb,kmlb->mlb', A, dA)
            sumprod_slice_async(
                1.0,
                self.a.value,
                self.a.grad,
                0.0,
                self.a_sumprod_slice,
                0,
                redux=self.redux,
            )
            # dA += -bias('kmlb,mlb->kmlb', dA, A_sumprod_slice)
            add_slice_inplace_async(-1.0, self.a_sumprod_slice, 1.0, self.a.grad, 0)
            # A_sumprod_slice can be deleted
            # self.a_sumprod_slice.wont_use()
            self.a_sumprod_slice.invalidate_submit()
            # dA *= A
            prod_inplace_async(self.a.value, self.a.grad)

            if self.has_relative_bias:
                self._add_positional_bias_backward()

        # A can be deleted
        # self.a.value.wont_use()
        self.a.value.invalidate_submit()
        # Backward for mask if needed
        if self.mask:
            mask_scalar_async(self.mask, 0, self.a.grad, 2)
            self.mask.wont_use()
        # Backward for:
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K, Q)
        if self.k.grad_required:
            # dK = 1.0/sqrt(head_size) * einsum('jmlb,kmlb->jklb', Q, dA)
            gemm_async(
                1.0 / self.head_size**0.5 if self.scale_attn else 1.0,
                notrans,
                self.q.value,
                trans,
                self.a.grad,
                0.0,
                self.k.grad,
                1,
                2,
                redux=self.redux,
            )
        # Q can be deleted
        # self.q.value.wont_use()
        self.q.value.invalidate_submit()
        if self.q.grad_required:
            # dQ = 1.0/sqrt(head_size) * einsum('jklb,kmlb->jmlb', K, dA)
            gemm_async(
                1.0 / self.head_size**0.5 if self.scale_attn else 1.0,
                notrans,
                self.k.value,
                notrans,
                self.a.grad,
                0.0,
                self.q.grad,
                1,
                2,
                redux=self.redux,
            )
        # K can be deleted
        # self.k.value.wont_use()
        self.k.value.invalidate_submit()
        # dA can be deleted
        # self.a.grad.wont_use()
        self.a.grad.invalidate_submit()
        # Backward for bias of V
        if self.in_proj_bias_v is not None:
            if self.in_proj_bias_v.grad_required:
                sum_fiber_async(
                    1,
                    self.v.grad,
                    1,
                    self.in_proj_bias_v.grad,
                    0,
                    1,
                    redux=self.redux,
                )
                self.in_proj_bias_v.grad.wont_use()
        # Backward for axes rotation (V_transposed->V)
        if self.v_transposed.grad_required:
            # Rotate axes (head_size, n_seq, n_batch, n_head) into
            # (n_head, head_size, n_seq, n_batch)
            transpose_async(1.0, self.v.grad, self.v_transposed.grad, 3)
        # dV can be deleted
        # self.v.grad.wont_use()
        self.v.grad.invalidate_submit()
        # Backward for V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V)
        if self.x_v.grad_required:
            # dX_V += einsum('jkl,jkmn->lmn', W_V, dV_transposed)
            gemm_async(
                1.0,
                trans,
                self.w_v.value,
                notrans,
                self.v_transposed.grad,
                1.0,
                self.x_v.grad,
                2,
                0,
                redux=self.redux,
            )
        # W_V can be offloaded from GPU
        self.w_v.value.wont_use()
        # dX_V can be offloaded from GPU
        self.x_v.grad.wont_use()
        if self.w_v.grad_required:
            # dW_V += einsum('jkmn,lmn->jkl', dV_transposed, X_V)
            gemm_async(
                1.0,
                notrans,
                self.v_transposed.grad,
                trans,
                self.x_v.value,
                1.0,
                self.w_v.grad,
                2,
                0,
                redux=self.redux,
            )
        # dW_V can be offloaded from GPU
        self.w_v.grad.wont_use()
        # X_V can be offloaded from GPU
        self.x_v.value.wont_use()
        # dV_transposed can be deleted
        # self.v_transposed.grad.wont_use()
        self.v_transposed.grad.invalidate_submit()
        # Backward for bias of K
        if self.in_proj_bias_k is not None:
            if self.in_proj_bias_k.grad_required:
                sum_fiber_async(
                    1,
                    self.k.grad,
                    1,
                    self.in_proj_bias_k.grad,
                    0,
                    1,
                    redux=self.redux,
                )
                self.in_proj_bias_k.grad.wont_use()
        # Backward for axes rotation (K_transposed->K)
        if self.k_transposed.grad_required:
            # Rotate axes (head_size, n_seq, n_batch, n_head) into
            # (n_head, head_size, n_seq, n_batch)
            transpose_async(1.0, self.k.grad, self.k_transposed.grad, 3)
        # dK can be deleted
        # self.k.grad.wont_use()
        self.k.grad.invalidate_submit()
        # Backward for K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K)
        if self.x_k.grad_required:
            # dX_K += einsum('jkl,jkmn->lmn', W_K, dK_transposed)
            gemm_async(
                1.0,
                trans,
                self.w_k.value,
                notrans,
                self.k_transposed.grad,
                1.0,
                self.x_k.grad,
                2,
                0,
                redux=self.redux,
            )
        # W_K can be offloaded from GPU
        self.w_k.value.wont_use()
        # dX_K can be offloaded from GPU
        self.x_k.grad.wont_use()
        if self.w_k.grad_required:
            # dW_K += einsum('jkmn,lmn->jkl', dK_transposed, X_K)
            gemm_async(
                1.0,
                notrans,
                self.k_transposed.grad,
                trans,
                self.x_k.value,
                1.0,
                self.w_k.grad,
                2,
                0,
                redux=self.redux,
            )
        # dW_K can be offloaded from GPU
        self.w_k.grad.wont_use()
        # X_K can be offloaded from GPU
        self.x_k.value.wont_use()
        # dK_transposed can be deleted
        # self.k_transposed.grad.wont_use()
        self.k_transposed.grad.invalidate_submit()
        # Backward for bias of Q
        if self.in_proj_bias_q is not None:
            if self.in_proj_bias_q.grad_required:
                sum_fiber_async(
                    1,
                    self.q.grad,
                    1,
                    self.in_proj_bias_q.grad,
                    0,
                    1,
                    redux=self.redux,
                )
                self.in_proj_bias_q.grad.wont_use()
        # Backward for axes rotation (Q_transposed->Q)
        if self.q_transposed.grad_required:
            # Rotate axes (head_size, n_seq, n_batch, n_head) into
            # (n_head, head_size, n_seq, n_batch)
            transpose_async(1.0, self.q.grad, self.q_transposed.grad, 3)
        # dQ can be deleted
        # self.q.grad.wont_use()
        self.q.grad.invalidate_submit()
        # Backward for Q_transposed = einsum('jkl,lmn->jkmn', W_Q, X_Q)
        if self.x_q.grad_required:
            # dX_Q += einsum('jkl,jkmn->lmn', W_Q, dQ_transposed)
            gemm_async(
                1.0,
                trans,
                self.w_q.value,
                notrans,
                self.q_transposed.grad,
                1.0,
                self.x_q.grad,
                2,
                0,
                redux=self.redux,
            )
            self.x_q.grad.wont_use()
        # W_Q can be offloaded from GPU
        self.w_q.value.wont_use()
        # dX_Q can be offloaded from GPU
        self.x_q.grad.wont_use()
        if self.w_q.grad_required:
            # dW_Q += einsum('jkmn,lmn->jkl', dQ_transposed, X_Q)
            gemm_async(
                1.0,
                notrans,
                self.q_transposed.grad,
                trans,
                self.x_q.value,
                1.0,
                self.w_q.grad,
                2,
                0,
                redux=self.redux,
            )
        # dW_Q can be offloaded from GPU
        self.w_q.grad.wont_use()
        # X_Q can be offloaded from GPU
        self.x_q.value.wont_use()
        # dQ_transposed can be deleted
        # self.q_transposed.grad.wont_use()
        self.q_transposed.grad.invalidate_submit()
