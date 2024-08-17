# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/llama_attention.py
# LlamaAttention layer of NNTile Python package
#
# @version 1.1.0

import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig as LlamaConfig_torch)

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, Tensor_bool, TensorMoments, TensorOrNone, TensorTraits,
    add_fiber_async, add_slice_async, clear_async, copy_intersection_async,
    flash_maxsumexp_async, flash_softmax_gemm_async,
    flash_softmax_gemm_backward_async, gemm_async, mask_scalar_async,
    maxsumexp_async, notrans, prod_inplace_async, rope_async,
    rope_backward_async, softmax_inplace_async, sum_fiber_async,
    sum_slice_async, sumprod_slice_async, to_numpy, trans, transpose_async)

from ..model.llama_config import LlamaConfigNNTile


# LLaMa Multi-Query Self-Attention with Rotary Embeddings
# Inputs:
#  x: (n_emb, n_seq, n_batch) tensor
# Output:
#  y: (n_emb, n_seq, n_batch) tensor
class LlamaAttention(BaseLayer):
    x: TensorMoments
    y: TensorMoments
    w_q: TensorMoments
    w_k: TensorMoments
    w_v: TensorMoments
    w: TensorMoments
    q_transposed: TensorMoments
    q: TensorMoments
    q_rope: TensorMoments
    k_transposed: TensorMoments
    k: TensorMoments
    k_rope: TensorMoments
    k_rep: TensorMoments
    v_transposed: TensorMoments
    v: TensorMoments
    v_rep: TensorMoments
    a: TensorMoments
    a_maxsumexp: Tensor
    a_sumprod_slice: Tensor
    b: TensorMoments
    b_transposed: TensorMoments
    sin: Tensor
    cos: Tensor
    mask: TensorOrNone
    n_emb: int
    n_emb_kv: int
    n_seq: int
    n_batch: int
    n_head: int
    n_head_kv: int
    kv_group_size: int
    head_size: int
    flash_attention: bool

    # Construct attention layer with all the provided data
    def __init__(
        self,
        x: TensorMoments,
        y: TensorMoments,
        w_q: TensorMoments,
        w_k: TensorMoments,
        w_v: TensorMoments,
        w: TensorMoments,
        q_transposed: TensorMoments,
        q: TensorMoments,
        q_rope: TensorMoments,
        k_transposed: TensorMoments,
        k: TensorMoments,
        k_rope: TensorMoments,
        k_rep: TensorMoments,
        v_transposed: TensorMoments,
        v: TensorMoments,
        v_rep: TensorMoments,
        a: TensorMoments,
        a_maxsumexp: Tensor,
        a_sumprod_slice: Tensor,
        b: TensorMoments,
        b_transposed: TensorMoments,
        sin: Tensor,
        cos: Tensor,
        in_proj_bias_q: TensorMoments,
        in_proj_bias_k: TensorMoments,
        in_proj_bias_v: TensorMoments,
        out_proj_bias: TensorMoments,
        mask: TensorOrNone = None,
        flash_attention: bool = True,
        redux: bool = False,
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
        # Redirect to BaseClass initialization
        super().__init__(
            [x],
            [y],
            [w_q, w_k, w_v] + qkv_bias_list + [w] + bias_list_out_proj,
            [
                q_transposed,
                q,
                q_rope,
                k_transposed,
                k,
                k_rope,
                k_rep,
                v_transposed,
                v,
                v_rep,
                a,
                a_maxsumexp,
                a_sumprod_slice,
                b,
                b_transposed,
                sin,
                cos
            ],
        )
        if mask is not None:
            self.temporaries.append(mask)
        self.x = x
        self.x.grad.set_reduction_add()
        # Aliases
        self.x_q = x
        self.x_k = x
        self.x_v = x
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
        self.q_rope = q_rope
        self.q.grad.set_reduction_add()
        self.k_transposed = k_transposed
        self.k_transposed.value.set_reduction_add()
        self.k = k
        self.k_rope = k_rope
        self.k.grad.set_reduction_add()
        self.k_rep = k_rep
        self.k_rep.grad.set_reduction_add()
        self.v_transposed = v_transposed
        self.v_transposed.value.set_reduction_add()
        self.v = v
        self.v.grad.set_reduction_add()
        self.v_rep = v_rep
        self.v_rep.grad.set_reduction_add()
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
        self.sin = sin
        self.cos = cos
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        self.out_proj_bias = out_proj_bias
        self.n_head = w_q.value.shape[0] * w_q.value.shape[1]
        self.n_head_kv = w_k.value.shape[0]
        self.kv_group_size = self.n_head // self.n_head_kv
        if self.n_head != self.kv_group_size * self.n_head_kv:
            raise ValueError("Wrong number of heads of W_k")
        n_emb, n_seq, n_batch = x.value.shape
        head_size = n_emb // self.n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * self.n_head:
            raise RuntimeError
        self.n_emb = n_emb
        self.n_emb_kv = head_size * self.n_head_kv
        self.n_seq = n_seq
        self.n_batch = n_batch
        self.head_size = head_size
        self.mask = mask
        if mask:
            self.val = -np.float32(np.inf)
        if redux:
            self.redux = 1
        else:
            self.redux = 0
        self.flash_attention = flash_attention

        # need to fill with valid values for dynamic api usage
        clear_async(self.q.value)
        clear_async(self.k.value)
        clear_async(self.v.value)

        self.reset_cache()

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple(
        x: TensorMoments,
        n_head: int,
        n_head_tile: int,
        n_head_kv: int,
        position_ids: np.ndarray,
        theta: float,
        next_tag: int,
        bias: bool = False,
        mask: np.ndarray = None,
        flash_attention: bool = True,
        redux: bool = False,
    ):
        # Get sizes
        n_emb, n_seq, n_batch = x.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x.value.basetile_shape
        n_emb_k = n_emb_v = n_emb
        n_emb_k_tile = n_emb_v_tile = n_emb_tile
        head_size = n_emb // n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * n_head:
            raise RuntimeError
        # KV group is NOT split into tiles
        kv_group_size = n_head // n_head_kv
        kv_group_size_tile = kv_group_size
        # n_head_kv is split into tiles in such a way, that
        # n_head_kv_tile*kv_group_size_tile=n_head_tile
        if n_head_tile % kv_group_size != 0:
            raise ValueError("Invalid value of n_head_kv")
        n_head_kv_tile = n_head_tile // kv_group_size
        # Fixed for now
        head_size_tile = head_size
        # Define shape of each tensor
        w_q_shape = [kv_group_size, n_head_kv, head_size, n_emb]
        w_k_shape = [n_head_kv, head_size, n_emb_k]
        w_v_shape = [n_head_kv, head_size, n_emb_v]
        w_shape = [n_emb, kv_group_size, n_head_kv, head_size]
        q_transposed_shape = [
            kv_group_size,
            n_head_kv,
            head_size,
            n_seq,
            n_batch,
        ]
        q_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
        q_rope_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
        k_transposed_shape = [n_head_kv, head_size, n_seq, n_batch]
        k_shape = [head_size, n_seq, n_batch, n_head_kv]
        k_rope_shape = [head_size, n_seq, n_batch, n_head_kv]
        k_rep_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
        v_transposed_shape = [n_head_kv, head_size, n_seq, n_batch]
        v_shape = [head_size, n_seq, n_batch, n_head_kv]
        v_rep_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
        a_shape = [n_seq, n_seq, n_batch, kv_group_size, n_head_kv]
        a_maxsumexp_shape = [2, n_seq, n_batch, kv_group_size, n_head_kv]
        a_sumprod_slice_shape = [n_seq, n_batch, kv_group_size, n_head_kv]
        b_shape = [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
        b_transposed_shape = [
            kv_group_size,
            n_head_kv,
            head_size,
            n_seq,
            n_batch,
        ]
        cos_shape = [head_size // 2, n_seq, n_batch]
        sin_shape = [head_size // 2, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [
            kv_group_size_tile,
            n_head_kv_tile,
            head_size_tile,
            n_emb_tile,
        ]
        w_k_basetile = [n_head_kv_tile, head_size_tile, n_emb_k_tile]
        w_v_basetile = [n_head_kv_tile, head_size_tile, n_emb_v_tile]
        w_basetile = [
            n_emb_tile,
            kv_group_size_tile,
            n_head_kv_tile,
            head_size_tile,
        ]
        q_transposed_basetile = [
            kv_group_size_tile,
            n_head_kv_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        q_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        q_rope_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        k_transposed_basetile = [
            n_head_kv_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        k_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_kv_tile]
        k_rope_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_kv_tile,
        ]
        k_rep_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        v_transposed_basetile = [
            n_head_kv_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        v_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head_kv_tile]
        v_rep_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        a_basetile = [
            n_seq_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        a_maxsumexp_basetile = [
            2,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        a_sumprod_slice_basetile = [
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        b_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            kv_group_size_tile,
            n_head_kv_tile,
        ]
        b_transposed_basetile = [
            kv_group_size_tile,
            n_head_kv_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
        ]
        cos_basetile = [head_size_tile // 2, n_seq_tile, n_batch_tile]
        sin_basetile = [head_size_tile // 2, n_seq_tile, n_batch_tile]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        w_traits = TensorTraits(w_shape, w_basetile)
        q_transposed_traits = TensorTraits(
            q_transposed_shape, q_transposed_basetile
        )
        q_traits = TensorTraits(q_shape, q_basetile)
        q_rope_traits = TensorTraits(q_rope_shape, q_rope_basetile)
        k_transposed_traits = TensorTraits(
            k_transposed_shape, k_transposed_basetile
        )
        k_traits = TensorTraits(k_shape, k_basetile)
        k_rope_traits = TensorTraits(k_rope_shape, k_rope_basetile)
        k_rep_traits = TensorTraits(k_rep_shape, k_rep_basetile)
        v_transposed_traits = TensorTraits(
            v_transposed_shape, v_transposed_basetile
        )
        v_traits = TensorTraits(v_shape, v_basetile)
        v_rep_traits = TensorTraits(v_rep_shape, v_rep_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(
            a_maxsumexp_shape, a_maxsumexp_basetile
        )
        a_sumprod_slice_traits = TensorTraits(
            a_sumprod_slice_shape, a_sumprod_slice_basetile
        )
        b_traits = TensorTraits(b_shape, b_basetile)
        b_transposed_traits = TensorTraits(
            b_transposed_shape, b_transposed_basetile
        )
        cos_traits = TensorTraits(cos_shape, cos_basetile)
        sin_traits = TensorTraits(sin_shape, sin_basetile)
        # TODO change distribution
        w_q_distr = [0] * w_q_traits.grid.nelems
        w_k_distr = [0] * w_k_traits.grid.nelems
        w_v_distr = [0] * w_v_traits.grid.nelems
        w_distr = [0] * w_traits.grid.nelems
        q_transposed_distr = [0] * q_transposed_traits.grid.nelems
        q_distr = [0] * q_traits.grid.nelems
        q_rope_distr = [0] * q_rope_traits.grid.nelems
        k_transposed_distr = [0] * k_transposed_traits.grid.nelems
        k_distr = [0] * k_traits.grid.nelems
        k_rope_distr = [0] * k_rope_traits.grid.nelems
        k_rep_distr = [0] * k_rep_traits.grid.nelems
        v_transposed_distr = [0] * v_transposed_traits.grid.nelems
        v_distr = [0] * v_traits.grid.nelems
        v_rep_distr = [0] * v_rep_traits.grid.nelems
        a_distr = [0] * a_traits.grid.nelems
        a_maxsumexp_distr = [0] * a_maxsumexp_traits.grid.nelems
        a_sumprod_slice_distr = [0] * a_sumprod_slice_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        b_transposed_distr = [0] * b_transposed_traits.grid.nelems
        if bias:
            in_proj_bias_q_traits = TensorTraits(
                [head_size, kv_group_size, n_head_kv],
                [head_size_tile, kv_group_size_tile, n_head_kv_tile],
            )
            in_proj_bias_q_distr = [0] * in_proj_bias_q_traits.grid.nelems
            in_proj_bias_kv_traits = TensorTraits(
                [head_size, n_head_kv], [head_size_tile, n_head_kv_tile]
            )
            in_proj_bias_kv_distr = [0] * in_proj_bias_kv_traits.grid.nelems
        cos_distr = [0] * cos_traits.grid.nelems
        sin_distr = [0] * sin_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_value.next_tag
        w_q_grad = type(x.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_grad.next_tag
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if bias:
            in_proj_bias_q_value = type(x.value)(
                in_proj_bias_q_traits, in_proj_bias_q_distr, next_tag
            )
            next_tag = in_proj_bias_q_value.next_tag
            in_proj_bias_q_grad = type(x.value)(
                in_proj_bias_q_traits, in_proj_bias_q_distr, next_tag
            )
            next_tag = in_proj_bias_q_grad.next_tag
            bias_inproj_q = TensorMoments(
                in_proj_bias_q_value, in_proj_bias_q_grad, True
            )
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_value.next_tag
        w_k_grad = type(x.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_grad.next_tag
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if bias:
            in_proj_bias_k_value = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr, next_tag
            )
            next_tag = in_proj_bias_k_value.next_tag
            in_proj_bias_k_grad = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr, next_tag
            )
            next_tag = in_proj_bias_k_grad.next_tag
            bias_inproj_k = TensorMoments(
                in_proj_bias_k_value, in_proj_bias_k_grad, True
            )
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_value.next_tag
        w_v_grad = type(x.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_grad.next_tag
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if bias:
            in_proj_bias_v_value = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr, next_tag
            )
            next_tag = in_proj_bias_v_value.next_tag
            in_proj_bias_v_grad = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr, next_tag
            )
            next_tag = in_proj_bias_v_grad.next_tag
            bias_inproj_v = TensorMoments(
                in_proj_bias_v_value, in_proj_bias_v_grad, True
            )
        else:
            bias_inproj_v = None
        # w
        w_value = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        w_grad = type(x.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        w = TensorMoments(w_value, w_grad, True)
        # q_transposed
        q_transposed_value = type(x.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_value.next_tag
        q_transposed_grad = type(x.value)(
            q_transposed_traits, q_transposed_distr, next_tag
        )
        next_tag = q_transposed_grad.next_tag
        q_transposed = TensorMoments(
            q_transposed_value, q_transposed_grad, True
        )
        # q
        q_value = type(x.value)(q_traits, q_distr, next_tag)
        next_tag = q_value.next_tag
        q_grad = type(x.value)(q_traits, q_distr, next_tag)
        next_tag = q_grad.next_tag
        q = TensorMoments(q_value, q_grad, True)
        # q_rope
        q_rope_value = type(x.value)(q_rope_traits, q_rope_distr, next_tag)
        next_tag = q_rope_value.next_tag
        q_rope_grad = type(x.value)(q_rope_traits, q_rope_distr, next_tag)
        next_tag = q_rope_grad.next_tag
        q_rope = TensorMoments(q_rope_value, q_rope_grad, True)
        # k_transposed
        k_transposed_value = type(x.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_value.next_tag
        k_transposed_grad = type(x.value)(
            k_transposed_traits, k_transposed_distr, next_tag
        )
        next_tag = k_transposed_grad.next_tag
        k_transposed = TensorMoments(
            k_transposed_value, k_transposed_grad, True
        )
        # k
        k_value = type(x.value)(k_traits, k_distr, next_tag)
        next_tag = k_value.next_tag
        k_grad = type(x.value)(k_traits, k_distr, next_tag)
        next_tag = k_grad.next_tag
        k = TensorMoments(k_value, k_grad, True)
        # k_rope
        k_rope_value = type(x.value)(k_rope_traits, k_rope_distr, next_tag)
        next_tag = k_rope_value.next_tag
        k_rope_grad = type(x.value)(k_rope_traits, k_rope_distr, next_tag)
        next_tag = k_rope_grad.next_tag
        k_rope = TensorMoments(k_rope_value, k_rope_grad, True)
        # k_rep
        k_rep_value = type(x.value)(k_rep_traits, k_rep_distr, next_tag)
        next_tag = k_rep_value.next_tag
        k_rep_grad = type(x.value)(k_rep_traits, k_rep_distr, next_tag)
        next_tag = k_rep_grad.next_tag
        k_rep = TensorMoments(k_rep_value, k_rep_grad, True)
        # v_transposed
        v_transposed_value = type(x.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_value.next_tag
        v_transposed_grad = type(x.value)(
            v_transposed_traits, v_transposed_distr, next_tag
        )
        next_tag = v_transposed_grad.next_tag
        v_transposed = TensorMoments(
            v_transposed_value, v_transposed_grad, True
        )
        # v
        v_value = type(x.value)(v_traits, v_distr, next_tag)
        next_tag = v_value.next_tag
        v_grad = type(x.value)(v_traits, v_distr, next_tag)
        next_tag = v_grad.next_tag
        v = TensorMoments(v_value, v_grad, True)
        # v_rep
        v_rep_value = type(x.value)(v_rep_traits, v_rep_distr, next_tag)
        next_tag = v_rep_value.next_tag
        v_rep_grad = type(x.value)(v_rep_traits, v_rep_distr, next_tag)
        next_tag = v_rep_grad.next_tag
        v_rep = TensorMoments(v_rep_value, v_rep_grad, True)
        # a
        a_value = type(x.value)(a_traits, a_distr, next_tag)
        next_tag = a_value.next_tag
        a_grad = type(x.value)(a_traits, a_distr, next_tag)
        next_tag = a_grad.next_tag
        a = TensorMoments(a_value, a_grad, True)
        # a_maxsumexp
        a_maxsumexp = type(x.value)(
            a_maxsumexp_traits, a_maxsumexp_distr, next_tag
        )
        next_tag = a_maxsumexp.next_tag
        # a_sumprod_slice
        a_sumprod_slice = type(x.value)(
            a_sumprod_slice_traits, a_sumprod_slice_distr, next_tag
        )
        next_tag = a_sumprod_slice.next_tag
        # b
        b_value = type(x.value)(b_traits, b_distr, next_tag)
        next_tag = b_value.next_tag
        b_grad = type(x.value)(b_traits, b_distr, next_tag)
        next_tag = b_grad.next_tag
        b = TensorMoments(b_value, b_grad, True)
        # b_transposed
        b_transposed_value = type(x.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_value.next_tag
        b_transposed_grad = type(x.value)(
            b_transposed_traits, b_transposed_distr, next_tag
        )
        next_tag = b_transposed_grad.next_tag
        b_transposed = TensorMoments(
            b_transposed_value, b_transposed_grad, True
        )
        cos = type(x.value)(cos_traits, cos_distr, next_tag)
        next_tag = cos.next_tag

        sin = type(x.value)(sin_traits, sin_distr, next_tag)
        next_tag = sin.next_tag
        # Allocate tensors for bias for q, k, v and output projection
        if bias:
            out_proj_bias_traits = TensorTraits([n_emb], [n_emb_tile])
            out_proj_bias_distr = [0] * out_proj_bias_traits.grid.nelems
            out_proj_bias_value = type(x.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_value.next_tag
            out_proj_bias_grad = type(x.value)(
                out_proj_bias_traits, out_proj_bias_distr, next_tag
            )
            next_tag = out_proj_bias_grad.next_tag
            out_proj_bias = TensorMoments(
                out_proj_bias_value, out_proj_bias_grad, True
            )
        else:
            out_proj_bias = None
        # Allocate tensor for output y
        y_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        y_value = type(x.value)(y_traits, x.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x.value)(y_traits, x.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)

        # Fill sin, cos tensors:
        inv_freq = 1.0 / (theta
                ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size))
        freq_frame = np.empty((head_size // 2, n_seq, n_batch))
        for i in range(n_batch):
            freq_frame[:, :, i] = np.outer(inv_freq, position_ids[i, :])
        np_freqs = np.array(freq_frame, dtype=np.float32, order='F')
        np_cos = np.cos(np_freqs)
        np_sin = np.sin(np_freqs)

        cos.from_array(np_cos)
        sin.from_array(np_sin)

        # Mask
        if mask is not None:
            layer_mask_shape = [n_seq, n_seq]
            layer_mask_basetile = [n_seq_tile, n_seq_tile]
            layer_mask_traits = TensorTraits(
                    layer_mask_shape,
                    layer_mask_basetile
            )
            layer_mask = Tensor_bool(
                    layer_mask_traits,
                    [0] * layer_mask_traits.grid.nelems,
                    next_tag
            )
            next_tag = layer_mask.next_tag
            layer_mask.from_array(mask)
        else:
            layer_mask = None

        # Create attention layer with all the provided data
        layer = LlamaAttention(
            x,
            y,
            w_q,
            w_k,
            w_v,
            w,
            q_transposed,
            q,
            q_rope,
            k_transposed,
            k,
            k_rope,
            k_rep,
            v_transposed,
            v,
            v_rep,
            a,
            a_maxsumexp,
            a_sumprod_slice,
            b,
            b_transposed,
            sin,
            cos,
            bias_inproj_q,
            bias_inproj_k,
            bias_inproj_v,
            out_proj_bias,
            layer_mask,
            flash_attention=flash_attention,
            redux=redux,
        )
        # Return layer and next tag to be used
        return (layer, next_tag)

    def reset_cache(self, value=0):
        self.kv_cache_size = value

    # Forward propagation of the attention layer
    def forward_async(self):
        # Compute query, key and value tensors
        # Q_transposed = einsum('ijkl,lmn->ijkmn', W_Q, X_Q)
        # gemm (kv_group_size, n_head_kv, head_size, n_emb)
        # by (n_emb, n_seq, n_batch)
        # into (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
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
        # Rotate axes into
        # (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        transpose_async(1.0, self.q_transposed.value, self.q.value, 2)
        # Q_transposed can be deleted
        self.q_transposed.value.invalidate_submit()
        # X_Q and W_Q can be offloaded from GPU
        self.x_q.value.wont_use()
        self.w_q.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_q is not None:
            # batched add_fiber (head_size, batch=(kv_group_size, n_head_kv))
            # into
            # (head_size, n_seq, n_batch, batch=(kv_group_size, n_head_kv))
            add_fiber_async(
                1, self.in_proj_bias_q.value, 1, self.q.value, 0, 2
            )
            self.in_proj_bias_q.value.wont_use()
        # Perform RoPE on Q
        rope_async(self.sin, self.cos, self.q.value, self.q_rope.value)
        # Q can be deleted
        self.q.value.invalidate_submit()
        # K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head_kv, head_size, n_seq, n_batch)
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
        # Rotate axes into (head_size, n_seq, n_batch, n_head_kv)
        transpose_async(1.0, self.k_transposed.value, self.k.value, 1)
        # K_transposed can be deleted
        self.k_transposed.value.invalidate_submit()
        # X_K and W_K can be offloaded from GPU
        self.x_k.value.wont_use()
        self.w_k.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_k is not None:
            # batched add_fiber (head_size, batch=n_head_kv) into
            # (head_size, n_seq, n_batch, batch=n_head_kv)
            add_fiber_async(
                1, self.in_proj_bias_k.value, 1, self.k.value, 0, 1
            )
            self.in_proj_bias_k.value.wont_use()
        # Perform RoPE on K
        rope_async(self.sin, self.cos, self.k.value, self.k_rope.value)
        # K can be deleted
        self.k.value.invalidate_submit()
        # Repeat K_rope along fibers of proper axis
        # from (head_size, n_seq, n_batch, n_head_kv)
        # into (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        add_slice_async(1.0, self.k_rope.value, 0.0, self.k_rep.value, 3)
        # K_rope can be offloaded from GPU
        self.k_rope.value.wont_use()
        # V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head_kv, head_size, n_seq, n_batch)
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
        # Rotate axes into (head_size, n_seq, n_batch, n_head_kv)
        transpose_async(1.0, self.v_transposed.value, self.v.value, 1)
        # V_transposed can be deleted
        self.v_transposed.value.invalidate_submit()
        # X_V and W_V can be offloaded from GPU
        self.x_v.value.wont_use()
        self.w_v.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_v is not None:
            # batched add_fiber (head_size, batch=n_head_kv) into
            # (head_size, n_seq, n_batch, batch=n_head_kv)
            add_fiber_async(
                1, self.in_proj_bias_v.value, 1, self.v.value, 0, 1
            )
            self.in_proj_bias_v.value.wont_use()
        # Repeat V along fibers of proper axis
        # from (head_size, n_seq, n_batch, n_head_kv)
        # into (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        add_slice_async(1.0, self.v.value, 0.0, self.v_rep.value, 3)
        # V can be offloaded from GPU
        self.v.value.wont_use()

        # Apply attention to Q_rope, K_rep and V_rep into B
        if self.flash_attention:
            self._flash_attention_fwd()
        else:
            self._attention_fwd()
        # Repeated tensors can be deleted now
        self.k_rep.value.invalidate_submit()
        self.v_rep.value.invalidate_submit()

        # Rotate axes (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        # into (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
        transpose_async(1.0, self.b.value, self.b_transposed.value, 3)
        # B can be deleted
        self.b.value.invalidate_submit()
        # Y = einsum('jklm,klmni->jni', W, B_transposed)
        # gemm (n_emb, kv_group_size, n_head_kv, head_size) by
        # (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
        # into (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w.value,
            notrans,
            self.b_transposed.value,
            0.0,
            self.y.value,
            3,
            0,
            redux=self.redux,
        )
        # W and B_transposed can be offloaded from GPU
        self.w.value.wont_use()
        self.b_transposed.value.wont_use()
        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_async(
                1.0, self.out_proj_bias.value, 1.0, self.y.value, 0, 0
            )
            self.out_proj_bias.value.wont_use()
        self.y.value.wont_use()

    def _forward_mlp_q_dynamic(self, x):
        q_partial_tr_bt_shape = tuple(
            self.q_transposed.value.basetile_shape[:-2]
        ) + tuple(x.shape[-2:])
        q_partial_tr_shape = tuple(self.q_transposed.value.shape[:-2]) + tuple(
            x.shape[-2:]
        )
        q_partial_tr = nntc.empty(
            q_partial_tr_shape,
            basetile_shape=q_partial_tr_bt_shape,
            dtype=type(x),
        )  # (kv_group_size, n_head_kv, head_size, n_seq_dyn, n_batch_dyn)

        q_partial_bt_shape = (
            (self.q.value.basetile_shape[0],)
            + tuple(x.shape[-2:])
            + tuple(self.q.value.basetile_shape[-2:])
        )
        q_partial_shape = (
            (self.q.value.shape[0],)
            + tuple(x.shape[-2:])
            + tuple(self.q.value.shape[-2:])
        )
        q_partial = nntc.empty(
            q_partial_shape, basetile_shape=q_partial_bt_shape, dtype=type(x)
        )  # (head_size, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv)

        # Q_transposed = einsum('ijkl,lmn->ijkmn', W_Q, X_Q_dyn)
        # gemm (kv_group_size, n_head_kv, head_size, n_emb)
        # by (n_emb, n_seq_dyn, n_batch_dyn)
        # into (kv_group_size, n_head_kv, head_size, n_seq_dyn, n_batch_dyn)
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

        # Rotate axes into
        # (head_size, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv)
        transpose_async(1.0, q_partial_tr, q_partial, 2)
        q_partial_tr.invalidate_submit()

        if self.in_proj_bias_q is not None:
            # batched add_fiber (head_size, batch=(kv_group_size, n_head_kv))
            # into
            # (head_size, n_seq_dyn, n_batch_dyn, batch=(kv_group_size, n_head_kv)) # noqa: E501
            add_fiber_async(
                1.0, self.in_proj_bias_q.value, 1.0, q_partial, 0, 2
            )

        return q_partial

    def _forward_mlp_k_dynamic(self, x):
        k_partial_tr_bt_shape = tuple(
            self.k_transposed.value.basetile_shape[:-2]
        ) + tuple(x.shape[-2:])
        k_partial_tr_shape = tuple(self.k_transposed.value.shape[:-2]) + tuple(
            x.shape[-2:]
        )
        k_partial_tr = nntc.empty(
            k_partial_tr_shape,
            basetile_shape=k_partial_tr_bt_shape,
            dtype=type(x),
        )  # (n_head_kv, head_size, n_seq_dyn, n_batch_dyn)

        k_partial_bt_shape = (
            (self.k.value.basetile_shape[0],)
            + tuple(x.shape[-2:])
            + (self.k.value.basetile_shape[-1],)
        )
        k_partial_shape = (
            (self.k.value.shape[0],)
            + tuple(x.shape[-2:])
            + (self.k.value.shape[-1],)
        )
        k_partial = nntc.empty(
            k_partial_shape, basetile_shape=k_partial_bt_shape, dtype=type(x)
        )  # (head_size, n_seq_dyn, n_batch_dyn, n_head_kv)

        # K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K_dyn)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq_dyn, n_batch_dyn) into # noqa: E501
        # (n_head_kv, head_size, n_seq_dyn, n_batch_dyn)
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
        # Rotate axes into (head_size, n_seq_dyn, n_batch_dyn, n_head_kv)
        transpose_async(1.0, k_partial_tr, k_partial, 1)
        if self.in_proj_bias_k is not None:
            # batched add_fiber (head_size, batch=n_head_kv) into
            # (head_size, n_seq_dyn, n_batch_dyn, batch=n_head_kv)
            add_fiber_async(
                1.0, self.in_proj_bias_k.value, 1.0, k_partial, 0, 1
            )

        return k_partial

    def _forward_mlp_v_dynamic(self, x):
        v_partial_tr_bt_shape = tuple(
            self.v_transposed.value.basetile_shape[:-2]
        ) + tuple(x.shape[-2:])
        v_partial_tr_shape = tuple(self.v_transposed.value.shape[:-2]) + tuple(
            x.shape[-2:]
        )
        v_partial_tr = nntc.empty(
            v_partial_tr_shape,
            basetile_shape=v_partial_tr_bt_shape,
            dtype=type(x),
        )  # (n_head_kv, head_size, n_seq_dyn, n_batch_dyn)

        v_partial_bt_shape = (
            (self.v.value.basetile_shape[0],)
            + tuple(x.shape[-2:])
            + (self.v.value.basetile_shape[-1],)
        )
        v_partial_shape = (
            (self.v.value.shape[0],)
            + tuple(x.shape[-2:])
            + (self.v.value.shape[-1],)
        )
        v_partial = nntc.empty(
            v_partial_shape, basetile_shape=v_partial_bt_shape, dtype=type(x)
        )  # (head_size, n_seq_dyn, n_batch_dyn, n_head_kv)

        # V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V_dyn)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq_dyn, n_batch_dyn) into # noqa: E501
        # (n_head_kv, head_size, n_seq_dyn, n_batch_dyn)
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
        # Rotate axes into (head_size, n_seq_dyn, n_batch_dyn, n_head_kv)
        transpose_async(1.0, v_partial_tr, v_partial, 1)
        if self.in_proj_bias_v is not None:
            # batched add_fiber (head_size, batch=n_head_kv) into
            # (head_size, n_seq_dyn, n_batch_dyn, batch=n_head_kv)
            add_fiber_async(
                1.0, self.in_proj_bias_v.value, 1.0, v_partial, 0, 1
            )

        v_partial_tr.invalidate_submit()

        return v_partial

    def _forward_attn_dynamic(self, q, k, v):
        a_tmp = nntc.empty(
            (k.shape[1],) + tuple(q.shape[1:3]) + tuple(k.shape[3:]),
            dtype=type(q),
            basetile_shape=(k.basetile_shape[1],)
            + tuple(q.basetile_shape[1:3])
            + tuple(k.basetile_shape[3:]),
        )  # (n_seq_kvcached, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv)
        a_maxsumexp_tmp = nntc.empty(
            (2,) + tuple(a_tmp.shape[1:]),
            dtype=type(q),
            basetile_shape=(2,) + tuple(a_tmp.basetile_shape[1:]),
        )  # (2, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv)
        b_tmp = nntc.empty(
            q.shape,
            dtype=type(q),
            basetile_shape=q.basetile_shape,
        )  # (head_size, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv)
        b_tr_tmp = nntc.empty(
            tuple(b_tmp.shape[3:]) + tuple(b_tmp.shape[:3]),
            dtype=type(q),
            basetile_shape=tuple(b_tmp.basetile_shape[3:])
            + tuple(b_tmp.basetile_shape[:3]),
        )  # (n_head, head_size, n_seq_dyn, n_batch_dyn)

        y_tensor = nntc.empty(
            (self.w.value.shape[0],) + tuple(q.shape[1:3]),
            dtype=type(q),
            basetile_shape=(self.w.value.basetile_shape[0],)
            + tuple(q.basetile_shape[1:3]),
        )  # [n_emb, n_seq_dyn, n_batch_dyn] == x.shape

        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklbi,jmlbi->kmlbi', K_rep, Q_rope)
        # single batched gemm
        # (head_size, n_seq_cached, batch=(n_batch_cached, kv_group_size, n_head_kv)) # noqa: E501
        # by (head_size, n_seq_dyn, batch=(n_batch_dyn, kv_group_size, n_head_kv)) # noqa: E501
        # into (n_seq_cached, n_seq_dyn, batch=(n_batch_dyn, kv_group_size, n_head_kv)) # noqa: E501
        # note: n_batch_cached == n_batch_dyn
        gemm_async(
            1.0 / self.head_size**0.5,
            trans,
            k,
            notrans,
            q,
            0.0,
            a_tmp,
            1,
            3,
            redux=self.redux,
        )
        clear_async(a_maxsumexp_tmp)
        if self.mask:
            mask_tmp = nntc.empty(a_tmp.shape[:2], dtype=Tensor_bool)
            copy_intersection_async(
                self.mask, [0, 0], mask_tmp, [0, k.shape[1] - q.shape[1]]
            )
            mask_scalar_async(mask_tmp, self.val, a_tmp, 3)

        maxsumexp_async(a_tmp, a_maxsumexp_tmp, 0, redux=self.redux)
        softmax_inplace_async(a_maxsumexp_tmp, 1.0, a_tmp, 0)

        # Apply value tensor
        # B = einsum('jklbi,kmlbi->jmlbi', V_rep, A)
        # batched gemm
        # (head_size, n_seq_cached, batch=(n_batch_cached, kv_group_size, n_head_kv)) # noqa: E501
        # by (n_seq_cached, n_seq_dyn, batch=(n_batch_dyn, kv_group_size, n_head_kv)) # noqa: E501
        # into (head_size, n_seq_dyn, batch=(n_batch_dyn, kv_group_size, n_head_kv)) # noqa: E501
        gemm_async(
            1.0,
            notrans,
            v,
            notrans,
            a_tmp,
            0.0,
            b_tmp,
            1,
            3,
            redux=self.redux,
        )

        # Rotate axes (head_size, n_seq_dyn, n_batch_dyn, kv_group_size, n_head_kv) # noqa: E501
        # into (kv_group_size, n_head_kv, head_size, n_seq_dyn, n_batch_dyn)
        transpose_async(1.0, b_tmp, b_tr_tmp, 3)

        # Y = einsum('jklm,klmni->jni', W, B_transposed)
        # gemm (n_emb, kv_group_size, n_head_kv, head_size) by
        # (kv_group_size, n_head_kv, head_size, n_seq_dyn, n_batch_dyn)
        # into (n_emb, n_seq_dyn, n_batch_dyn)
        gemm_async(
            1.0,
            notrans,
            self.w.value,
            notrans,
            b_tr_tmp,
            0.0,
            y_tensor,
            3,
            0,
            redux=self.redux,
        )

        if self.out_proj_bias is not None:
            add_fiber_async(1.0, self.out_proj_bias.value, 1.0, y_tensor, 0, 0)

        return y_tensor

    def _apply_rope_dynamic(
        self, x: Tensor, q_partial: Tensor, k_partial: Tensor
    ):
        q_rope_partial = nntc.empty(
            q_partial.shape,
            basetile_shape=q_partial.basetile_shape,
            dtype=type(x),
        )
        k_rope_partial = nntc.empty(
            k_partial.shape,
            basetile_shape=k_partial.basetile_shape,
            dtype=type(x),
        )

        sin_partial = nntc.zeros((self.sin.shape[0],) + tuple(x.shape[-2:]))
        cos_partial = nntc.zeros((self.cos.shape[0],) + tuple(x.shape[-2:]))
        copy_intersection_async(
            self.sin, [0, 0, 0], sin_partial, [0, self.kv_cache_size, 0]
        )
        copy_intersection_async(
            self.cos, [0, 0, 0], cos_partial, [0, self.kv_cache_size, 0]
        )

        rope_async(sin_partial, cos_partial, q_partial, q_rope_partial)
        rope_async(sin_partial, cos_partial, k_partial, k_rope_partial)

        sin_partial.invalidate_submit()
        cos_partial.invalidate_submit()
        return q_rope_partial, k_rope_partial

    def _storeload_kvcache(
        self,
        x: Tensor,
        k_rope_partial: Tensor,
        v_partial: Tensor,
        use_cache: bool,
    ):
        """
        handles kv-cache routine.
        1. Save new partials to cache
        2. Load K,V from cache
        """
        if k_rope_partial.shape[1] != v_partial.shape[1]:
            raise Exception(
                "Current kvcache code assumes equal seq_size for K,V: ",
                f"{k_rope_partial.shape[1]} != {v_partial.shape[1]}",
            )

        if v_partial.shape[1] + self.kv_cache_size > self.x_v.value.shape[1]:
            raise Exception(
                "Overload internal state: "
                f"try add {v_partial.shape[1]} "
                f"to {self.kv_cache_size}, max: {self.x_v.value.shape[1]}. "
                "Maybe you forgot to call reset_cache between iterations?"
            )

        copy_intersection_async(
            k_rope_partial,
            [0, self.kv_cache_size, 0, 0],
            self.k.value,
            [0, 0, 0, 0],
        )

        copy_intersection_async(
            v_partial,
            [0, self.kv_cache_size, 0, 0],
            self.v.value,
            [0, 0, 0, 0],
        )
        self.kv_cache_size += v_partial.shape[1]

        if not use_cache:
            return k_rope_partial, v_partial

        # For correct softmax we should next use only currently cached seq_size
        # So copy here
        k_cached_shape = self.k.value.shape
        k_cached_shape[1] = self.kv_cache_size
        k_cached_shape[2] = x.shape[2]
        k_partial_cached = nntc.empty(
            k_cached_shape,
            dtype=type(k_rope_partial),
            basetile_shape=tuple(k_cached_shape[:-1])
            + (k_rope_partial.basetile_shape[-1],),
        )
        copy_intersection_async(
            self.k.value, [0, 0, 0, 0], k_partial_cached, [0, 0, 0, 0]
        )

        cached_shape = self.v.value.shape
        cached_shape[1] = self.kv_cache_size
        cached_shape[2] = x.shape[2]
        v_partial_cached = nntc.empty(
            cached_shape,
            dtype=type(v_partial),
            basetile_shape=tuple(cached_shape[:-1])
            + (v_partial.basetile_shape[-1],),
        )
        copy_intersection_async(
            self.v.value, [0, 0, 0, 0], v_partial_cached, [0, 0, 0, 0]
        )

        return k_partial_cached, v_partial_cached

    def _broadcast_kv_dynamic(
        self, x: Tensor, k_rope_partial: Tensor, v_partial: Tensor
    ):
        k_rep_bt_shape = (
            (self.q_rope.value.basetile_shape[0],)
            + tuple(k_rope_partial.shape[-3:-1])
            + tuple(self.q_rope.value.basetile_shape[-2:])
        )
        K_rep_shape = (
            (self.q_rope.value.shape[0],)
            + tuple(k_rope_partial.shape[-3:-1])
            + tuple(self.q_rope.value.shape[-2:])
        )
        k_rep_partial = nntc.empty(
            K_rep_shape, basetile_shape=k_rep_bt_shape, dtype=type(x)
        )

        v_rep_bt_shape = (
            (self.v_rep.value.basetile_shape[0],)
            + tuple(v_partial.shape[-3:-1])
            + tuple(self.v_rep.value.basetile_shape[-2:])
        )
        v_rep_shape = (
            (self.v_rep.value.shape[0],)
            + tuple(v_partial.shape[-3:-1])
            + tuple(self.v_rep.value.shape[-2:])
        )
        v_rep_partial = nntc.empty(
            v_rep_shape, basetile_shape=v_rep_bt_shape, dtype=type(x)
        )
        add_slice_async(1.0, k_rope_partial, 0.0, k_rep_partial, 3)
        add_slice_async(1.0, v_partial, 0.0, v_rep_partial, 3)

        return k_rep_partial, v_rep_partial

    def forward_dynamic(self, x: TensorMoments, use_cache: bool = False):
        if not use_cache:
            self.reset_cache()

        q_partial = self._forward_mlp_q_dynamic(x.value)
        k_partial = self._forward_mlp_k_dynamic(x.value)
        v_partial = self._forward_mlp_v_dynamic(x.value)

        q_rope_partial, k_rope_partial = self._apply_rope_dynamic(
            x.value, q_partial, k_partial
        )
        q_partial.invalidate_submit()
        k_partial.invalidate_submit()

        k_rope_partial, v_partial = self._storeload_kvcache(
            x.value, k_rope_partial, v_partial, use_cache
        )

        k_rep_partial, v_rep_partial = self._broadcast_kv_dynamic(
            x.value, k_rope_partial, v_partial
        )
        v_partial.invalidate_submit()
        k_rope_partial.invalidate_submit()

        y_tensor = self._forward_attn_dynamic(
            q_rope_partial, k_rep_partial, v_rep_partial
        )

        return TensorMoments(y_tensor, None, False)

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
        # Backward for Y = einsum('jklm,klmni->jni', W, B_transposed)
        if self.w.grad_required:
            # dW += einsum('jni,klmni->jklm', dY, B_transposed)
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
        self.b_transposed.value.invalidate_submit()
        self.w.grad.wont_use()
        if self.b_transposed.grad_required:
            # dB_transposed = einsum('jklm,jni->klmni', W, dY)
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
            # rotate axes (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
            # into (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
            transpose_async(1.0, self.b_transposed.grad, self.b.grad, 2)
        # self.b_transposed.grad.wont_use()
        self.b_transposed.grad.invalidate_submit()

        # Repeat K along fibers of proper axis
        # from (head_size, n_seq, n_batch, n_head_kv)
        # into (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        add_slice_async(1.0, self.k_rope.value, 0.0, self.k_rep.value, 3)
        # K_rope can be deleted
        self.k_rope.value.invalidate_submit()
        # Repeat V along fibers of proper axis
        # from (head_size, n_seq, n_batch, n_head_kv)
        # into (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
        add_slice_async(1.0, self.v.value, 0.0, self.v_rep.value, 3)
        # V can be deleted
        self.v.value.invalidate_submit()
        # Apply backward to (attention to Q_rope, K_rep and V_rep into B)
        if self.flash_attention:
            self._flash_attention_bwd()
        else:
            self._attention_bwd()

        # Backward for repeating V along fibers of proper axis
        sum_slice_async(1.0, self.v_rep.grad, 0.0, self.v.grad, 3)
        # dV_rep can be deleted
        self.v_rep.grad.invalidate_submit()
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
            # Rotate axes (head_size, n_seq, n_batch, n_head_kv) into
            # (n_head_kv, head_size, n_seq, n_batch)
            transpose_async(1.0, self.v.grad, self.v_transposed.grad, 3)
        # dV can be deleted
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
        self.v_transposed.grad.invalidate_submit()

        # Backward for repeating K along fibers of proper axis
        if self.k_rope.grad_required:
            sum_slice_async(1.0, self.k_rep.grad, 0.0, self.k_rope.grad, 3)
        # Backward for RoPE
        if self.k.grad_required:
            rope_backward_async(
                    self.sin, self.cos, self.k_rope.grad, self.k.grad
            )
        # dK_rep can be deleted
        self.k_rep.grad.invalidate_submit()
        # #dK_rope can be deleted
        self.k_rope.grad.invalidate_submit()
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
            # Rotate axes (head_size, n_seq, n_batch, n_head_kv) into
            # (n_head_kv, head_size, n_seq, n_batch)
            transpose_async(1.0, self.k.grad, self.k_transposed.grad, 3)
        # dK can be deleted
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
        self.k_transposed.grad.invalidate_submit()
        # Backward for RoPE for Q
        if self.q.grad_required:
            rope_backward_async(
                    self.sin, self.cos, self.q_rope.grad, self.q.grad
            )
        # Backward for bias of Q
        if self.in_proj_bias_q is not None:
            if self.in_proj_bias_q.grad_required:
                sum_fiber_async(
                    1,
                    self.q.grad,
                    1,
                    self.in_proj_bias_q.grad,
                    0,
                    2,
                    redux=self.redux,
                )
                self.in_proj_bias_q.grad.wont_use()
        # Backward for axes rotation (Q_transposed->Q)
        if self.q_transposed.grad_required:
            # Rotate axes (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
            # into (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
            transpose_async(1.0, self.q.grad, self.q_transposed.grad, 3)
        # dQ can be deleted
        self.q.grad.invalidate_submit()
        # dQ_rope can be deleted
        self.q_rope.grad.invalidate_submit()
        # Backward for Q_transposed = einsum('ijkl,lmn->ijkmn', W_Q, X_Q)
        if self.x_q.grad_required:
            # dX_Q += einsum('ijkl,ijkmn->lmn', W_Q, dQ_transposed)
            gemm_async(
                1.0,
                trans,
                self.w_q.value,
                notrans,
                self.q_transposed.grad,
                1.0,
                self.x_q.grad,
                3,
                0,
                redux=self.redux,
            )
            self.x_q.grad.wont_use()
        # W_Q can be offloaded from GPU
        self.w_q.value.wont_use()
        # dX_Q can be offloaded from GPU
        self.x_q.grad.wont_use()
        if self.w_q.grad_required:
            # dW_Q += einsum('ijkmn,lmn->ijkl', dQ_transposed, X_Q)
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
        self.q_transposed.grad.invalidate_submit()

    @staticmethod
    def rotate_tensor_in(x: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            new_shape = (1, x.shape[0], np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), x.shape[-1], 1)
        else:
            new_shape = (
                    np.prod(x.shape[:axis]),
                    x.shape[axis],
                    np.prod(x.shape[axis + 1:])
            )
        x_reshaped = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
        y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]
        return y_reshaped.reshape(x.shape)

    @staticmethod
    def rotate_tensor_out(x: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            new_shape = (1, x.shape[0], np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), x.shape[-1], 1)
        else:
            new_shape = (
                    np.prod(x.shape[:axis]),
                    x.shape[axis],
                    np.prod(x.shape[axis + 1:])
            )
        x_reshaped = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, :mid, :] = x_reshaped[:, 0::2, :]
        y_reshaped[:, mid:, :] = x_reshaped[:, 1::2, :]
        return y_reshaped.reshape(x.shape)

    @classmethod
    def from_torch(cls,
        torch_layer: LlamaAttention_torch,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: LlamaConfigNNTile,
        next_tag: int
    ):  # -> Self: does not work with Python 3.10
        layer, next_tag = cls.generate_simple(
            x,
            n_head=torch_layer.num_heads,
            n_head_tile=config.n_head_tile,
            n_head_kv=torch_layer.num_key_value_heads,
            position_ids=position_ids,
            theta=config.rope_theta,
            next_tag=next_tag,
            bias=torch_layer.q_proj.bias is not None,
            mask=mask,
            flash_attention=config.flash_attention,
            redux=config.redux,
        )
        tmp_q_shape = layer.w_q.value.shape.copy()
        tmp_q_shape[:2] = tmp_q_shape[1::-1]
        layer.w_q.value.from_array(
            cls.rotate_tensor_in(
                np.moveaxis(
                    torch_layer.q_proj.weight.detach().cpu().numpy()
                    .reshape(*tmp_q_shape),
                    0,
                    1,
                ),
                2
            )
        )
        layer.w_k.value.from_array(
            cls.rotate_tensor_in(
                torch_layer.k_proj.weight.detach().cpu().numpy()
                .reshape(*layer.w_k.value.shape),
                1
            )
        )
        layer.w_v.value.from_array(
                torch_layer.v_proj.weight.detach().cpu().numpy()
                .reshape(*layer.w_v.value.shape)
        )
        tmp_w_shape = layer.w.value.shape.copy()
        tmp_w_shape[1:3] = tmp_w_shape[2:0:-1]
        layer.w.value.from_array(
            np.moveaxis(
                torch_layer.o_proj.weight.detach()
                .cpu()
                .numpy()
                .reshape(*tmp_w_shape),
                1,
                2,
            )
        )
        if layer.out_proj_bias is not None:
            layer.out_proj_bias.value.from_array(
                torch_layer.o_proj.bias.detach()
                .cpu()
                .numpy()
                .reshape(*layer.out_proj_bias.value.shape)
            )
            layer.in_proj_bias_q.value.from_array(
                cls.rotate_tensor_in(
                    torch_layer.q_proj.bias.detach()
                    .cpu()
                    .numpy()
                    .reshape(*layer.in_proj_bias_q.value.shape[::-1])
                    .T,
                    0
                )
            )
            layer.in_proj_bias_k.value.from_array(
                cls.rotate_tensor_in(
                    torch_layer.k_proj.bias.detach()
                    .cpu()
                    .numpy()
                    .reshape(*layer.in_proj_bias_k.value.shape[::-1])
                    .T,
                    0
                )
            )
            layer.in_proj_bias_v.value.from_array(
                torch_layer.v_proj.bias.detach()
                .cpu()
                .numpy()
                .reshape(*layer.in_proj_bias_v.value.shape[::-1])
                .T
            )
        return layer, next_tag

    def to_torch(self) -> LlamaAttention_torch:
        bias = self.in_proj_bias_q is not None
        torch_layer_config = LlamaConfig_torch(
            hidden_size=self.n_emb,
            num_attention_heads=self.n_head,
            num_key_value_heads=self.n_head_kv,
            attention_bias=bias,
            use_cache=False,
            attention_dropout=0.0,
        )
        torch_layer = LlamaAttention_torch(torch_layer_config, layer_idx=0)
        torch_layer.q_proj.weight.data = torch.tensor(
            np.moveaxis(
                __class__.rotate_tensor_out(to_numpy(self.w_q.value), 2),
                0,
                1
            ).reshape(self.n_emb, self.n_emb),
            requires_grad=True,
        )
        torch_layer.k_proj.weight.data = torch.tensor(
            __class__.rotate_tensor_out(to_numpy(self.w_k.value), 1)
            .reshape(self.n_emb_kv, self.n_emb),
            requires_grad=True,
        )
        torch_layer.v_proj.weight.data = torch.tensor(
            to_numpy(self.w_v.value).reshape(self.n_emb_kv, self.n_emb),
            requires_grad=True,
        )
        torch_layer.o_proj.weight.data = torch.tensor(
            np.moveaxis(to_numpy(self.w.value), 1, 2).reshape(
                self.n_emb, self.n_emb
            ),
            requires_grad=True,
        )
        if bias:
            torch_layer.o_proj.bias.data = torch.tensor(
                to_numpy(self.out_proj_bias.value).flatten(),
                requires_grad=True,
            )
            torch_layer.q_proj.bias.data = torch.tensor(
                __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_q.value),
                    0
                ).T.flatten(),
                requires_grad=True,
            )
            torch_layer.k_proj.bias.data = torch.tensor(
                __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_k.value),
                    0
                ).T.flatten(),
                requires_grad=True,
            )
            torch_layer.v_proj.bias.data = torch.tensor(
                to_numpy(self.in_proj_bias_v.value).T.flatten(),
                requires_grad=True,
            )
        return torch_layer

    def to_torch_with_grads(self) -> LlamaAttention_torch:
        bias = self.in_proj_bias_q is not None
        torch_layer = self.to_torch()
        torch_layer.q_proj.weight.grad = torch.tensor(
            np.moveaxis(
                __class__.rotate_tensor_out(to_numpy(self.w_q.grad), 2),
                0,
                1
            ).reshape(self.n_emb, self.n_emb)
        )
        torch_layer.k_proj.weight.grad = torch.tensor(
            __class__.rotate_tensor_out(to_numpy(self.w_k.grad), 1)
            .reshape(self.n_emb_kv, self.n_emb)
        )
        torch_layer.v_proj.weight.grad = torch.tensor(
            to_numpy(self.w_v.grad).reshape(self.n_emb_kv, self.n_emb)
        )
        torch_layer.o_proj.weight.grad = torch.tensor(
            np.moveaxis(to_numpy(self.w.grad), 1, 2).reshape(
                self.n_emb, self.n_emb
            )
        )
        if bias:
            torch_layer.o_proj.bias.grad = torch.tensor(
                to_numpy(self.out_proj_bias.grad).flatten()
            )
            torch_layer.q_proj.bias.grad = torch.tensor(
                __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_q.grad),
                    0
                ).T.flatten()
            )
            torch_layer.k_proj.bias.grad = torch.tensor(
                __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_k.grad),
                    0
                ).T.flatten()
            )
            torch_layer.v_proj.bias.grad = torch.tensor(
                to_numpy(self.in_proj_bias_v.grad).T.flatten()
            )
        return torch_layer

    def get_forward_flops(self):
        total_forward_flops = 0
        # Compute Q_transposed
        # Q_transposed = einsum('ijkl,lmn->ijkmn', W_Q, X_Q)
        # gemm (kv_group_size, n_head_kv, head_size, n_emb)
        # by (n_emb, n_seq, n_batch)
        # into (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
        w_q_shape = self.w_q.value.shape
        x_q_shape = self.x_q.value.shape
        qt_flops = 2 * np.prod(w_q_shape) * np.prod(x_q_shape[1:])
        total_forward_flops += qt_flops
        # Compute K_transposed
        # K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head_kv, head_size, n_seq, n_batch)
        w_k_shape = self.w_k.value.shape
        x_k_shape = self.x_k.value.shape
        kt_flops = 2 * np.prod(w_k_shape) * np.prod(x_k_shape[1:])
        total_forward_flops += kt_flops
        # Compute V_transposed
        # V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V)
        # gemm (n_head_kv, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head_kv, head_size, n_seq, n_batch)
        w_v_shape = self.w_v.value.shape
        x_v_shape = self.x_v.value.shape
        vt_flops = 2 * np.prod(w_v_shape) * np.prod(x_v_shape[1:])
        total_forward_flops += vt_flops
        # Compute tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklbi,jmlbi->kmlbi', K_rep, Q_rope)
        # single batched gemm
        # (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # by (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # into (n_seq, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        k_rep_shape = self.k_rep.value.shape
        q_rope_shape = self.q_rope.value.shape
        a_flops = 2 * np.prod(k_rep_shape[:2]) * np.prod(q_rope_shape[1:])
        total_forward_flops += a_flops
        # Apply value tensor
        # B = einsum('jklbi,kmlbi->jmlbi', V_rep, A)
        # batched gemm
        # (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # by (n_seq, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # into (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        v_rep_shape = self.v_rep.value.shape
        a_shape = self.a.value.shape
        b_flops = 2 * np.prod(a_shape) * v_rep_shape[0]
        total_forward_flops += b_flops
        # Gemm for accumulate result for all the heads
        # Y = einsum('jklm,klmni->jni', W, B_transposed)
        # gemm (n_emb, kv_group_size, n_head_kv, head_size) by
        # (kv_group_size, n_head_kv, head_size, n_seq, n_batch)
        # into (n_emb, n_seq, n_batch)
        w_shape = self.w.value.shape
        bt_shape = self.b_transposed.value.shape
        y_flops = 2 * np.prod(w_shape) * np.prod(bt_shape[3:])
        total_forward_flops += y_flops
        return total_forward_flops

    def get_backward_flops(self):
        total_backward_flops = 0
        if self.w.grad_required:
            # dW += einsum('jni,klmni->jklm', dY, B_transposed)
            y_grad_shape = self.y.grad.shape
            bt_shape = self.b_transposed.value.shape
            w_grad_flops = 2 * np.prod(bt_shape) * np.prod(y_grad_shape[0])
            total_backward_flops += w_grad_flops
        if self.b_transposed.grad_required:
            # dB_transposed = einsum('jklm,jni->klmni', W, dY)
            w_shape = self.w.value.shape
            y_grad_shape = self.y.grad.shape
            bt_grad_flops = 2 * np.prod(w_shape) * np.prod(y_grad_shape[1:])
            total_backward_flops += bt_grad_flops
        if self.a.grad_required:
            # dA = einsum('jklbi,jmlbi->kmlbi', V_rep, dB)
            # ndim = 1
            # batch_ndim = 3
            b_grad_shape = self.b.grad.shape
            v_rep_shape = self.v_rep.value.shape
            a_grad_flops = 2 * np.prod(v_rep_shape) * b_grad_shape[1]
            total_backward_flops += a_grad_flops
        if self.v_rep.grad_required:
            # dV_rep = einsum('jmlbi,kmlbi->jklbi', dB, A)
            # ndim = 1
            # batch_ndim = 3
            a_shape = self.a.value.shape
            b_grad_shape = self.b.grad.shape
            v_rep_grad_flops = 2 * np.prod(a_shape) * b_grad_shape[0]
            total_backward_flops += v_rep_grad_flops
        if self.k_rep.grad_required:
            # dK_rep = 1.0/sqrt(head_size)
            #          * einsum('jmlbi,kmlbi->jklbi', Q_rope, dA)
            # ndim = 1
            # batch_ndim = 3
            q_rope_shape = self.q_rope.value.shape
            a_grad_shape = self.a.grad.shape
            k_rep_grad_flops = 2 * np.prod(a_grad_shape) * q_rope_shape[0]
            total_backward_flops += k_rep_grad_flops
        if self.q_rope.grad_required:
            # dQ_rope = 1.0/sqrt(head_size)
            #      * einsum('jklbi,kmlbi->jmlbi', K_rep, dA)
            # ndim = 1
            # batch_ndim = 3
            k_rep_shape = self.k_rep.value.shape
            a_grad_shape = self.a.grad.shape
            q_rope_grad_flops = 2 * np.prod(a_grad_shape) * k_rep_shape[0]
            total_backward_flops += q_rope_grad_flops
        if self.x_v.grad_required:
            # dX_V += einsum('jkl,jkmn->lmn', W_V, dV_transposed)
            # ndim = 2
            v_t_grad_shape = self.v_transposed.grad.shape
            w_v_shape = self.w_v.value.shape
            x_v_grad_flops = (2 * np.prod(w_v_shape) *
                              np.prod(v_t_grad_shape[2:]))
            total_backward_flops += x_v_grad_flops
        if self.w_v.grad_required:
            # dW_V += einsum('jkmn,lmn->jkl', dV_transposed, X_V)
            # ndim = 2
            x_v_shape = self.x_v.value.shape
            v_t_grad_shape = self.v_transposed.grad.shape
            w_v_grad_flops = (2 * np.prod(x_v_shape) *
                              np.prod(v_t_grad_shape[:-2]))
            total_backward_flops += w_v_grad_flops
        if self.x_k.grad_required:
            # dX_K += einsum('jkl,jkmn->lmn', W_K, dK_transposed)
            # ndim = 2
            kt_grad_shape = self.k_transposed.grad.shape
            w_k_shape = self.w_k.value.shape
            x_k_grad_flops = (2 * np.prod(w_k_shape) *
                              np.prod(kt_grad_shape[2:]))
            total_backward_flops += x_k_grad_flops
        if self.w_k.grad_required:
            # dW_K += einsum('jkmn,lmn->jkl', dK_transposed, X_K)
            # ndim = 2
            x_k_shape = self.x_k.value.shape
            kt_grad_shape = self.k_transposed.grad.shape
            w_k_grad_flops = 2 * np.prod(kt_grad_shape) * x_k_shape[0]
            total_backward_flops += w_k_grad_flops
        if self.x_q.grad_required:
            # dX_Q += einsum('ijkl,ijkmn->lmn', W_Q, dQ_transposed)
            # ndim = 3
            qt_grad_shape = self.q_transposed.grad.shape
            w_q_shape = self.w_q.value.shape
            x_q_grad_flops = (2 * np.prod(w_q_shape) *
                              np.prod(qt_grad_shape[3:]))
            total_backward_flops += x_q_grad_flops
        if self.w_q.grad_required:
            # dW_Q += einsum('ijkmn,lmn->ijkl', dQ_transposed, X_Q)
            # ndim = 2
            x_q_shape = self.x_q.value.shape
            qt_grad_shape = self.q_transposed.grad.shape
            w_q_grad_flops = (2 * np.prod(x_q_shape) *
                              np.prod(qt_grad_shape[:-2]))
            total_backward_flops += w_q_grad_flops
        return total_backward_flops

    def _flash_attention_fwd(self):
        # Use flash-like maxsumexp
        clear_async(self.a_maxsumexp)
        flash_maxsumexp_async(
            self.q_rope.value,
            self.k_rep.value,
            self.mask,
            self.a_maxsumexp,
            self.a.value,
            redux=self.redux,
        )
        # Use flash-like softmax+gemm
        flash_softmax_gemm_async(
            self.q_rope.value,
            self.k_rep.value,
            self.v_rep.value,
            self.mask,
            self.a_maxsumexp,
            self.b.value,
            self.a.value,
            redux=self.redux,
        )
        # Q_rope, K_rep, V_rep, mask and A_maxsumexp can be offloaded from GPU
        self.q_rope.value.wont_use()
        self.k_rep.value.wont_use()
        self.v_rep.value.wont_use()
        self.mask.wont_use()
        self.a_maxsumexp.wont_use()
        # A can be deleted
        self.a.value.invalidate_submit()

    def _flash_attention_bwd(self):
        # Flash-like backward of softmax+gemm
        clear_async(self.a_sumprod_slice)
        flash_softmax_gemm_backward_async(
            self.q_rope.value,
            self.q_rope.grad,
            self.k_rep.value,
            self.k_rep.grad,
            self.v_rep.value,
            self.v_rep.grad,
            self.mask,
            self.a_maxsumexp,
            self.b.grad,
            self.a.value,
            self.a.grad,
            self.a_sumprod_slice,
            redux=self.redux,
        )
        # Q_rope can be deleted
        self.q_rope.value.invalidate_submit()
        # K_rep can be deleted
        self.k_rep.value.invalidate_submit()
        # V_rep can be deleted
        self.v_rep.value.invalidate_submit()
        # mask can be offloaded from GPU
        self.mask.wont_use()
        # A_maxsumexp can be deleted
        self.a_maxsumexp.invalidate_submit()
        # dB can be deleted
        self.b.grad.invalidate_submit()
        # A can be deleted
        self.a.value.invalidate_submit()
        # dA can be deleted
        self.a.grad.invalidate_submit()
        # A_sumprod_slice can be deleted
        self.a_sumprod_slice.invalidate_submit()

    def _attention_fwd(self):
        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklbi,jmlbi->kmlbi', K_rep, Q_rope)
        # single batched gemm
        # (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # by (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # into (n_seq, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        gemm_async(
            1.0 / self.head_size**0.5,
            trans,
            self.k_rep.value,
            notrans,
            self.q_rope.value,
            0.0,
            self.a.value,
            1,
            3,
            redux=self.redux,
        )
        clear_async(self.a_maxsumexp)
        # Q_rope, K_rep can be offloaded from GPU
        self.q_rope.value.wont_use()
        self.k_rep.value.wont_use()
        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.mask:
            mask_scalar_async(self.mask, self.val, self.a.value, 3)
            self.mask.wont_use()
        # Calculate max and sumexp along axis
        maxsumexp_async(self.a.value, self.a_maxsumexp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(self.a_maxsumexp, 1.0, self.a.value, 0)
        # A_maxsumexp can be deleted
        self.a_maxsumexp.invalidate_submit()
        # Apply value tensor
        # B = einsum('jklbi,kmlbi->jmlbi', V_rep, A)
        # batched gemm
        # (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # by (n_seq, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        # into (head_size, n_seq, batch=(n_batch, kv_group_size, n_head_kv))
        gemm_async(
            1.0,
            notrans,
            self.v_rep.value,
            notrans,
            self.a.value,
            0.0,
            self.b.value,
            1,
            3,
            redux=self.redux,
        )
        # V_rep and A can be offloaded from GPU
        self.v_rep.value.wont_use()
        self.a.value.wont_use()

    def _attention_bwd(self):
        # Backward for B = einsum('jklbi,kmlbi->jmlbi', V_rep, A)
        if self.a.grad_required:
            # dA = einsum('jklbi,jmlbi->kmlbi', V_rep, dB)
            gemm_async(
                1.0,
                trans,
                self.v_rep.value,
                notrans,
                self.b.grad,
                0.0,
                self.a.grad,
                1,
                3,
                redux=self.redux,
            )
        # V_rep can be deleted
        self.v_rep.value.invalidate_submit()
        if self.v_rep.grad_required:
            # dV_rep = einsum('jmlbi,kmlbi->jklbi', dB, A)
            gemm_async(
                1.0,
                notrans,
                self.b.grad,
                trans,
                self.a.value,
                0.0,
                self.v_rep.grad,
                1,
                3,
                redux=self.redux,
            )
        # dB can be deleted
        self.b.grad.invalidate_submit()
        # Backward for A = softmax(A, axis=0)
        if self.a.grad_required:
            # A_sumprod_slice = einsum('kmlbi,kmlbi->mlbi', A, dA)
            sumprod_slice_async(
                1.0,
                self.a.value,
                self.a.grad,
                0.0,
                self.a_sumprod_slice,
                0,
                redux=self.redux,
            )
            # dA += -bias('kmlbi,mlbi->kmlbi', dA, A_sumprod_slice)
            add_slice_async(-1.0, self.a_sumprod_slice, 1.0, self.a.grad, 0)
            # A_sumprod_slice can be deleted
            self.a_sumprod_slice.invalidate_submit()
            # dA *= A
            prod_inplace_async(self.a.value, self.a.grad)
        # A can be deleted
        self.a.value.invalidate_submit()
        # Backward for mask if needed
        if self.mask:
            mask_scalar_async(self.mask, 0.0, self.a.grad, 3)
            self.mask.wont_use()
        # Backward for:
        # A = 1.0/sqrt(head_size) * einsum('jklbi,jmlbi->kmlbi', K_rep, Q_rope)
        if self.k_rep.grad_required:
            # dK_rep = 1.0/sqrt(head_size)
            #          * einsum('jmlbi,kmlbi->jklbi', Q_rope, dA)
            gemm_async(
                1.0 / self.head_size**0.5,
                notrans,
                self.q_rope.value,
                trans,
                self.a.grad,
                0.0,
                self.k_rep.grad,
                1,
                3,
                redux=self.redux,
            )
        # Q_rope can be deleted
        self.q_rope.value.invalidate_submit()
        if self.q_rope.grad_required:
            # dQ_rope = 1.0/sqrt(head_size)
            #      * einsum('jklbi,kmlbi->jmlbi', K_rep, dA)
            gemm_async(
                1.0 / self.head_size**0.5,
                notrans,
                self.k_rep.value,
                notrans,
                self.a.grad,
                0.0,
                self.q_rope.grad,
                1,
                3,
                redux=self.redux,
            )
        # K_rep can be deleted
        self.k_rep.value.invalidate_submit()
        # dA can be deleted
        self.a.grad.invalidate_submit()
