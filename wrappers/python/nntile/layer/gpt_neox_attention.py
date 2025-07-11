# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/gpt_neox_attention.py
# GPTNeoX layer of NNTile Python package
#
# @version 1.1.0

from typing import Optional

import numpy as np
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as GPTNeoXAttention_torch,
    GPTNeoXConfig as GPTNeoXConfig_torch)

import nntile.utils.constructors as nntc
from nntile.layer.base_layer import BaseLayer
from nntile.layer.cache_utils import KVCache
from nntile.tensor import (
    Tensor, Tensor_bool, TensorMoments, TensorOrNone, TensorTraits,
    add_fiber_inplace_async, add_slice_inplace_async, clear_async,
    copy_intersection_async, gemm_async, mask_scalar_async, maxsumexp_async,
    notrans, prod_inplace_async, rope_async, rope_backward_async,
    softmax_inplace_async, sum_fiber_async, sumprod_slice_async, to_numpy,
    trans, transpose_async)

from ..model.gpt_neox_config import GPTNeoXConfig


# GPTNeoX Self-Attention with Rotary Embeddings
# Inputs:
#  x: (n_emb, n_seq, n_batch) tensor
# Output:
#  y: (n_emb, n_seq, n_batch) tensor
class GPTNeoXAttention(BaseLayer):
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
    v_transposed: TensorMoments
    v: TensorMoments
    a: TensorMoments
    a_maxsumexp: Tensor
    a_sumprod_slice: Tensor
    b: TensorMoments
    b_transposed: TensorMoments
    sin: Tensor
    cos: Tensor
    causal_mask: TensorOrNone
    n_emb: int
    n_emb_kv: int
    n_seq: int
    n_batch: int
    n_head: int
    head_size: int
    redux: bool
    n_head_tile: int
    n_emb_tile: int
    rotary_pct: float

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
        v_transposed: TensorMoments,
        v: TensorMoments,
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
        rotary_pct: float,
        mask: TensorOrNone = None,
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
                v_transposed,
                v,
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
        self.sin = sin
        self.cos = cos
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        self.out_proj_bias = out_proj_bias
        self.n_head = w_q.value.shape[0]
        self.n_head_tile = w_q.value.basetile_shape[0]
        self.n_emb_tile = x.value.basetile_shape[0]
        n_emb, n_seq, n_batch = x.value.shape
        head_size = n_emb // self.n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * self.n_head:
            raise RuntimeError
        self.n_emb = n_emb
        self.n_seq = n_seq
        self.n_batch = n_batch
        self.head_size = head_size
        self.rotary_pct = rotary_pct
        self.causal_mask = mask
        if mask:
            self.val = -np.float32(np.inf)
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple(
        x: TensorMoments,
        n_head: int,
        n_head_tile: int,
        position_ids: np.ndarray,
        rotary_pct: float,
        theta: float,
        attention_bias: bool = False,
        mask: np.ndarray = None,
        redux: bool = False,
    ):
        # Get sizes
        n_emb, n_seq, n_batch = x.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x.value.basetile_shape
        head_size = n_emb // n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * n_head:
            raise RuntimeError

        # Fixed for now
        head_size_tile = head_size
        # Define shape of each tensor
        w_q_shape = [n_head, head_size, n_emb]
        w_k_shape = [n_head, head_size, n_emb]
        w_v_shape = [n_head, head_size, n_emb]
        w_shape = [n_emb, n_head, head_size]
        q_transposed_shape = [
            n_head,
            head_size,
            n_seq,
            n_batch,
        ]
        q_shape = [head_size, n_seq, n_batch, n_head]
        q_rope_shape = [head_size, n_seq, n_batch, n_head]
        k_transposed_shape = [n_head, head_size, n_seq, n_batch]
        k_shape = [head_size, n_seq, n_batch, n_head]
        k_rope_shape = [head_size, n_seq, n_batch, n_head]
        v_transposed_shape = [n_head, head_size, n_seq, n_batch]
        v_shape = [head_size, n_seq, n_batch, n_head]
        a_shape = [n_seq, n_seq, n_batch, n_head]
        a_maxsumexp_shape = [2, n_seq, n_batch, n_head]
        a_sumprod_slice_shape = [n_seq, n_batch, n_head]
        b_shape = [head_size, n_seq, n_batch, n_head]
        b_transposed_shape = [
            n_head,
            head_size,
            n_seq,
            n_batch,
        ]
        cos_shape = [head_size // 2, n_seq, n_batch]
        sin_shape = [head_size // 2, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [
            n_head_tile,
            head_size_tile,
            n_emb_tile,
        ]
        w_k_basetile = [n_head_tile, head_size_tile, n_emb_tile]
        w_v_basetile = [n_head_tile, head_size_tile, n_emb_tile]
        w_basetile = [n_emb_tile, n_head_tile, head_size_tile]
        q_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile
        ]
        q_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile
        ]
        q_rope_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile,
        ]
        k_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile
        ]
        k_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile
        ]
        k_rope_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile,
        ]
        v_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile
        ]
        v_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile
        ]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile, n_head_tile]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile, n_head_tile]
        a_sumprod_slice_basetile = [n_seq_tile, n_batch_tile, n_head_tile]
        b_basetile = [
            head_size_tile,
            n_seq_tile,
            n_batch_tile,
            n_head_tile
        ]
        b_transposed_basetile = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile
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
        v_transposed_traits = TensorTraits(
            v_transposed_shape, v_transposed_basetile
        )
        v_traits = TensorTraits(v_shape, v_basetile)
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
        v_transposed_distr = [0] * v_transposed_traits.grid.nelems
        v_distr = [0] * v_traits.grid.nelems
        a_distr = [0] * a_traits.grid.nelems
        a_maxsumexp_distr = [0] * a_maxsumexp_traits.grid.nelems
        a_sumprod_slice_distr = [0] * a_sumprod_slice_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        b_transposed_distr = [0] * b_transposed_traits.grid.nelems
        if attention_bias:
            in_proj_bias_q_traits = TensorTraits(
                [head_size, n_head],
                [head_size_tile, n_head_tile],
            )
            in_proj_bias_q_distr = [0] * in_proj_bias_q_traits.grid.nelems
            in_proj_bias_kv_traits = TensorTraits(
                [head_size, n_head], [head_size_tile, n_head_tile]
            )
            in_proj_bias_kv_distr = [0] * in_proj_bias_kv_traits.grid.nelems
        cos_distr = [0] * cos_traits.grid.nelems
        sin_distr = [0] * sin_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x.value)(w_q_traits, w_q_distr)
        w_q_grad = type(x.value)(w_q_traits, w_q_distr)
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if attention_bias:
            in_proj_bias_q_value = type(x.value)(
                in_proj_bias_q_traits, in_proj_bias_q_distr
            )
            in_proj_bias_q_grad = type(x.value)(
                in_proj_bias_q_traits, in_proj_bias_q_distr
            )
            bias_inproj_q = TensorMoments(
                in_proj_bias_q_value, in_proj_bias_q_grad, True
            )
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x.value)(w_k_traits, w_k_distr)
        w_k_grad = type(x.value)(w_k_traits, w_k_distr)
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if attention_bias:
            in_proj_bias_k_value = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr
            )
            in_proj_bias_k_grad = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr
            )
            bias_inproj_k = TensorMoments(
                in_proj_bias_k_value, in_proj_bias_k_grad, True
            )
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x.value)(w_v_traits, w_v_distr)
        w_v_grad = type(x.value)(w_v_traits, w_v_distr)
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if attention_bias:
            in_proj_bias_v_value = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr
            )
            in_proj_bias_v_grad = type(x.value)(
                in_proj_bias_kv_traits, in_proj_bias_kv_distr
            )
            bias_inproj_v = TensorMoments(
                in_proj_bias_v_value, in_proj_bias_v_grad, True
            )
        else:
            bias_inproj_v = None
        # w
        w_value = type(x.value)(w_traits, w_distr)
        w_grad = type(x.value)(w_traits, w_distr)
        w = TensorMoments(w_value, w_grad, True)
        # q_transposed
        q_transposed_value = type(x.value)(
            q_transposed_traits, q_transposed_distr
        )
        q_transposed_grad = type(x.value)(
            q_transposed_traits, q_transposed_distr
        )
        q_transposed = TensorMoments(
            q_transposed_value, q_transposed_grad, True
        )
        # q
        q_value = type(x.value)(q_traits, q_distr)
        q_grad = type(x.value)(q_traits, q_distr)
        q = TensorMoments(q_value, q_grad, True)
        # q_rope
        q_rope_value = type(x.value)(q_rope_traits, q_rope_distr)
        q_rope_grad = type(x.value)(q_rope_traits, q_rope_distr)
        q_rope = TensorMoments(q_rope_value, q_rope_grad, True)
        # k_transposed
        k_transposed_value = type(x.value)(
            k_transposed_traits, k_transposed_distr
        )
        k_transposed_grad = type(x.value)(
            k_transposed_traits, k_transposed_distr
        )
        k_transposed = TensorMoments(
            k_transposed_value, k_transposed_grad, True
        )
        # k
        k_value = type(x.value)(k_traits, k_distr)
        k_grad = type(x.value)(k_traits, k_distr)
        k = TensorMoments(k_value, k_grad, True)
        # k_rope
        k_rope_value = type(x.value)(k_rope_traits, k_rope_distr)
        k_rope_grad = type(x.value)(k_rope_traits, k_rope_distr)
        k_rope = TensorMoments(k_rope_value, k_rope_grad, True)
        # v_transposed
        v_transposed_value = type(x.value)(
            v_transposed_traits, v_transposed_distr
        )
        v_transposed_grad = type(x.value)(
            v_transposed_traits, v_transposed_distr
        )
        v_transposed = TensorMoments(
            v_transposed_value, v_transposed_grad, True
        )
        # v
        v_value = type(x.value)(v_traits, v_distr)
        v_grad = type(x.value)(v_traits, v_distr)
        v = TensorMoments(v_value, v_grad, True)
        # a
        a_value = type(x.value)(a_traits, a_distr)
        a_grad = type(x.value)(a_traits, a_distr)
        a = TensorMoments(a_value, a_grad, True)
        # a_maxsumexp
        a_maxsumexp = type(x.value)(
            a_maxsumexp_traits, a_maxsumexp_distr
        )
        # a_sumprod_slice
        a_sumprod_slice = type(x.value)(
            a_sumprod_slice_traits, a_sumprod_slice_distr
        )
        # b
        b_value = type(x.value)(b_traits, b_distr)
        b_grad = type(x.value)(b_traits, b_distr)
        b = TensorMoments(b_value, b_grad, True)
        # b_transposed
        b_transposed_value = type(x.value)(
            b_transposed_traits, b_transposed_distr
        )
        b_transposed_grad = type(x.value)(
            b_transposed_traits, b_transposed_distr
        )
        b_transposed = TensorMoments(
            b_transposed_value, b_transposed_grad, True
        )
        cos = type(x.value)(cos_traits, cos_distr)
        sin = type(x.value)(sin_traits, sin_distr)
        # Allocate tensors for bias for q, k, v and output projection
        if attention_bias:
            out_proj_bias_traits = TensorTraits([n_emb], [n_emb_tile])
            out_proj_bias_distr = [0] * out_proj_bias_traits.grid.nelems
            out_proj_bias_value = type(x.value)(
                out_proj_bias_traits, out_proj_bias_distr
            )
            out_proj_bias_grad = type(x.value)(
                out_proj_bias_traits, out_proj_bias_distr
            )
            out_proj_bias = TensorMoments(
                out_proj_bias_value, out_proj_bias_grad, True
            )
        else:
            out_proj_bias = None
        # Allocate tensor for output y
        y_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        y_value = type(x.value)(y_traits, x.value.distribution)
        y_grad = type(x.value)(y_traits, x.value.distribution)
        y = TensorMoments(y_value, y_grad, True)

        # Fill sin, cos tensors:
        rot_imitation_border = int((head_size // 2) * rotary_pct)
        rope_size = int(head_size * rotary_pct)
        inv_freq = 1.0 / (theta
                ** (np.arange(0, head_size, 2, dtype=np.float32) / rope_size))
        freq_frame = np.empty((head_size // 2, n_seq, n_batch))
        for i in range(n_batch):
            freq_frame[:, :, i] = np.outer(inv_freq, position_ids[i, :])
        np_freqs = np.array(freq_frame, dtype=np.float32, order='F')
        np_cos = np.cos(np_freqs)
        np_sin = np.sin(np_freqs)

        # Set up the "dummy" rotations so trigonometric tensors
        # are always of the same shape but elements outside the "rotary" part
        # in the target tensors Q and K are not actually moved
        np_cos[rot_imitation_border:, :, :] = 1.0
        np_sin[rot_imitation_border:, :, :] = 0.0
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
            )
            layer_mask.from_array(mask)
        else:
            layer_mask = None

        # Create attention layer with all the provided data
        layer = GPTNeoXAttention(
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
            v_transposed,
            v,
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
            rotary_pct,
            layer_mask,
            redux=redux,
        )
        # Return layer and next tag to be used
        return layer

    # Forward propagation of the attention layer
    def forward_async(self):
        # Compute query, key and value tensors
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
        # Q_transposed can be deleted
        self.q_transposed.value.invalidate_submit()
        # X_Q and W_Q can be offloaded from GPU
        self.x_q.value.wont_use()
        self.w_q.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_q is not None:
            # batched add_fiber_inplace
            # (head_size, n_head)
            # into
            # (head_size, n_seq, n_batch, n_head)
            add_fiber_inplace_async(
                1, self.in_proj_bias_q.value, 1, self.q.value, 0, 1
            )
            self.in_proj_bias_q.value.wont_use()
        # Perform RoPE on Q
        rope_async(self.sin, self.cos, self.q.value, self.q_rope.value)
        # Q can be deleted
        self.q.value.invalidate_submit()
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
        # K_transposed can be deleted
        self.k_transposed.value.invalidate_submit()
        # X_K and W_K can be offloaded from GPU
        self.x_k.value.wont_use()
        self.w_k.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_k is not None:
            # batched add_fiber_inplace (head_size, n_head) into
            # (head_size, n_seq, n_batch, n_head)
            add_fiber_inplace_async(
                1, self.in_proj_bias_k.value, 1, self.k.value, 0, 1
            )
            self.in_proj_bias_k.value.wont_use()
        # Perform RoPE on K
        rope_async(self.sin, self.cos, self.k.value, self.k_rope.value)
        # K can be deleted
        self.k.value.invalidate_submit()

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
            # batched add_fiber_inplace (head_size, n_head) into
            # (head_size, n_seq, n_batch, n_head)
            add_fiber_inplace_async(
                1, self.in_proj_bias_v.value, 1, self.v.value, 0, 1
            )
            self.in_proj_bias_v.value.wont_use()

        self._attention_fwd()

        # Rotate axes (head_size, n_seq, n_batch, n_head) into
        # (n_head, head_size, n_seq, n_batch)
        transpose_async(1.0, self.b.value, self.b_transposed.value, 3)
        # B can be offloaded from GPU
        self.b.value.wont_use()
        # Y = einsum('jklm,klmni->jni', W, B_transposed)
        # gemm (n_emb, n_head, head_size) by
        # (n_head, head_size, n_seq, n_batch)
        # into (n_emb, n_seq, n_batch)
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
        # W and B_transposed can be offloaded from GPU
        self.w.value.wont_use()
        self.b_transposed.value.wont_use()
        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_inplace_async(
                1.0, self.out_proj_bias.value, 1.0, self.y.value, 0, 0
            )
            self.out_proj_bias.value.wont_use()
        self.y.value.wont_use()

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
        self.b_transposed.value.invalidate_submit()
        # W_out can be offloaded from GPU
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
            # (head_size, n_seq, n_batch, n_head)
            transpose_async(1.0, self.b_transposed.grad, self.b.grad, 1)
        # dB_transposed can be deleted
        self.b_transposed.grad.invalidate_submit()

        # Apply backward to (attention to Q_rope, K_rope and V into B)
        self._attention_bwd()

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

        # Backward for RoPE
        if self.k.grad_required:
            rope_backward_async(
                    self.sin, self.cos, self.k_rope.grad, self.k.grad
            )
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
            # Rotate axes (head_size, n_seq, n_batch, n_head) into
            # (n_head, head_size, n_seq, n_batch)
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
        self.q.grad.invalidate_submit()
        # dQ_rope can be deleted
        self.q_rope.grad.invalidate_submit()
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
        self.q_transposed.grad.invalidate_submit()

    @staticmethod
    def rotate_tensor_in(
        x: np.ndarray,
        axis: int,
        rotary_pct: float
    ) -> np.ndarray:
        # Calculate the number of elements to rotate based on the rotary_pct
        k_elements = int(x.shape[axis] * rotary_pct)
        if axis == 0:
            new_shape = (1, k_elements, np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), k_elements, 1)
        else:
            new_shape = (
                    np.prod(x.shape[:axis]),
                    k_elements,
                    np.prod(x.shape[axis + 1:])
            )
        # Select first k_elements from the target axis
        if axis == 0:
            x_selected = x[:k_elements, ...]
        elif axis == x.ndim - 1:
            x_selected = x[..., :k_elements]
        else:
            # Create slice for the target axis
            slice_obj = [slice(None)] * x.ndim
            slice_obj[axis] = slice(0, k_elements)
            x_selected = x[tuple(slice_obj)]

        x_reshaped = x_selected.reshape(new_shape)
        mid = k_elements // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
        y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]

        # Create output tensor with same shape as input
        result = x.copy()

        # Place the rotated sub-tensor into the output tensor
        if axis == 0:
            result[:k_elements, ...] = y_reshaped.reshape(x_selected.shape)
        elif axis == x.ndim - 1:
            result[..., :k_elements] = y_reshaped.reshape(x_selected.shape)
        else:
            slice_obj = [slice(None)] * x.ndim
            slice_obj[axis] = slice(0, k_elements)
            result[tuple(slice_obj)] = y_reshaped.reshape(x_selected.shape)
        return result

    @staticmethod
    def rotate_tensor_out(
        x: np.ndarray,
        axis: int,
        rotary_pct: float
    ) -> np.ndarray:
        # Calculate the number of elements to rotate based on the rotary_pct
        k_elements = int(x.shape[axis] * rotary_pct)
        if axis == 0:
            new_shape = (1, k_elements, np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), k_elements, 1)
        else:
            new_shape = (
                    np.prod(x.shape[:axis]),
                    k_elements,
                    np.prod(x.shape[axis + 1:])
            )
        # Select first k_elements from the target axis
        if axis == 0:
            x_selected = x[:k_elements, ...]
        elif axis == x.ndim - 1:
            x_selected = x[..., :k_elements]
        else:
            # Create slice for the target axis
            slice_obj = [slice(None)] * x.ndim
            slice_obj[axis] = slice(0, k_elements)
            x_selected = x[tuple(slice_obj)]

        x_reshaped = x_selected.reshape(new_shape)
        mid = k_elements // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, :mid, :] = x_reshaped[:, 0::2, :]
        y_reshaped[:, mid:, :] = x_reshaped[:, 1::2, :]

        # Create output tensor with same shape as input
        result = x.copy()

        # Place the rotated sub-tensor into the output tensor
        if axis == 0:
            result[:k_elements, ...] = y_reshaped.reshape(x_selected.shape)
        elif axis == x.ndim - 1:
            result[..., :k_elements] = y_reshaped.reshape(x_selected.shape)
        else:
            slice_obj = [slice(None)] * x.ndim
            slice_obj[axis] = slice(0, k_elements)
            result[tuple(slice_obj)] = y_reshaped.reshape(x_selected.shape)
        return result

    @classmethod
    def from_torch(cls,
        torch_layer: GPTNeoXAttention_torch,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: GPTNeoXConfig,
    ):  # -> Self: does not work with Python 3.10
        n_emb, _, _ = x.value.shape
        layer = cls.generate_simple(
            x,
            n_head=torch_layer.config.num_attention_heads,
            n_head_tile=config.num_heads_tile,
            position_ids=position_ids,
            rotary_pct=torch_layer.config.rotary_pct,
            theta=torch_layer.config.rotary_emb_base,
            attention_bias=torch_layer.config.attention_bias,
            mask=mask,
            redux=config.redux,
        )
        weight_torch = (
            torch_layer.query_key_value.weight.cpu().detach().numpy()
        )
        w_qkv = weight_torch.reshape(layer.n_head, 3 * layer.head_size, n_emb)
        w_q_parts = w_qkv[:, :layer.head_size, :]
        w_k_parts = w_qkv[:, layer.head_size: 2 * layer.head_size, :]
        w_v_parts = w_qkv[:, 2 * layer.head_size: 3 * layer.head_size, :]
        layer.w_q.value.from_array(
            cls.rotate_tensor_in(
                w_q_parts.reshape(*layer.w_q.value.shape),
                1,
                layer.rotary_pct
            )
        )
        layer.w_k.value.from_array(
            cls.rotate_tensor_in(
                w_k_parts.reshape(*layer.w_k.value.shape),
                1,
                layer.rotary_pct
            )
        )
        layer.w_v.value.from_array(
            w_v_parts.reshape(*layer.w_v.value.shape)
        )

        layer.w.value.from_array(
            torch_layer.dense.weight
            .cpu()
            .detach()
            .numpy()
            .reshape(n_emb, layer.n_head, layer.head_size)
        )

        if layer.out_proj_bias is not None:
            layer.out_proj_bias.value.from_array(
                torch_layer.dense.bias
                .cpu()
                .detach()
                .numpy()
            )
            bias_torch = (
                torch_layer.query_key_value.bias.cpu().detach().numpy()
            )
            bias_qkv = bias_torch.reshape(layer.n_head, 3 * layer.head_size)
            bias_q = bias_qkv[:, :layer.head_size]
            bias_k = bias_qkv[:, layer.head_size: 2 * layer.head_size]
            bias_v = bias_qkv[:, 2 * layer.head_size: 3 * layer.head_size]
            layer.in_proj_bias_q.value.from_array(
                cls.rotate_tensor_in(
                    bias_q.reshape(layer.n_head, layer.head_size).T,
                    0,
                    layer.rotary_pct
                )
            )
            layer.in_proj_bias_k.value.from_array(
                cls.rotate_tensor_in(
                    bias_k.reshape(layer.n_head, layer.head_size).T,
                    0,
                    layer.rotary_pct
                )
            )
            layer.in_proj_bias_v.value.from_array(
                bias_v.reshape(layer.n_head, layer.head_size).T
            )
        return layer

    def to_torch(self) -> GPTNeoXAttention_torch:
        bias = self.in_proj_bias_q is not None
        torch_layer_config = GPTNeoXConfig_torch(
            hidden_size=self.n_emb,
            num_attention_heads=self.n_head,
            attention_bias=bias,
            use_cache=False,
            attention_dropout=0.0,
            rotary_pct=self.rotary_pct,
        )
        torch_layer = GPTNeoXAttention_torch(torch_layer_config)
        w_qkv_np = np.empty((self.n_head, 3 * self.head_size, self.n_emb))

        w_q_parts = __class__.rotate_tensor_out(
            to_numpy(self.w_q.value), 1, self.rotary_pct
        ).reshape(self.n_head, self.head_size, self.n_emb)
        w_k_parts = __class__.rotate_tensor_out(
            to_numpy(self.w_k.value), 1, self.rotary_pct
        ).reshape(self.n_head, self.head_size, self.n_emb)
        w_v_parts = to_numpy(self.w_v.value).reshape(
            self.n_head, self.head_size, self.n_emb
        )

        w_qkv_np[:, :self.head_size, :] = w_q_parts
        w_qkv_np[:, self.head_size: 2 * self.head_size, :] = w_k_parts
        w_qkv_np[:, 2 * self.head_size: 3 * self.head_size, :] = w_v_parts
        w_qkv_np = w_qkv_np.reshape(3 * self.n_emb, self.n_emb)

        torch_layer.query_key_value.weight.data = torch.tensor(
            w_qkv_np,
            requires_grad=True,
        )

        torch_layer.dense.weight.data = torch.tensor(
            to_numpy(self.w.value)
            .reshape(self.n_emb, self.n_emb),
            requires_grad=True,
        )

        if bias:
            torch_layer.dense.bias.data = torch.tensor(
                to_numpy(self.out_proj_bias.value).flatten(),
                requires_grad=True,
            )
            bias_qkv = np.empty((self.n_head, 3 * self.head_size))
            bias_q = __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_q.value),
                    0,
                    self.rotary_pct
                ).T
            bias_k = __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_k.value),
                    0,
                    self.rotary_pct
                ).T
            bias_v = to_numpy(
                self.in_proj_bias_v.value
            ).T
            bias_qkv[:, :self.head_size] = bias_q
            bias_qkv[:, self.head_size: 2 * self.head_size] = bias_k
            bias_qkv[:, 2 * self.head_size: 3 * self.head_size] = bias_v
            bias_qkv = bias_qkv.reshape(3 * self.n_emb)
            torch_layer.query_key_value.bias.data = torch.tensor(
                bias_qkv,
                requires_grad=True,
            )
        return torch_layer

    def to_torch_with_grads(self) -> GPTNeoXAttention_torch:
        bias = self.in_proj_bias_q is not None
        torch_layer = self.to_torch()
        w_qkv_np = np.empty((self.n_head, 3 * self.head_size, self.n_emb))

        w_q_parts = __class__.rotate_tensor_out(
            to_numpy(self.w_q.grad), 1, self.rotary_pct
        ).reshape(self.n_head, self.head_size, self.n_emb)
        w_k_parts = __class__.rotate_tensor_out(
            to_numpy(self.w_k.grad), 1, self.rotary_pct
        ).reshape(self.n_head, self.head_size, self.n_emb)
        w_v_parts = to_numpy(self.w_v.grad).reshape(
            self.n_head, self.head_size, self.n_emb
        )

        w_qkv_np[:, :self.head_size, :] = w_q_parts
        w_qkv_np[:, self.head_size: 2 * self.head_size, :] = w_k_parts
        w_qkv_np[:, 2 * self.head_size: 3 * self.head_size, :] = w_v_parts
        w_qkv_np = w_qkv_np.reshape(3 * self.n_emb, self.n_emb)

        torch_layer.query_key_value.weight.grad = torch.tensor(
            w_qkv_np
        )

        torch_layer.dense.weight.grad = torch.tensor(
            to_numpy(self.w.grad)
            .reshape(self.n_emb, self.n_emb)
        )

        if bias:
            torch_layer.dense.bias.grad = torch.tensor(
                to_numpy(self.out_proj_bias.grad).flatten()
            )
            bias_qkv = np.empty((self.n_head, 3 * self.head_size))
            bias_q = __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_q.grad),
                    0,
                    self.rotary_pct
                ).T
            bias_k = __class__.rotate_tensor_out(
                    to_numpy(self.in_proj_bias_k.grad),
                    0,
                    self.rotary_pct
                ).T
            bias_v = to_numpy(
                self.in_proj_bias_v.grad
            ).T
            bias_qkv[:, :self.head_size] = bias_q
            bias_qkv[:, self.head_size: 2 * self.head_size] = bias_k
            bias_qkv[:, 2 * self.head_size: 3 * self.head_size] = bias_v
            bias_qkv = bias_qkv.reshape(3 * self.n_emb)
            torch_layer.query_key_value.bias.grad = torch.tensor(
                bias_qkv
            )
        return torch_layer

    def _attention_fwd(self):
        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K, Q)
        # single batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (head_size, n_seq, batch=n_batch, batch=n_head) into
        # (n_seq, n_seq, batch=n_batch, batch=n_head)
        gemm_async(
            1.0 / (self.head_size ** 0.5),
            trans,
            self.k_rope.value,
            notrans,
            self.q_rope.value,
            0.0,
            self.a.value,
            1,
            2,
            redux=self.redux,
        )
        clear_async(self.a_maxsumexp)
        # Q_rope can be offloaded from GPU
        self.q_rope.value.wont_use()
        # K can be offloaded from GPU
        self.k_rope.value.wont_use()

        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.causal_mask:
            mask_scalar_async(self.causal_mask, self.val, self.a.value, 2)
            self.causal_mask.wont_use()

        # Calculate max and sumexp along axis
        maxsumexp_async(self.a.value, self.a_maxsumexp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(self.a_maxsumexp, 1.0, self.a.value, 0)
        # A_maxsumexp can be deleted
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

    def _attention_bwd(self):
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
        # Calculate A_sumprod_slice based on B and its grad
        if self.a.grad_required:
            # A_sumprod_slice = einsum('kmlb,kmlb->mlb', B, dB)
            sumprod_slice_async(1.0, self.b.value, self.b.grad,
                    0.0, self.a_sumprod_slice, 0, redux=self.redux)
        # B and dB can be deleted
        self.b.value.invalidate_submit()
        self.b.grad.invalidate_submit()
        # Backward for A = softmax(A, axis=0)
        if self.a.grad_required:
            # dA += -bias('kmlb,mlb->kmlb', dA, A_sumprod_slice)
            add_slice_inplace_async(
                -1.0, self.a_sumprod_slice, 1.0, self.a.grad, 0
            )
            # A_sumprod_slice can be deleted
            self.a_sumprod_slice.invalidate_submit()
            # dA *= A
            prod_inplace_async(self.a.value, self.a.grad)
        # A can be deleted
        self.a.value.invalidate_submit()
        # Backward for mask if needed
        if self.causal_mask:
            mask_scalar_async(self.causal_mask, 0.0, self.a.grad, 2)
            self.causal_mask.wont_use()
        # Backward for:
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K_rope, Q_rope)
        if self.k_rope.grad_required:
            # dK = 1.0/sqrt(head_size)
            #          * einsum('jmlb,kmlb->jklb', Q_rope, dA)
            gemm_async(
                1.0 / self.head_size**0.5,
                notrans,
                self.q_rope.value,
                trans,
                self.a.grad,
                0.0,
                self.k_rope.grad,
                1,
                2,
                redux=self.redux,
            )
        if self.q_rope.grad_required:
            # dQ_rope = 1.0/sqrt(head_size)
            #      * einsum('jklb,kmlb->jmlb', K_rope, dA)
            gemm_async(
                1.0 / self.head_size**0.5,
                notrans,
                self.k_rope.value,
                notrans,
                self.a.grad,
                0.0,
                self.q_rope.grad,
                1,
                2,
                redux=self.redux,
            )
        # Q_rope can be deleted
        self.q_rope.value.invalidate_submit()
        # K can be deleted
        self.k_rope.value.invalidate_submit()
        # dA can be deleted
        self.a.grad.invalidate_submit()

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

    def _forward_mlp_q_dynamic(self, x: Tensor, kv_cache_size: int):
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

        # Apply bias if needed
        if self.in_proj_bias_q is not None:
            add_fiber_inplace_async(
                1, self.in_proj_bias_q.value, 1, q_partial, 0, 1
            )

        # Apply RoPE to Q
        q_rope_partial = self._get_tmp_for_cache(x)
        current_seq_len = q_partial.shape[1]

        # Create sliced sin/cos tensors to match current sequence length
        sin_sliced = nntc.empty(
            (self.sin.shape[0], current_seq_len, q_partial.shape[2]),
            dtype=type(self.sin),
            basetile_shape=(
                self.sin.shape[0],
                current_seq_len,
                q_partial.shape[2]
            ),
        )
        cos_sliced = nntc.empty(
            (self.cos.shape[0], current_seq_len, q_partial.shape[2]),
            dtype=type(self.cos),
            basetile_shape=(
                self.cos.shape[0],
                current_seq_len,
                q_partial.shape[2]
            ),
        )

        # Copy the relevant slice from the original sin/cos tensors
        copy_intersection_async(
            self.sin, [0, 0, 0], sin_sliced, [0, kv_cache_size, 0]
        )
        copy_intersection_async(
            self.cos, [0, 0, 0], cos_sliced, [0, kv_cache_size, 0]
        )

        rope_async(sin_sliced, cos_sliced, q_partial, q_rope_partial)

        return q_rope_partial

    def _forward_mlp_k_dynamic(self, x: Tensor, kv_cache_size: int):
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

        # Apply bias if needed
        if self.in_proj_bias_k is not None:
            add_fiber_inplace_async(
                1, self.in_proj_bias_k.value, 1, k_partial, 0, 1
            )

        # Apply RoPE to K
        k_rope_partial = self._get_tmp_for_cache(x)
        current_seq_len = k_partial.shape[1]

        # Create sliced sin/cos tensors to match current sequence length
        sin_sliced = nntc.empty(
            (self.sin.shape[0], current_seq_len, k_partial.shape[2]),
            dtype=type(self.sin),
            basetile_shape=(
                self.sin.shape[0],
                current_seq_len,
                k_partial.shape[2]
            ),
        )
        cos_sliced = nntc.empty(
            (self.cos.shape[0], current_seq_len, k_partial.shape[2]),
            dtype=type(self.cos),
            basetile_shape=(
                self.cos.shape[0],
                current_seq_len,
                k_partial.shape[2]
            ),
        )

        # Copy the relevant slice from the original sin/cos tensors
        copy_intersection_async(
            self.sin, [0, 0, 0], sin_sliced, [0, kv_cache_size, 0]
        )
        copy_intersection_async(
            self.cos, [0, 0, 0], cos_sliced, [0, kv_cache_size, 0]
        )

        rope_async(sin_sliced, cos_sliced, k_partial, k_rope_partial)

        return k_rope_partial

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

        # Apply bias if needed
        if self.in_proj_bias_v is not None:
            add_fiber_inplace_async(
                1, self.in_proj_bias_v.value, 1, v_partial, 0, 1
            )

        return v_partial

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
            basetile_shape=(2,)
            + tuple(a_tmp.shape[1:-1])
            + (self.n_head_tile,),
        )
        b_tmp = nntc.empty(
            q.shape,
            dtype=type(q),
            basetile_shape=tuple(q.shape[:-1]) + (self.n_head_tile,),
        )  # (head_size, n_seq, n_batch, n_head)
        b_tr_tmp = nntc.empty(
            (self.n_head, self.head_size) + tuple(q.shape[1:3]),
            dtype=type(q),
            basetile_shape=(self.n_head_tile, self.head_size)
            + tuple(q.shape[1:3]),
        )  # (n_head, head_size, n_seq, n_batch)
        self.y_tensor = nntc.empty(
            (self.n_emb,) + tuple(q.shape[1:3]),
            dtype=type(q),
            basetile_shape=(self.n_emb_tile,) + tuple(q.shape[1:3]),
        )  # (n_emb, n_seq, n_batch)
        y_tensor = self.y_tensor

        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K_rope, Q_rope)
        # single batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (head_size, n_seq, batch=n_batch, batch=n_head) into
        # (n_seq, n_seq, batch=n_batch, batch=n_head)
        # Note: q and k already have RoPE applied from the MLP functions
        gemm_async(
            1.0 / (self.head_size ** 0.5),
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
        if self.causal_mask:
            mask_tmp = nntc.empty(a_tmp.shape[:2], dtype=Tensor_bool)
            copy_intersection_async(
                self.causal_mask,
                [0, 0],
                mask_tmp,
                [0, k.shape[1] - q.shape[1]]
            )
            mask_scalar_async(mask_tmp, self.val, a_tmp, 2)

        # Calculate max and sumexp along axis
        maxsumexp_async(a_tmp, a_maxsumexp_tmp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(a_maxsumexp_tmp, 1.0, a_tmp, 0)

        # Apply value tensor
        # B = einsum('jklb,kmlb->jmlb', V, A)
        # batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (n_seq, n_seq, batch=n_batch, batch=n_head) into
        # (head_size, n_seq, batch=n_batch, batch=n_head)
        gemm_async(
            1.0, notrans, v, notrans, a_tmp, 0.0, b_tmp, 1, 2, redux=self.redux
        )
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
        b_tr_tmp.wont_use()

        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_inplace_async(
                1.0, self.out_proj_bias.value, 1.0, y_tensor, 0, 0
            )
            self.out_proj_bias.value.wont_use()
        return y_tensor

    def forward_dynamic(
            self, x: TensorMoments, kv_cache: Optional[KVCache] = None
        ):
        if (kv_cache is not None) and (x.value.shape[1] + len(kv_cache) > self.x.value.shape[1]):  # noqa: E501
            raise Exception(
                "Overload internal state: "
                f"try add {x.value.shape[1]} "
                f"to {len(kv_cache)}, max: {self.x.value.shape[1]}. "
            )

        # Compute query, key and value tensors
        if kv_cache is not None:
            q_partial = self._forward_mlp_q_dynamic(x.value, len(kv_cache))
            k_partial = self._forward_mlp_k_dynamic(x.value, len(kv_cache))
        else:
            q_partial = self._forward_mlp_q_dynamic(x.value, 0)
            k_partial = self._forward_mlp_k_dynamic(x.value, 0)
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
