# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/bert_selfattention.py
# Selfattention layer for the Bert model of NNTile Python package
#
# @version 1.1.0

import numpy as np
import torch
from transformers.models.bert.modeling_bert import (
    BertConfig as BertConfig_torch,
    BertSelfAttention as BertSelfAttention_torch)

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, Tensor_bool, TensorMoments, TensorTraits, add_fiber_inplace_async,
    add_slice_inplace_async, clear_async, gemm_async, mask_scalar_async,
    maxsumexp_async, notrans, prod_inplace_async, softmax_inplace_async,
    sum_fiber_async, sumprod_slice_async, to_numpy, trans, transpose_async)

from ..model.bert_config import BertConfigNNTile


# Multi-head attention
# Inputs:
#  x_q: (n_emb, n_seq, n_batch) tensor
#  x_k: (n_emb_k, n_seq, n_batch) tensor
#  x_v: (n_emb_v, n_seq, n_batch) tensor
# Output:
#  y: () tensor
class BertSelfAttention(BaseLayer):
    x_q: TensorMoments
    x_k: TensorMoments
    x_v: TensorMoments
    y: TensorMoments
    w_q: TensorMoments
    w_k: TensorMoments
    w_v: TensorMoments
    # w: TensorMoments
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
    # b_transposed: TensorMoments
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
        # w: TensorMoments,
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
        # b_transposed: TensorMoments,
        in_proj_bias_q: TensorMoments,
        in_proj_bias_k: TensorMoments,
        in_proj_bias_v: TensorMoments,
        # out_proj_bias: TensorMoments,
        mask=None,
        redux: bool = False
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
        # if out_proj_bias:
        #     bias_list_out_proj = [out_proj_bias]
        #     out_proj_bias.grad.set_reduction_add()
        # else:
        #     bias_list_out_proj = []
        # Redirect to BaseClass initialization
        super().__init__(
            [x_q, x_k, x_v],
            [y],
            [w_q, w_k, w_v] + qkv_bias_list,
            [
                q_transposed,
                q,
                k_transposed,
                k, v_transposed,
                v,
                a,
                a_maxsumexp,
                a_sumprod_slice,
                b,
                # b_transposed
            ]
        )
        if mask is not None:
            self.temporaries.append(mask)
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
        # self.w = w
        # self.w.grad.set_reduction_add()
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
        # self.b_transposed = b_transposed
        # self.b_transposed.grad.set_reduction_add()
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        # self.out_proj_bias = out_proj_bias
        self.n_head = w_q.value.shape[0]
        n_emb = x_q.value.shape[0]
        head_size = n_emb // self.n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * self.n_head:
            raise RuntimeError
        self.head_size = head_size
        self.mask = mask
        if mask:
            self.val = -np.float32(np.inf)
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the BertSelfAttention layer
    @staticmethod
    def generate_simple(x_q: TensorMoments,
                        x_k: TensorMoments,
                        x_v: TensorMoments,
                        n_head: int, n_head_tile: int,
                        bias=False, mask=None, redux: bool = False):
        # Get sizes
        n_emb, n_seq, n_batch = x_q.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x_q.value.basetile_shape
        head_size = n_emb // n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * n_head:
            raise RuntimeError
        n_emb_k = x_k.value.shape[0]
        n_emb_k_tile = x_k.value.basetile_shape[0]
        if [n_seq, n_batch] != x_k.value.shape[1:]:
            raise ValueError("Invalid shape of x_k")
        if [n_seq_tile, n_batch_tile] != x_k.value.basetile_shape[1:]:
            raise ValueError("Invalid basetile shape of x_k")
        n_emb_v = x_v.value.shape[0]
        n_emb_v_tile = x_v.value.basetile_shape[0]
        if [n_seq, n_batch] != x_v.value.shape[1:]:
            raise ValueError("Invalid shape of x_v")
        if [n_seq_tile, n_batch_tile] != x_v.value.basetile_shape[1:]:
            raise ValueError("Invalid basetile shape of x_v")
        # Fixed for now
        head_size_tile = head_size
        # Define shape of each tensor
        w_q_shape = [n_head, head_size, n_emb]
        w_k_shape = [n_head, head_size, n_emb_k]
        w_v_shape = [n_head, head_size, n_emb_v]
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
        # b_transposed_shape = [n_head, head_size, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [n_head_tile, head_size_tile, n_emb_tile]
        w_k_basetile = [n_head_tile, head_size_tile, n_emb_k_tile]
        w_v_basetile = [n_head_tile, head_size_tile, n_emb_v_tile]
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
        # b_transposed_basetile = [
        #     n_head_tile,
        #     head_size_tile,
        #     n_seq_tile,
        #     n_batch_tile
        # ]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        q_transposed_traits = TensorTraits(
            q_transposed_shape,
            q_transposed_basetile
        )
        q_traits = TensorTraits(q_shape, q_basetile)
        k_transposed_traits = TensorTraits(
            k_transposed_shape,
            k_transposed_basetile
        )
        k_traits = TensorTraits(k_shape, k_basetile)
        v_transposed_traits = TensorTraits(
            v_transposed_shape,
            v_transposed_basetile
        )
        v_traits = TensorTraits(v_shape, v_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(a_maxsumexp_shape,
                a_maxsumexp_basetile)
        a_sumprod_slice_traits = TensorTraits(
            a_sumprod_slice_shape,
            a_sumprod_slice_basetile
        )
        b_traits = TensorTraits(b_shape, b_basetile)
        # b_transposed_traits = TensorTraits(
        #     b_transposed_shape,
        #     b_transposed_basetile
        # )
        # TODO change distribution
        w_q_distr = [0] * w_q_traits.grid.nelems
        w_k_distr = [0] * w_k_traits.grid.nelems
        w_v_distr = [0] * w_v_traits.grid.nelems
        # w_distr = [0] * w_traits.grid.nelems
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
        # b_transposed_distr = [0] * b_transposed_traits.grid.nelems
        if bias:
            in_proj_bias_qkv_traits = TensorTraits([head_size, n_head],
                                        [head_size_tile, n_head_tile])
            in_proj_bias_qkv_distr = [0] * in_proj_bias_qkv_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x_q.value)(w_q_traits, w_q_distr)
        w_q_grad = type(x_q.value)(w_q_traits, w_q_distr)
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if bias:
            in_proj_bias_q_value = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            in_proj_bias_q_grad = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            bias_inproj_q = TensorMoments(
                in_proj_bias_q_value,
                in_proj_bias_q_grad,
                True
            )
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x_q.value)(w_k_traits, w_k_distr)
        w_k_grad = type(x_q.value)(w_k_traits, w_k_distr)
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if bias:
            in_proj_bias_k_value = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            in_proj_bias_k_grad = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            bias_inproj_k = TensorMoments(
                in_proj_bias_k_value,
                in_proj_bias_k_grad,
                True
            )
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x_q.value)(w_v_traits, w_v_distr)
        w_v_grad = type(x_q.value)(w_v_traits, w_v_distr)
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if bias:
            in_proj_bias_v_value = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            in_proj_bias_v_grad = type(x_q.value)(
                in_proj_bias_qkv_traits,
                in_proj_bias_qkv_distr
            )
            bias_inproj_v = TensorMoments(
                in_proj_bias_v_value,
                in_proj_bias_v_grad,
                True
            )
        else:
            bias_inproj_v = None
        # q_transposed
        q_transposed_value = type(x_q.value)(
            q_transposed_traits,
            q_transposed_distr
        )
        q_transposed_grad = type(x_q.value)(
            q_transposed_traits,
            q_transposed_distr
        )
        q_transposed = TensorMoments(
            q_transposed_value,
            q_transposed_grad,
            True
        )
        # q
        q_value = type(x_q.value)(q_traits, q_distr)
        q_grad = type(x_q.value)(q_traits, q_distr)
        q = TensorMoments(q_value, q_grad, True)
        # k_transposed
        k_transposed_value = type(x_q.value)(
            k_transposed_traits,
            k_transposed_distr
        )
        k_transposed_grad = type(x_q.value)(
            k_transposed_traits,
            k_transposed_distr
        )
        k_transposed = TensorMoments(
            k_transposed_value,
            k_transposed_grad,
            True
        )
        # k
        k_value = type(x_q.value)(k_traits, k_distr)
        k_grad = type(x_q.value)(k_traits, k_distr)
        k = TensorMoments(k_value, k_grad, True)
        # v_transposed
        v_transposed_value = type(x_q.value)(
            v_transposed_traits,
            v_transposed_distr
        )
        v_transposed_grad = type(x_q.value)(
            v_transposed_traits,
            v_transposed_distr
        )
        v_transposed = TensorMoments(
            v_transposed_value,
            v_transposed_grad,
            True
        )
        # v
        v_value = type(x_q.value)(v_traits, v_distr)
        v_grad = type(x_q.value)(v_traits, v_distr)
        v = TensorMoments(v_value, v_grad, True)
        # a
        a_value = type(x_q.value)(a_traits, a_distr)
        a_grad = type(x_q.value)(a_traits, a_distr)
        a = TensorMoments(a_value, a_grad, True)
        # a_maxsumexp
        a_maxsumexp = type(x_q.value)(a_maxsumexp_traits, a_maxsumexp_distr)
        # a_sumprod_slice
        a_sumprod_slice = type(x_q.value)(
            a_sumprod_slice_traits,
            a_sumprod_slice_distr
        )
        # b
        b_value = type(x_q.value)(b_traits, b_distr)
        b_grad = type(x_q.value)(b_traits, b_distr)
        b = TensorMoments(b_value, b_grad, True)
        # Allocate tensor for output y
        y_shape = [n_head, head_size, n_seq, n_batch]
        y_basetile_shape = [
            n_head_tile,
            head_size_tile,
            n_seq_tile,
            n_batch_tile
        ]
        y_traits = TensorTraits(y_shape, y_basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x_q.value)(y_traits, y_distr)
        y_grad = type(x_q.value)(y_traits, y_distr)
        y = TensorMoments(y_value, y_grad, True)
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
                    [0] * layer_mask_traits.grid.nelems
            )
            layer_mask.from_array(mask)
        else:
            layer_mask = None
        # Create attention layer with all the provided data
        layer = BertSelfAttention(x_q, x_k, x_v, y, w_q, w_k, w_v,
                q_transposed, q, k_transposed, k, v_transposed,
                v, a, a_maxsumexp, a_sumprod_slice, b,
                bias_inproj_q, bias_inproj_k, bias_inproj_v,
                layer_mask, redux=redux)
        # Return layer
        return layer

    # Forward propagation of the attention layer
    def forward_async(self):
        # Compute query, key and value tensors
        # Q_transposed = einsum('jkl,lmn->jkmn', W_Q, X_Q)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(1.0, notrans, self.w_q.value, notrans,
                    self.x_q.value, 0.0, self.q_transposed.value, 1, 0,
                    redux=self.redux)
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
            add_fiber_inplace_async(1, self.in_proj_bias_q.value, 1,
                    self.q.value, 0, 1)
            self.in_proj_bias_q.value.wont_use()
        # K_transposed = einsum('jkl,lmn->jkmn', W_K, X_K)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(1.0, notrans, self.w_k.value, notrans,
                    self.x_k.value, 0.0, self.k_transposed.value, 1, 0,
                    redux=self.redux)
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
            add_fiber_inplace_async(1, self.in_proj_bias_k.value, 1,
                            self.k.value, 0, 1)
            self.in_proj_bias_k.value.wont_use()
        # V_transposed = einsum('jkl,lmn->jkmn', W_V, X_V)
        # gemm (n_head, head_size, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_head, head_size, n_seq, n_batch)
        gemm_async(1.0, notrans, self.w_v.value, notrans,
                    self.x_v.value, 0.0, self.v_transposed.value, 1, 0,
                    redux=self.redux)
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
            add_fiber_inplace_async(1, self.in_proj_bias_v.value, 1,
                            self.v.value, 0, 1)
            self.in_proj_bias_v.value.wont_use()
        # Get tensor for softmax
        # A = 1.0/sqrt(head_size) * einsum('jklb,jmlb->kmlb', K, Q)
        # single batched gemm (head_size, n_seq, batch=n_batch, batch=n_head)
        # by (head_size, n_seq, batch=n_batch, batch=n_head) into
        # (n_seq, n_seq, batch=n_batch, batch=n_head)
        gemm_async(1.0 / self.head_size ** 0.5, trans, self.k.value,
                    notrans, self.q.value, 0.0, self.a.value, 1, 2,
                    redux=self.redux)
        clear_async(self.a_maxsumexp)
        # Q and K can be offloaded from GPU
        self.q.value.wont_use()
        self.k.value.wont_use()
        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.mask:
            mask_scalar_async(self.mask, self.val, self.a.value, 2)
            self.mask.wont_use()
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
        gemm_async(1.0, notrans, self.v.value, notrans,
                    self.a.value, 0.0, self.b.value, 1, 2, redux=self.redux)
        # V and A can be offloaded from GPU
        self.v.value.wont_use()
        self.a.value.wont_use()
        # Accumulate result from all the heads
        # rotate axes (head_size, n_seq, n_batch, n_head) into
        # (n_head, head_size, n_seq, n_batch) and then
        # transpose_async(1.0, self.b.value, self.b_transposed.value, 3)
        transpose_async(1.0, self.b.value, self.y.value, 3)
        # Y = einsum('jkl,klmn->jmn', W, B_transposed)
        # gemm (n_emb, n_head, head_size) by
        # (n_head, head_size, n_seq, n_batch) into (n_emb, n_seq, n_batch)
        # gemm_async(1.0, notrans, self.w.value, notrans,
        #             self.b_transposed.value, 0.0, self.y.value, 2, 0,
        #             redux=self.redux)
        # W, B and B_transposed can be offloaded from GPU
        # self.w.value.wont_use()
        self.b.value.wont_use()
        self.b.value.invalidate_submit()
        # self.b_transposed.value.wont_use()
        # Apply bias if needed
        # if self.out_proj_bias is not None:
        #     add_fiber_inplace_async(
        #         1.0, self.out_proj_bias.value, 1.0, self.y.value, 0, 0
        #     )
        #     self.out_proj_bias.value.wont_use()
        self.y.value.wont_use()

    # Backward propagation of the linear layer
    def backward_async(self):
        # Apply backward of bias if needed
        # if self.out_proj_bias is not None:
        #     if self.out_proj_bias.grad_required:
        #         sum_fiber_async(1.0, self.y.grad, 1.0,
        #                 self.out_proj_bias.grad, 0, 0, redux=self.redux)
        #         self.out_proj_bias.grad.wont_use()
        # Backward for Y = einsum('jkl,klmn->jmn', W, B_transposed)
        # if self.w.grad_required:
        #     # dW += einsum('jmn,klmn->jkl', dY, B_transposed)
        #     gemm_async(1.0, notrans, self.y.grad, trans,
        #                 self.b_transposed.value, 1.0, self.w.grad, 2, 0,
        #                 redux=self.redux)
        # B_transposed can be deleted
        # self.b_transposed.value.wont_use()
        # self.b_transposed.value.invalidate_submit()
        # self.w.grad.wont_use()
        # if self.b_transposed.grad_required:
        #     # dB_transposed = einsum('jkl,jmn->klmn', W, dY)
        #     gemm_async(1.0, trans, self.w.value, notrans, self.y.grad,
        #                 0.0, self.b_transposed.grad, 1, 0, redux=self.redux)
        # W can be offloaded from GPU
        # self.w.value.wont_use()
        # dY can be offloaded from GPU
        # self.y.grad.wont_use()
        # Backward for axes rotation
        if self.b.grad_required:
            # rotate axes (n_head, head_size, n_seq, n_batch) into
            # (head_size, n_seq, n_batch, n_head) and then
            # transpose_async(1.0, self.b_transposed.grad, self.b.grad, 1)
            transpose_async(1.0, self.y.grad, self.b.grad, 1)
        # self.b_transposed.grad.wont_use()
        # self.b_transposed.grad.invalidate_submit()
        self.y.grad.invalidate_submit()
        # Backward for B = einsum('jklb,kmlb->jmlb', V, A)
        if self.a.grad_required:
            # dA = einsum('jklb,jmlb->kmlb', V, dB)
            gemm_async(1.0, trans, self.v.value, notrans,
                        self.b.grad, 0.0, self.a.grad, 1, 2, redux=self.redux)
        # V can be deleted
        # self.v.value.wont_use()
        self.v.value.invalidate_submit()
        if self.v.grad_required:
            # dV = einsum('jmlb,kmlb->jklb', dB, A)
            gemm_async(1.0, notrans, self.b.grad, trans,
                       self.a.value, 0.0, self.v.grad, 1, 2, redux=self.redux)
        # dB can be deleted
        # self.b.grad.wont_use()
        self.b.grad.invalidate_submit()
        # Backward for A = softmax(A, axis=0)
        if self.a.grad_required:
            # A_sumprod_slice = einsum('kmlb,kmlb->mlb', A, dA)
            sumprod_slice_async(1.0, self.a.value, self.a.grad,
                    0.0, self.a_sumprod_slice, 0, redux=self.redux)
            # dA += -bias('kmlb,mlb->kmlb', dA, A_sumprod_slice)
            add_slice_inplace_async(
                -1.0, self.a_sumprod_slice, 1.0, self.a.grad, 0
            )
            # A_sumprod_slice can be deleted
            # self.a_sumprod_slice.wont_use()
            self.a_sumprod_slice.invalidate_submit()
            # dA *= A
            prod_inplace_async(self.a.value, self.a.grad)
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
            gemm_async(1.0 / self.head_size ** 0.5, notrans, self.q.value,
                        trans, self.a.grad, 0.0, self.k.grad, 1, 2,
                        redux=self.redux)
        # Q can be deleted
        # self.q.value.wont_use()
        self.q.value.invalidate_submit()
        if self.q.grad_required:
            # dQ = 1.0/sqrt(head_size) * einsum('jklb,kmlb->jmlb', K, dA)
            gemm_async(1.0 / self.head_size ** 0.5, notrans, self.k.value,
                        notrans, self.a.grad, 0.0, self.q.grad, 1, 2,
                        redux=self.redux)
        # K can be deleted
        # self.k.value.wont_use()
        self.k.value.invalidate_submit()
        # dA can be deleted
        # self.a.grad.wont_use()
        self.a.grad.invalidate_submit()
        # Backward for bias of V
        if self.in_proj_bias_v is not None:
            if self.in_proj_bias_v.grad_required:
                sum_fiber_async(1, self.v.grad, 1, self.in_proj_bias_v.grad,
                        0, 1, redux=self.redux)
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
            gemm_async(1.0, trans, self.w_v.value, notrans,
                        self.v_transposed.grad, 1.0, self.x_v.grad, 2, 0,
                        redux=self.redux)
        # W_V can be offloaded from GPU
        self.w_v.value.wont_use()
        # dX_V can be offloaded from GPU
        self.x_v.grad.wont_use()
        if self.w_v.grad_required:
            # dW_V += einsum('jkmn,lmn->jkl', dV_transposed, X_V)
            gemm_async(1.0, notrans, self.v_transposed.grad, trans,
                        self.x_v.value, 1.0, self.w_v.grad, 2, 0,
                        redux=self.redux)
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
                sum_fiber_async(1, self.k.grad, 1, self.in_proj_bias_k.grad,
                        0, 1, redux=self.redux)
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
            gemm_async(1.0, trans, self.w_k.value, notrans,
                        self.k_transposed.grad, 1.0, self.x_k.grad, 2, 0,
                        redux=self.redux)
        # W_K can be offloaded from GPU
        self.w_k.value.wont_use()
        # dX_K can be offloaded from GPU
        self.x_k.grad.wont_use()
        if self.w_k.grad_required:
            # dW_K += einsum('jkmn,lmn->jkl', dK_transposed, X_K)
            gemm_async(1.0, notrans, self.k_transposed.grad, trans,
                        self.x_k.value, 1.0, self.w_k.grad, 2, 0,
                        redux=self.redux)
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
                sum_fiber_async(1, self.q.grad, 1, self.in_proj_bias_q.grad,
                        0, 1, redux=self.redux)
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
            gemm_async(1.0, trans, self.w_q.value, notrans,
                        self.q_transposed.grad, 1.0, self.x_q.grad, 2, 0,
                        redux=self.redux)
            self.x_q.grad.wont_use()
        # W_Q can be offloaded from GPU
        self.w_q.value.wont_use()
        # dX_Q can be offloaded from GPU
        self.x_q.grad.wont_use()
        if self.w_q.grad_required:
            # dW_Q += einsum('jkmn,lmn->jkl', dQ_transposed, X_Q)
            gemm_async(1.0, notrans, self.q_transposed.grad, trans,
                        self.x_q.value, 1.0, self.w_q.grad, 2, 0,
                        redux=self.redux)
        # dW_Q can be offloaded from GPU
        self.w_q.grad.wont_use()
        # X_Q can be offloaded from GPU
        self.x_q.value.wont_use()
        # dQ_transposed can be deleted
        # self.q_transposed.grad.wont_use()
        self.q_transposed.grad.invalidate_submit()

    @classmethod
    def from_torch(cls,
        torch_layer: BertSelfAttention_torch,
        x_q: TensorMoments, x_k: TensorMoments, x_v: TensorMoments,
        config: BertConfigNNTile
    ):  # -> Self: does not work with Python 3.10
        n_emb, n_seq, _ = x_q.value.shape

        mask_np = np.array(
            np.ones((n_seq, n_seq)), dtype=bool, order="F"
        )
        layer = cls.generate_simple(
            x_q,
            x_k,
            x_v,
            n_head=config.num_attention_heads,
            n_head_tile=config.n_head_tile,
            bias=True,
            mask=mask_np,
            redux=config.redux,
        )

        weight_query_torch_np = torch_layer.query.weight.cpu().detach().numpy()
        layer.w_q.value.from_array(
            weight_query_torch_np.reshape(
                layer.n_head, layer.head_size, n_emb
            )
        )
        weight_key_torch_np = torch_layer.key.weight.cpu().detach().numpy()
        layer.w_k.value.from_array(
            weight_key_torch_np.reshape(
                layer.n_head, layer.head_size, n_emb
            )
        )
        weight_value_torch_np = torch_layer.value.weight.cpu().detach().numpy()
        layer.w_v.value.from_array(
            weight_value_torch_np.reshape(
                layer.n_head, layer.head_size, n_emb
            )
        )

        bias_query_torch_np = torch_layer.query.bias.cpu().detach().numpy()
        layer.in_proj_bias_q.value.from_array(
            bias_query_torch_np.reshape(layer.n_head, layer.head_size).T
        )
        bias_key_torch_np = torch_layer.key.bias.cpu().detach().numpy()
        layer.in_proj_bias_k.value.from_array(
            bias_key_torch_np.reshape(layer.n_head, layer.head_size).T
        )
        bias_value_torch_np = torch_layer.value.bias.cpu().detach().numpy()
        layer.in_proj_bias_v.value.from_array(
            bias_value_torch_np.reshape(layer.n_head, layer.head_size).T
        )
        return layer

    def to_torch(self) -> BertSelfAttention_torch:
        n_emb = self.head_size * self.n_head
        torch_layer_config = BertConfig_torch(
            hidden_size=self.head_size * self.n_head,
            num_attention_heads=self.n_head,
            use_cache=False,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        torch_layer = BertSelfAttention_torch(
            torch_layer_config)

        weight_torch_query_np = to_numpy(
            self.w_q.value
            ).reshape(-1, n_emb)
        torch_layer.query.weight.data = torch.tensor(
            weight_torch_query_np,
            requires_grad=True,
        )
        weight_torch_key_np = to_numpy(
            self.w_k.value
            ).reshape(-1, n_emb)
        torch_layer.key.weight.data = torch.tensor(
            weight_torch_key_np,
            requires_grad=True,
        )
        weight_torch_value_np = to_numpy(
            self.w_v.value
            ).reshape(-1, n_emb)
        torch_layer.value.weight.data = torch.tensor(
            weight_torch_value_np,
            requires_grad=True,
        )

        bias_torch_query_np = to_numpy(
            self.in_proj_bias_q.value
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.query.bias.data = torch.tensor(
            bias_torch_query_np,
            requires_grad=True,
        )
        bias_torch_key_np = to_numpy(
            self.in_proj_bias_k.value
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.key.bias.data = torch.tensor(
            bias_torch_key_np,
            requires_grad=True,
        )
        bias_torch_value_np = to_numpy(
            self.in_proj_bias_v.value
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.value.bias.data = torch.tensor(
            bias_torch_value_np,
            requires_grad=True,
        )
        return torch_layer

    def to_torch_with_grads(self) -> BertSelfAttention_torch:
        n_emb = self.head_size * self.n_head
        torch_layer = self.to_torch()

        weight_grad_torch_query_np = to_numpy(
            self.w_q.grad
            ).reshape(-1, n_emb)
        torch_layer.query.weight.grad = torch.tensor(
            weight_grad_torch_query_np
        )
        weight_grad_torch_key_np = to_numpy(
            self.w_k.grad
            ).reshape(-1, n_emb)
        torch_layer.key.weight.grad = torch.tensor(
            weight_grad_torch_key_np,
        )
        weight_grad_torch_value_np = to_numpy(
            self.w_v.grad
            ).reshape(-1, n_emb)
        torch_layer.value.weight.grad = torch.tensor(
            weight_grad_torch_value_np,
        )

        bias_grad_torch_query_np = to_numpy(
            self.in_proj_bias_q.grad
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.query.bias.grad = torch.tensor(
            bias_grad_torch_query_np,
        )
        bias_grad_torch_key_np = to_numpy(
            self.in_proj_bias_k.grad
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.key.bias.grad = torch.tensor(
            bias_grad_torch_key_np,
        )
        bias_grad_torch_value_np = to_numpy(
            self.in_proj_bias_v.grad
            ).T.reshape(self.n_head * self.head_size)
        torch_layer.value.bias.grad = torch.tensor(
            bias_grad_torch_value_np,
        )

        return torch_layer
