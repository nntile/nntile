# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/attention.py
# Attention layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-07-21

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, clear_async, gemm_async, randn_async, \
        maxsumexp_async, softmax_inplace_async, sumprod_slice_async, \
        add_slice_async, prod_async, mask_scalar_async, add_fiber_async, \
        sum_fiber_async, transpose_async
from nntile.layer.base_layer import BaseLayer
import numpy as np
from typing import List

# Multi-head attention
# Inputs:
#  x_q: (n_emb, n_seq, n_batch) tensor
#  x_k: (n_emb_k, n_seq, n_batch) tensor
#  x_v: (n_emb_v, n_seq, n_batch) tensor
# Output:
#  y: (n_emb, n_seq, n_batch) tensor
class Attention(BaseLayer):
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
    def __init__(self, x_q: TensorMoments, x_k: TensorMoments, \
            x_v: TensorMoments, y: TensorMoments, \
            w_q: TensorMoments, w_k: TensorMoments, \
            w_v: TensorMoments, w: TensorMoments, \
            q_transposed: TensorMoments, q: TensorMoments, \
            k_transposed: TensorMoments, k: TensorMoments, \
            v_transposed: TensorMoments, v: TensorMoments, \
            a: TensorMoments, a_maxsumexp: Tensor, a_sumprod_slice: Tensor, \
            b: TensorMoments, b_transposed: TensorMoments, \
            in_proj_bias_q: TensorMoments, in_proj_bias_k: TensorMoments, \
            in_proj_bias_v: TensorMoments, out_proj_bias: TensorMoments, \
            mask=None):
        qkv_bias_list = []
        if in_proj_bias_q:
            qkv_bias_list.append(in_proj_bias_q)
        if in_proj_bias_k:
            qkv_bias_list.append(in_proj_bias_k)
        if in_proj_bias_v:
            qkv_bias_list.append(in_proj_bias_v)
        if out_proj_bias:
            bias_list_out_proj = [out_proj_bias]
        else:
            bias_list_out_proj = []
        # Redirect to BaseClass initialization
        super().__init__([x_q, x_k, x_v], [y], [w_q, w_k, w_v] + \
                qkv_bias_list + [w] + bias_list_out_proj, \
                [q_transposed, q, k_transposed, k, v_transposed, v, a, \
                a_maxsumexp, a_sumprod_slice, b, b_transposed])
        self.x_q = x_q
        self.x_k = x_k
        self.x_v = x_v
        self.y = y
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w = w
        self.q_transposed = q_transposed
        self.q = q
        self.k_transposed = k_transposed
        self.k = k
        self.v_transposed = v_transposed
        self.v = v
        self.a = a
        self.a_maxsumexp = a_maxsumexp
        self.a_sumprod_slice = a_sumprod_slice
        self.b = b
        self.b_transposed = b_transposed
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        self.out_proj_bias = out_proj_bias
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

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple(x_q: TensorMoments, x_k: TensorMoments, \
            x_v: TensorMoments, n_head: int, next_tag: int, bias=False, \
            mask=None):
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
        # TODO: the following tile size is a hyperparameter
        head_size_tile = x_q.value.basetile_shape[0]
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
        a_sumprod_slice_shape = [n_seq, n_batch]
        b_shape = [head_size, n_seq, n_batch, n_head]
        b_transposed_shape = [n_head, head_size, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [n_head, head_size_tile, n_emb_tile]
        w_k_basetile = [n_head, head_size_tile, n_emb_k_tile]
        w_v_basetile = [n_head, head_size_tile, n_emb_v_tile]
        w_basetile = [n_emb_tile, n_head, head_size_tile]
        q_transposed_basetile = [n_head, head_size_tile, n_seq_tile, n_batch_tile]
        q_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head]
        k_transposed_basetile = [n_head, head_size_tile, n_seq_tile, n_batch_tile]
        k_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head]
        v_transposed_basetile = [n_head, head_size_tile, n_seq_tile, n_batch_tile]
        v_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile, n_head]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile, n_head]
        a_sumprod_slice_basetile = [n_seq_tile, n_batch_tile]
        b_basetile = [head_size_tile, n_seq_tile, n_batch_tile, n_head]
        b_transposed_basetile = [n_head, head_size_tile, n_seq_tile, n_batch_tile]
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
        a_maxsumexp_traits = TensorTraits(a_maxsumexp_shape,
                a_maxsumexp_basetile)
        a_sumprod_slice_traits = TensorTraits(a_sumprod_slice_shape, \
                a_sumprod_slice_basetile)
        b_traits = TensorTraits(b_shape, b_basetile)
        b_transposed_traits = TensorTraits(b_transposed_shape, b_transposed_basetile)
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
            in_proj_bias_qkv_traits = TensorTraits([head_size, n_head], \
                    [head_size_tile, n_head])
            in_proj_bias_qkv_distr = [0] * in_proj_bias_qkv_traits.grid.nelems
        # Define all the lists
        # w_q
        w_q_value = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_value.next_tag
        w_q_grad = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
        next_tag = w_q_grad.next_tag
        w_q = TensorMoments(w_q_value, w_q_grad, True)
        if bias:
            in_proj_bias_q_value = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_q_value.next_tag
            in_proj_bias_q_grad = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_q_grad.next_tag
            bias_inproj_q = TensorMoments(in_proj_bias_q_value, \
                    in_proj_bias_q_grad, True)
        else:
            bias_inproj_q = None
        # w_k
        w_k_value = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_value.next_tag
        w_k_grad = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
        next_tag = w_k_grad.next_tag
        w_k = TensorMoments(w_k_value, w_k_grad, True)
        if bias:
            in_proj_bias_k_value = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_k_value.next_tag
            in_proj_bias_k_grad = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_k_grad.next_tag
            bias_inproj_k = TensorMoments(in_proj_bias_k_value, \
                    in_proj_bias_k_grad, True)
        else:
            bias_inproj_k = None
        # w_v
        w_v_value = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_value.next_tag
        w_v_grad = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
        next_tag = w_v_grad.next_tag
        w_v = TensorMoments(w_v_value, w_v_grad, True)
        if bias:
            in_proj_bias_v_value = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_v_value.next_tag
            in_proj_bias_v_grad = type(x_q.value)( \
                    in_proj_bias_qkv_traits, in_proj_bias_qkv_distr, \
                    next_tag)
            next_tag = in_proj_bias_v_grad.next_tag
            bias_inproj_v = TensorMoments(in_proj_bias_v_value, \
                    in_proj_bias_v_grad, True)
        else:
            bias_inproj_v = None
        # w
        w_value = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_value.next_tag
        w_grad = type(x_q.value)(w_traits, w_distr, next_tag)
        next_tag = w_grad.next_tag
        w = TensorMoments(w_value, w_grad, True)
        # q_transposed
        q_transposed_value = type(x_q.value)(q_transposed_traits, q_transposed_distr, next_tag)
        next_tag = q_transposed_value.next_tag
        q_transposed_grad = type(x_q.value)(q_transposed_traits, q_transposed_distr, next_tag)
        next_tag = q_transposed_grad.next_tag
        q_transposed = TensorMoments(q_transposed_value, q_transposed_grad, True)
        # q
        q_value = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_value.next_tag
        q_grad = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_grad.next_tag
        q = TensorMoments(q_value, q_grad, True)
        # k_transposed
        k_transposed_value = type(x_q.value)(k_transposed_traits, k_transposed_distr, next_tag)
        next_tag = k_transposed_value.next_tag
        k_transposed_grad = type(x_q.value)(k_transposed_traits, k_transposed_distr, next_tag)
        next_tag = k_transposed_grad.next_tag
        k_transposed = TensorMoments(k_transposed_value, k_transposed_grad, True)
        # k
        k_value = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_value.next_tag
        k_grad = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_grad.next_tag
        k = TensorMoments(k_value, k_grad, True)
        # v_transposed
        v_transposed_value = type(x_q.value)(v_transposed_traits, v_transposed_distr, next_tag)
        next_tag = v_transposed_value.next_tag
        v_transposed_grad = type(x_q.value)(v_transposed_traits, v_transposed_distr, next_tag)
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
        a_maxsumexp = type(x_q.value)(a_maxsumexp_traits, a_maxsumexp_distr, \
                next_tag)
        next_tag = a_maxsumexp.next_tag
        # a_sumprod_slice
        a_sumprod_slice = type(x_q.value)(a_sumprod_slice_traits, \
                a_sumprod_slice_distr, next_tag)
        next_tag = a_sumprod_slice.next_tag
        # b
        b_value = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_value.next_tag
        b_grad = type(x_q.value)(b_traits, b_distr, next_tag)
        next_tag = b_grad.next_tag
        b = TensorMoments(b_value, b_grad, True)
        # b_transposed
        b_transposed_value = type(x_q.value)(b_transposed_traits, b_transposed_distr, next_tag)
        next_tag = b_transposed_value.next_tag
        b_transposed_grad = type(x_q.value)(b_transposed_traits, b_transposed_distr, next_tag)
        next_tag = b_transposed_grad.next_tag
        b_transposed = TensorMoments(b_transposed_value, b_transposed_grad, True)
        # Allocate tensors for bias for q, k, v and output projection
        if bias:
            out_proj_bias_traits = TensorTraits([n_emb], [n_emb_tile])
            out_proj_bias_distr = [0] * out_proj_bias_traits.grid.nelems
            out_proj_bias_value = type(x_q.value)(out_proj_bias_traits, \
                    out_proj_bias_distr, next_tag)
            next_tag = out_proj_bias_value.next_tag
            out_proj_bias_grad = type(x_q.value)(out_proj_bias_traits, \
                    out_proj_bias_distr, next_tag)
            next_tag = out_proj_bias_grad.next_tag
            out_proj_bias = TensorMoments(out_proj_bias_value, \
                    out_proj_bias_grad, True)
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
        layer = Attention(x_q, x_k, x_v, y, w_q, w_k, w_v, w, q_transposed, \
                q, k_transposed, k, v_transposed, v, a, a_maxsumexp, \
                a_sumprod_slice, b, b_transposed, bias_inproj_q, \
                bias_inproj_k, bias_inproj_v, out_proj_bias, mask)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the attention layer
    def forward_async(self):
        # Clear output
        # Y = 0
        #clear_async(self.y.value)
        # Compute query, key and value tensors
        # Q[i] = einsum('jk,klm->jlm', W_Q[i], X_Q)
        gemm_async(1.0, notrans, self.w_q.value, notrans, \
                self.x_q.value, 0.0, self.q_transposed.value, 1, 0)
        # TODO: single gemm (n_head, head_size, n_emb) by
        #       (n_emb, n_seq, n_batch) into
        #       (n_head, head_size, n_seq, n_batch)
        #       then rotate axes into (head_size, n_seq, n_batch, n_head)
        transpose_async(1.0, self.q_transposed.value, self.q.value, 1)
        # X_Q can be offloaded from GPU
        self.x_q.value.wont_use()
        self.q_transposed.value.wont_use()
        # W_Q[i] can be offloaded from GPU
        self.w_q.value.wont_use()
        if self.in_proj_bias_q:
            # TODO: single batched add_fiber (head_size, batch=n_head) into
            #       (head_size, n_seq, n_batch, batch=n_head)
            add_fiber_async(1, self.in_proj_bias_q.value, 1, \
                    self.q.value, 0, 1)
            self.in_proj_bias_q.value.wont_use()
        # K[i] = einsum('jk,klm->jlm', W_K[i], X_K)
        gemm_async(1.0, notrans, self.w_k.value, notrans, \
                self.x_k.value, 0.0, self.k_transposed.value, 1, 0)
        transpose_async(1.0, self.k_transposed.value, self.k.value, 1)
        # X_K can be offloaded from GPU
        self.x_k.value.wont_use()
        self.k_transposed.value.wont_use()
        # W_K[i] can be offloaded from GPU
        self.w_k.value.wont_use()
        if self.in_proj_bias_k:
            add_fiber_async(1, self.in_proj_bias_k.value, 1, \
                    self.k.value, 0, 1)
            self.in_proj_bias_k.value.wont_use()
        # V[i] = einsum('jk,klm->jlm', W_V[i], X_V)
        gemm_async(1.0, notrans, self.w_v.value, notrans, \
                self.x_v.value, 0.0, self.v_transposed.value, 1, 0)
        transpose_async(1.0, self.v_transposed.value, self.v.value, 1)
        # X_V can be offloaded from GPU
        self.x_v.value.wont_use()
        self.v_transposed.value.wont_use()
        # W_V[i] can be offloaded from GPU
        self.w_v.value.wont_use()
        if self.in_proj_bias_v:
            add_fiber_async(1, self.in_proj_bias_v.value, 1, \
                    self.v.value, 0, 1)
            self.in_proj_bias_v.value.wont_use()
        # Get tensor for softmax
        # A[i] = 1.0/sqrt(head_size) * einsum('jkl,jml->kml', K[i], Q[i])
        gemm_async(1.0/self.head_size**0.5, trans, self.k.value, \
                notrans, self.q.value, 0.0, self.a.value, 1, 2)
        # TODO: single batched gemm (head_size, n_seq, batch=n_batch,
        #       batch=n_head) by (head_size, n_seq, batch=n_batch,
        #       batch=n_head) into (n_seq, n_seq, batch=n_batch,
        #       batch=n_head)
        # Q[i] can be offloaded from GPU
        self.q.value.wont_use()
        # K[i] can be offloaded from GPU
        self.k.value.wont_use()
        # Calculate softmax inplace
        # A[i] = softmax(A[i], axis=0)
        if self.mask:
            mask_scalar_async(self.mask, self.val, self.a.value, 2)
            # TODO: variable number of batch dimensions mask
            self.mask.wont_use()
        maxsumexp_async(self.a.value, self.a_maxsumexp, 0)
        softmax_inplace_async(self.a_maxsumexp, self.a.value, 0)
        # A_maxsumexp[i] can be deleted
        #self.a_maxsumexp[i].invalidate_submit()
        self.a_maxsumexp.wont_use()
        # Apply value tensor
        # B[i] = einsum('jkl,kml->jml', V[i], A[i])
        gemm_async(1.0, notrans, self.v.value, notrans, \
                self.a.value, 0.0, self.b.value, 1, 2)
        # TODO: single batched gemm (head_size, n_seq, batch=n_batch,
        #       batch=n_head) by (n_seq, n_seq, batch=n_batch,
        #       batch=n_head) into (head_size, n_seq, batch=n_batch,
        #       batch=n_head)
        # V[i] can be offloaded from GPU
        self.v.value.wont_use()
        # A[i] can be offloaded from GPU
        self.a.value.wont_use()
        # Accumulate result from the current head into output
        # Y += einsum('jk,kml->jml', W[i], B[i])
        transpose_async(1.0, self.b.value, self.b_transposed.value, 3)
        gemm_async(1.0, notrans, self.w.value, notrans, \
                self.b_transposed.value, 0.0, self.y.value, 2, 0)
        # TODO: rotate axes (head_size, n_seq, n_batch, n_head) into
        #       (n_head, head_size, n_seq, n_batch) and then
        #       single gemm (n_emb, n_head, head_size) by
        #       (n_head, head_size, n_seq, n_batch) into (n_emb,
        #       n_seq, n_batch)
        # W can be offloaded from GPU
        self.w.value.wont_use()
        # B[i] can be offloaded from GPU
        self.b.value.wont_use()
        self.b_transposed.value.wont_use()
        if self.out_proj_bias:
            add_fiber_async(1, self.out_proj_bias.value, 1, self.y.value, 0, 0)
            self.out_proj_bias.value.wont_use()

    # Backward propagation of the linear layer
    def backward_async(self):
        raise NotImplementedError
        if self.out_proj_bias is not None:
            if self.out_proj_bias.grad_required:
                sum_fiber_async(1, self.y.grad, 1, self.out_proj_bias.grad, 0)
                self.out_proj_bias.grad.wont_use()
        for i in range(self.n_head):
            # Backward for Y += einsum('jk,kml->jml', W[i], B[i])
            if self.w[i].grad_required:
                # dW[i] += einsum('jml,kml->jk', dY, B[i])
                gemm_async(1.0, notrans, self.y.grad, trans, self.b[i].value, \
                        1.0, self.w[i].grad, 2, 0)
            # B[i] can be deleted
            #self.b[i].value.invalidate_submit()
            self.b[i].value.wont_use()
            if self.b[i].grad_required:
                # dB[i] = einsum('jk,jml->kml', W[i], dY)
                gemm_async(1.0, trans, self.w[i].value, notrans, self.y.grad, \
                        0.0, self.b[i].grad, 1, 0)
            # W[i] can be offloaded from GPU
            self.w[i].value.wont_use()
            # Backward for B[i] = einsum('jkl,kml->jml', V[i], A[i])
            if self.a[i].grad_required:
                # dA[i] = einsum('jkl,jml->kml', V[i], dB[i])
                gemm_async(1.0, trans, self.v[i].value, notrans, \
                        self.b[i].grad, 0.0, self.a[i].grad, 1, 1)
            # V[i] can be deleted
            #self.v[i].value.invalidate_submit()
            self.v[i].value.wont_use()
            if self.v[i].grad_required:
                # dV[i] = einsum('jml,kml->jkl', dB[i], A[i])
                gemm_async(1.0, notrans, self.b[i].grad, trans, \
                        self.a[i].value, 0.0, self.v[i].grad, 1, 1)
            # dB[i] can be deleted
            #self.b[i].grad.invalidate_submit()
            self.b[i].grad.wont_use()
            # Backward for A[i] = softmax(A[i], axis=0)
            if self.a[i].grad_required:
                # A_sumprod_slice[i] = einsum('kml,kml->ml', A[i], dA[i])
                sumprod_slice_async(1.0, self.a[i].value, self.a[i].grad, \
                        0.0, self.a_sumprod_slice[i], 0)
                # dA[i] += -bias('kml,ml->kml', dA[i], A_sumprod_slice[i])
                add_slice_async(-1.0, self.a_sumprod_slice[i], 1.0, \
                        self.a[i].grad, 0)
                # A_sumprod_slice[i] can be deleted
                #self.a_sumprod_slice[i].invalidate_submit()
                self.a_sumprod_slice[i].wont_use()
                # dA[i] *= A[i]
                prod_async(self.a[i].value, self.a[i].grad)
            if self.mask:
                mask_scalar_async(self.mask, 0, self.a[i].grad)
                self.mask.wont_use()
            # A[i] can be deleted
            #self.a[i].value.invalidate_submit()
            self.a[i].value.wont_use()
            # Backward for:
            # A[i] = 1.0/sqrt(head_size) * einsum('jkl,jml->kml', K[i], Q[i])
            if self.k[i].grad_required:
                # dK[i] = 1.0/sqrt(head_size) * einsum('jml,kml->jkl', Q[i],
                #       dA[i])
                gemm_async(1.0/self.head_size**0.5, notrans, self.q[i].value, \
                        trans, self.a[i].grad, 0.0, self.k[i].grad, 1, 1)
            # Q[i] can be deleted
            #self.q[i].value.invalidate_submit()
            self.q[i].value.wont_use()
            if self.q[i].grad_required:
                # dQ[i] = 1.0/sqrt(head_size) * einsum('jkl,kml->jml', K[i],
                #       dA[i])
                gemm_async(1.0/self.head_size**0.5, notrans, self.k[i].value, \
                        notrans, self.a[i].grad, 0.0, self.q[i].grad, 1, 1)
            # K[i] can be deleted
            #self.k[i].value.invalidate_submit()
            self.k[i].value.wont_use()
            # dA[i] can be deleted
            #self.a[i].grad.invalidate_submit()
            self.a[i].grad.wont_use()
            # Backward for V[i] = einsum('jk,klm->jlm', W_V[i], X_V)
            if self.x_v.grad_required:
                # dX_V += einsum('jk,jlm->klm', W_V[i], dV[i])
                gemm_async(1.0, trans, self.w_v[i].value, notrans, \
                        self.v[i].grad, 1.0, self.x_v.grad, 1, 0)
            # W_V[i] can be offloaded from GPU
            self.w_v[i].value.wont_use()
            if self.in_proj_bias_v[i] is not None:
                if self.in_proj_bias_v[i].grad_required:
                    sum_fiber_async(1, self.v[i].grad, 1, \
                            self.in_proj_bias_v[i].grad, 0)
                    self.in_proj_bias_v[i].grad.wont_use()
            if self.w_v[i].grad_required:
                # dW_V[i] += einsum('jlm,klm->jk', dV[i], X_V)
                gemm_async(1.0, notrans, self.v[i].grad, trans, \
                        self.x_v.value, 1.0, self.w_v[i].grad, 2, 0)
            # dW_V[i] can be offloaded from GPU
            self.w_v[i].grad.wont_use()
            # dV[i] can be deleted
            #self.v[i].grad.invalidate_submit()
            self.v[i].grad.wont_use()
            # Backward for K[i] = einsum('jk,klm->jlm', W_K[i], X_K)
            if self.x_k.grad_required:
                # dX_K += einsum('jk,jlm->klm', W_K[i], dK[i])
                gemm_async(1.0, trans, self.w_k[i].value, notrans, \
                        self.k[i].grad, 1.0, self.x_k.grad, 1, 0)
            # W_K[i] can be offloaded from GPU
            self.w_k[i].value.wont_use()
            if self.in_proj_bias_k[i] is not None:
                if self.in_proj_bias_k[i].grad_required:
                    sum_fiber_async(1, self.k[i].grad, 1, \
                            self.in_proj_bias_k[i].grad, 0)
                    self.in_proj_bias_k[i].grad.wont_use()
            if self.w_k[i].grad_required:
                # dW_K[i] += einsum('jlm,klm->jk', dK[i], X_K)
                gemm_async(1.0, notrans, self.k[i].grad, trans, \
                        self.x_k.value, 1.0, self.w_k[i].grad, 2, 0)
            # dW_K[i] can be offloaded from GPU
            self.w_k[i].grad.wont_use()
            # dK[i] can be deleted
            #self.k[i].grad.invalidate_submit()
            self.k[i].grad.wont_use()
            # Backward for Q[i] = einsum('jk,klm->jlm', W_Q[i], X_Q)
            if self.x_q.grad_required:
                # dX_Q += einsum('jk,jlm->klm', W_Q[i], dQ[i])
                gemm_async(1.0, trans, self.w_q[i].value, notrans, \
                        self.q[i].grad, 1.0, self.x_q.grad, 1, 0)
            # W_Q[i] can be offloaded from GPU
            self.w_q[i].value.wont_use()
            if self.in_proj_bias_q[i] is not None:
                if self.in_proj_bias_q[i].grad_required:
                    sum_fiber_async(1, self.q[i].grad, 1, \
                            self.in_proj_bias_q[i].grad, 0)
                    self.in_proj_bias_q[i].grad.wont_use()
            if self.w_q[i].grad_required:
                # dW_Q[i] += einsum('jlm,klm->jk', dQ[i], X_Q)
                gemm_async(1.0, notrans, self.q[i].grad, trans, \
                        self.x_q.value, 1.0, self.w_q[i].grad, 2, 0)
            # dW_Q[i] can be offloaded from GPU
            self.w_q[i].grad.wont_use()
            # dQ[i] can be deleted
            #self.q[i].grad.invalidate_submit()
            self.q[i].grad.wont_use()

