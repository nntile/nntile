# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/attention_single_head.py
# Single-headed attention layer of NNTile Python package
#
# @version 1.1.0

import numpy as np

from nntile.layer.base_layer import BaseLayer
from nntile.tensor import (
    Tensor, TensorMoments, TensorTraits, add_fiber_async, add_slice_async,
    clear_async, gemm_async, mask_scalar_async, maxsumexp_async, notrans,
    prod_inplace_async, softmax_inplace_async, sum_fiber_async,
    sumprod_slice_async, trans)


# Single-head attention
# Inputs:
#  x_q: (n_emb, n_seq, n_batch) tensor
#  x_k: (n_emb_k, n_seq, n_batch) tensor
#  x_v: (n_emb_v, n_seq, n_batch) tensor
# Output:
#  y: (n_emb, n_seq, n_batch) tensor
class AttentionSingleHead(BaseLayer):
    x_q: TensorMoments
    x_k: TensorMoments
    x_v: TensorMoments
    y: TensorMoments
    w_q: TensorMoments
    w_k: TensorMoments
    w_v: TensorMoments
    w: TensorMoments
    q: TensorMoments
    k: TensorMoments
    v: TensorMoments
    a: TensorMoments
    a_maxsumexp: Tensor
    a_sumprod_slice: Tensor
    b: TensorMoments

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
        q: TensorMoments,
        k: TensorMoments,
        v: TensorMoments,
        a: TensorMoments,
        a_maxsumexp: Tensor,
        a_sumprod_slice: Tensor,
        b: TensorMoments,
        in_proj_bias_q: TensorMoments,
        in_proj_bias_k: TensorMoments,
        in_proj_bias_v: TensorMoments,
        out_proj_bias: TensorMoments,
        mask=None,
        redux: bool = False,
    ):
        # print("SINGLE HEAD")
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
            [x_q, x_k, x_v],
            [y],
            [w_q, w_k, w_v] + qkv_bias_list + [w] + bias_list_out_proj,
            [q, k, v, a, a_maxsumexp, a_sumprod_slice, b],
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
        self.q = q
        self.q.value.set_reduction_add()
        self.q.grad.set_reduction_add()
        self.k = k
        self.k.value.set_reduction_add()
        self.k.grad.set_reduction_add()
        self.v = v
        self.v.value.set_reduction_add()
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
        self.b.grad.set_reduction_add()
        self.in_proj_bias_q = in_proj_bias_q
        self.in_proj_bias_k = in_proj_bias_k
        self.in_proj_bias_v = in_proj_bias_v
        self.out_proj_bias = out_proj_bias
        self.mask = mask
        if mask:
            self.val = -np.float32(np.inf)
        if redux:
            self.redux = 1
        else:
            self.redux = 0

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple(
        x_q: TensorMoments,
        x_k: TensorMoments,
        x_v: TensorMoments,
        next_tag: int,
        bias=False,
        mask=None,
        redux: bool = False,
    ):
        # Get sizes
        n_emb, n_seq, n_batch = x_q.value.shape
        n_emb_tile, n_seq_tile, n_batch_tile = x_q.value.basetile_shape
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
        # Define shape of each tensor
        w_q_shape = [n_emb, n_emb]
        w_k_shape = [n_emb, n_emb_k]
        w_v_shape = [n_emb, n_emb_v]
        w_shape = [n_emb, n_emb]
        q_shape = [n_emb, n_seq, n_batch]
        k_shape = [n_emb, n_seq, n_batch]
        v_shape = [n_emb, n_seq, n_batch]
        a_shape = [n_seq, n_seq, n_batch]
        a_maxsumexp_shape = [2, n_seq, n_batch]
        a_sumprod_slice_shape = [n_seq, n_batch]
        b_shape = [n_emb, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [n_emb_tile, n_emb_tile]
        w_k_basetile = [n_emb_tile, n_emb_k_tile]
        w_v_basetile = [n_emb_tile, n_emb_v_tile]
        w_basetile = [n_emb_tile, n_emb_tile]
        q_basetile = [n_emb_tile, n_seq_tile, n_batch_tile]
        k_basetile = [n_emb_tile, n_seq_tile, n_batch_tile]
        v_basetile = [n_emb_tile, n_seq_tile, n_batch_tile]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile]
        a_sumprod_slice_basetile = [n_seq_tile, n_batch_tile]
        b_basetile = [n_emb_tile, n_seq_tile, n_batch_tile]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        w_traits = TensorTraits(w_shape, w_basetile)
        q_traits = TensorTraits(q_shape, q_basetile)
        k_traits = TensorTraits(k_shape, k_basetile)
        v_traits = TensorTraits(v_shape, v_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(
            a_maxsumexp_shape, a_maxsumexp_basetile
        )
        a_sumprod_slice_traits = TensorTraits(
            a_sumprod_slice_shape, a_sumprod_slice_basetile
        )
        b_traits = TensorTraits(b_shape, b_basetile)
        # TODO change distribution
        w_q_distr = [0] * w_q_traits.grid.nelems
        w_k_distr = [0] * w_k_traits.grid.nelems
        w_v_distr = [0] * w_v_traits.grid.nelems
        w_distr = [0] * w_traits.grid.nelems
        q_distr = [0] * q_traits.grid.nelems
        k_distr = [0] * k_traits.grid.nelems
        v_distr = [0] * v_traits.grid.nelems
        a_distr = [0] * a_traits.grid.nelems
        a_maxsumexp_distr = [0] * a_maxsumexp_traits.grid.nelems
        a_sumprod_slice_distr = [0] * a_sumprod_slice_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        if bias:
            in_proj_bias_qkv_traits = TensorTraits([n_emb], [n_emb_tile])
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
        # q
        q_value = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_value.next_tag
        q_grad = type(x_q.value)(q_traits, q_distr, next_tag)
        next_tag = q_grad.next_tag
        q = TensorMoments(q_value, q_grad, True)
        # k
        k_value = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_value.next_tag
        k_grad = type(x_q.value)(k_traits, k_distr, next_tag)
        next_tag = k_grad.next_tag
        k = TensorMoments(k_value, k_grad, True)
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
        a_maxsumexp = type(x_q.value)(
            a_maxsumexp_traits, a_maxsumexp_distr, next_tag
        )
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
            out_proj_bias = TensorMoments(
                out_proj_bias_value, out_proj_bias_grad, True
            )
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
        layer = AttentionSingleHead(
            x_q,
            x_k,
            x_v,
            y,
            w_q,
            w_k,
            w_v,
            w,
            q,
            k,
            v,
            a,
            a_maxsumexp,
            a_sumprod_slice,
            b,
            bias_inproj_q,
            bias_inproj_k,
            bias_inproj_v,
            out_proj_bias,
            mask,
            redux=redux,
        )
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the attention layer
    def forward_async(self):
        # Compute query, key and value tensors
        # Q = einsum('jk,klm->jlm', W_Q, X_Q)
        # gemm (n_emb, n_emb) by (n_emb, n_seq, n_batch) into
        # (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_q.value,
            notrans,
            self.x_q.value,
            0.0,
            self.q.value,
            1,
            0,
            redux=self.redux,
        )
        # X_Q, W_Q and Q_transposed can be offloaded from GPU
        self.x_q.value.wont_use()
        self.w_q.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_q is not None:
            # non-batched add_fiber (n_emb, batch_ndim=0) into
            # (n_emb, n_seq, n_batch, batch_ndim=0)
            add_fiber_async(
                1, self.in_proj_bias_q.value, 1, self.q.value, 0, 0
            )
            self.in_proj_bias_q.value.wont_use()
        # K = einsum('jk,klm->jlm', W_K, X_K)
        # gemm (n_emb, n_emb_k) by (n_emb_k, n_seq, n_batch) into
        # (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_k.value,
            notrans,
            self.x_k.value,
            0.0,
            self.k.value,
            1,
            0,
            redux=self.redux,
        )
        # X_K, W_K and K_transposed can be offloaded from GPU
        self.x_k.value.wont_use()
        self.w_k.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_k is not None:
            # non-batched add_fiber (n_emb, batch_ndim=0) into
            # (n_emb, n_seq, n_batch, batch_ndim=0)
            add_fiber_async(
                1, self.in_proj_bias_k.value, 1, self.k.value, 0, 0
            )
            self.in_proj_bias_k.value.wont_use()
        # V = einsum('jk,klm->jlm', W_V, X_V)
        # gemm (n_emb, n_emb_v) by (n_emb_v, n_seq, n_batch) into
        # (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w_v.value,
            notrans,
            self.x_v.value,
            0.0,
            self.v.value,
            1,
            0,
            redux=self.redux,
        )
        # X_V, W_V and V_transposed can be offloaded from GPU
        self.x_v.value.wont_use()
        self.w_v.value.wont_use()
        # Apply bias if needed
        if self.in_proj_bias_v is not None:
            # non-batched add_fiber (n_emb, batch_ndim=0) into
            # (n_emb, n_seq, n_batch, batch_ndim=0)
            add_fiber_async(
                1, self.in_proj_bias_v.value, 1, self.v.value, 0, 0
            )
            self.in_proj_bias_v.value.wont_use()
        # Get tensor for softmax
        # A = 1.0/sqrt(n_emb) * einsum('jkl,jml->kml', K, Q)
        # single batched gemm (n_emb, n_seq, batch=n_batch)
        # by (n_emb, n_seq, batch=n_batch) into
        # (n_seq, n_seq, batch=n_batch)
        n_emb = self.x_q.value.shape[0]
        gemm_async(
            1.0 / n_emb**0.5,
            trans,
            self.k.value,
            notrans,
            self.q.value,
            0.0,
            self.a.value,
            1,
            1,
            redux=self.redux,
        )
        # Q and K can be offloaded from GPU
        self.q.value.wont_use()
        self.k.value.wont_use()
        # Calculate softmax inplace
        # A = softmax(A, axis=0)
        # Apply mask if needed
        if self.mask:
            mask_scalar_async(self.mask, self.val, self.a.value, 1)
            self.mask.wont_use()
        # Calculate max and sumexp along axis
        clear_async(self.a_maxsumexp)
        maxsumexp_async(self.a.value, self.a_maxsumexp, 0, redux=self.redux)
        # Finally, get the inplace softmax
        softmax_inplace_async(self.a_maxsumexp, 1.0, self.a.value, 0)
        # A_maxsumexp can be deleted
        self.a_maxsumexp.invalidate_submit()
        # Apply value tensor
        # B = einsum('jkl,kml->jml', V, A)
        # batched gemm (n_emb, n_seq, batch=n_batch)
        # by (n_seq, n_seq, batch=n_batch) into
        # (n_emb, n_seq, batch=n_batch)
        gemm_async(
            1.0,
            notrans,
            self.v.value,
            notrans,
            self.a.value,
            0.0,
            self.b.value,
            1,
            1,
            redux=self.redux,
        )
        # V and A can be offloaded from GPU
        self.v.value.wont_use()
        self.a.value.wont_use()
        # Accumulate result from all the heads
        # Y = einsum('jk,klm->jlm', W, B)
        # gemm (n_emb, n_emb) by (n_emb, n_seq, n_batch)
        # into (n_emb, n_seq, n_batch)
        gemm_async(
            1.0,
            notrans,
            self.w.value,
            notrans,
            self.b.value,
            0.0,
            self.y.value,
            1,
            0,
            redux=self.redux,
        )
        # W and B can be offloaded from GPU
        self.w.value.wont_use()
        self.b.value.wont_use()
        # Apply bias if needed
        if self.out_proj_bias is not None:
            add_fiber_async(
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
        # Backward for Y = einsum('jk,klm->jlm', W, B)
        if self.w.grad_required:
            # dW += einsum('jlm,klm->jk', dY, B)
            gemm_async(
                1.0,
                notrans,
                self.y.grad,
                trans,
                self.b.value,
                1.0,
                self.w.grad,
                2,
                0,
                redux=self.redux,
            )
        self.w.grad.wont_use()
        if self.b.grad_required:
            # dB = einsum('jk,jlm->klm', W, dY)
            gemm_async(
                1.0,
                trans,
                self.w.value,
                notrans,
                self.y.grad,
                0.0,
                self.b.grad,
                1,
                0,
                redux=self.redux,
            )
        # W can be offloaded from GPU
        self.w.value.wont_use()
        # dY can be offloaded from GPU
        self.y.grad.wont_use()
        # Backward for B = einsum('jkl,kml->jml', V, A)
        if self.a.grad_required:
            # dA = einsum('jkl,jml->kml', V, dB)
            gemm_async(
                1.0,
                trans,
                self.v.value,
                notrans,
                self.b.grad,
                0.0,
                self.a.grad,
                1,
                1,
                redux=self.redux,
            )
        # V can be deleted
        self.v.value.invalidate_submit()
        if self.v.grad_required:
            # dV = einsum('jml,kml->jkl', dB, A)
            gemm_async(
                1.0,
                notrans,
                self.b.grad,
                trans,
                self.a.value,
                0.0,
                self.v.grad,
                1,
                1,
                redux=self.redux,
            )
        # dB can be deleted
        self.b.grad.invalidate_submit()
        # Backward for A = softmax(A, axis=0)
        if self.a.grad_required:
            # A_sumprod_slice = einsum('kml,kml->ml', A, dA)
            sumprod_slice_async(
                1.0,
                self.a.value,
                self.a.grad,
                0.0,
                self.a_sumprod_slice,
                0,
                redux=self.redux,
            )
            # dA += -bias('kml,ml->kml', dA, A_sumprod_slice)
            add_slice_async(-1.0, self.a_sumprod_slice, 1.0, self.a.grad, 0)
            # A_sumprod_slice can be deleted
            self.a_sumprod_slice.invalidate_submit()
            # dA *= A
            prod_inplace_async(self.a.value, self.a.grad)
        # A can be deleted
        self.a.value.invalidate_submit()
        # Backward for mask if needed
        if self.mask:
            mask_scalar_async(self.mask, 0, self.a.grad, 1)
            self.mask.wont_use()
        # Backward for:
        # A = 1.0/sqrt(n_emb) * einsum('jkl,jml->kml', K, Q)
        n_emb = self.x_q.value.shape[0]
        if self.k.grad_required:
            # dK = 1.0/sqrt(n_emb) * einsum('jml,kml->jkl', Q, dA)
            gemm_async(
                1.0 / n_emb**0.5,
                notrans,
                self.q.value,
                trans,
                self.a.grad,
                0.0,
                self.k.grad,
                1,
                1,
                redux=self.redux,
            )
        # Q can be deleted
        self.q.value.invalidate_submit()
        if self.q.grad_required:
            # dQ = 1.0/sqrt(n_emb) * einsum('jkl,kml->jml', K, dA)
            gemm_async(
                1.0 / n_emb**0.5,
                notrans,
                self.k.value,
                notrans,
                self.a.grad,
                0.0,
                self.q.grad,
                1,
                1,
                redux=self.redux,
            )
        # K can be deleted
        self.k.value.invalidate_submit()
        # dA can be deleted
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
                    0,
                    redux=self.redux,
                )
                self.in_proj_bias_v.grad.wont_use()
        # Backward for V = einsum('jk,klm->jlm', W_V, X_V)
        if self.x_v.grad_required:
            # dX_V += einsum('jk,jlm->klm', W_V, dV)
            gemm_async(
                1.0,
                trans,
                self.w_v.value,
                notrans,
                self.v.grad,
                1.0,
                self.x_v.grad,
                1,
                0,
                redux=self.redux,
            )
        # W_V can be offloaded from GPU
        self.w_v.value.wont_use()
        # dX_V can be offloaded from GPU
        self.x_v.grad.wont_use()
        if self.w_v.grad_required:
            # dW_V += einsum('jlm,klm->jk', dV, X_V)
            gemm_async(
                1.0,
                notrans,
                self.v.grad,
                trans,
                self.x_v.value,
                1.0,
                self.w_v.grad,
                2,
                0,
                redux=self.redux,
            )
        # dV can be deleted
        self.v.grad.invalidate_submit()
        # dW_V can be offloaded from GPU
        self.w_v.grad.wont_use()
        # X_V can be offloaded from GPU
        self.x_v.value.wont_use()
        # Backward for bias of K
        if self.in_proj_bias_k is not None:
            if self.in_proj_bias_k.grad_required:
                sum_fiber_async(
                    1,
                    self.k.grad,
                    1,
                    self.in_proj_bias_k.grad,
                    0,
                    0,
                    redux=self.redux,
                )
                self.in_proj_bias_k.grad.wont_use()
        # Backward for K = einsum('jk,klm->jlm', W_K, X_K)
        if self.x_k.grad_required:
            # dX_K += einsum('jk,jlm->klm', W_K, dK)
            gemm_async(
                1.0,
                trans,
                self.w_k.value,
                notrans,
                self.k.grad,
                1.0,
                self.x_k.grad,
                1,
                0,
                redux=self.redux,
            )
        # W_K can be offloaded from GPU
        self.w_k.value.wont_use()
        # dX_K can be offloaded from GPU
        self.x_k.grad.wont_use()
        if self.w_k.grad_required:
            # dW_K += einsum('jlm,klm->jk', dK, X_K)
            gemm_async(
                1.0,
                notrans,
                self.k.grad,
                trans,
                self.x_k.value,
                1.0,
                self.w_k.grad,
                2,
                0,
                redux=self.redux,
            )
        # dK can be deleted
        self.k.grad.invalidate_submit()
        # dW_K can be offloaded from GPU
        self.w_k.grad.wont_use()
        # X_K can be offloaded from GPU
        self.x_k.value.wont_use()
        # Backward for bias of Q
        if self.in_proj_bias_q is not None:
            if self.in_proj_bias_q.grad_required:
                sum_fiber_async(
                    1,
                    self.q.grad,
                    1,
                    self.in_proj_bias_q.grad,
                    0,
                    0,
                    redux=self.redux,
                )
                self.in_proj_bias_q.grad.wont_use()
        # Backward for Q = einsum('jk,klm->jlm', W_Q, X_Q)
        if self.x_q.grad_required:
            # dX_Q += einsum('jk,jlm->klm', W_Q, dQ)
            gemm_async(
                1.0,
                trans,
                self.w_q.value,
                notrans,
                self.q.grad,
                1.0,
                self.x_q.grad,
                1,
                0,
                redux=self.redux,
            )
            self.x_q.grad.wont_use()
        # W_Q can be offloaded from GPU
        self.w_q.value.wont_use()
        # dX_Q can be offloaded from GPU
        self.x_q.grad.wont_use()
        if self.w_q.grad_required:
            # dW_Q += einsum('jlm,klm->jk', dQ, X_Q)
            gemm_async(
                1.0,
                notrans,
                self.q.grad,
                trans,
                self.x_q.value,
                1.0,
                self.w_q.grad,
                2,
                0,
                redux=self.redux,
            )
        # dQ can be deleted
        self.q.grad.invalidate_submit()
        # dW_Q can be offloaded from GPU
        self.w_q.grad.wont_use()
        # X_Q can be offloaded from GPU
        self.x_q.value.wont_use()
