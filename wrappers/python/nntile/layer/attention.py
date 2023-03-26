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
# @date 2023-03-26

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        TransOp, trans, notrans, clear_async, gemm_async, randn_async, \
        maxsumexp_async, softmax_async, prod_async, sum_async, bias_async
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
    w_q: List[TensorMoments]
    w_k: List[TensorMoments]
    w_v: List[TensorMoments]
    w: List[TensorMoments]
    q: List[TensorMoments]
    k: List[TensorMoments]
    v: List[TensorMoments]
    a: List[TensorMoments]
    a_maxsumexp: List[Tensor]
    a_scalprod: List[Tensor]
    b: List[TensorMoments]
    n_head: int
    head_size: int

    # Construct attention layer with all the provided data
    def __init__(self, x_q: TensorMoments, x_k: TensorMoments, \
            x_v: TensorMoments, y: TensorMoments, \
            w_q: List[TensorMoments], w_k: List[TensorMoments], \
            w_v: List[TensorMoments], w: List[TensorMoments], \
            q: List[TensorMoments], k: List[TensorMoments], \
            v: List[TensorMoments], a: List[TensorMoments], \
            a_maxsumexp: List[Tensor], a_scalprod: List[Tensor], \
            b: List[TensorMoments]):
        # Redirect to BaseClass initialization
        super().__init__([x_q, x_k, x_v], [y], [*w_q, *w_k, *w_v, *w])
        self.x_q = x_q
        self.x_k = x_k
        self.x_v = x_v
        self.y = y
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w = w
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.a_maxsumexp = a_maxsumexp
        self.a_scalprod = a_scalprod
        self.b = b
        self.n_head = len(w_q)
        n_emb = x_q.value.shape[0]
        head_size = n_emb // self.n_head
        # Stupid check, that is not necessary, as the code shall work
        if n_emb != head_size * self.n_head:
            raise RuntimeError
        self.head_size = head_size

    # Simple generator for the linear layer
    @staticmethod
    def generate_simple_mpiroot(x_q: TensorMoments, x_k: TensorMoments, \
            x_v: TensorMoments, n_head: int, next_tag: int):
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
        w_q_shape = [head_size, n_emb]
        w_k_shape = [head_size, n_emb_k]
        w_v_shape = [head_size, n_emb_v]
        w_shape = [n_emb, head_size]
        q_shape = [head_size, n_seq, n_batch]
        k_shape = [head_size, n_seq, n_batch]
        v_shape = [head_size, n_seq, n_batch]
        a_shape = [n_seq, n_seq, n_batch]
        a_maxsumexp_shape = [2, n_seq, n_batch]
        a_scalprod_shape = [n_seq, n_batch]
        b_shape = [head_size, n_seq, n_batch]
        # Define tile shapes of each tensor
        w_q_basetile = [head_size_tile, n_emb_tile]
        w_k_basetile = [head_size_tile, n_emb_k_tile]
        w_v_basetile = [head_size_tile, n_emb_v_tile]
        w_basetile = [n_emb_tile, head_size_tile]
        q_basetile = [head_size_tile, n_seq_tile, n_batch_tile]
        k_basetile = [head_size_tile, n_seq_tile, n_batch_tile]
        v_basetile = [head_size_tile, n_seq_tile, n_batch_tile]
        a_basetile = [n_seq_tile, n_seq_tile, n_batch_tile]
        a_maxsumexp_basetile = [2, n_seq_tile, n_batch_tile]
        a_scalprod_basetile = [n_seq_tile, n_batch_tile]
        b_basetile = [head_size_tile, n_seq_tile, n_batch_tile]
        # Define traits
        w_q_traits = TensorTraits(w_q_shape, w_q_basetile)
        w_k_traits = TensorTraits(w_k_shape, w_k_basetile)
        w_v_traits = TensorTraits(w_v_shape, w_v_basetile)
        w_traits = TensorTraits(w_shape, w_basetile)
        q_traits = TensorTraits(q_shape, q_basetile)
        k_traits = TensorTraits(k_shape, k_basetile)
        v_traits = TensorTraits(v_shape, v_basetile)
        a_traits = TensorTraits(a_shape, a_basetile)
        a_maxsumexp_traits = TensorTraits(a_maxsumexp_shape,
                a_maxsumexp_basetile)
        a_scalprod_traits = TensorTraits(a_scalprod_shape, a_scalprod_basetile)
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
        a_scalprod_distr = [0] * a_scalprod_traits.grid.nelems
        b_distr = [0] * b_traits.grid.nelems
        # Define all the lists
        w_q = []
        w_k = []
        w_v = []
        w = []
        q = []
        k = []
        v = []
        a = []
        a_maxsumexp = []
        a_scalprod = []
        b = []
        for i in range(n_head):
            # w_q
            w_q_value = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
            next_tag = w_q_value.next_tag
            w_q_grad = type(x_q.value)(w_q_traits, w_q_distr, next_tag)
            next_tag = w_q_grad.next_tag
            w_q.append(TensorMoments(w_q_value, w_q_grad, True))
            # w_k
            w_k_value = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
            next_tag = w_k_value.next_tag
            w_k_grad = type(x_q.value)(w_k_traits, w_k_distr, next_tag)
            next_tag = w_k_grad.next_tag
            w_k.append(TensorMoments(w_k_value, w_k_grad, True))
            # w_v
            w_v_value = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
            next_tag = w_v_value.next_tag
            w_v_grad = type(x_q.value)(w_v_traits, w_v_distr, next_tag)
            next_tag = w_v_grad.next_tag
            w_v.append(TensorMoments(w_v_value, w_v_grad, True))
            # w
            w_value = type(x_q.value)(w_traits, w_distr, next_tag)
            next_tag = w_value.next_tag
            w_grad = type(x_q.value)(w_traits, w_distr, next_tag)
            next_tag = w_grad.next_tag
            w.append(TensorMoments(w_value, w_grad, True))
            # q
            q_value = type(x_q.value)(q_traits, q_distr, next_tag)
            next_tag = q_value.next_tag
            q_grad = type(x_q.value)(q_traits, q_distr, next_tag)
            next_tag = q_grad.next_tag
            q.append(TensorMoments(q_value, q_grad, True))
            # k
            k_value = type(x_q.value)(k_traits, k_distr, next_tag)
            next_tag = k_value.next_tag
            k_grad = type(x_q.value)(k_traits, k_distr, next_tag)
            next_tag = k_grad.next_tag
            k.append(TensorMoments(k_value, k_grad, True))
            # v
            v_value = type(x_q.value)(v_traits, v_distr, next_tag)
            next_tag = v_value.next_tag
            v_grad = type(x_q.value)(v_traits, v_distr, next_tag)
            next_tag = v_grad.next_tag
            v.append(TensorMoments(v_value, v_grad, True))
            # a
            a_value = type(x_q.value)(a_traits, a_distr, next_tag)
            next_tag = a_value.next_tag
            a_grad = type(x_q.value)(a_traits, a_distr, next_tag)
            next_tag = a_grad.next_tag
            a.append(TensorMoments(a_value, a_grad, True))
            # a_maxsumexp
            a_maxsumexp_value = type(x_q.value)(a_maxsumexp_traits,
                    a_maxsumexp_distr, next_tag)
            next_tag = a_maxsumexp_value.next_tag
            a_maxsumexp.append(a_maxsumexp_value)
            # a_scalprod
            a_scalprod_value = type(x_q.value)(a_scalprod_traits, \
                    a_scalprod_distr, next_tag)
            next_tag = a_scalprod_value.next_tag
            a_scalprod.append(a_scalprod_value)
            # b
            b_value = type(x_q.value)(b_traits, b_distr, next_tag)
            next_tag = b_value.next_tag
            b_grad = type(x_q.value)(b_traits, b_distr, next_tag)
            next_tag = b_grad.next_tag
            b.append(TensorMoments(b_value, b_grad, True))
        # Allocate tensor for output y
        y_traits = TensorTraits(x_q.value.shape, x_q.value.basetile_shape)
        y_value = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x_q.value)(y_traits, x_q.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        # Create attention layer with all the provided data
        layer = Attention(x_q, x_k, x_v, y, w_q, w_k, w_v, w, q, k, v, a, \
                a_maxsumexp, a_scalprod, b)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Random initialization of weights
    def init_randn_async(self):
        pass

    # Forward propagation of the attention layer
    def forward_async(self):
        # Clear output
        # Y = 0
        clear_async(self.y.value)
        # Workout each head separately
        for i in range(self.n_head):
            # Compute query, key and value tensors
            # Q[i] = W_Q[i] * X_Q
            gemm_async(1.0, notrans, self.w_q[i].value, notrans, \
                    self.x_q.value, 0.0, self.q[i].value, 1, 0)
            # K[i] = W_K[i] * X_K
            gemm_async(1.0, notrans, self.w_k[i].value, notrans, \
                    self.x_k.value, 0.0, self.k[i].value, 1, 0)
            # V[i] = W_V[i] * X_V
            gemm_async(1.0, notrans, self.w_v[i].value, notrans, \
                    self.x_v.value, 0.0, self.v[i].value, 1, 0)
            # Get tensor for softmax
            # batch A[i][j] = 1.0/sqrt(head_size) * K[i][j].T * Q[i][j]
            gemm_async(1.0/self.head_size**0.5, trans, self.k[i].value, \
                    notrans, self.q[i].value, 0.0, self.a[i].value, 1, 1)
            # Calculate softmax inplace
            # A[i] = softmax(A[i], axis=0)
            maxsumexp_async(self.a[i].value, self.a_maxsumexp[i], 0)
            softmax_async(self.a_maxsumexp[i], self.a[i].value, 0)
            # Apply value tensor
            # batch B[i][j] = V[i][j] * A[i][j]
            gemm_async(1.0, notrans, self.v[i].value, notrans, \
                    self.a[i].value, 0.0, self.b[i].value, 1, 1)
            # Accumulate result from the current head into output
            # Y += W[i] * B[i]
            gemm_async(1.0, notrans, self.w[i].value, notrans, \
                    self.b[i].value, 1.0, self.y.value, 1, 0)

    # Backward propagation of the linear layer
    def backward_async(self):
        for i in range(self.n_head):
            # Backward for Y += W[i] * B[i]
            if self.w[i].grad_required:
                # dW[i] = dY * B[i].T
                gemm_async(1.0, notrans, self.y.grad, trans, self.b[i].value, \
                        0.0, self.w[i].grad, 2, 0)
            # dB[i] = W[i].T * dY
            gemm_async(1.0, trans, self.w[i].value, notrans, self.y.grad, \
                    0.0, self.b[i].grad, 1, 0)
            # Backward for batch B[i][j] = V[i][j] * A[i][j]
            # batch dA[i][j] = V[i][j].T * dB[i][j]
            gemm_async(1.0, trans, self.v[i].value, notrans, self.b[i].grad, \
                    0.0, self.a[i].grad, 1, 1)
            if self.x_v.grad_required or self.w_v[i].grad_required:
                # batch dV[i][j] = dB[i][j] * A[i][j].T
                gemm_async(1.0, notrans, self.b[i].grad, trans, \
                        self.a[i].value, 0.0, self.v[i].grad, 1, 1)
            # Backward for A[i] = softmax(A[i], axis=0)
            # dA[i] = prod(dA[i], A[i])
            prod_async(self.a[i].value, self.a[i].grad)
            # A_scalprod[i] = sum(dA[i], axis=0)
            clear_async(self.a_scalprod[i])
            sum_async(self.a[i].grad, self.a_scalprod[i], 0)
            # dA[i] = bias(dA[i], -A_scalprod[i], axis=0)
            bias_async(-1.0, self.a_scalprod[i], self.a[i].grad, 0)
            if self.x_v.grad_required:
                # dX_V += W_V[i].T * dV[i]
                gemm_async(1.0, trans, self.w_v[i].value, notrans, \
                        self.v[i].grad, 1.0, self.x_v.grad, 1, 0)
            if self.w_v[i].grad_required:
                # dW_V[i] = dV[i] * X_V.T
                gemm_async(1.0, notrans, self.v[i].grad, trans, \
                        self.x_v.value, 0.0, self.w_v[i].grad, 2, 0)

    # Unregister all internal tensors
    def unregister(self):
        pass

