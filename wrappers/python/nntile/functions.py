# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/tensor.py
# Multiprecision tensor with operations
#
# @version 1.1.0

# TODO(@daskol): There are a lot of typing errors related to dispatching
# depending on operand types. Arguments are usually annotated as a `Tensor`
# (sum of types) while a specific tensor routine requires a specific type
# `Tensor_*` instead of `Tensor`.
#
# mypy: ignore-errors

from typing import Any, List, Sequence, Type, TypeGuard, TypeVar

import nntile.nntile_core.tensor as ops
from nntile.nntile_core import TransOp, tensor as core_tensor
from nntile.nntile_core.tensor import (
    Tensor_bf16, Tensor_bool, Tensor_fp32, Tensor_fp32_fast_tf32, Tensor_fp64,
    Tensor_int64)
from nntile.types import Tensor, TensorFloatOrInt, TensorOrFloat

T = TypeVar('T')


def is_tensor_of(tensors: Sequence[Any],
                 tensor_type: Type[T]) -> TypeGuard[Sequence[T]]:
    """A type guard to narrow type of of collection of uniformly typed
    tensors.

    >>> a: Tensor_fp32
    >>> b: Tensor_fp32
    >>> tensors: list[Tensor] = [a, b]
    >>> reveal_type(tensors)
    # Revealed type is "builtins.list[nntile.nntile_core.tensor.Tensor]
    >>> if is_tensor_of(tensors, Tensor_fp32):
    >>>     reveal_type(tensors)
    # Revealed type is "typing.Sequence[nntile.nntile_core.tensor.Tensor_fp32]
    """
    return all(isinstance(t, tensor_type) for t in tensors)


def gemm_async(
    alpha: float,
    trans_A: TransOp,
    A: Tensor,
    trans_B: TransOp,
    B: Tensor,
    beta: float,
    C: Tensor,
    ndim: int,
    batch_ndim: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision gemm
    """
    if type(A) is not type(B) or type(A) is not type(C):
        raise TypeError
    if type(A) is core_tensor.Tensor_fp32:
        core_tensor.gemm_async_fp32(
            alpha, trans_A, A, trans_B, B, beta, C, ndim, batch_ndim, redux
        )
    elif type(A) is core_tensor.Tensor_fp64:
        core_tensor.gemm_async_fp64(
            alpha, trans_A, A, trans_B, B, beta, C, ndim, batch_ndim, redux
        )
    elif type(A) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gemm_async_fp32_fast_tf32(
            alpha, trans_A, A, trans_B, B, beta, C, ndim, batch_ndim, redux
        )
    elif type(A) is core_tensor.Tensor_bf16:
        core_tensor.gemm_async_bf16(
            alpha, trans_A, A, trans_B, B, beta, C, ndim, batch_ndim, redux
        )
    else:
        raise TypeError


def relu_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision ReLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_async_fp64(x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_async_fp32_fast_tf32(x)
    else:
        raise TypeError


def relu_forward_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision forward ReLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_forward_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_forward_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_forward_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.relu_forward_async_bf16(x, y)
    else:
        raise TypeError


def silu_forward_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision forward SiLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.silu_forward_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.silu_forward_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.silu_forward_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.silu_forward_async_bf16(x, y)
    else:
        raise TypeError


def relu_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    """
    Wrapper for multiprecision backward ReLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_backward_async_fp64(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_backward_async_fp32_fast_tf32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.relu_backward_async_bf16(x, dy, dx)
    else:
        raise TypeError


def silu_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    """
    Wrapper for multiprecision backward SiLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.silu_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.silu_backward_async_fp64(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.silu_backward_async_fp32_fast_tf32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.silu_backward_async_bf16(x, dy, dx)
    else:
        raise TypeError


def drelu_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision derivative of ReLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.drelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.drelu_async_fp64(x)
    else:
        raise TypeError


def gelu_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision GELU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelu_async_fp64(x)
    else:
        raise TypeError


def dgelu_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision derivative of GELU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.dgelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.dgelu_async_fp64(x)
    else:
        raise TypeError


def gelu_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    """
    Wrapper for multiprecision backward GeLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelu_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelu_backward_async_fp64(x, dy, dx)
    else:
        raise TypeError


def gelutanh_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision approximated GELU
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gelutanh_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.gelutanh_async_bf16(x, y)
    else:
        raise TypeError


def gelutanh_inplace_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision approximated GELU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_inplace_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_inplace_async_fp64(x)
    else:
        raise TypeError


def dgelutanh_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision derivative of approximate GELU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.dgelutanh_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.dgelutanh_async_fp64(x)
    else:
        raise TypeError


def gelutanh_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    """
    Wrapper for multiprecision backward approximate GeLU
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gelutanh_backward_async_fp32_fast_tf32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_backward_async_fp64(x, dy, dx)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.gelutanh_backward_async_bf16(x, dy, dx)
    else:
        raise TypeError


def fill_async(val: float, x: Tensor) -> None:
    """
    Wrapper for multiprecision fill
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.fill_async_fp32(val, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.fill_async_fp32_fast_tf32(val, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.fill_async_fp64(val, x)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.fill_async_bf16(val, x)
    else:
        raise TypeError


def sum_slice_async(alpha: float, x: Tensor, beta: float, sum_slice: Tensor,
                    axis: int, redux: int = 0) -> None:
    """Wrapper for multiprecision `sum_slice`."""
    ts = (x, sum_slice)
    if is_tensor_of(ts, Tensor_bf16):
        ops.sum_slice_async_bf16(alpha, ts[0], beta, ts[1], axis, redux)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.sum_slice_async_fp32(alpha, ts[0], beta, ts[1], axis, redux)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.sum_slice_async_fp32_fast_tf32(alpha, ts[0], beta, ts[1], axis,
                                           redux)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.sum_slice_async_fp64(alpha, ts[0], beta, ts[1], axis, redux)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')


def sum_fiber_async(
    alpha: float,
    x: Tensor,
    beta: float,
    sum_fiber: Tensor,
    axis: int,
    batch_ndim: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision sum_fiber
    """
    if type(x) is not type(sum_fiber):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sum_fiber_async_fp32(
            alpha, x, beta, sum_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sum_fiber_async_fp32_fast_tf32(
            alpha, x, beta, sum_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sum_fiber_async_fp64(
            alpha, x, beta, sum_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.sum_fiber_async_bf16(
            alpha, x, beta, sum_fiber, axis, batch_ndim, redux
        )
    else:
        raise TypeError


def norm_fiber_async(
    alpha: float,
    x: Tensor,
    beta: float,
    norm_fiber: Tensor,
    axis: int,
    batch_ndim: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision norm_fiber
    """
    if type(x) is not type(norm_fiber):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.norm_fiber_async_fp32(
            alpha, x, beta, norm_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.norm_fiber_async_fp32_fast_tf32(
            alpha, x, beta, norm_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.norm_fiber_async_fp64(
            alpha, x, beta, norm_fiber, axis, batch_ndim, redux
        )
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.norm_fiber_async_bf16(
            alpha, x, beta, norm_fiber, axis, batch_ndim, redux
        )
    else:
        raise TypeError


def norm_slice_async(
    alpha: float,
    x: Tensor,
    beta: float,
    norm_slice: Tensor,
    axis: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision norm_slice
    """
    if type(x) is not type(norm_slice):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.norm_slice_async_fp32(
            alpha, x, beta, norm_slice, axis, redux
        )
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.norm_slice_async_fp32_fast_tf32(
            alpha, x, beta, norm_slice, axis, redux
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.norm_slice_async_fp64(
            alpha, x, beta, norm_slice, axis, redux
        )
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.norm_slice_async_bf16(
            alpha, x, beta, norm_slice, axis, redux
        )
    else:
        raise TypeError


def pow_async(alpha: float, exp: float, x: Tensor) -> None:
    """
    Wrapper for multiprecision pow
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.pow_async_fp32(alpha, exp, x)
    else:
        core_tensor.pow_async_fp64(alpha, exp, x)


def sumnorm_async(x: Tensor, sumnorm: Tensor, axis: int) -> None:
    """
    Wrapper for multiprecision sumnorm
    """
    if type(x) is not type(sumnorm):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sumnorm_async_fp32(x, sumnorm, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sumnorm_async_fp64(x, sumnorm, axis)
    else:
        raise TypeError


def flash_softmax_gemm_async(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor_bool,
    maxsumexp: Tensor,
    dst: Tensor,
    tmp: Tensor,
    redux: int = 0,
    fp32_fast_tf32: int = 0,
) -> None:
    """
    Wrapper for multiprecision fast fused softmax+gemm
    """
    if type(Q) is not type(K):
        raise TypeError
    if type(Q) is not type(V):
        raise TypeError
    if type(Q) is not type(maxsumexp):
        raise TypeError
    if type(Q) is not type(dst):
        raise TypeError
    if type(Q) is not type(tmp):
        raise TypeError
    if type(Q) is core_tensor.Tensor_fp32:
        core_tensor.flash_softmax_gemm_async_fp32(
            Q, K, V, mask, maxsumexp, dst, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_softmax_gemm_async_fp32_fast_tf32(
            Q, K, V, mask, maxsumexp, dst, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_softmax_gemm_async_fp64(
            Q, K, V, mask, maxsumexp, dst, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_bf16:
        core_tensor.flash_softmax_gemm_async_bf16(
            Q, K, V, mask, maxsumexp, dst, tmp, redux
        )
    else:
        raise TypeError


def flash_softmax_gemm_backward_async(
    Q: Tensor,
    dQ: Tensor,
    K: Tensor,
    dK: Tensor,
    V: Tensor,
    dV: Tensor,
    mask: Tensor_bool,
    maxsumexp: Tensor,
    dst_grad: Tensor,
    tmp: Tensor,
    tmp_grad: Tensor,
    tmp_sumprod_slice: Tensor,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision fast fused softmax+gemm
    """
    if type(Q) is not type(dQ):
        raise TypeError
    if type(Q) is not type(K):
        raise TypeError
    if type(Q) is not type(dK):
        raise TypeError
    if type(Q) is not type(V):
        raise TypeError
    if type(Q) is not type(dV):
        raise TypeError
    if type(Q) is not type(maxsumexp):
        raise TypeError
    if type(Q) is not type(dst_grad):
        raise TypeError
    if type(Q) is not type(tmp):
        raise TypeError
    if type(Q) is not type(tmp_grad):
        raise TypeError
    if type(Q) is not type(tmp_sumprod_slice):
        raise TypeError
    if type(Q) is core_tensor.Tensor_fp32:
        core_tensor.flash_softmax_gemm_backward_async_fp32(
            Q,
            dQ,
            K,
            dK,
            V,
            dV,
            mask,
            maxsumexp,
            dst_grad,
            tmp,
            tmp_grad,
            tmp_sumprod_slice,
            redux,
        )
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_softmax_gemm_backward_async_fp32_fast_tf32(
            Q,
            dQ,
            K,
            dK,
            V,
            dV,
            mask,
            maxsumexp,
            dst_grad,
            tmp,
            tmp_grad,
            tmp_sumprod_slice,
            redux,
        )
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_softmax_gemm_backward_async_fp64(
            Q,
            dQ,
            K,
            dK,
            V,
            dV,
            mask,
            maxsumexp,
            dst_grad,
            tmp,
            tmp_grad,
            tmp_sumprod_slice,
            redux,
        )
    elif type(Q) is core_tensor.Tensor_bf16:
        core_tensor.flash_softmax_gemm_backward_async_bf16(
            Q,
            dQ,
            K,
            dK,
            V,
            dV,
            mask,
            maxsumexp,
            dst_grad,
            tmp,
            tmp_grad,
            tmp_sumprod_slice,
            redux,
        )
    else:
        raise TypeError


def softmax_async(
    maxsumexp: Tensor, x: Tensor, alpha: float, y: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision softmax
    """
    if type(x) is not type(y):
        raise TypeError
    if type(maxsumexp) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.softmax_async_fp32(maxsumexp, x, alpha, y, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.softmax_async_fp32_fast_tf32(maxsumexp, x, alpha, y, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.softmax_async_fp64(maxsumexp, x, alpha, y, axis)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.softmax_async_bf16(maxsumexp, x, alpha, y, axis)
    else:
        raise TypeError


def softmax_inplace_async(
    maxsumexp: Tensor, alpha, x: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision softmax
    """
    if type(maxsumexp) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.softmax_inplace_async_fp32(maxsumexp, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.softmax_inplace_async_fp32_fast_tf32(
            maxsumexp, alpha, x, axis
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.softmax_inplace_async_fp64(maxsumexp, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.softmax_inplace_async_bf16(maxsumexp, alpha, x, axis)
    else:
        raise TypeError


def scatter_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
    """
    Wrapper for multiprecision scatter
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scatter_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.scatter_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.scatter_async_int64(x, y)
    elif type(x) is core_tensor.Tensor_bool:
        core_tensor.scatter_async_bool(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.scatter_async_bf16(x, y)
    else:
        raise TypeError


def randn_async(x: Tensor, start: Sequence[int], shape: Sequence[int],
                seed: int, mean: float, dev: float) -> None:
    """Wrapper for multiprecision randn."""
    if isinstance(x, Tensor_bf16):
        ops.randn_async_bf16(x, start, shape, seed, mean, dev)
    elif isinstance(x, Tensor_fp32):
        ops.randn_async_fp32(x, start, shape, seed, mean, dev)
    elif isinstance(x, Tensor_fp32_fast_tf32):
        ops.randn_async_fp32_fast_tf32(x, start, shape, seed, mean, dev)
    elif isinstance(x, Tensor_fp64):
        ops.randn_async_fp64(x, start, shape, seed, mean, dev)
    else:
        raise TypeError('Wrong tensor type {type(x)}.')


def prod_async(x: Tensor, y: Tensor, z: Tensor) -> None:
    """
    Wrapper for multiprecision prod
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is not type(z):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_async_fp32(x, y, z)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_async_fp32_fast_tf32(x, y, z)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_async_fp64(x, y, z)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.prod_async_bf16(x, y, z)
    else:
        raise TypeError


def prod_inplace_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision prod_inplace
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_inplace_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_inplace_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_inplace_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.prod_inplace_async_bf16(x, y)
    else:
        raise TypeError


def add_async(alpha: float, x: Tensor, beta: float, y: Tensor) -> None:
    """
    Wrapper for multiprecision add
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_async_fp32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_async_fp32_fast_tf32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_async_fp64(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.add_async_bf16(alpha, x, beta, y)
    else:
        raise TypeError


def nrm2_async(
    alpha: float, x: Tensor, beta: float, y: Tensor, tmp: Tensor
) -> None:
    """
    Wrapper for multiprecision nrm2
    """
    if type(x) is not type(y) or type(x) is not type(tmp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.nrm2_async_fp32(alpha, x, beta, y, tmp)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.nrm2_async_fp64(alpha, x, beta, y, tmp)
    else:
        raise TypeError


def normalize_async(
    gb: Tensor, x: Tensor, y: Tensor, n_elem: int, eps: float, axis: int
) -> None:
    """
    Wrapper for multiprecision normalize
    """
    if type(x) is not type(y) or type(x) is not type(gb):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.normalize_async_fp32(gb, x, y, n_elem, eps, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.normalize_async_fp64(gb, x, y, n_elem, eps, axis)
    else:
        raise TypeError


def flash_maxsumexp_async(
    Q: Tensor,
    K: Tensor,
    mask: Tensor_bool,
    maxsumexp: Tensor,
    tmp: Tensor,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision fast maxsumexp
    """
    if type(Q) is not type(K):
        raise TypeError
    if type(Q) is not type(maxsumexp):
        raise TypeError
    if type(Q) is not type(tmp):
        raise TypeError
    if type(Q) is core_tensor.Tensor_fp32:
        core_tensor.flash_maxsumexp_async_fp32(
            Q, K, mask, maxsumexp, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_maxsumexp_async_fp32_fast_tf32(
            Q, K, mask, maxsumexp, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_maxsumexp_async_fp64(
            Q, K, mask, maxsumexp, tmp, redux
        )
    elif type(Q) is core_tensor.Tensor_bf16:
        core_tensor.flash_maxsumexp_async_bf16(
            Q, K, mask, maxsumexp, tmp, redux
        )
    else:
        raise TypeError


def maxsumexp_async(
    x: Tensor, maxsumexp: Tensor, axis: int, redux: int = 0
) -> None:
    """
    Wrapper for multiprecision maxsumexp
    """
    if type(x) is not type(maxsumexp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maxsumexp_async_fp32(x, maxsumexp, axis, redux)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.maxsumexp_async_fp32_fast_tf32(x, maxsumexp, axis, redux)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.maxsumexp_async_fp64(x, maxsumexp, axis, redux)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.maxsumexp_async_bf16(x, maxsumexp, axis, redux)
    else:
        raise TypeError


def add_slice_async(
    alpha: float, add_slice: Tensor, beta, x: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision add_slice
    """
    if type(add_slice) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_slice_async_fp32(alpha, add_slice, beta, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_slice_async_fp64(alpha, add_slice, beta, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_slice_async_fp32_fast_tf32(
            alpha, add_slice, beta, x, axis
        )
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.add_slice_async_bf16(alpha, add_slice, beta, x, axis)
    else:
        raise TypeError


def add_slice3_async(
    alpha: float, add_slice: Tensor, beta, x: Tensor, y: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision add_slice3
    """
    if type(add_slice) is not type(x):
        raise TypeError
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_slice3_async_fp32(alpha, add_slice, beta, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_slice3_async_fp32_fast_tf32(
            alpha, add_slice, beta, x, y, axis
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_slice3_async_fp64(alpha, add_slice, beta, x, y, axis)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.add_slice3_async_bf16(alpha, add_slice, beta, x, y, axis)
    else:
        raise TypeError


def add_fiber_async(alpha: float, add_fiber: Tensor, beta, x: Tensor,
                    axis: int, batch_ndim: int) -> None:
    """Wrapper for multiprecision `add_fiber`."""
    ts = (add_fiber, x)
    if is_tensor_of(ts, Tensor_bf16):
        ops.add_fiber_async_bf16(alpha, ts[0], beta, ts[1], axis, batch_ndim)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.add_fiber_async_fp32(alpha, ts[0], beta, ts[1], axis, batch_ndim)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.add_fiber_async_fp32_fast_tf32(alpha, ts[0], beta, ts[1], axis,
                                           batch_ndim)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.add_fiber_async_fp64(alpha, ts[0], beta, ts[1], axis, batch_ndim)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')


def prod_slice_async(
    prod_slice: Tensor, alpha: float, x: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision prod_slice
    """
    if type(prod_slice) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_slice_async_fp32(prod_slice, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_slice_async_fp32_fast_tf32(prod_slice, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_slice_async_fp64(prod_slice, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.prod_slice_async_bf16(prod_slice, alpha, x, axis)
    else:
        raise TypeError


def prod_fiber_async(
    prod_fiber: Tensor, alpha: float, x: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision prod_fiber
    """
    if type(prod_fiber) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_fiber_async_fp32(prod_fiber, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_fiber_async_fp64(prod_fiber, alpha, x, axis)
    else:
        raise TypeError


def prod_fiber3_async(
    prod_fiber: Tensor, alpha: float, x: Tensor, y: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision prod_fiber3
    """
    if type(prod_fiber) is not type(x):
        raise TypeError
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_fiber3_async_fp32(prod_fiber, alpha, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_fiber3_async_fp32_fast_tf32(
            prod_fiber, alpha, x, y, axis
        )
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_fiber3_async_fp64(prod_fiber, alpha, x, y, axis)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.prod_fiber3_async_bf16(prod_fiber, alpha, x, y, axis)
    else:
        raise TypeError


def add_scalar_async(alpha: float, beta: float, x: Tensor) -> None:
    """
    Wrapper for multiprecision add_scalar
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_scalar_async_fp32(alpha, beta, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_scalar_async_fp64(alpha, beta, x)
    else:
        raise TypeError


def gather_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
    """
    Wrapper for multiprecision gather
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gather_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gather_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.gather_async_int64(x, y)
    elif type(x) is core_tensor.Tensor_bool:
        core_tensor.gather_async_bool(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.gather_async_bf16(x, y)
    else:
        raise TypeError


def copy_intersection_async(
    x: TensorFloatOrInt,
    x_offset: List[int],
    y: TensorFloatOrInt,
    y_offset: List[int],
) -> None:
    """
    Wrapper for multiprecision copy_intersection
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_intersection_async_fp32(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.copy_intersection_async_fp64(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.copy_intersection_async_int64(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_bool:
        core_tensor.copy_intersection_async_bool(x, x_offset, y, y_offset)
    else:
        raise TypeError


def copy_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
    """
    Wrapper for multiprecision copy
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.copy_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.copy_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.copy_async_int64(x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.copy_async_bf16(x, y)
    else:
        raise TypeError


def clear_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision clear
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.clear_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.clear_async_fp32_fast_tf32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.clear_async_fp64(x)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.clear_async_bf16(x)
    else:
        raise TypeError


def axpy_async(alpha: TensorOrFloat, x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision axpy
    """
    if type(x) is not type(y):
        raise TypeError
    if type(alpha) is Tensor:
        if type(alpha) is not type(x):
            raise TypeError
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy_async_fp32(alpha, x, y)
        elif type(x) is core_tensor.Tensor_fp64:
            core_tensor.axpy_async_fp64(alpha, x, y)
        else:
            raise TypeError
    else:
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy_async_fp32(alpha, x, y)
        elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
            core_tensor.axpy_async_fp32_fast_tf32(alpha, x, y)
        elif type(x) is core_tensor.Tensor_fp64:
            core_tensor.axpy_async_fp64(alpha, x, y)
        else:
            raise TypeError


def sqrt_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision square root
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sqrt_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sqrt_async_fp64(x, y)
    else:
        raise TypeError


def sqrt_inplace_async(x: Tensor) -> None:
    """
    Wrapper for multiprecision inplace square root
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sqrt_inplace_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sqrt_inplace_async_fp64(x)
    else:
        raise TypeError


def maximum_async(x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision elementwise maximum
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maximum_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.maximum_async_fp64(x, y)
    else:
        raise TypeError


def addcdiv_async(
    alpha: float, eps: float, nom: Tensor, denom: Tensor, src: Tensor
) -> None:
    """
    Wrapper for multiprecision addcdiv
    """
    if type(nom) is not type(denom):
        raise TypeError
    if type(nom) is not type(src):
        raise TypeError
    if type(nom) is core_tensor.Tensor_fp32:
        core_tensor.addcdiv_async_fp32(alpha, eps, nom, denom, src)
    elif type(nom) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.addcdiv_async_fp32_fast_tf32(alpha, eps, nom, denom, src)
    elif type(nom) is core_tensor.Tensor_fp64:
        core_tensor.addcdiv_async_fp64(alpha, eps, nom, denom, src)
    elif type(nom) is core_tensor.Tensor_bf16:
        core_tensor.addcdiv_async_bf16(alpha, eps, nom, denom, src)
    else:
        raise TypeError


def sumprod_slice_async(
    alpha: float,
    src1: Tensor,
    src2: Tensor,
    beta: float,
    dst: Tensor,
    axis: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision sumprod_slice
    """
    if type(src1) is not type(src2):
        raise TypeError
    if type(src1) is not type(dst):
        raise TypeError
    if type(src1) is core_tensor.Tensor_fp32:
        core_tensor.sumprod_slice_async_fp32(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sumprod_slice_async_fp32_fast_tf32(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_fp64:
        core_tensor.sumprod_slice_async_fp64(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_bf16:
        core_tensor.sumprod_slice_async_bf16(
            alpha, src1, src2, beta, dst, axis, redux
        )
    else:
        raise TypeError


def sumprod_fiber_async(
    alpha: float,
    src1: Tensor,
    src2: Tensor,
    beta: float,
    dst: Tensor,
    axis: int,
    redux: int = 0,
) -> None:
    """
    Wrapper for multiprecision sumprod_fiber
    """
    if type(src1) is not type(src2):
        raise TypeError
    if type(src1) is not type(dst):
        raise TypeError
    if type(src1) is core_tensor.Tensor_fp32:
        core_tensor.sumprod_fiber_async_fp32(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sumprod_fiber_async_fp32_fast_tf32(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_fp64:
        core_tensor.sumprod_fiber_async_fp64(
            alpha, src1, src2, beta, dst, axis, redux
        )
    elif type(src1) is core_tensor.Tensor_bf16:
        core_tensor.sumprod_fiber_async_bf16(
            alpha, src1, src2, beta, dst, axis, redux
        )
    else:
        raise TypeError


def logsumexp_async(maxsumexp: Tensor, logsumexp: Tensor) -> None:
    if type(maxsumexp) is not type(logsumexp):
        raise TypeError
    if type(maxsumexp) is core_tensor.Tensor_fp32:
        core_tensor.logsumexp_async_fp32(maxsumexp, logsumexp)
    elif type(maxsumexp) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.logsumexp_async_fp32_fast_tf32(maxsumexp, logsumexp)
    elif type(maxsumexp) is core_tensor.Tensor_fp64:
        core_tensor.logsumexp_async_fp64(maxsumexp, logsumexp)
    elif type(maxsumexp) is core_tensor.Tensor_bf16:
        core_tensor.logsumexp_async_bf16(maxsumexp, logsumexp)
    else:
        raise TypeError


def total_sum_accum_async(
    alpha: float,
    logsumexp: Tensor,
    src: Tensor,
    class_labels: Tensor_int64,
    val: Tensor,
):
    if type(logsumexp) is core_tensor.Tensor_fp32:
        core_tensor.total_sum_accum_async_fp32(
            alpha, logsumexp, src, class_labels, val
        )
    elif type(logsumexp) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.total_sum_accum_async_fp32_fast_tf32(
            alpha, logsumexp, src, class_labels, val
        )
    elif type(logsumexp) is core_tensor.Tensor_fp64:
        core_tensor.total_sum_accum_async_fp64(
            alpha, logsumexp, src, class_labels, val
        )
    elif type(logsumexp) is core_tensor.Tensor_bf16:
        core_tensor.total_sum_accum_async_bf16(
            alpha, logsumexp, src, class_labels, val
        )
    else:
        raise TypeError


def subtract_indexed_outputs_async(
    val: float, class_labels: Tensor_int64, dst: Tensor
):
    if type(dst) is core_tensor.Tensor_fp32:
        core_tensor.subtract_indexed_outputs_async_fp32(val, class_labels, dst)
    elif type(dst) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.subtract_indexed_outputs_async_fp32_fast_tf32(
            val, class_labels, dst
        )
    elif type(dst) is core_tensor.Tensor_fp64:
        core_tensor.subtract_indexed_outputs_async_fp64(val, class_labels, dst)
    elif type(dst) is core_tensor.Tensor_bf16:
        core_tensor.subtract_indexed_outputs_async_bf16(val, class_labels, dst)
    else:
        raise TypeError


def scal_async(alpha: float, x: Tensor, y: Tensor) -> None:
    """
    Wrapper for multiprecision scaling
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scal_async_fp32(alpha, x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.scal_async_fp64(alpha, x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.scal_async_bf16(alpha, x, y)
    else:
        raise TypeError


def scal_inplace_async(alpha: float, x: Tensor) -> None:
    """
    Wrapper for multiprecision scaling
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scal_inplace_async_fp32(alpha, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.scal_inplace_async_fp32_fast_tf32(alpha, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.scal_inplace_async_fp64(alpha, x)
    else:
        raise TypeError


def mask_scalar_async(mask: Tensor_bool, alpha: float, x: Tensor,
                      batch_ndim: int) -> None:
    """Wrapper for multiprecision scaling."""
    if isinstance(x, Tensor_bf16):
        ops.mask_scalar_async_bf16(mask, alpha, x, batch_ndim)
    elif isinstance(x, Tensor_fp32):
        ops.mask_scalar_async_fp32(mask, alpha, x, batch_ndim)
    elif isinstance(x, Tensor_fp32_fast_tf32):
        ops.mask_scalar_async_fp32_fast_tf32(mask, alpha, x, batch_ndim)
    elif isinstance(x, Tensor_fp64):
        ops.mask_scalar_async_fp64(mask, alpha, x, batch_ndim)
    else:
        raise TypeError('Wrong tensor type {type(x)}.')


def embedding_async(
    index: Tensor_int64, vocab: Tensor, embed: Tensor, axis: int
) -> None:
    """
    Wrapper for multiprecision embedding
    """
    if type(vocab) is not type(embed):
        raise TypeError
    if type(embed) is core_tensor.Tensor_fp32:
        core_tensor.embedding_async_fp32(index, vocab, embed, axis)
    elif type(embed) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.embedding_async_fp32_fast_tf32(index, vocab, embed, axis)
    elif type(embed) is core_tensor.Tensor_fp64:
        core_tensor.embedding_async_fp64(index, vocab, embed, axis)
    elif type(embed) is core_tensor.Tensor_bf16:
        core_tensor.embedding_async_bf16(index, vocab, embed, axis)
    else:
        raise TypeError


def embedding_backward_async(index: Tensor_int64, embed: Tensor, vocab: Tensor,
                             axis: int, redux: int = 0) -> None:
    """Wrapper for multiprecision `embedding_backward`."""
    ts = (embed, vocab)
    if is_tensor_of(ts, Tensor_bf16):
        ops.embedding_backward_async_bf16(index, ts[0], ts[1], axis, redux)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.embedding_backward_async_fp32(index, ts[0], ts[1], axis, redux)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.embedding_backward_async_fp32_fast_tf32(index, ts[0], ts[1], axis,
                                                    redux)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.embedding_backward_async_fp64(index, ts[0], ts[1], axis, redux)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')


def hypot_async(alpha: float, x: Tensor, beta: float, y: Tensor) -> None:
    """
    Wrapper for multiprecision hypot
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.hypot_async_fp32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.hypot_async_fp32_fast_tf32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.hypot_async_fp64(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.hypot_async_bf16(alpha, x, beta, y)
    else:
        raise TypeError


def hypot_scalar_inverse_async(eps: float, alpha: float, x: Tensor) -> None:
    """
    Wrapper for multiprecision hypot_scalar_inverse
    """
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.hypot_scalar_inverse_async_fp32(eps, alpha, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.hypot_scalar_inverse_async_fp32_fast_tf32(eps, alpha, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.hypot_scalar_inverse_async_fp64(eps, alpha, x)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.hypot_scalar_inverse_async_bf16(eps, alpha, x)
    else:
        raise TypeError


def fused_adam_step(
    p: Tensor,
    grad: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    lr: float,
    eps: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    num_iter: int,
):
    if type(p) is not type(grad):
        raise TypeError
    if type(p) is not type(first_moment):
        raise TypeError
    if type(p) is not type(second_moment):
        raise TypeError
    if type(p) is core_tensor.Tensor_fp32:
        core_tensor.adam_step_async_fp32(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.adam_step_async_fp32_fast_tf32(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_fp64:
        core_tensor.adam_step_async_fp64(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_bf16:
        core_tensor.adam_step_async_bf16(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    else:
        raise TypeError


def fused_adamw_step(
    p: Tensor,
    grad: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    lr: float,
    eps: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    num_iter: int,
):
    if type(p) is not type(grad):
        raise TypeError
    if type(p) is not type(first_moment):
        raise TypeError
    if type(p) is not type(second_moment):
        raise TypeError
    if type(p) is core_tensor.Tensor_fp32:
        core_tensor.adamw_step_async_fp32(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.adamw_step_async_fp32_fast_tf32(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_fp64:
        core_tensor.adamw_step_async_fp64(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    elif type(p) is core_tensor.Tensor_bf16:
        core_tensor.adamw_step_async_bf16(
            num_iter,
            beta1,
            beta2,
            eps,
            lr,
            weight_decay,
            grad,
            first_moment,
            second_moment,
            p,
        )
    else:
        raise TypeError


def transpose_async(alpha: float, src: Tensor, dst: Tensor, ndim: int) -> None:
    """
    Wrapper for multiprecision transpose
    """
    if type(src) is not type(dst):
        raise TypeError
    if type(src) is core_tensor.Tensor_fp32:
        core_tensor.transpose_async_fp32(alpha, src, dst, ndim)
    elif type(src) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.transpose_async_fp32_fast_tf32(alpha, src, dst, ndim)
    elif type(src) is core_tensor.Tensor_fp64:
        core_tensor.transpose_async_fp64(alpha, src, dst, ndim)
    elif type(src) is core_tensor.Tensor_bf16:
        core_tensor.transpose_async_bf16(alpha, src, dst, ndim)
    else:
        raise TypeError


def rope_async(
        sin: Tensor,
        cos: Tensor,
        x: Tensor,
        y: Tensor
) -> None:
    """
    Wrapper for multiprecision rope
    """
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.rope_async_fp32(sin, cos, x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.rope_async_fp64(sin, cos, x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.rope_async_fp32_fast_tf32(sin, cos, x, y)
    elif type(x) is core_tensor.Tensor_bf16:
        core_tensor.rope_async_bf16(sin, cos, x, y)
    else:
        raise TypeError


def rope_backward_async(
        sin: Tensor,
        cos: Tensor,
        dy: Tensor,
        dx: Tensor
) -> None:
    """
    Wrapper for multiprecision rope
    """
    if type(dx) is not type(dy):
        raise TypeError
    if type(dx) is core_tensor.Tensor_fp32:
        core_tensor.rope_backward_async_fp32(sin, cos, dy, dx)
    elif type(dx) is core_tensor.Tensor_fp64:
        core_tensor.rope_backward_async_fp64(sin, cos, dy, dx)
    elif type(dx) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.rope_backward_async_fp32_fast_tf32(sin, cos, dy, dx)
    elif type(dx) is core_tensor.Tensor_bf16:
        core_tensor.rope_backward_async_bf16(sin, cos, dy, dx)
    else:
        raise TypeError


def conv2d_inplace_async(
        alpha: float,
        X: Tensor,
        C: Tensor,
        beta: float,
        Y: Tensor,
        padding: Sequence[int] = [0, 0],
        stride: Sequence[int] = [1, 1],
        dilation: Sequence[int] = [1, 1]
) -> None:
    """Wrapper for multiprecision conv2d_inplace"""
    ts = (X, C, Y)
    if is_tensor_of(ts, Tensor_bf16):
        ops.conv2d_inplace_async_bf16(alpha, X, C, beta, Y,
                padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.conv2d_inplace_async_fp32(alpha, X, C, beta, Y,
                padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.conv2d_inplace_async_fp32_fast_tf32(alpha, X, C, beta,
                Y, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.conv2d_inplace_async_fp64(alpha, X, C, beta, Y,
                padding, stride, dilation)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')


def conv2d_bwd_input_inplace_async(
        alpha: float,
        dY: Tensor,
        C: Tensor,
        beta: float,
        dX: Tensor,
        padding: Sequence[int] = [0, 0],
        stride: Sequence[int] = [1, 1],
        dilation: Sequence[int] = [1, 1]
) -> None:
    """Wrapper for multiprecision conv2d_bwd_input_inplace"""
    ts = (dY, C, dX)
    if is_tensor_of(ts, Tensor_bf16):
        ops.conv2d_bwd_input_inplace_async_bf16(alpha, dY, C, beta,
                dX, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.conv2d_bwd_input_inplace_async_fp32(alpha, dY, C, beta,
                dX, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.conv2d_bwd_input_inplace_async_fp32_fast_tf32(alpha, dY,
                C, beta, dX, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.conv2d_bwd_input_inplace_async_fp64(alpha, dY, C, beta,
                dX, padding, stride, dilation)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')


def conv2d_bwd_weight_inplace_async(
        alpha: float,
        X: Tensor,
        dY: Tensor,
        beta: float,
        dC: Tensor,
        padding: Sequence[int] = [0, 0],
        stride: Sequence[int] = [1, 1],
        dilation: Sequence[int] = [1, 1]
) -> None:
    """Wrapper for multiprecision conv2d_bwd_weight_inplace"""
    ts = (X, dY, dC)
    if is_tensor_of(ts, Tensor_bf16):
        ops.conv2d_bwd_weight_inplace_async_bf16(alpha, X, dY, beta,
                dC, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32):
        ops.conv2d_bwd_weight_inplace_async_fp32(alpha, X, dY, beta,
                dC, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp32_fast_tf32):
        ops.conv2d_bwd_weight_inplace_async_fp32_fast_tf32(alpha, X,
                dY, beta, dC, padding, stride, dilation)
    elif is_tensor_of(ts, Tensor_fp64):
        ops.conv2d_bwd_weight_inplace_async_fp64(alpha, X, dY, beta,
                dC, padding, stride, dilation)
    else:
        types = ', '.join(str(type(t)) for t in ts)
        raise TypeError(
            f'Tensor must share the same type but actual types are {types}.')
