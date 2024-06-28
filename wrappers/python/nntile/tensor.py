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
# @version 1.0.0

from .nntile_core import tensor as core_tensor
from .nntile_core.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, \
        Tensor_int64, Tensor_fp16, Tensor_bool, Tensor_fp32_fast_tf32
from .nntile_core import TransOp, notrans, trans
from typing import Union, List

# Multiprecision tensor as a union type for all precisions
Tensor = Union[core_tensor.Tensor_fp32, core_tensor.Tensor_fp64, core_tensor.Tensor_fp32_fast_tf32]
# Optional tensor argument
TensorOrNone = Union[Tensor, None]
# Union of multiprecision tensor and float
TensorOrFloat = Union[Tensor, float]
TensorFloatOrInt = Union[Tensor, core_tensor.Tensor_int64]

# Struct meant for tensor, its gradient and a flag if gradient is required
class TensorMoments(object):
    value: TensorOrNone
    grad: TensorOrNone
    grad_required: bool

    def __init__(self, value: TensorOrNone, grad: TensorOrNone,
            grad_required: bool):
        self.value = value
        self.grad = grad
        self.grad_required = grad_required

    def __del__(self):
        self.unregister()

    def unregister(self):
        if self.value is not None:
            self.value.unregister()
        if self.grad is not None:
            self.grad.unregister()


# Wrapper for multiprecision gemm
def gemm_async(alpha: float, trans_A: TransOp, A: Tensor, trans_B: TransOp, \
        B: Tensor, beta: float, C: Tensor, ndim: int, \
        batch_ndim: int, redux: int=0) -> None:
    if type(A) is not type(B) or type(A) is not type(C):
        raise TypeError
    if type(A) is core_tensor.Tensor_fp32:
        core_tensor.gemm_async_fp32(alpha, trans_A, A, trans_B, B, beta, C,
                ndim, batch_ndim, redux)
    elif type(A) is core_tensor.Tensor_fp64:
        core_tensor.gemm_async_fp64(alpha, trans_A, A, trans_B, B, beta, C,
                ndim, batch_ndim, redux)
    elif type(A) is core_tensor.Tensor_fp16:
        core_tensor.gemm_async_fp16(alpha, trans_A, A, trans_B, B, beta, C,
                ndim, batch_ndim, redux)
    elif type(A) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gemm_async_fp32_fast_tf32(alpha, trans_A, A, trans_B, B, beta, C,
                ndim, batch_ndim, redux)
    else:
        raise TypeError

# Wrapper for multiprecision ReLU
def relu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_async_fp64(x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_async_fp32_fast_tf32(x)
    else:
        raise TypeError

# Wrapper for multiprecision forward ReLU
def relu_forward_async(x: Tensor, y: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_forward_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_forward_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_forward_async_fp32_fast_tf32(x, y)
    else:
        raise TypeError

# Wrapper for multiprecision backward ReLU
def relu_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.relu_backward_async_fp64(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.relu_backward_async_fp32_fast_tf32(x, dy, dx)
    else:
        raise TypeError

# Wrapper for multiprecision derivative of ReLU
def drelu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.drelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.drelu_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision GELU
def gelu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelu_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision derivative of GELU
def dgelu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.dgelu_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.dgelu_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision backward GeLU
def gelu_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelu_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelu_backward_async_fp64(x, dy, dx)
    else:
        raise TypeError

# Wrapper for multiprecision approximated GELU
def gelutanh_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gelutanh_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_async_fp64(x, y)
    else:
        raise TypeError

# Wrapper for multiprecision approximated GELU
def gelutanh_inplace_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_inplace_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_inplace_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision derivative of approximate GELU
def dgelutanh_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.dgelutanh_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.dgelutanh_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision backward approximate GeLU
def gelutanh_backward_async(x: Tensor, dy: Tensor, dx: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gelutanh_backward_async_fp32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.gelutanh_backward_async_fp32_fast_tf32(x, dy, dx)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.gelutanh_backward_async_fp64(x, dy, dx)
    else:
        raise TypeError

# Wrapper for multiprecision fill
def fill_async(val: float, x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.fill_async_fp32(val, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.fill_async_fp32_fast_tf32(val, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.fill_async_fp64(val, x)
    else:
        raise TypeError

# Wrapper for multiprecision sum_slice
def sum_slice_async(alpha: float, x: Tensor, beta: float, sum_slice: Tensor, \
        axis: int, redux: int=0) -> None:
    if type(x) is not type(sum_slice):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sum_slice_async_fp32(alpha, x, beta, sum_slice, axis, \
                redux)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sum_slice_async_fp32_fast_tf32(alpha, x, beta, sum_slice, axis, \
                redux)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sum_slice_async_fp64(alpha, x, beta, sum_slice, axis, \
                redux)
    else:
        raise TypeError

# Wrapper for multiprecision sum_fiber
def sum_fiber_async(alpha: float, x: Tensor, beta: float, sum_fiber: Tensor, \
        axis: int, batch_ndim: int, redux: int=0) -> None:
    if type(x) is not type(sum_fiber):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sum_fiber_async_fp32(alpha, x, beta, sum_fiber, axis, \
                batch_ndim, redux)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sum_fiber_async_fp32_fast_tf32(alpha, x, beta, sum_fiber, axis, \
                batch_ndim, redux)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sum_fiber_async_fp64(alpha, x, beta, sum_fiber, axis, \
                batch_ndim, redux)
    else:
        raise TypeError

# Wrapper for multiprecision norm_slice
def norm_slice_async(alpha: float, x: Tensor, beta: float, \
        norm_slice: Tensor, axis: int, redux: int=0) -> None:
    if type(x) is not type(norm_slice):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.norm_slice_async_fp32(alpha, x, beta, norm_slice, axis, \
                redux)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.norm_slice_async_fp32_fast_tf32(alpha, x, beta, norm_slice, axis, \
                redux)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.norm_slice_async_fp64(alpha, x, beta, norm_slice, axis, \
                redux)
    else:
        raise TypeError

# Wrapper for multiprecision pow
def pow_async(alpha: float, exp: float, x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.pow_async_fp32(alpha, exp, x)
    else:
        core_tensor.pow_async_fp64(alpha, exp, x)

# Wrapper for multiprecision sumnorm
def sumnorm_async(x: Tensor, sumnorm: Tensor, axis: int) -> None:
    if type(x) is not type(sumnorm):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sumnorm_async_fp32(x, sumnorm, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sumnorm_async_fp64(x, sumnorm, axis)
    else:
        raise TypeError

# Wrapper for multiprecision fast fused softmax+gemm
def flash_softmax_gemm_async(Q: Tensor, K: Tensor, V: Tensor, \
        mask: Tensor_bool, maxsumexp: Tensor, dst: Tensor, tmp: Tensor, \
        redux: int=0, fp32_fast_tf32: int=0) -> None:
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
        core_tensor.flash_softmax_gemm_async_fp32(Q, K, V, mask, maxsumexp, \
                dst, tmp, redux)
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_softmax_gemm_async_fp32_fast_tf32(Q, K, V, mask, maxsumexp, \
                dst, tmp, redux)
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_softmax_gemm_async_fp64(Q, K, V, mask, maxsumexp, \
                dst, tmp, redux)
    else:
        raise TypeError

# Wrapper for multiprecision fast fused softmax+gemm
def flash_softmax_gemm_backward_async(Q: Tensor, dQ: Tensor, K: Tensor, \
        dK: Tensor, V: Tensor, dV: Tensor, mask: Tensor_bool, \
        maxsumexp: Tensor, dst_grad: Tensor, tmp: Tensor, tmp_grad: Tensor, \
        tmp_sumprod_slice: Tensor, redux: int=0) -> None:
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
        core_tensor.flash_softmax_gemm_backward_async_fp32(Q, dQ, K, dK, V, \
                dV, mask, maxsumexp, dst_grad, tmp, tmp_grad, \
                tmp_sumprod_slice, redux)
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_softmax_gemm_backward_async_fp32_fast_tf32(Q, dQ, K, dK, V, \
                dV, mask, maxsumexp, dst_grad, tmp, tmp_grad, \
                tmp_sumprod_slice, redux)
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_softmax_gemm_backward_async_fp64(Q, dQ, K, dK, V, \
                dV, mask, maxsumexp, dst_grad, tmp, tmp_grad, \
                tmp_sumprod_slice, redux)
    else:
        raise TypeError

# Wrapper for multiprecision softmax
def softmax_async(maxsumexp: Tensor, x: Tensor, alpha: float, y: Tensor, \
        axis: int) -> None:
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
    else:
        raise TypeError

# Wrapper for multiprecision softmax
def softmax_inplace_async(maxsumexp: Tensor, alpha, x: Tensor, \
        axis: int) -> None:
    if type(maxsumexp) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.softmax_inplace_async_fp32(maxsumexp, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.softmax_inplace_async_fp32_fast_tf32(maxsumexp, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.softmax_inplace_async_fp64(maxsumexp, alpha, x, axis)
    else:
        raise TypeError

# Wrapper for multiprecision scatter
def scatter_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
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
    else:
        raise TypeError

# Wrapper for multiprecision randn
def randn_async(x: Tensor, start: List[int], shape: List[int], seed: int,
        mean: float, dev: float) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.randn_async_fp32(x, start, shape, seed, mean, dev)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.randn_async_fp32_fast_tf32(x, start, shape, seed, mean, dev)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.randn_async_fp64(x, start, shape, seed, mean, dev)
    else:
        raise TypeError

# Wrapper for multiprecision prod
def prod_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_async_fp32_fast_tf32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_async_fp64(x, y)
    else:
        raise TypeError

# Wrapper for multiprecision add
def add_async(alpha: float, x: Tensor, beta: float, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_async_fp32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_async_fp32_fast_tf32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_async_fp64(alpha, x, beta, y)
    else:
        raise TypeError

# Wrapper for multiprecision nrm2
def nrm2_async(alpha:float, x: Tensor, beta: float, y: Tensor,
               tmp: Tensor) -> None:
    if type(x) is not type(y) or type(x) is not type(tmp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.nrm2_async_fp32(alpha, x, beta, y, tmp)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.nrm2_async_fp64(alpha, x, beta, y, tmp)
    else:
        raise TypeError

# Wrapper for multiprecision normalize
def normalize_async(gb: Tensor, x: Tensor, y: Tensor, l: int, eps: float,
        axis: int) -> None:
    if type(x) is not type(y) or type(x) is not type(gb):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.normalize_async_fp32(gb, x, y, l, eps, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.normalize_async_fp64(gb, x, y, l, eps, axis)
    else:
        raise TypeError

# Wrapper for multiprecision fast maxsumexp
def flash_maxsumexp_async(Q: Tensor, K: Tensor, mask: Tensor_bool, \
        maxsumexp: Tensor, tmp: Tensor, redux: int=0) -> None:
    if type(Q) is not type(K):
        raise TypeError
    if type(Q) is not type(maxsumexp):
        raise TypeError
    if type(Q) is not type(tmp):
        raise TypeError
    if type(Q) is core_tensor.Tensor_fp32:
        core_tensor.flash_maxsumexp_async_fp32(Q, K, mask, maxsumexp, tmp, redux)
    elif type(Q) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.flash_maxsumexp_async_fp32_fast_tf32(Q, K, mask, maxsumexp, tmp, redux)
    elif type(Q) is core_tensor.Tensor_fp64:
        core_tensor.flash_maxsumexp_async_fp64(Q, K, mask, maxsumexp, tmp, \
                redux, 0)
    else:
        raise TypeError

# Wrapper for multiprecision maxsumexp
def maxsumexp_async(x: Tensor, maxsumexp: Tensor, axis: int, \
        redux: int=0) -> None:
    if type(x) is not type(maxsumexp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maxsumexp_async_fp32(x, maxsumexp, axis, redux)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.maxsumexp_async_fp32_fast_tf32(x, maxsumexp, axis, redux)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.maxsumexp_async_fp64(x, maxsumexp, axis, redux)
    else:
        raise TypeError

# Wrapper for multiprecision add_slice
def add_slice_async(alpha: float, add_slice: Tensor, beta, x: Tensor, \
        axis: int) -> None:
    if type(add_slice) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_slice_async_fp32(alpha, add_slice, beta, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_slice_async_fp64(alpha, add_slice, beta, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_slice_async_fp32_fast_tf32(alpha, add_slice, beta, x, axis)
    else:
        raise TypeError

# Wrapper for multiprecision add_slice3
def add_slice3_async(alpha: float, add_slice: Tensor, beta, x: Tensor, \
        y: Tensor, axis: int) -> None:
    if type(add_slice) is not type(x):
        raise TypeError
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_slice3_async_fp32(alpha, add_slice, beta, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_slice3_async_fp32_fast_tf32(alpha, add_slice, beta, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_slice3_async_fp64(alpha, add_slice, beta, x, y, axis)
    else:
        raise TypeError

# Wrapper for multiprecision add_fiber
def add_fiber_async(alpha: float, add_fiber: Tensor, beta, x: Tensor, \
        axis: int, batch_ndim: int) -> None:
    if type(add_fiber) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_fiber_async_fp32(alpha, add_fiber, beta, x, axis, \
                batch_ndim)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_fiber_async_fp64(alpha, add_fiber, beta, x, axis, \
                batch_ndim)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.add_fiber_async_fp32_fast_tf32(alpha, add_fiber, beta, x, axis, \
                batch_ndim)
    else:
        raise TypeError

# Wrapper for multiprecision prod_slice
def prod_slice_async(prod_slice: Tensor, alpha: float, x: Tensor, \
        axis: int) -> None:
    if type(prod_slice) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_slice_async_fp32(prod_slice, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_slice_async_fp32_fast_tf32(prod_slice, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_slice_async_fp64(prod_slice, alpha, x, axis)
    else:
        raise TypeError

# Wrapper for multiprecision prod_fiber
def prod_fiber_async(prod_fiber: Tensor, alpha: float, x: Tensor, \
        axis: int) -> None:
    if type(prod_fiber) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_fiber_async_fp32(prod_fiber, alpha, x, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_fiber_async_fp64(prod_fiber, alpha, x, axis)
    else:
        raise TypeError

# Wrapper for multiprecision prod_fiber3
def prod_fiber3_async(prod_fiber: Tensor, alpha: float, x: Tensor, \
        y: Tensor, axis: int) -> None:
    if type(prod_fiber) is not type(x):
        raise TypeError
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_fiber3_async_fp32(prod_fiber, alpha, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.prod_fiber3_async_fp32_fast_tf32(prod_fiber, alpha, x, y, axis)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.prod_fiber3_async_fp64(prod_fiber, alpha, x, y, axis)
    else:
        raise TypeError

# Wrapper for multiprecision add_scalar
def add_scalar_async(alpha:float, beta: float, x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.add_scalar_async_fp32(alpha, beta, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.add_scalar_async_fp64(alpha, beta, x)
    else:
        raise TypeError

# Wrapper for multiprecision gather
def gather_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
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
    else:
        raise TypeError

# Wrapper for multiprecision copy_intersection
def copy_intersection_async(x: TensorFloatOrInt, x_offset: List[int], \
        y: TensorFloatOrInt, y_offset: List[int]) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_intersection_async_fp32(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.copy_intersection_async_fp64(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.copy_intersection_async_int64(x, x_offset, y, y_offset)
    else:
        raise TypeError

# Wrapper for multiprecision copy
def copy_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
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
    else:
        raise TypeError

# Wrapper for multiprecision clear
def clear_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.clear_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.clear_async_fp32_fast_tf32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.clear_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision axpy
def axpy_async(alpha: TensorOrFloat, x: Tensor, y: Tensor) -> None:
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

# Wrapper for multiprecision square root
def sqrt_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sqrt_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sqrt_async_fp64(x, y)
    else:
        raise TypeError

# Wrapper for multiprecision inplace square root
def sqrt_inplace_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sqrt_inplace_async_fp32(x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.sqrt_inplace_async_fp64(x)
    else:
        raise TypeError

# Wrapper for multiprecision elementwise maximum
def maximum_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maximum_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.maximum_async_fp64(x, y)
    else:
        raise TypeError

# Wrapper for multiprecision addcdiv
def addcdiv_async(alpha: float, eps: float, nom: Tensor, denom: Tensor, \
        src: Tensor) -> None:
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
    else:
        raise TypeError

# Wrapper for multiprecision sumprod_slice
def sumprod_slice_async(alpha: float, src1: Tensor, src2: Tensor, \
        beta: float, dst: Tensor, axis: int, redux: int=0) -> None:
    if type(src1) is not type(src2):
        raise TypeError
    if type(src1) is not type(dst):
        raise TypeError
    if type(src1) is core_tensor.Tensor_fp32:
        core_tensor.sumprod_slice_async_fp32(alpha, src1, src2, beta, dst, \
                axis, redux)
    elif type(src1) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sumprod_slice_async_fp32_fast_tf32(alpha, src1, src2, beta, dst, \
                axis, redux)
    elif type(src1) is core_tensor.Tensor_fp64:
        core_tensor.sumprod_slice_async_fp64(alpha, src1, src2, beta, dst, \
                axis, redux)
    else:
        raise TypeError

# Wrapper for multiprecision sumprod_fiber
def sumprod_fiber_async(alpha: float, src1: Tensor, src2: Tensor, \
        beta: float, dst: Tensor, axis: int, redux: int=0) -> None:
    if type(src1) is not type(src2):
        raise TypeError
    if type(src1) is not type(dst):
        raise TypeError
    if type(src1) is core_tensor.Tensor_fp32:
        core_tensor.sumprod_fiber_async_fp32(alpha, src1, src2, beta, dst, \
                axis, redux)
    elif type(src1) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.sumprod_fiber_async_fp32_fast_tf32(alpha, src1, src2, beta, dst, \
                axis, redux)
    elif type(src1) is core_tensor.Tensor_fp64:
        core_tensor.sumprod_fiber_async_fp64(alpha, src1, src2, beta, dst, \
                axis, redux)
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
    else:
        raise TypeError

def total_sum_accum_async(alpha: float, logsumexp: Tensor, src: Tensor, \
        class_labels: Tensor_int64, val: Tensor):
    if type(logsumexp) is core_tensor.Tensor_fp32:
        core_tensor.total_sum_accum_async_fp32(alpha, logsumexp, src, class_labels, \
                val)
    elif type(logsumexp) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.total_sum_accum_async_fp32_fast_tf32(alpha, logsumexp, src, class_labels, \
                val)
    elif type(logsumexp) is core_tensor.Tensor_fp64:
        core_tensor.total_sum_accum_async_fp64(alpha, logsumexp, src, class_labels, \
                val)
    else:
        raise TypeError

def subtract_indexed_outputs_async(val: float, class_labels: Tensor_int64, \
        dst: Tensor):
    if type(dst) is core_tensor.Tensor_fp32:
        core_tensor.subtract_indexed_outputs_async_fp32(val, class_labels, dst)
    elif type(dst) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.subtract_indexed_outputs_async_fp32_fast_tf32(val, class_labels, dst)
    elif type(dst) is core_tensor.Tensor_fp64:
        core_tensor.subtract_indexed_outputs_async_fp64(val, class_labels, dst)
    else:
        raise TypeError

# Wrapper for multiprecision scaling
def scal_async(alpha: float, x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scal_async_fp32(alpha, x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.scal_async_fp64(alpha, x, y)
    else:
        raise TypeError

# Wrapper for multiprecision scaling
def scal_inplace_async(alpha: float, x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scal_inplace_async_fp32(alpha, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.scal_inplace_async_fp32_fast_tf32(alpha, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.scal_inplace_async_fp64(alpha, x)
    else:
        raise TypeError

# Wrapper for multiprecision scaling
def mask_scalar_async(mask: Tensor_bool, alpha: float, x: Tensor, \
        batch_ndim: int) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.mask_scalar_async_fp32(mask, alpha, x, batch_ndim)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.mask_scalar_async_fp32_fast_tf32(mask, alpha, x, batch_ndim)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.mask_scalar_async_fp64(mask, alpha, x, batch_ndim)
    else:
        raise TypeError

# Wrapper for multiprecision embedding
def embedding_async(index: Tensor_int64, vocab: Tensor, embed: Tensor, \
        axis: int) -> None:
    if type(vocab) is not type(embed):
        raise TypeError
    if type(embed) is core_tensor.Tensor_fp32:
        core_tensor.embedding_async_fp32(index, vocab, embed, axis)
    elif type(embed) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.embedding_async_fp32_fast_tf32(index, vocab, embed, axis)
    elif type(embed) is core_tensor.Tensor_fp64:
        core_tensor.embedding_async_fp64(index, vocab, embed, axis)
    else:
        raise TypeError

# Wrapper for multiprecision embedding_backward
def embedding_backward_async(index: Tensor_int64, embed: Tensor, \
        vocab: Tensor, axis: int, redux: int=0) -> None:
    if type(vocab) is not type(embed):
        raise TypeError
    if type(embed) is core_tensor.Tensor_fp32:
        core_tensor.embedding_backward_async_fp32(index, embed, vocab, axis, \
                redux)
    elif type(embed) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.embedding_backward_async_fp32_fast_tf32(index, embed, vocab, axis, \
                redux)
    elif type(embed) is core_tensor.Tensor_fp64:
        core_tensor.embedding_backward_async_fp64(index, embed, vocab, axis, \
                redux)
    else:
        raise TypeError

# Wrapper for multiprecision hypot
def hypot_async(alpha: float, x: Tensor, beta: float, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.hypot_async_fp32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.hypot_async_fp32_fast_tf32(alpha, x, beta, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.hypot_async_fp64(alpha, x, beta, y)
    else:
        raise TypeError

# Wrapper for multiprecision hypot_scalar_inverse
def hypot_scalar_inverse_async(eps: float, alpha: float, x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.hypot_scalar_inverse_async_fp32(eps, alpha, x)
    elif type(x) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.hypot_scalar_inverse_async_fp32_fast_tf32(eps, alpha, x)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.hypot_scalar_inverse_async_fp64(eps, alpha, x)
    else:
        raise TypeError

def fused_adam_step(p: Tensor, grad: Tensor, first_moment: Tensor, second_moment: Tensor,
                   lr: float, eps: float, beta1: float, beta2: float, weight_decay: float, num_iter: int):
    if type(p) is not type(grad):
        raise TypeError
    if type(p) is not type(first_moment):
        raise TypeError
    if type(p) is not type(second_moment):
        raise TypeError
    if type(p) is core_tensor.Tensor_fp32:
        core_tensor.adam_step_async_fp32(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    elif type(p) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.adam_step_async_fp32_fast_tf32(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    elif type(p) is core_tensor.Tensor_fp64:
        core_tensor.adam_step_async_fp64(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    else:
        raise TypeError

def fused_adamw_step(p: Tensor, grad: Tensor, first_moment: Tensor, second_moment: Tensor,
                   lr: float, eps: float, beta1: float, beta2: float, weight_decay: float, num_iter: int):
    if type(p) is not type(grad):
        raise TypeError
    if type(p) is not type(first_moment):
        raise TypeError
    if type(p) is not type(second_moment):
        raise TypeError
    if type(p) is core_tensor.Tensor_fp32:
        core_tensor.adamw_step_async_fp32(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    elif type(p) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.adamw_step_async_fp32_fast_tf32(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    elif type(p) is core_tensor.Tensor_fp64:
        core_tensor.adamw_step_async_fp64(num_iter, beta1, beta2, eps, lr, weight_decay,
                                         grad, first_moment, second_moment, p)
    else:
        raise TypeError

# Wrapper for multiprecision transpose
def transpose_async(alpha: float, src: Tensor, dst: Tensor, ndim: int) -> None:
    if type(src) is not type(dst):
        raise TypeError
    if type(src) is core_tensor.Tensor_fp32:
        core_tensor.transpose_async_fp32(alpha, src, dst, ndim)
    elif type(src) is core_tensor.Tensor_fp32_fast_tf32:
        core_tensor.transpose_async_fp32_fast_tf32(alpha, src, dst, ndim)
    elif type(src) is core_tensor.Tensor_fp64:
        core_tensor.transpose_async_fp64(alpha, src, dst, ndim)
    else:
        raise TypeError
