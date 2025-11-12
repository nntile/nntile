# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/sdpa.py
# Scaled dot-product attention (SDPA) layer of NNTile Python package
#
# @version 1.1.0

from __future__ import annotations

from typing import Optional

import numpy as np

from nntile.functions import flash_sdpa_fwd_cudnn_async, is_tensor_of
from nntile.layer.base_layer import BaseLayer
from nntile.nntile_core.tensor import Tensor_bf16, Tensor_fp16, Tensor_fp32
from nntile.tensor import (
    Tensor, TensorMoments, TensorOrNone, TensorTraits, add_slice_inplace_async,
    clear_async, gemm_async, mask_scalar_async, maxsumexp_async,
    multiply_inplace_async, notrans, softmax_inplace_async,
    sumprod_slice_async, trans)


class Sdpa(BaseLayer):
    """
    Scaled dot-product attention layer with vanilla and flash implementations.

    Args:
        q: Query tensor moments (head_size, n_seq, n_batch, n_head[, ...]).
        k: Key tensor moments with shape compatible with q.
        v: Value tensor moments matching q layout.
        y: Output tensor moments, shape must match v.
        attn: Attention logits buffer (required for vanilla SDPA).
        attn_maxsumexp: Workspace for numerically stable softmax.
        attn_sumprod_slice: Workspace for softmax backward pass.
        mask: Optional causal mask broadcastable to attention logits.
        flash_attention: Enables cuDNN FlashAttention forward path.
        flash_logsumexp: fp32 scratch buffer required by cuDNN kernel.
        redux: Enables reduction semantics for distributed training.
    """

    def __init__(
        self,
        q: TensorMoments,
        k: TensorMoments,
        v: TensorMoments,
        y: TensorMoments,
        attn: Optional[TensorMoments] = None,
        attn_maxsumexp: Optional[Tensor] = None,
        attn_sumprod_slice: Optional[Tensor] = None,
        mask: TensorOrNone = None,
        flash_attention: bool = False,
        flash_logsumexp: Optional[Tensor] = None,
        redux: bool = False,
    ):
        q_shape = q.value.shape

        if not flash_attention and (
            attn is None or
            attn_maxsumexp is None or
            attn_sumprod_slice is None
        ):
            raise ValueError(
                "Vanilla SDPA requires attn, attn_maxsumexp and "
                "attn_sumprod_slice"
            )
        if flash_attention and flash_logsumexp is None:
            raise ValueError("Flash SDPA requires flash_logsumexp workspace")
        if flash_logsumexp is not None and not isinstance(
            flash_logsumexp, Tensor_fp32
        ):
            raise TypeError("flash_logsumexp must be a Tensor_fp32 instance")
        if flash_attention and flash_logsumexp is not None:
            expected_shape = tuple(q_shape[1:])
            actual_shape = tuple(flash_logsumexp.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    "flash_logsumexp must match q shape without the head "
                    f"dimension ({expected_shape}), got {actual_shape}"
                )

        temporaries = []
        for tmp in (attn, attn_maxsumexp, attn_sumprod_slice, flash_logsumexp):
            if tmp is not None:
                temporaries.append(tmp)

        super().__init__([q, k, v], [y], [], temporaries)

        if len(q_shape) != 5:
            raise ValueError(
                "SDPA tensors must have shape "
                "[head_size, n_seq, n_batch, kv_group_size, n_head_kv]"
            )

        self.q = q
        if self.q.grad is not None:
            self.q.grad.set_reduction_add()
        self.k = k
        if self.k.grad is not None:
            self.k.grad.set_reduction_add()
        self.v = v
        if self.v.grad is not None:
            self.v.grad.set_reduction_add()
        self.y = y
        self.y.value.set_reduction_add()
        if self.y.grad is not None:
            self.y.grad.set_reduction_add()

        self.attn = attn
        if self.attn is not None:
            self.attn.value.set_reduction_add()
            if self.attn.grad is not None:
                self.attn.grad.set_reduction_add()
        self.attn_maxsumexp = attn_maxsumexp
        if self.attn_maxsumexp is not None:
            self.attn_maxsumexp.set_reduction_maxsumexp()
        self.attn_sumprod_slice = attn_sumprod_slice
        if self.attn_sumprod_slice is not None:
            self.attn_sumprod_slice.set_reduction_add()

        self.mask = mask
        self.flash_attention = flash_attention
        self.flash_logsumexp = flash_logsumexp
        self.head_size = q_shape[0]
        if self.head_size <= 0:
            raise ValueError("Query tensor must have non-zero head size")
        self.scale = float(1.0 / np.float32(self.head_size ** 0.5))
        self.val = -np.float32(np.inf)
        self.redux = 1 if redux else 0
        self.batch_ndim = len(q_shape) - 2

        # Validate tensor dtypes for flash attention
        if self.flash_attention:
            tensors = (self.q.value, self.k.value, self.v.value, self.y.value)
            if not (
                is_tensor_of(tensors, Tensor_bf16)
                or is_tensor_of(tensors, Tensor_fp16)
            ):
                raise TypeError(
                    "Flash SDPA currently supports only bf16 or fp16 tensors"
                )

    def forward_async(self, effective_size=None):
        if self.flash_attention:
            self._forward_flash()
        else:
            self._forward_vanilla()

    def backward_async(self):
        if self.flash_attention:
            raise NotImplementedError(
                "Flash SDPA backward is not implemented yet."
            )
        self._backward_vanilla()

    @staticmethod
    def generate_simple(
        q: TensorMoments,
        k: TensorMoments,
        v: TensorMoments,
        mask: TensorOrNone = None,
        flash_attention: bool = False,
        redux: bool = False,
    ) -> "Sdpa":
        """Utility constructor that allocates outputs/temporaries."""
        if not (type(q.value) is type(k.value) is type(v.value)):
            raise TypeError("Q, K and V tensors must share the same dtype")

        q_shape = list(q.value.shape)
        k_shape = list(k.value.shape)
        v_shape = list(v.value.shape)
        if q_shape != k_shape or q_shape != v_shape:
            raise ValueError("Q, K and V tensors must share the same shape")
        if len(q_shape) != 5:
            raise ValueError(
                "SDPA tensors must have shape "
                "[head_size, n_seq, n_batch, kv_group_size, n_head_kv]"
            )

        q_basetile = list(q.value.basetile_shape)
        batch_shape = q_shape[2:]
        batch_basetile = q_basetile[2:]
        k_seq = k_shape[1]
        q_seq = q_shape[1]

        tensor_type = type(q.value)

        # Output tensor
        y_traits = TensorTraits(q_shape, q_basetile)
        y_distr = [0] * y_traits.grid.nelems
        y_value = tensor_type(y_traits, y_distr)
        y_grad = tensor_type(y_traits, y_distr)
        y = TensorMoments(y_value, y_grad, True)

        attn = None
        attn_max = None
        attn_sum = None
        logsumexp = None

        if flash_attention:
            logsumexp_shape = q_shape[1:]
            logsumexp_basetile = q_basetile[1:]
            logsumexp_traits = TensorTraits(
                logsumexp_shape, logsumexp_basetile
            )
            logsumexp = Tensor_fp32(
                logsumexp_traits, [0] * logsumexp_traits.grid.nelems
            )
        else:
            attn_shape = [k_seq, q_seq] + batch_shape
            attn_basetile = [k.value.basetile_shape[1], q_basetile[1]] \
                + batch_basetile
            attn_traits = TensorTraits(attn_shape, attn_basetile)
            attn_distr = [0] * attn_traits.grid.nelems
            attn_value = tensor_type(attn_traits, attn_distr)
            attn_grad = tensor_type(attn_traits, attn_distr)
            attn = TensorMoments(attn_value, attn_grad, True)

            attn_max_shape = [2, q_seq] + batch_shape
            attn_max_basetile = [2, q_basetile[1]] + batch_basetile
            attn_max_traits = TensorTraits(attn_max_shape, attn_max_basetile)
            attn_max = tensor_type(
                attn_max_traits, [0] * attn_max_traits.grid.nelems
            )

            attn_sum_shape = [q_seq] + batch_shape
            attn_sum_basetile = [q_basetile[1]] + batch_basetile
            attn_sum_traits = TensorTraits(attn_sum_shape, attn_sum_basetile)
            attn_sum = tensor_type(
                attn_sum_traits, [0] * attn_sum_traits.grid.nelems
            )

        return Sdpa(
            q=q,
            k=k,
            v=v,
            y=y,
            attn=attn,
            attn_maxsumexp=attn_max,
            attn_sumprod_slice=attn_sum,
            mask=mask,
            flash_attention=flash_attention,
            flash_logsumexp=logsumexp,
            redux=redux,
        )

    # ---- Internal helpers -------------------------------------------------

    def _forward_vanilla(self):
        if self.attn is None or self.attn_maxsumexp is None:
            raise RuntimeError("Vanilla SDPA buffers are not initialized")

        gemm_async(
            self.scale,
            trans,
            self.k.value,
            notrans,
            self.q.value,
            0.0,
            self.attn.value,
            1,
            self.batch_ndim,
            redux=self.redux,
        )
        clear_async(self.attn_maxsumexp)
        self.q.value.wont_use()
        self.k.value.wont_use()

        if self.mask is not None:
            mask_scalar_async(
                self.mask, self.val, self.attn.value, self.batch_ndim
            )
            self.mask.wont_use()
        maxsumexp_async(
            self.attn.value,
            self.attn_maxsumexp,
            0,
            redux=self.redux
        )
        softmax_inplace_async(self.attn_maxsumexp, 1.0, self.attn.value, 0)
        # self.attn_maxsumexp.invalidate_submit()

        gemm_async(
            1.0,
            notrans,
            self.v.value,
            notrans,
            self.attn.value,
            0.0,
            self.y.value,
            1,
            self.batch_ndim,
            redux=self.redux,
        )
        self.v.value.wont_use()
        self.attn.value.wont_use()
        self.y.value.wont_use()

    def _forward_flash(self):
        if self.flash_logsumexp is None:
            raise RuntimeError("flash_logsumexp buffer is missing")

        # clear_async(self.flash_logsumexp)
        # clear_async(self.y.value)
        mask = self.mask if self.mask is not None else None
        flash_sdpa_fwd_cudnn_async(
            self.k.value,
            self.q.value,
            mask,
            self.flash_logsumexp,
            self.v.value,
            self.y.value,
        )
        # self.q.value.wont_use()
        # self.k.value.wont_use()
        # self.v.value.wont_use()
        # self.y.value.wont_use()

    def _backward_vanilla(self):
        if (
            self.attn is None
            or self.attn_sumprod_slice is None
            or self.attn_maxsumexp is None
        ):
            raise RuntimeError("Vanilla SDPA buffers are not initialized")
        if self.y.grad is None:
            raise RuntimeError("Output gradient tensor is missing")

        if self.attn.grad_required:
            if self.attn.grad is None:
                raise RuntimeError("Attention gradient tensor is missing")
            gemm_async(
                1.0,
                trans,
                self.v.value,
                notrans,
                self.y.grad,
                0.0,
                self.attn.grad,
                1,
                self.batch_ndim,
                redux=self.redux,
            )

        if self.v.grad_required:
            if self.v.grad is None:
                raise RuntimeError("Value gradient tensor is missing")
            gemm_async(
                1.0,
                notrans,
                self.y.grad,
                trans,
                self.attn.value,
                0.0,
                self.v.grad,
                1,
                self.batch_ndim,
                redux=self.redux,
            )

        if self.attn.grad_required:
            if self.attn.grad is None:
                raise RuntimeError("Attention gradient tensor is missing")
            sumprod_slice_async(
                1.0,
                self.attn.value,
                self.attn.grad,
                0.0,
                self.attn_sumprod_slice,
                0,
                redux=self.redux,
            )
            add_slice_inplace_async(
                -1.0, self.attn_sumprod_slice, 1.0, self.attn.grad, 0
            )
            self.attn_sumprod_slice.invalidate_submit()
            multiply_inplace_async(1.0, self.attn.value, self.attn.grad)
        self.attn.value.invalidate_submit()

        if self.mask is not None and self.attn.grad_required:
            mask_scalar_async(
                self.mask, 0, self.attn.grad, self.batch_ndim
            )
            self.mask.wont_use()

        if self.k.grad_required:
            if self.k.grad is None:
                raise RuntimeError("Key gradient tensor is missing")
            if self.attn.grad is None:
                raise RuntimeError("Attention gradient tensor is missing")
            gemm_async(
                self.scale,
                notrans,
                self.q.value,
                trans,
                self.attn.grad,
                0.0,
                self.k.grad,
                1,
                self.batch_ndim,
                redux=self.redux,
            )

        if self.q.grad_required:
            if self.q.grad is None:
                raise RuntimeError("Query gradient tensor is missing")
            if self.attn.grad is None:
                raise RuntimeError("Attention gradient tensor is missing")
            gemm_async(
                self.scale,
                notrans,
                self.k.value,
                notrans,
                self.attn.grad,
                0.0,
                self.q.grad,
                1,
                self.batch_ndim,
                redux=self.redux,
            )
        if self.attn.grad_required and self.attn.grad is not None:
            self.attn.grad.invalidate_submit()
