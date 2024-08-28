# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/loss/crossentropy.py
# Crossentropy loss of NNTile Python package
#
# @version 1.1.0

import nntile.utils.constructors as nntc
from nntile.tensor import (
    Tensor, Tensor_fp32, Tensor_int64, TensorMoments, TensorTraits,
    clear_async, logsumexp_async, maxsumexp_async, softmax_async,
    subtract_indexed_outputs_async, total_sum_accum_async)


class CrossEntropy:
    model_output: TensorMoments
    y: Tensor_int64  # labels
    val: Tensor
    tmp: Tensor
    maxsumexp: Tensor

    # Constructor of loss with all the provided data
    def __init__(
        self,
        model_output: TensorMoments,
        labels: Tensor_int64,
        val: Tensor,
        maxsumexp: Tensor,
        logsumexp: Tensor,
        redux: bool = False,
        scale: float = 1.0,
    ):
        self.model_output = model_output
        self.val = val
        self.val.set_reduction_add()
        self.logsumexp = logsumexp
        self.maxsumexp = maxsumexp
        self.maxsumexp.set_reduction_maxsumexp()
        self.y = labels
        if redux:
            self.redux = 1
        else:
            self.redux = 0
        self.scale = scale

    # Simple generator
    @staticmethod
    def generate_simple(
        model_output: TensorMoments,
        next_tag: int,
        redux: bool = False,
        scale: float = 1.0,
    ) -> tuple:
        shape = model_output.value.shape[1:]
        basetile = model_output.value.basetile_shape[1:]
        labels_traits = TensorTraits(shape, basetile)
        labels = Tensor_int64(
            labels_traits, model_output.value.distribution, next_tag
        )
        next_tag = labels.next_tag
        maxsumexp_traits = TensorTraits([2] + shape, [2] + basetile)
        maxsumexp = type(model_output.value)(
            maxsumexp_traits, model_output.value.distribution, next_tag
        )
        next_tag = maxsumexp.next_tag
        val_traits = TensorTraits([], [])
        val = Tensor_fp32(val_traits, [0], next_tag)
        next_tag = val.next_tag
        logsumexp = type(model_output.value)(
            labels_traits, model_output.value.distribution, next_tag
        )
        next_tag = logsumexp.next_tag
        loss = CrossEntropy(
            model_output,
            labels,
            val,
            maxsumexp,
            logsumexp,
            redux=redux,
            scale=scale,
        )
        return loss, next_tag

    def unregister(self):
        self.logsumexp.unregister()
        self.maxsumexp.unregister()
        self.val.unregister()
        self.y.unregister()

    def get_val(self):
        return nntc.to_numpy(self.val)

    def get_grad(self):
        return nntc.to_numpy(self.model_output.grad)

    # Get value and gradient if needed
    def calc_async(self):
        clear_async(self.maxsumexp)
        maxsumexp_async(
            self.model_output.value, self.maxsumexp, 0, redux=self.redux
        )
        logsumexp_async(self.maxsumexp, self.logsumexp)
        total_sum_accum_async(
            self.scale,
            self.logsumexp,
            self.model_output.value,
            self.y,
            self.val,
        )
        if self.model_output.grad_required is True:
            softmax_async(
                self.maxsumexp,
                self.model_output.value,
                self.scale,
                self.model_output.grad,
                0,
            )
            subtract_indexed_outputs_async(
                self.scale, self.y, self.model_output.grad
            )
        self.model_output.value.wont_use()
        self.model_output.grad.wont_use()
        self.maxsumexp.wont_use()
        self.logsumexp.wont_use()
        self.val.wont_use()
        self.y.wont_use()
