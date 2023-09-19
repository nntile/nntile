# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/loss/crossentropy.py
# Crossentropy loss of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @author Aleksandr Mikhalev
# @date 2023-09-19

from nntile.tensor import softmax_async, clear_async, copy_async, \
        subtract_indexed_outputs_async, logsumexp_async, maxsumexp_async, \
        total_sum_accum_async
from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        Tensor_int64
import numpy as np

class CrossEntropy:
    model_output: TensorMoments
    y: Tensor_int64 # labels
    val: Tensor
    tmp: Tensor
    maxsumexp: Tensor

    # Constructor of loss with all the provided data
    def __init__(self, model_output: TensorMoments, labels: Tensor_int64, \
            val: Tensor, maxsumexp: Tensor, logsumexp: Tensor):
        self.model_output = model_output
        self.val = val
        self.val.set_reduction_add()
        self.logsumexp = logsumexp
        self.maxsumexp = maxsumexp
        self.y = labels

    # Simple generator
    @staticmethod
    def generate_simple(model_output: TensorMoments, next_tag: int) -> tuple:
        shape = model_output.value.shape[1:]
        basetile = model_output.value.basetile_shape[1:]
        labels_traits = TensorTraits(shape, basetile)
        labels = Tensor_int64(labels_traits, model_output.value.distribution, \
                next_tag)
        next_tag = labels.next_tag 
        maxsumexp_traits = TensorTraits([2]+shape, [2]+basetile)
        maxsumexp = type(model_output.value)(maxsumexp_traits, \
                model_output.value.distribution, next_tag)
        next_tag = maxsumexp.next_tag
        val_traits = TensorTraits([], [])
        val = type(model_output.value)(val_traits, [0], next_tag)
        next_tag = val.next_tag
        logsumexp = type(model_output.value)(labels_traits, \
                model_output.value.distribution, next_tag)
        next_tag = logsumexp.next_tag
        loss = CrossEntropy(model_output, labels, val, maxsumexp, logsumexp)
        return loss, next_tag
    
    def unregister(self):
        self.logsumexp.unregister()
        self.maxsumexp.unregister()
        self.val.unregister()
        self.y.unregister()

    def get_val(self, val_np):
        self.val.to_array(val_np)

    def get_grad(self, grad_np):
        self.model_output.grad.to_array(grad_np)

    # Get value and gradient if needed
    def calc_async(self):
        maxsumexp_async(self.model_output.value, self.maxsumexp, 0)
        logsumexp_async(self.maxsumexp, self.logsumexp)
        clear_async(self.val)
        total_sum_accum_async(self.logsumexp, self.model_output.value, \
                self.y, self.val)
        if self.model_output.grad_required is True:
            softmax_async(self.maxsumexp, self.model_output.value, \
                    self.model_output.grad, 0)
            subtract_indexed_outputs_async(1., self.y, self.model_output.grad)
        self.model_output.value.wont_use()
        self.model_output.grad.wont_use()
        self.maxsumexp.wont_use()
        self.logsumexp.wont_use()
        self.val.wont_use()
        self.y.wont_use()

