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
# @date 2023-03-17

from nntile.tensor import softmax_async, clear_async, copy_async, nrm2_async, prod_async, \
                          logsumexp_async, maxsumexp_async, total_sum_accum_async
from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, Tensor_int64
import numpy as np

class CrossEntropy:
    final_layer_output: TensorMoments
    class_labels: Tensor_int64
    val: Tensor
    tmp: Tensor
    maxsumexp: Tensor

    # Constructor of loss with all the provided data
    def __init__(self, final_layer_output: TensorMoments, class_labels: Tensor_int64, val: Tensor,
                 maxsumexp: Tensor, logsumexp: Tensor):
        self.final_layer_output = final_layer_output
        self.class_labels = class_labels
        self.val = val
        self.logsumexp = logsumexp
        self.maxsumexp = maxsumexp

    # Simple geenrator
    @staticmethod
    def generate_simple(final_layer_output: TensorMoments,
                        class_labels: Tensor_int64, next_tag: int) -> tuple:
        
        # ndim = len(x.value.grid.shape)
        maxsumexp_traits = TensorTraits((2, class_labels.shape[0]),
                                        (2, class_labels.basetile_shape[0]))
        maxsumexp = type(final_layer_output.value)(maxsumexp_traits,
                                                   final_layer_output.value.distribution,
                                                   next_tag)
        next_tag = maxsumexp.next_tag
        val_traits = TensorTraits([], [])
        val = type(final_layer_output.value)(val_traits, [0], next_tag)
        next_tag = val.next_tag

        logsumexp_traits = TensorTraits((class_labels.shape[0], ),
                                        (class_labels.basetile_shape[0], ))
        logsumexp = type(final_layer_output.value)(logsumexp_traits,
                                                   final_layer_output.value.distribution,
                                                   next_tag)
        next_tag = logsumexp.next_tag
        loss = CrossEntropy(final_layer_output, class_labels, val, maxsumexp, logsumexp)
        return loss, next_tag
    
    def unregister(self):
        self.logsumexp.unregister()
        self.maxsumexp.unregister()
        self.val.unregister()

    def get_val(self, val_np):
        self.val.to_array(val_np)

    def get_grad(self, grad_np):
        self.final_layer_output.grad.to_array(grad_np)

    # Get value and gradient if needed
    def calc_async(self):

        maxsumexp_async(self.final_layer_output.value, self.maxsumexp, 1)
        copy_async(self.final_layer_output.value, self.final_layer_output.grad)
        softmax_async(self.maxsumexp, self.final_layer_output.grad, 1)

        logsumexp_async(self.maxsumexp, self.logsumexp)

        clear_async(self.val)

        total_sum_accum_async(self.logsumexp, self.final_layer_output.value,
                              self.class_labels, self.val)

        # Put X into gradient grad X
        # copy_async(self.x.value, self.x.grad)
        # Define gradient dX as X-Y
        # axpy_async(-1, self.y, self.x.grad)
        # Values Y are not needed anymore
        #self.y.invalidate_submit()
        # Get value ||grad X||
        # nrm2_async(self.x.grad, self.val, self.tmp)
        # Ignore temporary values
        #self.tmp.invalidate_submit()
        # Invalidate gradient if it is unnecessary
        #if self.x.grad_required is False:
        #    self.x.grad.invalidate_submit()
        # Compute loss as 0.5*||dX||^2
        # prod_async(self.val, self.val)
        # axpy_async(-0.5, self.val, self.val)
