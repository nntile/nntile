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

from nntile.tensor import softmax_async, clear_async, copy_async, subtract_indexed_column_async, \
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
        self.val = val
        self.logsumexp = logsumexp
        self.maxsumexp = maxsumexp
        self.y = class_labels
        

    # Simple generator
    @staticmethod
    def generate_simple(final_layer_output: TensorMoments,
                        next_tag: int) -> tuple:
        print(final_layer_output.value.shape)
        class_labels_traits = TensorTraits((final_layer_output.value.shape[0], ),
                                           (final_layer_output.value.basetile_shape[0], ))
        # print(final_layer_output.value.basetile_shape, class_labels_traits.basetile_shape)
        class_labels = Tensor_int64(class_labels_traits,
                                    final_layer_output.value.distribution,
                                    next_tag)
        next_tag = class_labels.next_tag
        
        maxsumexp_traits = TensorTraits((2, final_layer_output.value.shape[0]),
                                        (2, final_layer_output.value.basetile_shape[0]))
        maxsumexp = type(final_layer_output.value)(maxsumexp_traits,
                                                   final_layer_output.value.distribution,
                                                   next_tag)
        next_tag = maxsumexp.next_tag

        val_traits = TensorTraits([], [])
        val = type(final_layer_output.value)(val_traits, [0], next_tag)
        next_tag = val.next_tag

        logsumexp_traits = TensorTraits((final_layer_output.value.shape[0], ),
                                        (final_layer_output.value.basetile_shape[0], ))
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
        self.y.unregister()

    def get_val(self, val_np):
        self.val.to_array(val_np)

    def get_grad(self, grad_np):
        self.final_layer_output.grad.to_array(grad_np)

    # Get value and gradient if needed
    def calc_async(self):

        maxsumexp_async(self.final_layer_output.value, self.maxsumexp, 1)
        logsumexp_async(self.maxsumexp, self.logsumexp)
        clear_async(self.val)
        total_sum_accum_async(self.logsumexp, self.final_layer_output.value,
                              self.y, self.val)
        if self.final_layer_output.grad_required is True:
            copy_async(self.final_layer_output.value, self.final_layer_output.grad)
            softmax_async(self.maxsumexp, self.final_layer_output.grad, 1)
            subtract_indexed_column_async(1., self.y, self.final_layer_output.grad)
