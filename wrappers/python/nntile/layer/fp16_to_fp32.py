# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/fp16_to_fp32.py
# Conversion layer from fp16_t to fp32_t of NNTile Python package
#
# @version 1.1.0

from nntile.layer.base_layer import BaseLayer
from nntile.nntile_core.tensor import fp16_to_fp32_async, fp32_to_fp16_async
from nntile.tensor import Tensor_fp32, TensorMoments, TensorTraits


class FP16_to_FP32(BaseLayer):
    x: TensorMoments
    y: TensorMoments

    # Construct activation layer with all the provided data
    def __init__(self, x: TensorMoments, y: TensorMoments):
        # Redirect to BaseLayer initialization
        super().__init__([x], [y], [], [])
        # Set up local named parameters
        self.x = x
        self.y = y

    # Simple generator for the conversion layer
    @staticmethod
    def generate_simple(x: TensorMoments, next_tag: int):
        # Get traits of X
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        y_value = Tensor_fp32(x_traits, x.value.distribution, next_tag)
        next_tag = y_value.next_tag
        y_grad = Tensor_fp32(x_traits, x.value.distribution, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        # Create activation layer with all the provided tensors
        layer = FP16_to_FP32(x, y)
        # Return layer and next tag to be used
        return (layer, next_tag)

    # Forward propagation of the activation layer
    def forward_async(self):
        fp16_to_fp32_async(self.x.value, self.y.value)

    # Backward propagation of the activation layer
    def backward_async(self):
        fp32_to_fp16_async(self.y.grad, self.x.grad)
