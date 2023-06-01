# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/act.py
# Activation layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-05-31

from .base_layer import BaseLayer
from nntile.tensor import add_async, copy_async, add_slice_async, sum_slice_async
from nntile.tensor import TensorTraits, Tensor_fp32, TensorMoments

class AddSlice(BaseLayer):

    def __init__(self, x: TensorMoments, y: TensorMoments, u: TensorMoments):
        super().__init__([x, y], [u], [], [])
        # Set up local named parameters
        self.x = x
        self.y = y
        self.u = u

    # Forward propagation of the add_slice layer
    def forward_async(self):
        # Init Y as a copy of X
        copy_async(self.x.value, self.u.value)
        # Add slice operation
        add_slice_async(1, self.y.value, 1, self.u.value, 0)

    def backward_async(self):
        add_async(1, self.u.grad, 1, self.x.grad)
        sum_slice_async(1, self.u.grad, 1, self.y.grad, 0)
        
    # Simple generator for the add_slice layer
    @staticmethod
    def generate_simple(x: TensorMoments, y: TensorMoments, next_tag: int):
        # Get traits of X
        u_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        # Create Y with the same traits and distribution as X
        u_value = Tensor_fp32(u_traits, x.value.distribution, next_tag)
        next_tag = u_value.next_tag
        u_grad = Tensor_fp32(u_traits, x.value.distribution, next_tag)
        next_tag = u_grad.next_tag
        u = TensorMoments(u_value, u_grad, True)
        # Create activation layer with all the provided tensors
        layer = AddSlice(x, y, u)
        # Return layer and next tag to be used
        return (layer, next_tag)

