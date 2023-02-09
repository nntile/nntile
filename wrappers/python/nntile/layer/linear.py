# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/linear.py
# Linear layer of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-09

import nntile.nntile_core.tensor as tensor
import numpy as np
from typing import List, Str

class Linear_fp32:
    x: tensor.Tensor_fp32
    dx: tensor.Tensor_fp32
    y: tensor.Tensor_fp32
    dy: tensor.Tensor_fp32
    w: tensor.Tensor_fp32
    dw: tensor.Tensor_fp32
    b: None | tensor.Tensor_fp32
    db: None | tensor.Tensor_fp32
    side: Str
    trans_x: Str
    trans_w: Str
    params: List[tensor.Tensor_fp32]
    grads: List[tensor.Tensor_fp32]

    # Construct linear layer with all the provided data
    def __init__(self, x, dx, y, dy, w, dw, b, db, side, trans_x, trans_w):
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self.w = w
        self.dw = dw
        self.b = b
        self.db = db
        self.side = side
        self.trans_x = trans_x
        self.trans_w = trans_w
        self.params = [w, b]
        self.grads = [dw, db]

#    # Simple generator for the normalization layer
#    @staticmethod
#    def generate_block_cyclic(x, dx, funcname, next_tag):
#        # Check if X and dX correspond to each other
#        if x.shape != dx.shape or x.basetile_shape != dx.basetile_shape:
#            raise ValueError
#        # Get traits of X
#        x_traits = tensor.TensorTraits(x.shape, x.basetile_shape)
#        # Create Y with the same traits and distribution as X
#        y = tensor.Tensor_fp32(x_traits, x.distribution, next_tag)
#        next_tag = y.next_tag
#        # Create dY with the same traits and distribution as X
#        dy = tensor.Tensor_fp32(x_traits, x.distribution, next_tag)
#        next_tag = dy.next_tag
#        # Create activation layer with all the provided tensors
#        layer = Act_fp32(x, dx, y, dy, funcname)
#        # Return layer and next tag to be used
#        return (layer, next_tag)
#
#    # Forward propagation of the activation layer
#    def forward_async(self):
#        # Init Y as a copy of X
#        tensor.copy_async_fp32(self.x, self.y)
#        # Non-linear activation of Y inplace
#        self.func(self.y)
#        # Copy X into dX to utilize it during backward propagation
#        tensor.copy_async_fp32(self.x, self.dx)
#        # Destroy values stored in tensor X
#        self.x.invalidate_submit()
#        # Hint for StarPU that dX tensor will
#        # not be used soon and it is advised to offload data from GPU 
#        self.dx.wont_use()
#
#    # Backward propagation of the activation layer
#    def backward_async(self):
#        # Get derivative of activation functions at X inplace of dX
#        self.dfunc(self.dx)
#        # Per-element product of dY and f'(x)
#        tensor.prod_async_fp32(self.dy, self.dx)
#        # Destroy values stored in tensor dY
#        self.dy.invalidate_submit()
#
