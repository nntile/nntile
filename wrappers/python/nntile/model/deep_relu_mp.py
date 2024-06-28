# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/deep_relu_mp.py
# Deep ReLU model of NNTile Python package (NOT YET READY)
#
# @version 1.0.0

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        notrans, trans, Tensor_fp32
from nntile.model.base_model import BaseModel
from nntile.layer.linear import Linear
from nntile.layer.act import Act
from nntile.layer.fp32_to_fp16 import FP32_to_FP16
from nntile.layer.fp16_to_fp32 import FP16_to_FP32
import numpy as np
from typing import List

class DeepReLU_mp(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, side: str, ndim: int, \
            add_shape: int, add_basetile_shape: int, nlayers: int, \
            n_classes:int, next_tag: int):
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check number of layers
        if nlayers < 2:
            raise ValueError("nlayers must be at least 2")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        self.fp32_fast_tf32 = fp32_fast_tf32
        # Init activations and list of layers
        activations = [x]
        layers = []
        # Initial linear layer that converts input to internal shape
        # new_layer, next_tag = FP32_to_FP16.generate_simple(x, next_tag)
        # layers.append(new_layer)
        # activations.extend(new_layer.activations_output)
        new_layer, next_tag = Linear.generate_simple(activations[-1], side, \
                notrans, ndim, [add_shape], [add_basetile_shape], next_tag,
                fp32_fast_tf32=fp32_fast_tf32)
        print("Layer 0 shape", new_layer.w.value.shape, new_layer.y.value.shape)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # new_layer, next_tag = FP16_to_FP32.generate_simple(activations[-1], next_tag)
        # layers.append(new_layer)
        # activations.extend(new_layer.activations_output)
        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                                                  next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Internal linear layers with the same internal shape
        for i in range(1, nlayers-1):
            new_layer, next_tag = Linear.generate_simple( \
                    activations[-1], side, notrans, 1, [add_shape], \
                    [add_basetile_shape], next_tag, fp32_fast_tf32=fp32_fast_tf32)
            print("Layer {} shape".format(i), new_layer.w.value.shape, new_layer.y.value.shape)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)
            new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                                                      next_tag)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)
        # Finalizing linear layer that converts result back to proper shape
        # if side == 'L':
        #     new_shape = x.value.shape[-ndim:]
        #     new_base = x.value.basetile_shape[-ndim:]
        # else:
        #     new_shape = x.value.shape[:ndim]
        #     new_base = x.value.basetile_shape[:ndim]

        new_layer, next_tag = Linear.generate_simple(activations[-1], \
                side, notrans, 1, [n_classes], [n_classes], next_tag, fp32_fast_tf32=fp32_fast_tf32)
        print("Last layer shape", new_layer.w.value.shape, new_layer.y.value.shape)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    # Randomly init all linear layers
    def init_randn_async(self):
        for l in self.layers:
            if type(l) is Linear:
                l.init_randn_async()
    @staticmethod
    def from_torch(torch_mlp, batch_size: int, n_classes: int, nonlinearity: str, next_tag: int):
        '''
        torch_mlp is PyTorch MLP where all intermediate dimensions are the same and no biases in linear layers
        '''
        print("Call from torch static method")
        gemm_ndim = 1
        n_layers = len(list(torch_mlp.parameters()))
        for i, p in enumerate(torch_mlp.parameters()):
            if i == 0:
                hidden_layer_dim = p.shape[0]
                n_pixels = p.shape[1]
            elif hidden_layer_dim != p.shape[1]:
                print(p.shape, hidden_layer_dim)
                raise ValueError("PyTorch model has different hidden dims")
            if i == n_layers - 1 and p.shape[0] != n_classes:
                raise ValueError("Last layer of PyTorch model does not correspond to the target number of classes")
        hidden_layer_dim_tile = hidden_layer_dim

        x_traits = TensorTraits([batch_size, n_pixels], \
        [batch_size, n_pixels])
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_fp32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)
        if nonlinearity == "relu":
            mlp_nntile = DeepReLU(x_moments, 'L', gemm_ndim, hidden_layer_dim,
            hidden_layer_dim_tile, n_layers, n_classes, next_tag)
            for p, p_torch in zip(mlp_nntile.parameters, torch_mlp.parameters()):
                p.value.from_array(p_torch.detach().numpy().T)
            return mlp_nntile, mlp_nntile.next_tag
