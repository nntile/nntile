# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/deep_relu.py
# Deep ReLU model of NNTile Python package
#
# @version 1.1.0

from nntile.layer.act import Act
from nntile.layer.linear import Linear
from nntile.model.base_model import BaseModel
from nntile.tensor import (
    Tensor_fp32, Tensor_fp32_fast_tf32, TensorMoments, TensorTraits, notrans)


class DeepReLU(BaseModel):
    next_tag: int

    # Construct model with all the provided data
    def __init__(self, x: TensorMoments, side: str, ndim: int,
            add_shape: int, add_basetile_shape: int, nlayers: int,
            n_classes: int, next_tag: int, bias: bool = False):
        # Check parameter side
        if side != 'L' and side != 'R':
            raise ValueError("side must be either 'L' or 'R'")
        # Check number of layers
        if nlayers < 2:
            raise ValueError("nlayers must be at least 2")
        # Check parameter ndim
        if ndim <= 0:
            raise ValueError("ndim must be positive integer")
        # Init activations and list of layers
        activations = [x]
        layers = []
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple(x, side, notrans, ndim,
                [add_shape], [add_basetile_shape], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Internal linear layers with the same internal shape
        for i in range(1, nlayers - 1):
            new_layer, next_tag = Linear.generate_simple(
                    activations[-1], side, notrans, 1, [add_shape],
                    [add_basetile_shape], next_tag, bias)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)
            new_layer, next_tag = Act.generate_simple(activations[-1],
                    "relu", next_tag)
            layers.append(new_layer)
            activations.extend(new_layer.activations_output)
        # Finalizing linear layer that converts result back to proper shape
        new_layer, next_tag = Linear.generate_simple(activations[-1],
                side, notrans, 1, [n_classes], [n_classes], next_tag, bias)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_mlp, batch_size: int, n_classes: int,
            nonlinearity: str, dtype: str, next_tag: int):
        """
        torch_mlp is PyTorch MLP where all intermediate dimensions are the
        same and no biases in linear layers
        """
        torch_parameters = list(torch_mlp.parameters())

        if len(torch_parameters[1].shape) == 2:
            bias = False
            n_layers = len(torch_parameters)
        elif len(torch_parameters[1].shape) == 1:
            bias = True
            n_layers = len(torch_parameters) // 2

        input_dim = torch_parameters[0].shape[1]
        hidden_layer_dim = torch_parameters[0].shape[0]
        x_traits = TensorTraits([input_dim, batch_size],
                [input_dim, batch_size])
        x_distr = [0] * x_traits.grid.nelems
        if dtype == "fp32":
            x = Tensor_fp32(x_traits, x_distr, next_tag)
        elif dtype == "tf32":
            x = Tensor_fp32_fast_tf32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x_moments = TensorMoments(x, x_grad, x_grad_required)
        hidden_layer_dim_tile = hidden_layer_dim
        gemm_ndim = 1
        if nonlinearity == "relu":
            mlp_nntile = DeepReLU(x_moments, 'R', gemm_ndim,
                    hidden_layer_dim, hidden_layer_dim_tile, n_layers,
                    n_classes, next_tag, bias=bias)
            for p, p_torch in zip(mlp_nntile.parameters,
                    torch_parameters):
                p.value.from_array(p_torch.detach().cpu().numpy())
            return mlp_nntile, mlp_nntile.next_tag
