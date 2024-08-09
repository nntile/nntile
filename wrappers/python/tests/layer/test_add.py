# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_add.py
# Test for nntile.layer.add
#
# @version 1.1.0

import time

import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
from nntile.layer.act import Act
from nntile.layer.add import Add
from nntile.layer.linear import Linear
from nntile.model.base_model import BaseModel
from nntile.tensor import Tensor_fp32, TensorMoments, TensorTraits, notrans


class ToyFC_SkipConnectionTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin3 = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        y = self.act(x)
        y = self.lin2(y)
        u = y + x
        u = self.act(u)
        u = self.lin3(u)
        return u


class ToyFC_SkipConnection(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, hidden_dim: int, next_tag: int):
        activations = [x]
        layers = []
        # Initial linear layer that converts input to internal shape
        new_layer, next_tag = Linear.generate_simple(x, "L", notrans,
                1, [hidden_dim], [hidden_dim], next_tag, bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU activation
        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                                                  next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Linear layer
        new_layer, next_tag = Linear.generate_simple(
                    activations[-1], "L", notrans, 1, [hidden_dim],
                    [hidden_dim], next_tag, bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Add operation
        new_layer, next_tag = Add.generate_simple(
            activations[1], activations[-1], next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU activation
        new_layer, next_tag = \
            Act.generate_simple(activations[-1], "relu", next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple(activations[-1], "L",
            notrans, 1, [x.value.shape[1]], [x.value.shape[1]], next_tag,
            bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        self.next_tag = next_tag
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_model, input_moment, hidden_dim, next_tag: int):
        """`torch_mlp` is PyTorch MLP where all intermediate dimensions are the
        same and no biases in linear layers
        """
        print("Call from torch static method")
        nntile_model = ToyFC_SkipConnection(input_moment, hidden_dim, next_tag)
        pairs = zip(nntile_model.parameters, torch_model.parameters())
        for p, p_torch in pairs:
            p.value.from_array(p_torch.detach().numpy().T)
        return nntile_model, nntile_model.next_tag


@pytest.mark.xfail(reason='not implemented')
def test_add(n=100, hidden_dim=50, num_samples=1000):
    torch_input = torch.randn(num_samples, n)
    torch_model = ToyFC_SkipConnectionTorch(n, hidden_dim)
    torch_output = torch_model(torch_input)
    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()
    print("Torch loss = {}".format(torch_loss.item()))

    time0 = -time.time()
    # Set up StarPU+MPI and init codelets
    _config = nntile.starpu.Config(1, -1, 1)
    nntile.starpu.init()
    time0 += time.time()
    print("StarPU + NNTile + MPI init in {} seconds".format(time0))
    next_tag = 0

    x_traits = TensorTraits([num_samples, n], [num_samples, n])
    x_distr = [0] * x_traits.grid.nelems
    nntile_input = Tensor_fp32(x_traits, x_distr, next_tag)
    nntile_input.from_array(torch_input.numpy())
    next_tag = nntile_input.next_tag
    nntile_input_moment = TensorMoments(nntile_input, None, False)
    nntile_model, next_tag = ToyFC_SkipConnection \
        .from_torch(torch_model, nntile_input_moment, hidden_dim, next_tag)
    nntile_model.clear_gradients()
    nntile_model.forward_async()
    fro_loss, next_tag = nntile.loss.Frob \
        .generate_simple(nntile_model.activations[-1], next_tag)
    np_zero = np.zeros(nntile_model.activations[-1].value.shape,
                       dtype=np.float32, order='F')
    fro_loss.y.from_array(np_zero)
    fro_loss.calc_async()
    nntile_model.backward_async()

    np_loss_nntile = np.zeros((1,), dtype=np.float32, order="F")
    fro_loss.val.to_array(np_loss_nntile)
    print("NNTile loss = {}".format(np_loss_nntile[0]))
    print("Relative error in the loss for torch and nntile models = {}".format(
        abs(torch_loss.item() - np_loss_nntile[0]) / torch_loss.item()))

    param_pairs = zip(nntile_model.parameters, torch_model.parameters())
    for i, (p_nntile, p_torch) in enumerate(param_pairs):
        p_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
        p_nntile.grad.to_array(p_np)
        p_torch_np = p_torch.grad.cpu().detach().numpy()
        abs_error = np.linalg.norm(p_np - p_torch_np.T, "fro")
        rel_error = abs_error / np.linalg.norm(p_torch_np, "fro")
        print("Relative error in layer {} gradient = {}".format(i, rel_error))

    nntile_model.unregister()
    fro_loss.unregister()
