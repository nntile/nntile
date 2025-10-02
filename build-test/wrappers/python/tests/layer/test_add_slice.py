# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_add_slice.py
# Test for nntile.layer.add_slice
#
# @version 1.1.0

import numpy as np
import pytest
import torch
import torch.nn as nn

from nntile.layer.act import Act
from nntile.layer.add_slice import AddSlice
from nntile.layer.linear import Linear
from nntile.loss.frob import Frob
from nntile.model.base_model import BaseModel
from nntile.tensor import Tensor_fp32, TensorMoments, TensorTraits, notrans


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(15, 20, bias=False)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(10, 20, bias=False)
        self.lin3 = nn.Linear(20, 10, bias=False)

    def forward(self, x, y):
        x = self.lin1(x)
        x = self.relu(x)
        y = self.lin2(y)
        x = x + y.unsqueeze(0)
        x = self.lin3(x)
        return x


class NNTileToyModel(BaseModel):

    def __init__(self, x: TensorMoments, y: TensorMoments, axis: int) -> None:
        activations = [x, y]
        layers = []
        new_layer = Linear.generate_simple(
            x, "L", notrans, 1, [20], [20], bias=False)

        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Act.generate_simple(activations[-1], "relu")
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Linear.generate_simple(
            activations[1], "L", notrans, 1, [20], [20], bias=False)

        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = AddSlice.generate_simple(
            activations[3], activations[-1], axis)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Linear.generate_simple(
            activations[-1], "L", notrans, 1, [10], [10], bias=False)

        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_model, batch_size, input_dim, axis):

        x_traits = TensorTraits([batch_size, input_dim],
        [batch_size, input_dim])
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_fp32(x_traits, x_distr)
        x_grad = None
        x_grad_required = False
        x = TensorMoments(x, x_grad, x_grad_required)

        y_traits = TensorTraits([10], [10])
        y_distr = [0] * y_traits.grid.nelems
        y = Tensor_fp32(y_traits, y_distr)
        y_grad = None
        y_grad_required = False
        y = TensorMoments(y, y_grad, y_grad_required)

        nntile_model = NNTileToyModel(x, y, axis)

        param_pairs = zip(nntile_model.parameters, torch_model.parameters())
        for p_nntile, p_torch in param_pairs:
            p_nntile.value.from_array(p_torch.detach().numpy().T)

        return nntile_model


@pytest.mark.skip(reason='Frob loss is not working now')
def test_add_slice(context, batch_size=10, input_dim=15):
    x_input_torch = torch.randn(batch_size, input_dim)
    y_input_torch = torch.randn(10)
    torch_model = ToyModel()
    torch_output = torch_model(x_input_torch, y_input_torch)
    torch_loss = torch.sum(torch.square(torch_output)) * 0.5
    torch_loss.backward()
    torch_loss_val = torch_loss.item()
    print("Torch loss = {}".format(torch_loss_val))

    axis = 0
    nntile_model = NNTileToyModel.from_torch(
        torch_model, batch_size, input_dim, axis
    )

    nntile_model.activations[0].value.from_array(x_input_torch.numpy())
    nntile_model.activations[1].value.from_array(y_input_torch.numpy())

    nntile_model.forward_async()
    nntile_model.clear_gradients()

    nntile_loss_func = Frob.generate_simple(nntile_model.activations[-1])
    nntile_loss_func.y.from_array(np.zeros((batch_size, 10), np.float32, 'F'))

    nntile_loss_func.calc_async()

    nntile_model.backward_async()

    val_np = np.zeros((1,), order="F", dtype=np.float32)
    nntile_loss_func.val.to_array(val_np)
    print("NNTile loss = {}".format(val_np[0]))
    rerror = abs(val_np[0] - torch_loss_val) / torch_loss_val
    print(f'Relative diff between Pytorch and NNTile models losses = {rerror}')

    param_pairs = zip(nntile_model.parameters, torch_model.parameters())
    for i, (p_nntile, p_torch) in enumerate(param_pairs):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, np.float32, 'F')
        p_nntile.grad.to_array(p_nntile_grad_np)
        # print(p_nntile_grad_np)
        grad_diff = p_torch.grad - torch.from_numpy(p_nntile_grad_np).T
        rel_error = torch.norm(grad_diff) / torch.norm(p_torch.grad)
        print(f'Relative error in gradient in layer {i} = {rel_error}')

    nntile_model.unregister()
    nntile_loss_func.unregister()
