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

dtype2nntile = {
    'fp16': nntile.tensor.Tensor_fp16,
    'bf16': nntile.tensor.Tensor_bf16,
    'fp32': nntile.tensor.Tensor_fp32,
}

dtype2np = {
    'fp16': np.float16,
    'bf16': np.float16,
    'fp32': np.float32,
}

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

    def __init__(self, x: TensorMoments, hidden_dim: int):
        activations = [x]
        layers = []
        # Initial linear layer that converts input to internal shape
        new_layer = Linear.generate_simple(x, "L", notrans,
                1, [hidden_dim], [hidden_dim], bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU activation
        new_layer = Act.generate_simple(activations[-1], "relu")
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Linear layer
        new_layer = Linear.generate_simple(
                    activations[-1], "L", notrans, 1, [hidden_dim],
                    [hidden_dim], bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Add operation
        new_layer = Add.generate_simple(
            activations[1], activations[-1])
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # ReLU activation
        new_layer = Act.generate_simple(activations[-1], "relu")
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer = Linear.generate_simple(activations[-1], "L",
            notrans, 1, [x.value.shape[1]], [x.value.shape[1]], bias=False)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)
        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(torch_model, input_moment, hidden_dim):
        """`torch_mlp` is PyTorch MLP where all intermediate dimensions are the
        same and no biases in linear layers
        """
        print("Call from torch static method")
        nntile_model = ToyFC_SkipConnection(input_moment, hidden_dim)
        pairs = zip(nntile_model.parameters, torch_model.parameters())
        for p, p_torch in pairs:
            p.value.from_array(p_torch.detach().numpy().T)
        return nntile_model


@pytest.mark.skip(reason='Frob loss is not working now')
def test_add(context, n=100, hidden_dim=50, num_samples=1000):
    torch_input = torch.randn(num_samples, n)
    torch_model = ToyFC_SkipConnectionTorch(n, hidden_dim)
    torch_output = torch_model(torch_input)
    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()
    print("Torch loss = {}".format(torch_loss.item()))

    x_traits = TensorTraits([num_samples, n], [num_samples, n])
    x_distr = [0] * x_traits.grid.nelems
    nntile_input = Tensor_fp32(x_traits, x_distr)
    nntile_input.from_array(torch_input.numpy())
    nntile_input_moment = TensorMoments(nntile_input, None, False)
    nntile_model = ToyFC_SkipConnection \
        .from_torch(torch_model, nntile_input_moment, hidden_dim)
    nntile_model.clear_gradients()
    nntile_model.forward_async()
    fro_loss = nntile.loss.Frob.generate_simple(nntile_model.activations[-1])
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


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_bench_add_forward_async(context_cuda, benchmark_operation, dtype: str):
    shape = [128, 128]
    traits = TensorTraits(shape, shape)
    distr = [0]

    # Create inputs
    tensor_type = dtype2nntile[dtype]
    X1 = tensor_type(traits, distr)
    X2 = tensor_type(traits, distr)
    G1 = tensor_type(traits, distr)
    G2 = tensor_type(traits, distr)

    rng = np.random.default_rng(42)
    np_dtype = dtype2np[dtype]
    x1_np = np.array(rng.standard_normal(shape), dtype=np_dtype, order='F')
    x2_np = np.array(rng.standard_normal(shape), dtype=np_dtype, order='F')
    X1.from_array(x1_np)
    X2.from_array(x2_np)

    x1_tm = TensorMoments(X1, G1, True)
    x2_tm = TensorMoments(X2, G2, True)

    layer = Add.generate_simple(x1_tm, x2_tm)

    out_np = np.zeros(shape, dtype=np_dtype, order='F')

    def bench_fn():
        layer.forward_async()
        layer.res.value.to_array(out_np)

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)

@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_bench_add_backward_async(context_cuda, benchmark_operation, dtype: str):
    shape = [128, 128]
    traits = TensorTraits(shape, shape)
    distr = [0]

    tensor_type = dtype2nntile[dtype]
    X1 = tensor_type(traits, distr)
    X2 = tensor_type(traits, distr)
    G1 = tensor_type(traits, distr)
    G2 = tensor_type(traits, distr)

    rng = np.random.default_rng(42)
    np_dtype = dtype2np[dtype]
    x1_np = np.array(rng.standard_normal(shape), dtype=np_dtype, order='F')
    x2_np = np.array(rng.standard_normal(shape), dtype=np_dtype, order='F')
    X1.from_array(x1_np)
    X2.from_array(x2_np)

    x1_tm = TensorMoments(X1, G1, True)
    x2_tm = TensorMoments(X2, G2, True)

    layer = Add.generate_simple(x1_tm, x2_tm)

    # Ensure grads are zeroed
    nntile.tensor.clear_async(G1)
    nntile.tensor.clear_async(G2)
    layer.clear_gradients()

    # forward once and prepare grad
    layer.forward_async()
    grad_np = np.array(rng.standard_normal(shape), dtype=np_dtype, order='F')
    layer.res.grad.from_array(grad_np)

    def bench_fn():
        layer.backward_async()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
