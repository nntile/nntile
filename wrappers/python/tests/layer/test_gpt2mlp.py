# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gpt2mlp.py
# Test for nntile.layer.gpt2mlp
# Here we use transformers-4.28.1 from https://github.com/huggingface/transformers
#
# @version 1.1.0

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import nntile
import nntile.utils.constructors as nntc
from gpt2_config import GPT2Config
from huggingface_activations import ACT2FN
from nntile.model.gpt2 import GPT2MLP as GPT2MLP_nntile

dtype2np = {
    'fp16': np.float32,
    'bf16': np.float16,
    'fp32': np.float32,
}

class Conv1D(nn.Module):
    """1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and
    also used in GPT-2). Basically works like a linear layer but the weights
    are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: tuple[Tensor, ...] | None) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def test_gpt2mlp(
    context,
    batch_size=100,
    batch_size_tile=10,
    interm_size=1000,
    interm_size_tile=100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_config = GPT2Config(activation_function="relu", resid_pdrop=0.0)

    input_dim = gpt2_config.n_embd
    input_dim_tile = input_dim // 4
    gpt2mlp_hug = GPT2MLP(interm_size, gpt2_config).to(device)

    rng = np.random.default_rng(42)
    test_input_np = rng.standard_normal((input_dim, batch_size))
    test_input_np = test_input_np.astype(np.float32, "F")
    test_input = torch.from_numpy(test_input_np.T).to(device)
    hug_result = gpt2mlp_hug(test_input)
    print(
        "Norm of the output of PyTorch GPT2MLP", torch.norm(hug_result).item()
    )

    torch_val = torch.square(torch.norm(hug_result)) * 0.5
    torch_val.backward()

    nntile_config = {
        "embed_dim": input_dim,
        "embed_dim_tile": input_dim_tile,
        "inner_dim": interm_size,
        "inner_dim_tile": interm_size_tile,
        "interm_size": interm_size,
        "interm_size_tile": interm_size_tile,
        "activation_function": gpt2_config.activation_function,
        "redux": False,
    }

    x_traits = nntile.tensor.TensorTraits(
        [input_dim, batch_size], [input_dim_tile, batch_size_tile]
    )
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(test_input_np)
    x_grad = None
    x_grad_required = False
    x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

    print("Create model...")
    gpt2mlp_nntile = GPT2MLP_nntile.from_torch(
        gpt2mlp_hug, x_moments, nntile_config
    )
    print("Create model...done")
    print("Forward model...")
    gpt2mlp_nntile.forward_async()
    print("Forward model...done")
    gpt2mlp_nntile.clear_gradients()
    nntile.tensor.copy_async(
        gpt2mlp_nntile.activations[-1].value,
        gpt2mlp_nntile.activations[-1].grad,
    )
    gpt2mlp_nntile.backward_async()

    output = np.zeros(
        gpt2mlp_nntile.activations[-1].value.shape, order="F", dtype=np.float32
    )
    gpt2mlp_nntile.activations[-1].value.to_array(output)
    print("Norm of the output of NNtile GPT2MLP", np.linalg.norm(output))
    print(
        "Norm of difference =",
        np.linalg.norm(output.T - hug_result.cpu().detach().numpy()),
    )

    for i, (p_nntile, p_torch) in enumerate(
        zip(gpt2mlp_nntile.parameters, gpt2mlp_hug.parameters())
    ):
        p_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
        p_nntile.grad.to_array(p_np)
        p_torch_np = p_torch.grad.cpu().detach().numpy().T
        rel_error = np.linalg.norm(p_np - p_torch_np) / np.linalg.norm(
            p_torch_np
        )
        print("Relative error in layer {} gradient = {}".format(i, rel_error))

    gpt2mlp_nntile.unregister()
    x.unregister()


def test_gpt2mlp_dynamic(
    context, batch_size=100, interm_size=1000, interm_size_tile=1000
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_config = GPT2Config(activation_function="relu", resid_pdrop=0.0)

    input_dim = gpt2_config.n_embd
    input_dim_tile = input_dim
    gpt2mlp_hug = GPT2MLP(interm_size, gpt2_config).to(device)

    rng = np.random.default_rng(42)
    test_input_np = rng.standard_normal((input_dim, batch_size)).astype(
        np.float32, "F"
    )
    test_input_half_np = rng.standard_normal(
        (input_dim, batch_size // 2)
    ).astype(np.float32, "F")

    hug_result = gpt2mlp_hug(torch.from_numpy(test_input_np.T).to(device))
    hug_result_half = gpt2mlp_hug(
        torch.from_numpy(test_input_half_np.T).to(device)
    )

    nntile_config = {
        "embed_dim": input_dim,
        "embed_dim_tile": input_dim_tile,
        "inner_dim": interm_size,
        "inner_dim_tile": interm_size_tile,
        "interm_size": interm_size,
        "interm_size_tile": interm_size_tile,
        "activation_function": gpt2_config.activation_function,
        "redux": False,
    }

    x = nntc.from_array(test_input_np)
    x_moments = nntile.tensor.TensorMoments(x, None, False)

    gpt2mlp_nntile = GPT2MLP_nntile.from_torch(
        gpt2mlp_hug, x_moments, nntile_config
    )

    # test with same tensor
    output_nnt = gpt2mlp_nntile.forward_dynamic(x_moments)
    output = nntc.to_numpy(output_nnt.value)
    assert np.linalg.norm(output.T - hug_result.cpu().detach().numpy()) < 1e-04

    # test with half size tensor
    x_half = nntc.from_array(test_input_half_np)
    x_half_moments = nntile.tensor.TensorMoments(x_half, None, False)
    output_half_nnt = gpt2mlp_nntile.forward_dynamic(x_half_moments)
    output_half_nnt = nntc.to_numpy(output_half_nnt.value)
    assert (
        np.linalg.norm(
            output_half_nnt.T - hug_result_half.cpu().detach().numpy()
        )
        < 1e-04
    )

    gpt2mlp_nntile.unregister()
    x.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_bench_gpt2mlp_forward_async(context_cuda, benchmark_operation, dtype: str):
    device = "cuda"
    gpt2_config = GPT2Config(activation_function="relu", resid_pdrop=0.0)

    input_dim = gpt2_config.n_embd
    input_dim_tile = input_dim
    interm_size = 512
    interm_size_tile = 512
    batch_size = 64

    gpt2mlp_hug = GPT2MLP(interm_size, gpt2_config).to(device)

    rng = np.random.default_rng(42)
    test_input_np = rng.standard_normal((input_dim, batch_size)).astype(np.float32, "F")

    x_traits = nntile.tensor.TensorTraits(
        [input_dim, batch_size], [input_dim_tile, batch_size]
    )
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(test_input_np)
    x_moments = nntile.tensor.TensorMoments(x, None, False)

    nntile_config = {
        "embed_dim": input_dim,
        "embed_dim_tile": input_dim_tile,
        "inner_dim": interm_size,
        "inner_dim_tile": interm_size_tile,
        "interm_size": interm_size,
        "interm_size_tile": interm_size_tile,
        "activation_function": gpt2_config.activation_function,
        "redux": False,
    }

    gpt2mlp_nntile = GPT2MLP_nntile.from_torch(
        gpt2mlp_hug, x_moments, nntile_config
    )

    np_dtype = dtype2np[dtype]
    out_np = np.zeros(
        gpt2mlp_nntile.activations[-1].value.shape,
        order="F",
        dtype=np_dtype,
    )

    def bench_fn():
        gpt2mlp_nntile.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_bench_gpt2mlp_forward_backward_async(context_cuda, benchmark_operation, dtype: str):
    device = "cuda"
    gpt2_config = GPT2Config(activation_function="relu", resid_pdrop=0.0)

    input_dim = gpt2_config.n_embd
    input_dim_tile = input_dim
    interm_size = 512
    interm_size_tile = 512
    batch_size = 64

    gpt2mlp_hug = GPT2MLP(interm_size, gpt2_config).to(device)

    rng = np.random.default_rng(42)
    test_input_np = rng.standard_normal((input_dim, batch_size)).astype(np.float32, "F")

    x_traits = nntile.tensor.TensorTraits(
        [input_dim, batch_size], [input_dim_tile, batch_size]
    )
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x_grad = None
    x_moments = nntile.tensor.TensorMoments(x, x_grad, False)
    x.from_array(test_input_np)

    nntile_config = {
        "embed_dim": input_dim,
        "embed_dim_tile": input_dim_tile,
        "inner_dim": interm_size,
        "inner_dim_tile": interm_size_tile,
        "interm_size": interm_size,
        "interm_size_tile": interm_size_tile,
        "activation_function": gpt2_config.activation_function,
        "redux": False,
    }

    gpt2mlp_nntile = GPT2MLP_nntile.from_torch(
        gpt2mlp_hug, x_moments, nntile_config
    )

    gpt2mlp_nntile.clear_gradients()
    # forward once and prepare grad
    gpt2mlp_nntile.forward_async()
    out_tm = gpt2mlp_nntile.activations[-1]
    np_dtype = dtype2np[dtype]
    grad_np = np.ones(out_tm.value.shape, dtype=np_dtype, order="F")
    out_tm.grad.from_array(grad_np)

    def bench_fn():
        gpt2mlp_nntile.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
