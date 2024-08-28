# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/optimizer/test_adam.py
# Test for Adam optimizer
#
# @version 1.1.0


import numpy as np
import pytest
import torch
import torch.optim as optim

import nntile


@pytest.fixture(scope='module')
def starpu():
    nntile_config = nntile.starpu.Config(1, 0, 0)
    nntile.starpu.init()
    yield nntile_config


@pytest.mark.xfail(reason='not implemented')
@pytest.mark.parametrize('dim,num_steps,device,lr', [
    (1000, 100, 'cpu', 1),
    (1000, 10, 'cpu', 1e-1),
    (1000, 10, 'cpu', 1e-4),
    (1000, 100, 'cpu', 1e-4),
])
def test_adam(starpu, dim, num_steps, device, lr, tol=1e-5):
    torch_param = torch.randn((dim, ), device=device, requires_grad=True,
                              dtype=torch.float32)
    next_tag = 0
    x_traits = nntile.tensor.TensorTraits([dim], [dim])
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x.from_array(torch_param.detach().cpu().numpy())
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x_grad.next_tag
    nntile_param = nntile.tensor.TensorMoments(x, x_grad, True)
    nntile_optimizer = nntile.optimizer.FusedAdam([nntile_param], lr, next_tag)
    next_tag = nntile_optimizer.get_next_tag()

    torch_optimizer = optim.Adam([torch_param], lr=lr)
    nntile_param_np = np.zeros((dim,), dtype=np.float32, order="F")
    for i_step in range(num_steps):
        torch_param.grad = torch.randn((dim, ), device=device)
        nntile_param.grad.from_array(torch_param.grad.detach().cpu().numpy())
        torch_optimizer.step()
        nntile_optimizer.step()
        nntile_param.value.to_array(nntile_param_np)
        top = np.linalg.norm(torch_param.data.cpu().numpy() - nntile_param_np)
        bottom = np.linalg.norm(torch_param.data.cpu().numpy())
        assert top / bottom < tol

    nntile_optimizer.unregister()
    nntile_param.unregister()
