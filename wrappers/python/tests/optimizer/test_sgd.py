# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/optimizer/test_sgd.py
# Test for SGD optimizer
#
# @version 1.1.0


import numpy as np
import pytest
import torch
import torch.optim as optim

import nntile


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('dim', [10, 100, 1000, 10000])
@pytest.mark.parametrize('num_steps', [10, 100])
@pytest.mark.parametrize('lr', [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize('momentum,dampening,nesterov', [
    [0.0, 0.0, False],
    [0.9, 0.0, False],
    [0.9, 0.0, True],
    [0.0, 0.9, False],
    [0.9, 0.1, False],
])
@pytest.mark.parametrize('weight_decay', [0.0, 1e-4, 1e-2])
def test_sgd(context, dim, num_steps, device, lr, momentum,
             weight_decay, dampening, nesterov, tol=1e-5):
    # Set up PyTorch parameter
    torch_param = torch.randn((dim, ), device=device, requires_grad=True,
                              dtype=torch.float32)

    # Set up NNTile parameter
    x_traits = nntile.tensor.TensorTraits([dim], [dim])
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(torch_param.detach().cpu().numpy())
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    nntile_param = nntile.tensor.TensorMoments(x, x_grad, True)

    # Create optimizers
    nntile_optimizer = nntile.optimizer.SGD([nntile_param], lr=lr,
                                           momentum=momentum,
                                           weight_decay=weight_decay,
                                           dampening=dampening,
                                           nesterov=nesterov)
    torch_optimizer = optim.SGD([torch_param], lr=lr, momentum=momentum,
                                weight_decay=weight_decay,
                                dampening=dampening,
                                nesterov=nesterov)

    # Run optimization steps
    nntile_param_np = np.zeros((dim,), dtype=np.float32, order="F")
    for i_step in range(num_steps):
        # Generate random gradient
        torch_param.grad = torch.randn((dim, ), device=device)
        nntile_param.grad.from_array(torch_param.grad.detach().cpu().numpy())

        # Step both optimizers
        torch_optimizer.step()
        nntile_optimizer.step()

        # Compare parameters
        nntile_param.value.to_array(nntile_param_np)
        top = np.linalg.norm(torch_param.data.cpu().numpy() - nntile_param_np)
        bottom = np.linalg.norm(torch_param.data.cpu().numpy())
        assert top / bottom < tol

    # Clean up
    nntile_optimizer.unregister()
    nntile_param.unregister()
