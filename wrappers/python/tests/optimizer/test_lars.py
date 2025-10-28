# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/optimizer/test_lars.py
# Test for LARS optimizer
#
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile


class LarsTorch:
    """Reference PyTorch implementation of LARS optimizer"""

    def __init__(self, params, lr, trust_ratio=0.02, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.trust_ratio = trust_ratio
        self.weight_decay = weight_decay

    def step(self):
        """Perform LARS step"""
        for param in self.params:
            if param.grad is None:
                continue

            # Compute weight norm and gradient norm
            weight_norm = torch.norm(param.data)
            grad_norm = torch.norm(param.grad)

            # LARS local learning rate (matches nntile implementation)
            if grad_norm > 0:
                local_lr = self.lr * weight_norm / grad_norm
            else:
                local_lr = self.lr

            # Apply trust ratio as upper bound
            adapted_lr = min(local_lr, self.lr * self.trust_ratio)

            # Apply weight decay
            if self.weight_decay != 0:
                param.grad.add_(param.data, alpha=self.weight_decay)

            # Update parameters
            param.data.add_(param.grad, alpha=-adapted_lr)


@pytest.mark.parametrize('dim,num_steps,lr,trust_ratio,weight_decay', [
    (1000, 10, 1e-3, 0.02, 0.0),
    (1000, 10, 1e-3, 0.02, 1e-4),
    (1000, 10, 1e-4, 0.01, 0.0),
    (500, 5, 1e-2, 0.05, 1e-5),
])
def test_lars_against_pytorch(context, dim, num_steps, lr, trust_ratio, weight_decay, tol=1e-4):
    """Test nntile LARS against reference PyTorch implementation"""

    # Initialize PyTorch parameter
    torch_param = torch.randn((dim,), requires_grad=True, dtype=torch.float32)
    torch_optimizer = LarsTorch([torch_param], lr=lr, trust_ratio=trust_ratio, weight_decay=weight_decay)

    # Initialize nntile parameter
    x_traits = nntile.tensor.TensorTraits([dim], [dim])
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(torch_param.detach().numpy())
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    nntile_param = nntile.tensor.TensorMoments(x, x_grad, True)
    nntile_optimizer = nntile.optimizer.Lars([nntile_param], lr=lr,
                                           trust_ratio=trust_ratio, weight_decay=weight_decay)

    # Test multiple steps
    nntile_param_np = np.zeros((dim,), dtype=np.float32, order="F")
    for i_step in range(num_steps):
        # Generate random gradient
        grad_data = torch.randn((dim,), dtype=torch.float32)

        # Set gradients for both optimizers
        torch_param.grad = grad_data
        nntile_param.grad.from_array(grad_data.numpy())

        # Compute norms for nntile (it requires pre-computed norms)
        weight_norm = np.linalg.norm(torch_param.data.numpy())
        grad_norm = np.linalg.norm(grad_data.numpy())

        # Step both optimizers
        torch_optimizer.step()
        nntile_optimizer.step([weight_norm], [grad_norm])

        # Compare results
        nntile_param.value.to_array(nntile_param_np)
        diff = np.linalg.norm(torch_param.data.numpy() - nntile_param_np)
        norm_ref = np.linalg.norm(torch_param.data.numpy())

        if norm_ref > 0:
            rel_error = diff / norm_ref
            assert rel_error < tol, f"Step {i_step}: relative error {rel_error} > {tol}"
        else:
            assert diff < tol, f"Step {i_step}: absolute error {diff} > {tol}"

    # Cleanup
    nntile_optimizer.unregister()
    nntile_param.unregister()


@pytest.mark.parametrize('dim,lr,trust_ratio,weight_decay', [
    (100, 1e-3, 0.02, 0.0),
    (50, 1e-4, 0.01, 1e-4),
])
def test_lars_zero_grad_norm(context, dim, lr, trust_ratio, weight_decay, tol=1e-6):
    """Test LARS behavior when gradient norm is zero"""

    # Initialize parameters
    torch_param = torch.randn((dim,), requires_grad=True, dtype=torch.float32)
    initial_param = torch_param.data.clone()
    torch_optimizer = LarsTorch([torch_param], lr=lr, trust_ratio=trust_ratio, weight_decay=weight_decay)

    x_traits = nntile.tensor.TensorTraits([dim], [dim])
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(torch_param.detach().numpy())
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    nntile_param = nntile.tensor.TensorMoments(x, x_grad, True)
    nntile_optimizer = nntile.optimizer.Lars([nntile_param], lr=lr,
                                           trust_ratio=trust_ratio, weight_decay=weight_decay)

    # Set zero gradient
    torch_param.grad = torch.zeros_like(torch_param)
    nntile_param.grad.from_array(np.zeros((dim,), dtype=np.float32))

    # Compute norms
    weight_norm = np.linalg.norm(torch_param.data.numpy())
    grad_norm = 0.0  # Zero gradient norm

    # Step optimizers
    torch_optimizer.step()
    nntile_optimizer.step([weight_norm], [grad_norm])

    # Parameters should remain unchanged (no update when grad_norm is zero)
    nntile_param_np = np.zeros((dim,), dtype=np.float32, order="F")
    nntile_param.value.to_array(nntile_param_np)

    torch_diff = torch.norm(torch_param.data - initial_param)
    nntile_diff = np.linalg.norm(nntile_param_np - initial_param.numpy())

    assert torch_diff < tol, f"PyTorch parameter changed: {torch_diff}"
    assert nntile_diff < tol, f"NNTile parameter changed: {nntile_diff}"

    nntile_optimizer.unregister()
    nntile_param.unregister()


@pytest.mark.parametrize('dim,lr,trust_ratio,weight_decay', [
    (100, 1e-3, 0.02, 0.0),
    (50, 1e-4, 0.01, 1e-4),
])
def test_lars_zero_weight_norm(context, dim, lr, trust_ratio, weight_decay, tol=1e-6):
    """Test LARS behavior when weight norm is zero"""

    # Initialize parameters with zeros
    torch_param = torch.zeros((dim,), requires_grad=True, dtype=torch.float32)
    torch_optimizer = LarsTorch([torch_param], lr=lr, trust_ratio=trust_ratio, weight_decay=weight_decay)

    x_traits = nntile.tensor.TensorTraits([dim], [dim])
    x_distr = [0] * x_traits.grid.nelems
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    x.from_array(torch_param.detach().numpy())
    x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr)
    nntile_param = nntile.tensor.TensorMoments(x, x_grad, True)
    nntile_optimizer = nntile.optimizer.Lars([nntile_param], lr=lr,
                                           trust_ratio=trust_ratio, weight_decay=weight_decay)

    # Set gradient
    grad_data = torch.randn((dim,), dtype=torch.float32)
    torch_param.grad = grad_data
    nntile_param.grad.from_array(grad_data.numpy())

    # Compute norms
    weight_norm = 0.0  # Zero weight norm
    grad_norm = np.linalg.norm(grad_data.numpy())

    # Step optimizers
    torch_optimizer.step()
    nntile_optimizer.step([weight_norm], [grad_norm])

    # Parameters should not change when weight_norm is zero (local_lr becomes 0)
    nntile_param_np = np.zeros((dim,), dtype=np.float32, order="F")
    nntile_param.value.to_array(nntile_param_np)

    # Compare with initial zero parameters
    torch_diff = torch.norm(torch_param.data - torch.zeros_like(torch_param.data))
    nntile_diff = np.linalg.norm(nntile_param_np)

    assert torch_diff < tol, f"PyTorch parameter changed: {torch_diff}"
    assert nntile_diff < tol, f"NNTile parameter changed: {nntile_diff}"

    nntile_optimizer.unregister()
    nntile_param.unregister()


def test_lars_multiple_parameters(context):
    """Test LARS with multiple parameters"""

    dims = [100, 50, 200]
    lr = 1e-3
    trust_ratio = 0.02
    weight_decay = 1e-4

    # PyTorch parameters
    torch_params = []
    torch_optimizer_params = []
    for dim in dims:
        param = torch.randn((dim,), requires_grad=True, dtype=torch.float32)
        torch_params.append(param)
        torch_optimizer_params.append(param)
    torch_optimizer = LarsTorch(torch_optimizer_params, lr=lr, trust_ratio=trust_ratio, weight_decay=weight_decay)

    # NNTile parameters
    nntile_params = []
    nntile_optimizer_params = []
    for dim in dims:
        x_traits = nntile.tensor.TensorTraits([dim], [dim])
        x_distr = [0] * x_traits.grid.nelems
        x = nntile.tensor.Tensor_fp32(x_traits, x_distr)
        x.from_array(torch_params[len(nntile_params)].detach().numpy())
        x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr)
        param = nntile.tensor.TensorMoments(x, x_grad, True)
        nntile_params.append(param)
        nntile_optimizer_params.append(param)

    nntile_optimizer = nntile.optimizer.Lars(nntile_optimizer_params, lr=lr,
                                           trust_ratio=trust_ratio, weight_decay=weight_decay)

    # Test one step
    grads = []
    weight_norms = []
    grad_norms = []

    for i, param in enumerate(torch_params):
        grad = torch.randn_like(param)
        param.grad = grad
        grads.append(grad)

        # Set nntile gradient
        nntile_params[i].grad.from_array(grad.numpy())

        # Compute norms
        weight_norms.append(np.linalg.norm(param.data.numpy()))
        grad_norms.append(np.linalg.norm(grad.numpy()))

    # Step optimizers
    torch_optimizer.step()
    nntile_optimizer.step(weight_norms, grad_norms)

    # Compare results
    for i, (torch_param, nntile_param) in enumerate(zip(torch_params, nntile_params)):
        nntile_param_np = np.zeros((dims[i],), dtype=np.float32, order="F")
        nntile_param.value.to_array(nntile_param_np)

        diff = np.linalg.norm(torch_param.data.numpy() - nntile_param_np)
        norm_ref = np.linalg.norm(torch_param.data.numpy())

        if norm_ref > 0:
            rel_error = diff / norm_ref
            assert rel_error < 1e-4, f"Parameter {i}: relative error {rel_error} > 1e-4"

    # Cleanup
    nntile_optimizer.unregister()
    for param in nntile_params:
        param.unregister()
