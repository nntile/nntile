import torch.optim as optim
import torch.nn as nn
import torch
import nntile
import time
import copy
import numpy as np

nntile_config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()

def run_test(dim, num_steps, device, lr, tol=1e-5):
    torch_param = torch.randn((dim, ), device=device, requires_grad=True, dtype=torch.float32)
    next_tag = 0
    x_traits = nntile.tensor.TensorTraits( \
                [dim], \
                [dim])
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
        assert np.linalg.norm(torch_param.data.cpu().numpy() - nntile_param_np) / \
            np.linalg.norm(torch_param.data.cpu().numpy()) < tol


    nntile_optimizer.unregister()
    nntile_param.unregister()

if __name__ == "__main__":

    run_test(dim=1000, num_steps=100, device="cpu", lr=1)
    run_test(dim=1000, num_steps=100, device="cuda", lr=1)

    run_test(dim=1000, num_steps=10, device="cuda", lr=1e-1)
    run_test(dim=1000, num_steps=10, device="cpu", lr=1e-1)

    run_test(dim=1000, num_steps=10, device="cpu", lr=1e-4)
    run_test(dim=1000, num_steps=10, device="cuda", lr=1e-4)

    run_test(dim=1000, num_steps=100, device="cpu", lr=1e-4)
    run_test(dim=1000, num_steps=100, device="cuda", lr=1e-4)