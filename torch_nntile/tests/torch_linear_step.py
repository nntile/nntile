# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine: one in-order CUDA
# stream executes submitted tasks while a reactive data manager moves
# bytes between the device and pinned host memory on demand.
#
# @file tests/torch_linear_step.py
# Classic linear forward and backward on an NNHaul CUDA worker, checked
# against a torch CPU reference; the capped run exercises the reactive
# offload path.
#
# @version 1.0.0

# Classic linear forward and backward on an NNHaul CUDA worker.
# One bias-free Linear plus mse_loss (sum of squares). The result and
# both gradients are checked against a torch CPU reference.
# With NNHAUL_LIMIT_CUDA_MEM=3 the backward GEMM no longer fits, so
# the reactive manager offloads one Belady victim to the host shadow
# and fetches it back when the launch cursor reaches the reader.

import os
import time

os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

import torch

import torch_nntile
from torch_nntile.nn.functional import mse_loss
from torch_nntile.nn.module import Linear


def main() -> None:
    if not torch_nntile.built_with_cuda():
        raise SystemExit("torch_nntile was not built with CUDA")
    if torch_nntile.is_context_initialized():
        raise SystemExit(
            "context already initialized; pass ncuda=1 before the first op"
        )
    cap = os.environ.get("NNHAUL_LIMIT_CUDA_MEM", "")
    print(f"nnhaul linear step env cap_mib={cap}")

    # x and W are 1 MiB. y is 256 KiB. Forward fits in 3 MiB.
    # dX adds grad_x (1 MiB) while x is still live for dW, and dX does
    # not read x, so the planner can move x before that GEMM.
    batch = 256
    features = 1024
    outputs = 256
    gen = torch.Generator().manual_seed(0)
    x_ref = torch.randn(
        batch, features, generator=gen, dtype=torch.float32
    )
    w_ref = torch.randn(
        outputs, features, generator=gen, dtype=torch.float32
    )
    x_ref.mul_(0.02)
    w_ref.mul_(0.02)
    x_ref.requires_grad_(True)
    w_ref.requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, w_ref)
    loss_ref = (y_ref * y_ref).sum()
    loss_ref.backward()

    torch_nntile.init_context(ncpu=1, ncuda=1)
    layer = Linear(features, outputs, bias=False)
    with torch.no_grad():
        layer.weight.copy_(w_ref.detach())
    layer = layer.to("nntile")
    x = x_ref.detach().to("nntile")
    x.requires_grad_(True)
    if not x.is_leaf or not layer.weight.is_leaf:
        raise SystemExit("expected leaf activations and weights")

    y = layer(x)
    loss = mse_loss(y, scale=1.0)
    loss.backward()
    names = torch_nntile.pending_op_names()
    print("pending", " ".join(names))
    native = [name for name in names if name.startswith("TORCH_")]
    if native:
        raise SystemExit(f"torch-native ops in the classic graph: {native}")
    if layer.weight.grad is None or x.grad is None:
        raise SystemExit("autograd did not attach a gradient")

    started = time.perf_counter()
    torch_nntile.compile_graph()
    torch_nntile.run()
    torch_nntile.wait()

    loss_got = loss.detach().cpu()
    w_grad = layer.weight.grad.detach().cpu()
    x_grad = x.grad.detach().cpu()
    print(f"step timing exec_ms={(time.perf_counter() - started) * 1e3:.1f}")
    loss_diff = (loss_got - loss_ref.detach()).abs().item()
    w_diff = (w_grad - w_ref.grad).abs().max().item()
    x_diff = (x_grad - x_ref.grad).abs().max().item()
    print(
        f"loss {loss_got.item()} ref {loss_ref.item()} diff {loss_diff}"
    )
    print(f"wgrad max_abs {w_diff}")
    print(f"xgrad max_abs {x_diff}")
    torch.testing.assert_close(loss_got, loss_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w_grad, w_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x_grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    print("nnhaul linear step ok")


if __name__ == "__main__":
    main()
