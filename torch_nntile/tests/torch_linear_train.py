# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine: one in-order CUDA
# stream executes submitted tasks while a reactive data manager moves
# bytes between the device and pinned host memory on demand.
#
# @file tests/torch_linear_train.py
# Multi-step training of a bias-free Linear layer on an NNHaul CUDA worker
# under a GPU cap, with the loss trajectory checked against a torch CPU
# reference.
#
# @version 1.0.0

# Multi-step training of a bias-free Linear layer on an NNHaul CUDA
# worker: forward, mse_loss, backward and a fused SGD step, repeated
# under NNHAUL_LIMIT_CUDA_MEM so the data manager moves tiles every
# step. Each iteration is its own batch for the runtime
# (compile_graph, run, wait), and the whole loss trajectory is
# checked against a torch CPU reference running the same optimizer.

import os
import time

os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

import torch

import torch_nntile
from torch_nntile.nn.functional import mse_loss
from torch_nntile.nn.module import Linear
from torch_nntile.training import SGD


def main() -> None:
    if not torch_nntile.built_with_cuda():
        raise SystemExit("torch_nntile was not built with CUDA")
    steps = int(os.environ.get("LINEAR_TRAIN_STEPS", "30"))
    lr = float(os.environ.get("LINEAR_TRAIN_LR", "0.5"))
    momentum = float(os.environ.get("LINEAR_TRAIN_MOMENTUM", "0.0"))
    cap = os.environ.get("NNHAUL_LIMIT_CUDA_MEM", "")

    print(
        f"nnhaul linear train env steps={steps} lr={lr} "
        f"momentum={momentum} cap_mib={cap}"
    )

    # x is 2 MiB, W is 2 MiB and y is 1 MiB, so the forward fits in
    # NNHAUL_LIMIT_CUDA_MEM while the backward gradients overflow it:
    # the data manager moves tiles every step, with the small y and
    # grad tiles as slack victims.
    batch = int(os.environ.get("LINEAR_TRAIN_BATCH", "512"))
    features = 1024
    outputs = int(os.environ.get("LINEAR_TRAIN_OUTPUTS", "512"))
    gen = torch.Generator().manual_seed(0)
    x_ref = torch.randn(
        batch, features, generator=gen, dtype=torch.float32
    )
    w_ref = torch.randn(
        outputs, features, generator=gen, dtype=torch.float32
    )
    x_ref.mul_(0.02)
    w_ref.mul_(0.02)
    # Snapshot before the reference loop trains w_ref in place.
    w_start = w_ref.detach().clone()
    w_ref.requires_grad_(True)

    ref_opt = torch.optim.SGD([w_ref], lr=lr, momentum=momentum)
    ref_losses = []
    for _ in range(steps):
        y = torch.nn.functional.linear(x_ref, w_ref)
        loss = (y * y).sum()
        loss.backward()
        ref_opt.step()
        ref_opt.zero_grad()
        ref_losses.append(loss.detach().clone())

    torch_nntile.init_context(ncpu=1, ncuda=1)
    layer = Linear(features, outputs, bias=False)
    with torch.no_grad():
        layer.weight.copy_(w_start)
    layer = layer.to("nntile")
    x = x_ref.detach().to("nntile")
    x.requires_grad_(True)
    opt = SGD([layer.weight], lr=lr, momentum=momentum)

    got_losses = []
    step_ns = []
    for step in range(steps):
        started = time.perf_counter_ns()
        y = layer(x)
        loss = mse_loss(y, scale=1.0)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step == 0:
            names = torch_nntile.pending_op_names()
            print("pending", " ".join(names))
        # The host readout flushes exactly one execution of the
        # recorded step; wait() compacts the history so the next
        # step does not replay it.
        got_losses.append(loss.detach().cpu())
        torch_nntile.wait()
        del y, loss
        step_ns.append(time.perf_counter_ns() - started)

    got = torch.stack(got_losses)
    ref = torch.stack(ref_losses)
    loss_diff = (got - ref).abs().max().item()
    weight_diff = (
        layer.weight.detach().cpu() - w_ref.detach()
    ).abs().max().item()
    print(f"loss first {got[0].item()} last {got[-1].item()}")
    print(f"ref  loss first {ref[0].item()} last {ref[-1].item()}")
    print(f"loss max_abs {loss_diff} weight max_abs {weight_diff}")
    wall_ms = sum(step_ns) / 1e6
    per_step_ms = wall_ms / steps
    print(
        f"train timing steps={steps} wall_ms={wall_ms:.1f} "
        f"per_step_ms={per_step_ms:.2f}"
    )
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        layer.weight.detach().cpu(), w_ref.detach(), rtol=1e-4, atol=1e-5
    )
    print("nnhaul linear train ok")


if __name__ == "__main__":
    main()
