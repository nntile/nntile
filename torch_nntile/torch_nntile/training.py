# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/training.py
# Training helpers for models on device="nntile".

"""Utilities for training on the nntile device.

NNTile cross-entropy lives in ``NNGraph`` and composes many tensor ops
(``maxsumexp``, ``logsumexp``, ``total_sum_accum``, ``softmax``,
``subtract_indexed_outputs``, INT64 labels). Wiring it into PyTorch autograd
would require a sizable set of new ATen kernels. Until that exists, these
helpers run cross-entropy on CPU for the loss value and ``grad_logits``, then
propagate through nntile linear/ReLU backward kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss_and_grad(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy loss and ``grad_logits`` with a CPU bridge.

    ``logits`` may live on ``nntile``; ``targets`` must be a CPU ``int64`` vector.
    Returns ``(loss_cpu, grad_logits_on_same_device_as_logits)``.
    """
    logits_cpu = logits.detach().cpu().requires_grad_(True)
    loss = F.cross_entropy(logits_cpu, targets)
    (grad_logits_cpu,) = torch.autograd.grad(loss, logits_cpu)
    grad_logits = grad_logits_cpu.to(device=logits.device)
    return loss.detach(), grad_logits


def manual_sgd_step(
    parameters: list[torch.Tensor],
    learning_rate: float,
) -> None:
    """In-place SGD without ``torch.optim`` (works for nntile parameters)."""
    with torch.no_grad():
        for param in parameters:
            if param.grad is None:
                continue
            if param.device.type == "nntile":
                updated = param.cpu() - learning_rate * param.grad.cpu()
                param.copy_(updated.to("nntile"))
            else:
                param.add_(param.grad, alpha=-learning_rate)


def train_full_batch_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float,
) -> float:
    """One full-batch SGD step; returns scalar loss."""
    logits = model(inputs)
    if inputs.device.type == "nntile":
        if targets.device.type != "cpu":
            targets = targets.cpu()
        loss, grad_logits = cross_entropy_loss_and_grad(logits, targets)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            logits,
            params,
            grad_outputs=grad_logits,
            retain_graph=False,
        )
        for param, grad in zip(params, grads):
            param.grad = grad
        manual_sgd_step(params, learning_rate)
        return float(loss.item())

    loss = F.cross_entropy(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()
    manual_sgd_step(list(model.parameters()), learning_rate)
    return float(loss.item())


def clone_model_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy weights to CPU tensors for checkpointing."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def max_weight_delta(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> float:
    """Maximum absolute difference between two CPU state dicts."""
    delta = 0.0
    for name in state_a:
        diff = (state_a[name] - state_b[name]).abs().max().item()
        delta = max(delta, diff)
    return delta
