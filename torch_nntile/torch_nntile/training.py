# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/training.py
# Training helpers for models on device="nntile".

"""Utilities for training on the nntile device.

Cross-entropy is implemented with libnntile tensor ops (``maxsumexp``,
``logsumexp``, ``total_sum_accum``, ``softmax``, ``subtract_indexed_outputs``),
mirroring ``NNCrossEntropyOp`` in the main package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_nntile import _C

# torch.nn._reduction: none=0, mean=1, sum=2
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


class _NntileCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: int,
        ignore_index: int,
    ) -> torch.Tensor:
        loss = _C.cross_entropy_forward(logits, target, reduction, ignore_index)
        ctx.save_for_backward(logits, target)
        ctx.reduction = int(reduction)
        ctx.ignore_index = int(ignore_index)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, target = ctx.saved_tensors
        grad_logits = _C.cross_entropy_backward(
            logits,
            target,
            grad_output,
            ctx.reduction,
            ctx.ignore_index,
        )
        return grad_logits, None, None, None


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy on ``device='nntile'`` via libnntile tensor ops."""
    if logits.device.type != "nntile":
        raise ValueError("cross_entropy expects nntile logits")
    if reduction == "mean":
        reduction_enum = _REDUCTION_MEAN
    elif reduction == "sum":
        reduction_enum = _REDUCTION_SUM
    else:
        raise ValueError("nntile cross_entropy supports reduction 'mean' or 'sum'")
    if target.device.type not in ("cpu", "nntile"):
        target = target.cpu()
    return _NntileCrossEntropy.apply(
        logits, target, reduction_enum, ignore_index
    )


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
    for param in model.parameters():
        param.grad = None

    logits = model(inputs)
    if inputs.device.type == "nntile":
        loss = cross_entropy(logits, targets)
        loss.backward()
        manual_sgd_step(
            [p for p in model.parameters() if p.requires_grad],
            learning_rate,
        )
        return float(loss.detach().cpu().item())

    loss = F.cross_entropy(logits, targets)
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
