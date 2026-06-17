# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/training.py
# Training helpers for models on device="nntile".

"""Utilities for training on the nntile device.

Cross-entropy uses libnntile tensor ops (``maxsumexp``, ``logsumexp``,
``total_sum_accum``, ``softmax``, ``subtract_indexed_outputs``).

SGD uses the fused ``tensor::sgd_step`` kernel (momentum, weight decay,
Nesterov), mirroring ``nntile::optim::SGD`` in the main package.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

import torch_nntile
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


class SGD:
    """Fused SGD for ``device='nntile'`` parameters (``tensor::sgd_step``)."""

    def __init__(
        self,
        parameters: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening == 0"
            )
        params = list(parameters)
        self.param_groups = [
            {
                "params": params,
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "dampening": float(dampening),
                "nesterov": bool(nesterov),
            }
        ]
        self._num_iter = 0
        self._velocity: dict[int, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._num_iter += 1
        with torch.no_grad():
            for group in self.param_groups:
                self._step_group(group)
        return loss

    def _velocity_for(self, param: torch.Tensor) -> torch.Tensor:
        key = id(param)
        velocity = self._velocity.get(key)
        if velocity is None:
            velocity = torch.zeros(
                list(param.shape),
                dtype=torch.float32,
                device="cpu",
            ).to("nntile")
            self._velocity[key] = velocity
        return velocity

    def _step_group(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.device.type == "nntile":
                velocity = self._velocity_for(param)
                _C.sgd_step(
                    param,
                    param.grad,
                    velocity,
                    self._num_iter,
                    lr,
                    momentum,
                    weight_decay,
                    dampening,
                    nesterov,
                )
            else:
                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)
                if momentum != 0:
                    velocity = self._velocity_for(param)
                    if self._num_iter == 1:
                        velocity.copy_(grad)
                    else:
                        velocity.mul_(momentum).add_(grad, alpha=1 - dampening)
                    if nesterov:
                        grad = grad.add(velocity, alpha=momentum)
                    else:
                        grad = velocity
                param.add_(grad, alpha=-lr)


def fused_sgd_step(
    parameters: list[torch.Tensor],
    learning_rate: float,
    *,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    optimizer: SGD | None = None,
) -> SGD:
    """One fused SGD step; returns the optimizer holding momentum state."""
    if optimizer is None:
        optimizer = SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
    else:
        optimizer.param_groups[0]["lr"] = float(learning_rate)
    optimizer.step()
    return optimizer


def manual_sgd_step(
    parameters: list[torch.Tensor],
    learning_rate: float,
) -> None:
    """Deprecated alias for a single plain fused SGD step (momentum=0)."""
    fused_sgd_step(parameters, learning_rate)


def _nntile_optimizer_for(
    model: torch.nn.Module,
    learning_rate: float,
) -> SGD:
    optimizer = getattr(model, "_nntile_optimizer", None)
    if optimizer is None or optimizer.param_groups[0]["lr"] != learning_rate:
        optimizer = SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        model._nntile_optimizer = optimizer
    return optimizer


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
        _nntile_optimizer_for(model, learning_rate).step()
        if torch_nntile.is_graph_mode():
            torch_nntile.execute()
        return float(loss.detach().cpu().item())

    loss = F.cross_entropy(logits, targets)
    loss.backward()
    fused_sgd_step(list(model.parameters()), learning_rate)
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
