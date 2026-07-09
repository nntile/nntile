# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/training.py
# Training helpers for models on device="nntile".

"""Utilities for training on the nntile device.

Cross-entropy uses libnntile tensor ops (``maxsumexp``, ``logsumexp``,
``total_sum_accum``, ``softmax``, ``subtract_indexed_outputs``). Logits must
have the class dimension last (``[..., C]``); labels match logits without that
axis. Backward broadcasts scalar ``grad_output`` with chained ``scale_slice``
(one per label dimension), then ``multiply_slice`` along the class axis.

SGD uses the fused ``tensor::sgd_step`` kernel (momentum, weight decay,
Nesterov), mirroring ``nntile::optim::SGD`` in the main package.

Adam and AdamW use fused ``tensor::adam_step`` / ``tensor::adamw_step``
kernels, mirroring ``nntile::optim::Adam`` / ``AdamW`` in libnntile.
"""

from __future__ import annotations

import gc
import math
from typing import Callable, Iterable, Mapping

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
    """Cross-entropy on ``device='nntile'`` via libnntile tensor ops.

    ``logits`` and ``target`` must both be on ``device='nntile'``.
    ``logits`` shape ``[..., C]`` (class dim last). ``target`` is int64 with
    shape matching logits without ``C``. Supports ``reduction='mean'`` or
    ``'sum'`` and ``ignore_index`` (default ``-100``).
    """
    if logits.device.type != "nntile":
        raise ValueError("cross_entropy expects nntile logits")
    if reduction == "mean":
        reduction_enum = _REDUCTION_MEAN
    elif reduction == "sum":
        reduction_enum = _REDUCTION_SUM
    else:
        raise ValueError("nntile cross_entropy supports reduction 'mean' or 'sum'")
    if target.device.type != "nntile":
        raise ValueError("cross_entropy expects nntile target")
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
            if param.device.type == "nntile":
                velocity = torch.empty(
                    list(param.shape),
                    dtype=torch.float32,
                    device="nntile",
                )
            else:
                velocity = torch.zeros(
                    list(param.shape),
                    dtype=torch.float32,
                    device=param.device,
                )
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


class _AdamBase:
    """Shared Adam / AdamW logic for ``device='nntile'`` parameters."""

    _use_decoupled_weight_decay: bool = False

    def __init__(
        self,
        parameters: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        params = list(parameters)
        self.param_groups = [
            {
                "params": params,
                "lr": float(lr),
                "betas": (float(betas[0]), float(betas[1])),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
            }
        ]
        self._num_iter = 0
        self._moments: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

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

    def _moments_for(
        self, param: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(param)
        moments = self._moments.get(key)
        if moments is None:
            if param.device.type == "nntile":
                zeros = torch.empty(
                    list(param.shape),
                    dtype=torch.float32,
                    device="nntile",
                )
                moments = (zeros, torch.empty(
                    list(param.shape),
                    dtype=torch.float32,
                    device="nntile",
                ))
            else:
                zeros = torch.zeros(
                    list(param.shape),
                    dtype=torch.float32,
                    device=param.device,
                )
                moments = (zeros, zeros.clone())
            self._moments[key] = moments
        return moments

    def _fused_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        first_moment: torch.Tensor,
        second_moment: torch.Tensor,
        num_iter: int,
        beta_1: float,
        beta_2: float,
        eps: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        if self._use_decoupled_weight_decay:
            _C.adamw_step(
                param,
                grad,
                first_moment,
                second_moment,
                num_iter,
                lr,
                beta_1,
                beta_2,
                eps,
                weight_decay,
            )
        else:
            _C.adam_step(
                param,
                grad,
                first_moment,
                second_moment,
                num_iter,
                lr,
                beta_1,
                beta_2,
                eps,
                weight_decay,
            )

    @staticmethod
    def _cpu_adam_values(
        param: torch.Tensor,
        grad: torch.Tensor,
        first_moment: torch.Tensor,
        second_moment: torch.Tensor,
        num_iter: int,
        beta_1: float,
        beta_2: float,
        eps: float,
        lr: float,
        weight_decay: float,
        *,
        decoupled_weight_decay: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_val = param.detach().clone()
        grad_val = grad.detach().clone()
        m_val = first_moment.detach().clone()
        v_val = second_moment.detach().clone()

        alpha = lr / (1.0 - beta_1 ** num_iter)
        beta_corr = 1.0 / math.sqrt(1.0 - beta_2 ** num_iter)

        if decoupled_weight_decay:
            if weight_decay != 0:
                p_val = p_val * (1.0 - lr * weight_decay)
        elif weight_decay != 0:
            grad_val = grad_val + weight_decay * p_val

        if num_iter == 1:
            m_new = (1.0 - beta_1) * grad_val
            v_new = math.sqrt(1.0 - beta_2) * grad_val.abs()
        else:
            m_new = beta_1 * m_val + (1.0 - beta_1) * grad_val
            v_new = torch.hypot(
                math.sqrt(beta_2) * v_val,
                math.sqrt(1.0 - beta_2) * grad_val,
            )

        denom = v_new * beta_corr + eps
        p_new = p_val - alpha * m_new / denom
        return p_new, m_new, v_new

    def _step_group(self, group: dict) -> None:
        lr = group["lr"]
        beta_1, beta_2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.device.type == "nntile":
                first_moment, second_moment = self._moments_for(param)
                self._fused_step(
                    param,
                    param.grad,
                    first_moment,
                    second_moment,
                    self._num_iter,
                    beta_1,
                    beta_2,
                    eps,
                    lr,
                    weight_decay,
                )
            else:
                first_moment, second_moment = self._moments_for(param)
                p_new, m_new, v_new = self._cpu_adam_values(
                    param,
                    param.grad,
                    first_moment,
                    second_moment,
                    self._num_iter,
                    beta_1,
                    beta_2,
                    eps,
                    lr,
                    weight_decay,
                    decoupled_weight_decay=self._use_decoupled_weight_decay,
                )
                param.copy_(p_new)
                first_moment.copy_(m_new)
                second_moment.copy_(v_new)


class Adam(_AdamBase):
    """Fused Adam for nntile parameters (``tensor::adam_step``)."""

    _use_decoupled_weight_decay = False


class AdamW(_AdamBase):
    """Fused AdamW on nntile (``tensor::adamw_step``)."""

    _use_decoupled_weight_decay = True

    def __init__(
        self,
        parameters: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(
            parameters,
            lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


def fused_adam_step(
    parameters: list[torch.Tensor],
    learning_rate: float,
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    optimizer: Adam | None = None,
) -> Adam:
    """One fused Adam step; returns the optimizer holding moment state."""
    if optimizer is None:
        optimizer = Adam(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer.param_groups[0]["lr"] = float(learning_rate)
    optimizer.step()
    return optimizer


def fused_adamw_step(
    parameters: list[torch.Tensor],
    learning_rate: float,
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    optimizer: AdamW | None = None,
) -> AdamW:
    """One fused AdamW step; returns the optimizer holding moment state."""
    if optimizer is None:
        optimizer = AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer.param_groups[0]["lr"] = float(learning_rate)
    optimizer.step()
    return optimizer


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
    *,
    name_axis_groups: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    axis_group_tiling: Mapping[str, int | list[int] | tuple[int, ...]] | None = None,
    print_axis_groups: bool = False,
) -> float:
    """One full-batch SGD step; returns scalar loss.

    When the model runs on ``device='nntile'``, ``inputs`` and ``targets`` must
    already be on nntile (use ``.to('nntile')`` explicitly).
    """
    for param in model.parameters():
        param.grad = None

    logits = model(inputs)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if logits.device.type == "nntile":
        loss = cross_entropy(logits, targets)
        loss.backward()
        _nntile_optimizer_for(model, learning_rate).step()
        if name_axis_groups is not None:
            name_axis_groups(inputs, logits)
        if axis_group_tiling is not None:
            for name, tile_sizes in axis_group_tiling.items():
                torch_nntile.set_axis_group_tiling(name, tile_sizes)
        if print_axis_groups:
            torch_nntile.print_axis_groups()
        # Keep logits marked through compile/run and loss readout so this
        # phase (and gather) can execute. Drop afterward so the next compile
        # can invalidate_submit the buffer via mark_output(false).
        torch_nntile.compile_graph()
        torch_nntile.run()

        torch_nntile.wait()
        with torch.no_grad():
            loss_cpu = loss.to("cpu")
        del logits
        del loss
        gc.collect()
        return float(loss_cpu.item())

    loss = F.cross_entropy(logits, targets)
    loss.backward()
    fused_sgd_step(list(model.parameters()), learning_rate)
    return float(loss.item())


def clone_model_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy weights to CPU tensors for checkpointing."""
    with torch.no_grad():
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
