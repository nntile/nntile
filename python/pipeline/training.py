# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file python/pipeline/training.py
# training.py
#
# @version 1.1.0

"""Shared graph training pipeline helpers (port from wrappers/python)."""

from __future__ import annotations

from typing import Any


def run_training_loop(model: Any, runtime: Any, **kwargs: Any) -> None:
    """Placeholder: bind model forward, lower, execute, optimizer step."""
    raise NotImplementedError(
        'Port training loop from wrappers/python after bindings are complete')
