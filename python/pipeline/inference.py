# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file python/pipeline/inference.py
# inference.py
#
# @version 1.1.0

"""Shared graph inference pipeline helpers."""

from __future__ import annotations

from typing import Any


def run_inference(model: Any, runtime: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        'Port inference loop from wrappers/python after bindings are complete')
