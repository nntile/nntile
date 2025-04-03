# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_config.py
# T5Config submodule of NNTile Python package
#
# @version 1.1.0

from dataclasses import dataclass


@dataclass
class T5ConfigNNTile:
    d_model: int
    d_model_tile: int
    d_ff: int
    d_ff_tile: int
    dense_act_fn: str = "gelu"
    dropout_rate: float = 0
    is_gated_act: bool = True
    layer_norm_epsilon: float = 1e-5

    redux: bool = False
