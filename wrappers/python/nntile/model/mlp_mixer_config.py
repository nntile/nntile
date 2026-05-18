# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/mlp_mixer_config.py
# MLP-Mixer model config
#
# @version 1.1.0

from dataclasses import dataclass


@dataclass
class MlpMixerConfig:
    channel_dim: int
    init_patch_dim: int
    projected_patch_dim: int
    num_mixer_layers: int
    n_classes: int
    layer_norm_epsilon: float = 1e-5
    dtype: str = "fp32"
