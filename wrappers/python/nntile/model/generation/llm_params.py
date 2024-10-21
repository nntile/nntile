# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/generation/llm_params.py
#
# @version 1.1.0

from dataclasses import dataclass
from enum import Enum


class GenerationMode(Enum):
    Greedy = "Greedy"
    TopK = "TopK"
    TopP = "TopP"


class ParallelSamplingMode(Enum):
    BeamSearch = "BeamSearch"
    BeamSearchBigrams = "BeamSearchBigrams"
    Parallel = "Parallel"


@dataclass
class GenerationParams:
    max_tokens: int
    use_cache: bool = True
    need_static_padding: bool = False
    top_k: int | None = None
    top_p_thr: float | None = None
    temperature: float = 1
    num_beams: int = 1
    parallel_sampling_mode: ParallelSamplingMode = (
        ParallelSamplingMode.BeamSearch
    )
