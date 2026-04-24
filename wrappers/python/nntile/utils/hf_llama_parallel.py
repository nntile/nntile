# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file wrappers/python/nntile/utils/hf_llama_parallel.py
# Hugging Face ``LlamaConfig`` helpers for NNTile interoperability.
#
# @version 1.1.0

from __future__ import annotations

from typing import Any


def disable_hf_llama_tensor_parallel(config: Any) -> None:
    """Clear tp and pp plans on a Hugging Face LlamaConfig.

    Recent ``transformers`` attaches default ``base_model_tp_plan`` and
    ``base_model_pp_plan`` to ``LlamaConfig``. ``PreTrainedModel.post_init``
    then validates TP styles; with some PyTorch builds ``ALL_PARALLEL_STYLES``
    is ``None``, and ``v not in None`` raises ``TypeError``. NNTile round-trips
    build single-process reference models without TP/PP.
    """
    if hasattr(config, "base_model_tp_plan"):
        config.base_model_tp_plan = None
    if hasattr(config, "base_model_pp_plan"):
        config.base_model_pp_plan = None
