# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/generation/llm.py
#
# @version 1.1.0

from dataclasses import dataclass
from enum import Enum

import numpy as np

import nntile
import nntile.utils.constructors as nntc
from nntile.tensor import Tensor
from nntile.utils import constructors as nnt_constructors


class GenerationMode(Enum):
    Greedy = "Greedy"


@dataclass
class GenerationParams:
    max_tokens: int
    use_cache: bool = True
    need_static_padding: bool = False


class LLMGenerationMixin:
    def generate(
        self,
        input_ids: Tensor,
        prefill_size: int,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        if mode == GenerationMode.Greedy:
            if params.need_static_padding:
                # This path only for compatibility with statically defined
                # model and not efficient on small examples
                if params.use_cache:
                    raise Exception(
                        "No support for kvcache for static inference"
                    )
                output_ids = generate_greedy(
                    self, input_ids, prefill_size, self.eos_token_id, params
                )
            else:
                output_ids = generate_greedy_dynamic(
                    self, input_ids, self.eos_token_id, params
                )
        else:
            raise Exception("Unsupported generation mode: ", mode)

        return output_ids


def generate_greedy(model, input_ids, prefill_size, eos_token_id, params):
    cur_seq_size = prefill_size

    output_ids = input_ids
    while cur_seq_size < params.max_tokens:
        logits = model.forward(output_ids)

        # TODO: add starpu function for argmax
        logits_np = nnt_constructors.to_numpy(logits)
        pred_token = np.argmax(logits_np[:, cur_seq_size - 1, :])

        if pred_token == eos_token_id:
            return output_ids, cur_seq_size

        # TODO: add starpu function for scalar assign
        output_ids_np = nnt_constructors.to_numpy(output_ids)
        output_ids_np[cur_seq_size, 0] = pred_token
        output_ids.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids, cur_seq_size


def generate_greedy_dynamic(model, input_ids, eos_token_id, params):
    cur_seq_size = input_ids.shape[0]

    is_prefill = True

    output_ids = input_ids
    while cur_seq_size < params.max_tokens:
        output_ids_np = nnt_constructors.to_numpy(output_ids)

        logits_nnt = model.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=(not is_prefill),
        )
        output_value_np = nntc.to_numpy(logits_nnt.value)
        if params.use_cache and is_prefill:
            is_prefill = False

        # TODO: add starpu function for argmax
        pred_token = np.argmax(output_value_np[:, -1, :])
        if pred_token == eos_token_id:
            return output_ids, cur_seq_size

        # TODO: add starpu function for concatenation
        output_ids_np = np.concatenate(
            [output_ids_np, pred_token[None, None]], axis=0
        )
        if params.use_cache:
            input_ids = nntc.from_array(
                pred_token[None, None].astype(np.int64)
            )
        else:
            input_ids = nntc.from_array(output_ids_np)
        output_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids, cur_seq_size
