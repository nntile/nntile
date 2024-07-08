from dataclasses import dataclass
from enum import Enum

import numpy as np

import nntile
from nntile.tensor import Tensor
from nntile.utils import constructors as nnt_constructors


class GenerationMode(Enum):
    Greedy = "Greedy"


@dataclass
class GenerationParams:
    max_tokens: int


class LLMGenerationMixin:
    def generate(
        self,
        input_ids: Tensor,
        prefill_size: int,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        if mode == GenerationMode.Greedy:
            output_ids = generate_greedy(self, input_ids, prefill_size, params)
        else:
            raise Exception("Unsupported generation mode: ", mode)

        return output_ids


def generate_greedy(model, input_ids, prefill_size, params):
    cur_seq_size = prefill_size

    output_ids = input_ids
    while cur_seq_size < params.max_tokens:
        logits = model.forward(output_ids)

        # TODO: add starpu function for argmax
        logits_np = nnt_constructors.to_numpy(logits)
        print(logits_np.shape)
        pred_token = np.argmax(logits_np[:, cur_seq_size - 1, :])
        print(pred_token)

        # TODO: add tokenizer.eos break

        # TODO: add starpu function for scalar assign
        output_ids_np = nnt_constructors.to_numpy(output_ids)
        output_ids_np[cur_seq_size, 0] = pred_token
        output_ids.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids, cur_seq_size
