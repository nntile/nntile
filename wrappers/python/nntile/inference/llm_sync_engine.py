# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/inference/llm_sync_engine.py
#
# @version 1.1.0

import numpy as np

import nntile.utils.constructors as nnt_constructors
from nntile.model.generation.llm import GenerationMode, GenerationParams


class LlmSyncInferenceEngine:
    def __init__(self, model, tokenizer, input_seq_size: int):
        """
        model - nntile model
        tokenizer - huggingface-like tokenizer
        input_seq_size - static size of input sequence.
        For now, need to manually pad sequence to it
        """
        self.model = model
        self.tokenizer = tokenizer
        self.input_seq_size = input_seq_size

    def generate(
        self,
        prompt: str,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        # tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        prefill_size = input_ids.shape[0]

        # transform to compatible input
        if params.need_static_padding:
            input_ids_np = _pad_input_numpy_tensor_to_sequence_size(
                input_ids, self.input_seq_size
            )
        else:
            input_ids_np = np.asfortranarray(input_ids).astype(np.int64)

        input_ids_nnt = nnt_constructors.from_array(input_ids_np.T)

        # generate ids
        output_ids, effective_size = self.model.generate(
            input_ids_nnt,
            prefill_size=prefill_size,
            params=params,
            mode=mode,
        )

        # decode
        output_ids_np = nnt_constructors.to_numpy(output_ids).astype(int)
        output_ids_np = output_ids_np[:effective_size]

        # construct generation result
        generation_result_list = self.tokenizer.batch_decode(output_ids_np)
        generated_text = "".join(generation_result_list)
        return generated_text


def _pad_input_numpy_tensor_to_sequence_size(
    input_ids: np.ndarray, input_seq_size: int
) -> np.ndarray:
    padded_input_ids = np.zeros((1, input_seq_size), dtype=int, order="F")
    padded_input_ids[0, : input_ids.shape[1]] = input_ids
    return padded_input_ids
