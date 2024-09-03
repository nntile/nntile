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


class LlmAsyncInferenceEngine:
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

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        # tokenize
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        prefill_size = input_ids.shape[1]

        # transform to compatible input
        if params.need_static_padding:
            raise Exception("static async inference is not supported")
        else:
            input_ids_np = np.asfortranarray(input_ids).astype(np.int64)

        input_ids_nnt = nnt_constructors.from_array(input_ids_np.T)

        # generate ids
        output_ids, effective_size = await self.model.generate_async(
            input_ids_nnt,
            prefill_size=prefill_size,
            params=params,
            mode=mode,
        )

        # decode
        output_ids_np = (
            await nnt_constructors.to_numpy_async(output_ids)
        ).astype(int)
        output_ids_np = output_ids_np[:effective_size]

        # construct generation result
        generated_text = self.tokenizer.batch_decode(output_ids_np)
        return generated_text
