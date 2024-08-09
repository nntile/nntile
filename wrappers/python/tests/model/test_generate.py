# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_generate.py
# GPT2 generate example
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
from transformers import GPT2Tokenizer

import nntile.utils.constructors as nnt_constructors
from nntile.model.generation.llm import GenerationMode, GenerationParams
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt


@dataclass
class GenerateTestParams:
    model_name: str
    prompt: str
    expected: str
    max_tokens: int

    minibatch_size: int = 1
    minibatch_size_tile: int = 1
    seq_len_tile: int = 1024


TEST_GENERATE_INPUT_PARAMS = [
    GenerateTestParams(
        "gpt2",
        "Hello, my dog is cute",
        "Hello, my dog is cute. I'm",  # not sure, if she's a puppy or not.",
        max_tokens=9,
    )
]


@pytest.mark.slow
@pytest.mark.parametrize("params", TEST_GENERATE_INPUT_PARAMS)
def test_generation_from_pretrained(starpu_simple, params):
    tokenizer = GPT2Tokenizer.from_pretrained(params.model_name)
    next_tag = 0
    model_nnt, next_tag = GPT2Model_nnt.from_pretrained(
        params.model_name,
        params.minibatch_size,
        params.minibatch_size_tile,
        params.seq_len_tile,
        next_tag,
    )

    inputs = tokenizer(params.prompt, return_tensors="np")
    input_ids = inputs["input_ids"]

    input_static_size = params.seq_len_tile
    padded_input_ids = np.zeros((1, input_static_size), dtype=int, order="F")
    padded_input_ids[0, : input_ids.shape[1]] = input_ids
    padded_input_ids = padded_input_ids.T

    padded_input = nnt_constructors.from_array(padded_input_ids)
    output_ids, effective_size = model_nnt.generate(
        padded_input,
        prefill_size=input_ids.shape[1],
        params=GenerationParams(max_tokens=params.max_tokens),
        mode=GenerationMode.Greedy,
    )

    output_ids_np = nnt_constructors.to_numpy(output_ids).astype(int)
    output_ids_np = output_ids_np[:effective_size]

    generation_result_list = tokenizer.batch_decode(output_ids_np)
    generated_text = "".join(generation_result_list)

    assert (
        generated_text == params.expected
    ), f"Got: {generated_text}. Expected {params.expected}"
