from dataclasses import dataclass

import pytest

from nntile.inference.llm_sync_engine import LlmSyncInferenceEngine
from nntile.model.generation.llm import GenerationMode, GenerationParams
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt


def get_tokenizer(model_name, cache_dir):
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)
    return tokenizer


@dataclass
class SyncLlmInferenceEngineTestParams:
    model_name: str
    prompt: str
    expected: str
    max_tokens: int

    minibatch_size: int = 1
    minibatch_size_tile: int = 1
    seq_len_tile: int = 1024


TEST_LLM_INF_ENGINE_INPUT_PARAMS = [
    SyncLlmInferenceEngineTestParams(
        "gpt2",
        "Are you big?\n",
        "Are you big?\n\nI'm",  # not big.
        max_tokens=8,
    )
]


@pytest.mark.parametrize("params", TEST_LLM_INF_ENGINE_INPUT_PARAMS)
def test_sync_llm_inference_engine_from_pretrained(
    starpu_simple, huggingface_local_cache_dir, params
):
    tokenizer = get_tokenizer(params.model_name, huggingface_local_cache_dir)
    next_tag = 0
    model_nnt, next_tag = GPT2Model_nnt.from_pretrained(
        params.model_name,
        params.minibatch_size,
        params.minibatch_size_tile,
        params.seq_len_tile,
        next_tag,
    )

    llm_engine = LlmSyncInferenceEngine(
        model_nnt, tokenizer, params.seq_len_tile
    )
    generated_text = llm_engine.generate(
        params.prompt,
        params=GenerationParams(max_tokens=params.max_tokens),
        mode=GenerationMode.Greedy,
    )

    assert (
        generated_text == params.expected
    ), f"Got: {generated_text}. Expected {params.expected}"
