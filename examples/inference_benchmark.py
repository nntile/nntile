import argparse
import logging

import os
import numpy as np
from tqdm import tqdm

from nntile.model.generation.llm import GenerationMode, GenerationParams
import nntile.utils.constructors as nntc
from nntile.inference.llm_api_server import (SimpleLlmApiServer,
                                             SimpleLlmApiServerParams)
from nntile.inference.llm_sync_engine import LlmSyncInferenceEngine
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt
from transformers import GPT2TokenizerFast

import nntile

starpu_config = nntile.starpu.Config(1, 0, 0)
# starpu_config = nntile.starpu.Config(1, 1, 1)
# nntile.starpu.restrict_cuda()
nntile.starpu.init()

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with llm inference server",
        usage="""
        Api will be available on http://host:port
        Interactive docs with methods and schemas (swagger) on http://host:port/docs 
        """,
    )

    parser.add_argument("--prompt_size", type=int, default=16)
    parser.add_argument("--static_alloc", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_hf",
        help="cache dir for huggingface objects",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.cache_dir)
    model_nnt, _ = GPT2Model_nnt.from_pretrained(
        "gpt2", 1, 1, 1024, 0, cache_dir=args.cache_dir
    )

    rng = np.random.default_rng()
    batch_for_test = rng.integers(10,tokenizer.vocab_size//2, args.prompt_size)

    print(f"PID: {os.getpid()}")
    nntile.starpu.profiling_enable()

    for iter in tqdm(range(args.n_iters)):
        input_ids_nnt = nntc.from_array(batch_for_test[:, None])
        output_ids, effective_size = model_nnt.generate(
            input_ids_nnt,
            prefill_size=1,
            params=GenerationParams(max_tokens=args.max_tokens),
            mode=GenerationMode.Greedy,
        )
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    nntile.starpu.profiling_disable()

if __name__ == "__main__":
    main()
