# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_generate.py
# GPT2 generate example
#
# @version 1.1.0

import argparse
import logging

from transformers import GPT2TokenizerFast

import nntile
from nntile.inference.llm_sync_engine import LlmSyncInferenceEngine
from nntile.model.generation.llm_params import GenerationParams
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt

starpu_config = nntile.starpu.Config(ncpus_=4, ncuda_=1, cublas_=1)
nntile.starpu.init()

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with llm inference for GPT2 model",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_hf",
        help="cache dir for huggingface objects",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dtype", choices=["fp32", "fp64", "tf32", "bf16"],
                        default="fp32")
    parser.add_argument("--restrict", choices=["cpu", "cuda", None],
            default=None)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--mode", choices=["Greedy"],
            default="Greedy")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p-thr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)

    args = parser.parse_args()
    print(f"Notice:\n {parser.usage}")
    return args


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",
            cache_dir=args.cache_dir)
    model_nnt, _ = GPT2Model_nnt.from_pretrained(
        args.model, 1, 1, args.max_seq_len, 0, cache_dir=args.cache_dir
    )

    llm_engine = LlmSyncInferenceEngine(
        model_nnt, tokenizer, args.max_seq_len
    )

    generation_params = GenerationParams(
        max_tokens=args.max_tokens,
        use_cache=args.use_cache,
        top_k=args.top_k,
        top_p_thr=args.top_p_thr,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )

    result = llm_engine.generate(args.prompt, generation_params)
    print(result)


if __name__ == "__main__":
    main()
