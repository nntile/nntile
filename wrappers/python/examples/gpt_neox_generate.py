# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt_neox_generate.py
# GPT-NeoX generate example
#
# @version 1.1.0

import argparse
import torch
import numpy as np
from transformers import GPTNeoXTokenizerFast

import nntile
import nntile.utils.constructors as nntc
from nntile.model.generation.llm_params import (
    GenerationMode, GenerationParams, ParallelSamplingMode
)
from nntile.model.gpt_neox_causal import (
    GPTNeoXForCausalLM as GPTNeoXModel
)

context = nntile.Context(
    ncpu=-1,
    ncuda=-1,
    ooc=0,
    logger=0,
    verbose=0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with LLM inference for GPT-NeoX model",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_hf",
        help="cache dir for huggingface objects",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--remote-model-name",
        type=str,
        default="EleutherAI/gpt-neox-20b"
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp64", "tf32", "bf16", "fp32_fast_fp16", "fp32_fast_bf16"],
        default="fp32",
    )
    parser.add_argument("--restrict", choices=["cpu", "cuda", None],
            default=None)
    parser.add_argument("--prompt", type=str,
            default="What do you think about dogs?")
    parser.add_argument(
            "--generation-mode",
            choices=[e.value for e in GenerationMode],
            default="Greedy"
    )
    parser.add_argument(
            "--parallel-sampling-mode",
            choices=[e.value for e in ParallelSamplingMode],
            default="BeamSearch"
    )
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

    if args.restrict == "cuda":
        context.restrict_cuda()
    elif args.restrict == "cpu":
        context.restrict_cpu()

    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        args.remote_model_name,
        cache_dir=args.cache_dir
    )
    model_nnt = GPTNeoXModel.from_pretrained(
        batch_size=1,
        batch_size_tile=1,
        seq_len=args.max_seq_len,
        cache_dir=args.cache_dir,
        remote_model_name=args.remote_model_name,
        dtype=args.dtype
    )

    mode = GenerationMode(args.generation_mode)
    sampling_mode = ParallelSamplingMode(args.parallel_sampling_mode)
    params = GenerationParams(
        max_tokens=args.max_tokens,
        use_cache=args.use_cache,
        top_k=args.top_k,
        top_p_thr=args.top_p_thr,
        temperature=args.temperature,
        num_beams=args.num_beams,
        parallel_sampling_mode=sampling_mode,
    )

    tokenized_input = tokenizer(args.prompt)
    input_ids = np.array(tokenized_input["input_ids"])
    input_ids_nnt = nntc.from_array(input_ids[:, None])
    prefill_size = -1

    outs = model_nnt.generate(
        input_ids_nnt, prefill_size, params=params, mode=mode
    )
    generated_tokens, _ = outs
    generated_tokens_np = nntc.to_numpy(generated_tokens)

    if params.num_beams > 1:
        generated_text = [tokenizer.decode(beam_text) for beam_text in
                generated_tokens_np.T]
    else:
        generated_text = tokenizer.decode(generated_tokens_np.flatten())

    print(generated_text)


if __name__ == "__main__":
    main()
