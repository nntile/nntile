# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/gpt2_inference_server.py
# GPT2 inference server example
#
# @version 1.1.0

import argparse
import logging

from transformers import GPT2TokenizerFast

import nntile
from nntile.inference.llm_api_server import (
    SimpleLlmApiServer, SimpleLlmApiServerParams)
from nntile.inference.llm_async_engine import LlmAsyncInferenceEngine
from nntile.inference.llm_sync_engine import LlmSyncInferenceEngine
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt

context = nntile.Context(ncpu=-1, ncuda=-1, ooc=0, logger=0, verbose=0)

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with llm inference server",
        usage="""Api will be available on http://host:port
        Interactive docs with methods and schemas (swagger) on
        http://host:port/docs
        """,
    )

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12224)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_hf",
        help="cache dir for huggingface objects",
    )
    parser.add_argument("--async-server", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=1024)

    args = parser.parse_args()
    print(f"Notice:\n {parser.usage}")
    return args


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",
            cache_dir=args.cache_dir)
    model_nnt = GPT2Model_nnt.from_pretrained(
        "gpt2", 1, 1, args.max_seq_len, cache_dir=args.cache_dir
    )

    if args.async_server:
        llm_engine = LlmAsyncInferenceEngine(
            model_nnt, tokenizer, args.max_seq_len
        )
    else:
        llm_engine = LlmSyncInferenceEngine(
            model_nnt, tokenizer, args.max_seq_len
        )

    server = SimpleLlmApiServer(
        llm_engine, params=SimpleLlmApiServerParams(host=args.host,
            port=args.port)
    )
    server.run()


if __name__ == "__main__":
    main()
