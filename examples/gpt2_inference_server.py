import argparse
import logging

from transformers import GPT2TokenizerFast

import nntile
from nntile.inference.llm_api_server import (
    SimpleLlmApiServer, SimpleLlmApiServerParams)
from nntile.inference.llm_sync_engine import LlmSyncInferenceEngine
from nntile.model.gpt2 import GPT2Model as GPT2Model_nnt

starpu_config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

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

    args = parser.parse_args()
    print(f"Notice:\n {parser.usage}")
    return args


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",
            cache_dir=args.cache_dir)
    model_nnt, _ = GPT2Model_nnt.from_pretrained(
        "gpt2", 1, 1, 1024, 0, cache_dir=args.cache_dir
    )

    llm_engine = LlmSyncInferenceEngine(model_nnt, tokenizer, 1024)

    server = SimpleLlmApiServer(
        llm_engine, params=SimpleLlmApiServerParams(host=args.host,
            port=args.port)
    )
    server.run()


if __name__ == "__main__":
    main()
