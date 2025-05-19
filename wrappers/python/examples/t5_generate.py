# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/t5_generate.py
# T5 generate example
#
# @version 1.1.0

import argparse

import numpy as np
from transformers import T5TokenizerFast

import nntile
import nntile.utils.constructors as nntc
from nntile.model.generation.llm_params import (
    GenerationMode, GenerationParams, ParallelSamplingMode)
from nntile.model.t5_model import T5ForSequenceClassification, T5ForConditionalGeneration
from transformers import T5ForConditionalGeneration as T5ForConditionalGenerationTorch
starpu_config = nntile.starpu.Config(ncpus_=4, ncuda_=1, cublas_=1)
nntile.starpu.init()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example with llm inference for T5 model",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_hf",
        help="cache dir for huggingface objects",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--restrict", choices=["cpu", "cuda", None],
            default=None)
    parser.add_argument("--prompt", type=str,
            default="translate English to German: The house is wonderful.")
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
        nntile.starpu.restrict_cuda()
    elif args.restrict == "cpu":
        nntile.starpu.restrict_cpu()

    tokenizer = T5TokenizerFast.from_pretrained(args.model,
            cache_dir=args.cache_dir)
    # First load the PyTorch model
    from transformers import T5Model as T5ModelTorch
    from transformers import T5Config as T5ConfigTorch
    from nntile.model.t5_model import T5Model as T5Model_nnt
    from nntile.model.t5_config import T5ConfigNNTile
    from nntile.tensor import TensorMoments, Tensor_fp32, Tensor_int64
    import nntile.utils.constructors as nntc
    # Load the PyTorch model and config
    torch_config = T5ConfigTorch.from_pretrained(args.model, cache_dir=args.cache_dir)
    # torch_model = T5ModelTorch.from_pretrained(args.model, cache_dir=args.cache_dir)
    torch_model = T5ForConditionalGenerationTorch.from_pretrained(args.model, cache_dir=args.cache_dir)
    
    # Manually create NNTile config from torch config
    config = T5ConfigNNTile(
        d_model=torch_config.d_model,
        d_model_tile=torch_config.d_model,
        d_kv=torch_config.d_kv,
        d_kv_tile=torch_config.d_kv,
        d_ff=torch_config.d_ff,
        d_ff_tile=torch_config.d_ff,
        num_layers=torch_config.num_layers,
        n_head=torch_config.num_heads,
        n_head_tile=torch_config.num_heads,
        vocab_size=torch_config.vocab_size,
        dropout_rate=0.0,
        layer_norm_epsilon=torch_config.layer_norm_epsilon,
        redux=False,
        dtype="fp32"
    )
    
    # Create input tensors using zeros from constructors
    x_tensor = nntc.empty([args.max_seq_len, 1], dtype=Tensor_int64)
    decoder_x_tensor = nntc.empty([args.max_seq_len, 1], dtype=Tensor_int64)
    x = TensorMoments(x_tensor, None, False)
    decoder_x = TensorMoments(decoder_x_tensor, None, False)
    
    # Convert from PyTorch to NNTile model
    # model_nnt, next_tag = T5ForSequenceClassification.from_torch(torch_model, x, decoder_x, config)
    model_nnt, next_tag = T5ForConditionalGeneration.from_torch(torch_model, x, decoder_x, config)

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
        need_static_padding=True  # Enable static padding for T5
    )

    tokenized_input = tokenizer(args.prompt)
    input_ids = np.array(tokenized_input["input_ids"])
    # Pad input_ids to max_seq_len
    padded_input_ids = np.zeros((args.max_seq_len, 1), dtype=np.int64)
    padded_input_ids[:len(input_ids), 0] = input_ids
    input_ids_nnt = nntc.from_array(padded_input_ids)
    prefill_size = len(input_ids)  # Set prefill size to input length for static padding

    outs = model_nnt.generate(
        input_ids_nnt, None, params=params, mode=mode
    )
    generated_tokens, _ = outs
    generated_tokens_np = nntc.to_numpy(generated_tokens)
    print("GENERATED TOKENS: ", generated_tokens_np)

    if params.num_beams > 1:
        generated_text = [tokenizer.decode(beam_text) for beam_text in
                generated_tokens_np.T]
    else:
        generated_text = tokenizer.decode(generated_tokens_np.flatten())

    print(generated_text)


if __name__ == "__main__":
    main()
