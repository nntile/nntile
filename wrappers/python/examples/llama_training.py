# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/llama_training.py
# Llama training example
#
# @version 1.0.0

import argparse
import json
import time

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

import nntile
from nntile.model.llama_causal import LlamaForCausalLM as Llama_nntile
from nntile.model.llama_config import LlamaConfigNNTile

# Create argument parser
parser = argparse.ArgumentParser(prog="LLaMa-based neural networks",
        description="This example presents an NNTile implementation of a "
        "LLaMa-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

# parser.add_argument("--remote_model_name", default="huggyllama/llama-7b")
parser.add_argument("--remote_model_name", default="kimihailv/llama-1.3b")

parser.add_argument("--pretrained", choices=["local", "remote"],
                    default="local")
parser.add_argument("--checkpoint-path", type=str, default="")
parser.add_argument("--config-path", type=str, default="")
parser.add_argument("--save-checkpoint-path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=-1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--hidden-size-tile", type=int, default=-1)
parser.add_argument("--intermediate-size-tile", type=int, default=-1)
parser.add_argument("--n-head-tile", type=int, default=-1)

parser.add_argument("--nntile-dtype", choices=["fp32", "fp64", "tf32", "bf16"],
                    default="fp32")
parser.add_argument("--nntile-restrict", choices=["cpu", "cuda", None],
        default=None)
parser.add_argument("--nntile-flashattention", action="store_true")
parser.add_argument("--nntile-use-redux", action="store_true")

parser.add_argument("--dataset", default="tinystories")
parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")
parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument("--save-dataset-bin", action="store_true")


parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--nntile-nepochs", type=int, default=0)

parser.add_argument("--nntile-logger", action="store_true")
parser.add_argument("--nntile-logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--nntile-logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args)

if args.seq_len_tile == -1:
    args.seq_len_tile = args.seq_len
if args.minibatch_size == -1:
    args.minibatch_size = args.batch_size
if args.minibatch_size_tile == -1:
    args.minibatch_size_tile = args.minibatch_size
# Check arguments
assert args.seq_len_tile > 0
assert args.batch_size > 0
assert args.minibatch_size > 0
assert args.minibatch_size_tile > 0
assert args.batch_size % args.minibatch_size == 0
num_minibatch = args.batch_size // args.minibatch_size
assert args.minibatch_size % args.minibatch_size_tile == 0
assert args.nntile_nepochs >= 0

# Load named pretrained PyTorch model
if args.pretrained == "remote":
    # Newer versions of transformers can use fast attention, so we disable it
    # through a parameter attn_implementation
    model_torch = LlamaForCausalLM.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path, local_files_only=True)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = LlamaConfig(**conf_dict)
        model_torch = LlamaForCausalLM(config)
        tokenizer = None
        if args.optimizer == "adam":
            optimizer = Adam(model_torch.parameters(), args.lr)
        elif args.optimizer == "sgd":
            optimizer = SGD(model_torch.parameters(), args.lr)
        elif args.optimizer == "adamw":
            optimizer = AdamW(model_torch.parameters(), args.lr)
        else:
            raise ValueError
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model_torch.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

tokenizer = LlamaTokenizerFast.from_pretrained(args.remote_model_name,
                                                cache_dir=args.model_path)
model_torch.eval()
print(model_torch.config)

if args.nntile_nepochs:
    # Initialize NNTile and StarPU
    time0 = time.time()
    # Set up StarPU+MPI and init codelets
    nntile_config = nntile.starpu.Config(-1, -1, 1, args.nntile_logger,
            args.nntile_logger_server_addr, args.nntile_logger_server_port)
    nntile.starpu.profiling_init()
    nntile.starpu.profiling_disable()
    nntile.starpu.init()
    # Restrict computations to CUDA if possible
    if args.nntile_restrict == "cuda":
        nntile.starpu.restrict_cuda()
    elif args.nntile_restrict == "cpu":
        nntile.starpu.restrict_cpu()
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
    next_tag = 0

    time0 = time.time()
    if args.n_head_tile == -1:
        args.n_head_tile = model_torch.config.num_attention_heads
    if args.hidden_size_tile == -1:
        args.hidden_size_tile = model_torch.config.hidden_size
    if args.intermediate_size_tile == -1:
        args.intermediate_size_tile = model_torch.config.intermediate_size

    llama_config_nntile = LlamaConfigNNTile(
        vocab_size=model_torch.vocab_size,
        vocab_embed_dim_tile=model_torch.config.hidden_size,
        hidden_size=model_torch.config.hidden_size,
        hidden_size_tile=args.hidden_size_tile,
        max_position_embeddings=model_torch.config.max_position_embeddings,
        num_hidden_layers=model_torch.config.num_hidden_layers,
        rms_norm_eps=model_torch.config.rms_norm_eps,
        n_attention_head=model_torch.config.num_attention_heads,
        num_key_value_heads=model_torch.config.num_key_value_heads,
        intermediate_size=model_torch.config.intermediate_size,
        intermediate_size_tile=args.intermediate_size_tile,
        n_head_tile=args.n_head_tile,
        dtype=args.nntile_dtype
        )

    print(llama_config_nntile)

    single_batch_pos_ids = np.arange(args.seq_len).reshape(1, args.seq_len)
    pos_ids = np.repeat(single_batch_pos_ids, args.minibatch_size, axis=0)

    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                        dtype=bool, order="F")
    llama_nntile, next_tag = Llama_nntile.from_torch(model_torch,
                                                    args.minibatch_size,
                                                    args.minibatch_size_tile,
                                                    args.seq_len,
                                                    args.seq_len_tile,
                                                    pos_ids,
                                                    mask,
                                                    llama_config_nntile,
                                                    next_tag)
    time1 = time.time() - time0
    print("Converting PyTorch model to NNTile",
          "requires {} seconds".format(time1))
    del model_torch

if args.nntile_nepochs > 0:
    from pathlib import Path

    import numpy as np
    from datasets import load_dataset
    if args.dataset == "tinystories":
        if args.dataset_file == args.dataset + "_train.bin":
            train_data = np.memmap(Path(args.dataset_path) /
                                    args.dataset_file,
                                   dtype=np.uint16, mode='r')
            train_tokens_raw = np.array(train_data, order='F', dtype=np.int64)
            del train_data
        elif args.dataset_file == "":
            train_dataset = load_dataset("roneneldan/TinyStories",
                            split='train', cache_dir=args.dataset_path)
            if args.dataset_select != -1:
                train_dataset = train_dataset.select(np.arange(
                                args.dataset_select, dtype=np.int64))

            map_train_tokens = map(lambda x: tokenizer(x["text"])["input_ids"],
                                    train_dataset)
            list_train_tokens = []
            for seq in map_train_tokens:
                list_train_tokens.extend(seq)
            train_tokens_raw = np.array(list_train_tokens, dtype=np.int64)
        else:
            raise ValueError("Incompatible datasetfile")

        num_train_tokens = train_tokens_raw.shape[0]

        num_train_seq = num_train_tokens // (args.seq_len + 1)
        num_train_batches = num_train_seq // args.batch_size
        num_train_tokens_truncated = num_train_batches * (args.batch_size
                * (args.seq_len + 1))
        train_tokens_trunc = np.array(
            train_tokens_raw[:num_train_tokens_truncated],
            order='F', dtype=np.int64)
        train_tokens = train_tokens_trunc.reshape(num_train_batches,
                                            num_minibatch,
                                            args.minibatch_size,
                                            args.seq_len + 1)

        if args.save_dataset_bin:

            # concatenate all the ids in each dataset
            # into one large file we can use for training
            arr_len = num_train_tokens_truncated
            filename = Path(args.dataset_path) / 'tinystories_train.bin'
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            idx = 0
            b_size_in_tok = args.batch_size * (args.seq_len + 1)
            for batch_idx in range(num_train_batches):
                # Batch together samples for faster write
                # batch = dset.shard(num_shards=num_train_batches,
                # index=batch_idx, contiguous=True).with_format('numpy')

                arr_batch = train_tokens_trunc[batch_idx * b_size_in_tok:
                                               (batch_idx + 1) * b_size_in_tok]
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

if args.nntile_nepochs > 0:
    time0 = time.time()
    batch_input = []
    batch_output = []
    x_traits = nntile.tensor.TensorTraits(
            [args.seq_len, args.minibatch_size],
            [args.seq_len_tile, args.minibatch_size_tile])
    x_distr = [0] * x_traits.grid.nelems
    for i in range(num_train_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(num_minibatch):
            x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            x.from_array(np.asfortranarray(train_tokens[i, j, :, :-1].T))
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(np.asfortranarray(train_tokens[i, j, :, 1:].T))
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)
    time1 = time.time() - time0
    print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
    # Set up learning rate and optimizer for training
    if args.optimizer == "adam":
        optimizer = nntile.optimizer.Adam(llama_nntile.get_parameters(),
                args.lr, next_tag)
    elif args.optimizer == "adamw":
        optimizer = nntile.optimizer.AdamW(llama_nntile.get_parameters(),
                args.lr, next_tag)
    elif args.optimizer == "sgd":
        optimizer = nntile.optimizer.SGD(llama_nntile.get_parameters(),
                args.lr, next_tag)
    next_tag = optimizer.get_next_tag()
    # Define Cross Entropy loss function
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
            llama_nntile.activations[-1], next_tag,
            scale=1.0 / (args.batch_size * args.seq_len))
    # Set up training pipeline
    pipeline = nntile.pipeline.Pipeline(batch_input, batch_output,
            llama_nntile, optimizer, loss, args.nntile_nepochs)
    # Warmup training
    # nntile.starpu.pause()
    nntile.starpu.profiling_enable()
    pipeline.train_async()
    # nntile.starpu.resume()
    nntile.starpu.wait_for_all()
    nntile.starpu.profiling_disable()
    time1 = time.time() - time0
    print("NNTile training time: {} seconds".format(time1))
    print("NNTile training throughput tokens/sec: {}".format(
            args.nntile_nepochs * num_train_batches * args.batch_size
            * args.seq_len / time1))
    nflops_seq = llama_nntile.get_model_forward_flops()
    print("NNTile performance: {} Tflops/s".format(nflops_seq
            * args.nntile_nepochs * num_train_batches * args.batch_size
            / time1 * 1e-12))
    loss_np = np.zeros((1), dtype=np.float32)
    loss.val.to_array(loss_np)
    print("NNTile loss on the last batch: {}".format(loss_np[0]))
    loss.unregister()
    optimizer.unregister()
    for batch in batch_input + batch_output:
        for x in batch:
            x.unregister()
    llama_nntile.unregister()
