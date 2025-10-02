# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/causal_lm_data_preparation.py
# Data preparation for causal language modeling
#
# @version 1.1.0

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Create argument parser
parser = argparse.ArgumentParser(prog="Data preparation",
        description="This example presents how to prepare and"
        "store data in .bin format after tokenization")

parser.add_argument("--hf-dataset", default="roneneldan/TinyStories")
parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument("--hf-tokenizer", type=str, default="kimihailv/llama-1.3b")
parser.add_argument("--tokenizer-path", type=str, default=".model")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()

train_dataset = load_dataset(args.hf_dataset,
                            split='train', cache_dir=args.dataset_path)
if args.dataset_select != -1:
    train_dataset = train_dataset.select(np.arange(
                    args.dataset_select, dtype=np.int64))

tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer,
                                               cache_dir=args.tokenizer_path)


map_train_tokens = map(lambda x: tokenizer(x["text"])["input_ids"],
                        train_dataset)
list_train_tokens = []
for seq in map_train_tokens:
    list_train_tokens.extend(seq)
train_tokens_raw = np.array(list_train_tokens, dtype=np.int64)

num_train_tokens = train_tokens_raw.shape[0]

num_train_seq = num_train_tokens // (args.seq_len + 1)
num_train_batches = num_train_seq // args.batch_size
num_train_tokens_truncated = num_train_batches * (args.batch_size
        * (args.seq_len + 1))
train_tokens_trunc = np.array(
    train_tokens_raw[:num_train_tokens_truncated],
    order='F', dtype=np.int64)


arr_len = num_train_tokens_truncated
ds_name = args.hf_dataset.split("/")[-1].lower()
filename = Path(args.dataset_path) / ds_name / 'train.bin'
if not (Path(args.dataset_path) / ds_name).exists():
    Path.mkdir(Path(args.dataset_path) / ds_name)
dtype = np.uint16
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

idx = 0
b_size_in_tok = args.batch_size * (args.seq_len + 1)
for batch_idx in range(num_train_batches):
    arr_batch = train_tokens_trunc[batch_idx * b_size_in_tok:
                                    (batch_idx + 1) * b_size_in_tok]
    # Write into mmap
    arr[idx : idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
arr.flush()
