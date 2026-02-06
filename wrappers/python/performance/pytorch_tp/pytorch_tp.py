# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/pytorch_tp/pytorch_tp.py
# PyTorch tensor parallel multi-GPU performance test
#
# @version 1.1.0

import argparse
import time

import numpy as np
import torch
import torch.distributed as torch_dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module)
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig, LlamaDecoderLayer,
    LlamaMLP, LlamaRotaryEmbedding)

parser = argparse.ArgumentParser()

parser.add_argument("--num-gpus",
                    choices=[1, 2, 4],
                    type=int,
                    default=1)
parser.add_argument("--module",
                    choices=["mlp", "attention", "decoder"],
                    type=str,
                    default="mlp")
args = parser.parse_args()

# Initialize tensor parallel mesh
tp_mesh = init_device_mesh("cuda", (args.num_gpus,))

# Large model configuration for Llama 405B scale
hidden_size = 16384
intermediate_size = 53248
seqlen = 4096

# Configure tensor parallelism based on module type
# Configure tensor parallelism based on module type
if args.module == "mlp":
    # Tensor parallel plan for MLP: split gate/up projections across GPUs,
    # gather down projection outputs
    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "LlamaMLP.gate_proj": ColwiseParallel(),
        "LlamaMLP.up_proj": ColwiseParallel(),
        "LlamaMLP.down_proj": RowwiseParallel(),
    }
    torch_layer_config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    torch_layer = LlamaMLP(
            torch_layer_config
    ).to("cuda")

elif args.module == "attention":
    torch_layer_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=128,
        num_key_value_heads=16,  # 8,
        attention_bias=False,
        use_cache=False,
        attention_dropout=0.0,
        rope_theta=500000.0,
    )

    torch_layer = LlamaAttention_torch(torch_layer_config,
                                       layer_idx=0).to("cuda")
    layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "LlamaAttention.q_proj": ColwiseParallel(),
    "LlamaAttention.k_proj": ColwiseParallel(),
    "LlamaAttention.v_proj": ColwiseParallel(),
    "LlamaAttention.o_proj": RowwiseParallel(),
    }
    gen = np.random.default_rng(42)
    pos_ids = gen.integers(seqlen,
                            size=(1, seqlen),
                            dtype=np.int64)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long).to("cuda")
    rotary_emb = LlamaRotaryEmbedding(config=torch_layer_config).to("cuda")
    pos_embs = rotary_emb(torch_layer.v_proj.weight,
                                    pos_ids_torch)

    mask = np.array(np.triu(np.ones((seqlen, seqlen))),
                        dtype=bool, order="F")
    mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
                * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(1,
                                                1, -1, -1).to("cuda")
elif args.module == "decoder":
    torch_layer_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=128,
        num_key_value_heads=8,
        attention_bias=False,
        use_cache=False,
        attention_dropout=0.0,
        rope_theta=500000.0,
    )

    torch_layer = LlamaDecoderLayer(torch_layer_config,
                                       layer_idx=0).to("cuda")
    layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "LlamaDecoderLayer.mlp.gate_proj": ColwiseParallel(),
    "LlamaDecoderLayer.mlp.up_proj": ColwiseParallel(),
    "LlamaDecoderLayer.mlp.down_proj": RowwiseParallel(),

    "LlamaDecoderLayer.self_attn.q_proj": ColwiseParallel(),
    "LlamaDecoderLayer.self_attn.k_proj": ColwiseParallel(),
    "LlamaDecoderLayer.self_attn.v_proj": ColwiseParallel(),
    "LlamaDecoderLayer.self_attn.o_proj": RowwiseParallel(),
    }
    gen = np.random.default_rng(42)
    pos_ids = gen.integers(seqlen,
                            size=(1, seqlen),
                            dtype=np.int64)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long).to("cuda")
    rotary_emb = LlamaRotaryEmbedding(config=torch_layer_config).to("cuda")
    pos_embs = rotary_emb(torch_layer.self_attn.v_proj.weight,
                                    pos_ids_torch)

    mask = np.array(np.triu(np.ones((seqlen, seqlen))),
                        dtype=bool, order="F")
    mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
                * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(1,
                                                1, -1, -1).to("cuda")

if args.num_gpus > 1:
    tp_model = parallelize_module(
                module=torch_layer,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )
else:
    tp_model = torch_layer

input_tensor = torch.randn((1, seqlen, hidden_size), requires_grad=True,
    device="cuda")

# Warmup iterations
for iter_idx in range(10):
    if args.module == "mlp":
        output = tp_model(input_tensor)
    elif args.module in ("attention", "decoder"):
        output = tp_model(input_tensor, position_embeddings=pos_embs,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
    loss = torch.sum(output)
    loss.backward()

# Performance measurement
torch.cuda.synchronize()
torch_dist.barrier()
st_time = time.time()
for iter_idx in range(100):
    if args.module == "mlp":
        output = tp_model(input_tensor)
    elif args.module in ("attention", "decoder"):
        output = tp_model(input_tensor, position_embeddings=pos_embs,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
    loss = torch.sum(output)
    loss.backward()
    tp_model.zero_grad()

torch.cuda.synchronize()
torch_dist.barrier()
print("TP model time = {}".format((time.time() - st_time) / 100))

torch_dist.destroy_process_group()
