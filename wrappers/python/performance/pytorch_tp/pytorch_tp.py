from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
import torch
import time
import torch.distributed as torch_dist
from torch.distributed.tensor.parallel import loss_parallel
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP
from transformers.models.llama.modeling_llama import LlamaAttention as LlamaAttention_torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import argparse
import numpy as np

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

tp_mesh = init_device_mesh("cuda", (args.num_gpus,))
hidden_size = 16384
intermediate_size = 53248
seqlen = 4096

if args.module == "mlp":

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

input_tensor = torch.randn((1, seqlen, hidden_size), requires_grad=True, device="cuda")

# Warmup
for iter_idx in range(10):
    if args.module == "mlp":
        output = tp_model(input_tensor)
    elif args.module in ("attention", "decoder"):
        output = tp_model(input_tensor,position_embeddings=pos_embs,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
    #with loss_parallel():
        # assuming pred and labels are of the shape [batch, seq, vocab]
    loss = torch.sum(output)
    loss.backward()

# Measurement
torch.cuda.synchronize()
torch_dist.barrier()
st_time = time.time()
for iter_idx in range(100):
    if args.module == "mlp":
        output = tp_model(input_tensor)
    elif args.module in ("attention", "decoder"):
        output = tp_model(input_tensor,position_embeddings=pos_embs,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
    #with loss_parallel():
        # assuming pred and labels are of the shape [batch, seq, vocab]
    loss = torch.sum(output)
    loss.backward()
    tp_model.zero_grad()

torch.cuda.synchronize()
torch_dist.barrier()
print("TP model time = {}".format((time.time() - st_time) / 100))

torch_dist.destroy_process_group()
