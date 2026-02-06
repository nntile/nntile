import time

import torch
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaDecoderLayer)

hidden_size = 1024
intermediate_size = 4 * hidden_size

torch_layer_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=8,
        num_key_value_heads=8,
        attention_bias=False,
        use_cache=False,
        attention_dropout=0.0,
        rope_theta=2.,
    )

# torch_layer = LlamaAttention_torch(
#         torch_layer_config, layer_idx=0
#     )
torch_layer = LlamaDecoderLayer(torch_layer_config,
                                     layer_idx=None)
# mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
#                     dtype=bool, order="F")
# pos_ids = gen.integers(args.seq_len,
#                         size=(args.minibatch_size, args.seq_len),
#                         dtype=np.int64)
#     mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
#             * torch.finfo(torch.float32).min
#     mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
#                                             1, -1, -1).to(torch_device)
#     pos_ids_torch = torch.tensor(pos_ids).to(torch_device)
#     rotary_emb = LlamaRotaryEmbedding(config=llama_torch_config) \
#       .to(torch_device)
#     pos_embs = rotary_emb(torch_layer_.self_attn.v_proj.weight,
#                                 pos_ids_torch)
print(torch_layer)

# torch_layer = LlamaMLP(
#         torch_layer_config
# ).to("cuda")
# print(torch_layer)
seqlen = 1024
input_tensor = torch.rand((4, seqlen, hidden_size)).to("cuda")


torch.cuda.synchronize()
st_time = time.time()
for iter_idx in range(100):
    output = torch_layer(input_tensor)
torch.cuda.synchronize()
print("No tp model time = {}".format(time.time() - st_time))
