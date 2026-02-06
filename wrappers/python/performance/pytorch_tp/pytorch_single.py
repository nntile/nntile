# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/pytorch_tp/pytorch_single.py
# PyTorch tensor parallel single GPU performance test
#
# @version 1.1.0

import time

import torch
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaDecoderLayer)

# Model configuration for small Llama model
hidden_size = 1024
intermediate_size = 4 * hidden_size

# Configure PyTorch Llama layer
torch_layer_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=8,
        num_key_value_heads=8,
        attention_bias=False,
        use_cache=False,
        attention_dropout=0.0,
        rope_theta=2.,
    )

# Create Llama decoder layer
torch_layer = LlamaDecoderLayer(torch_layer_config,
                                     layer_idx=None)
print(torch_layer)

# Prepare input tensor for inference
seqlen = 1024
input_tensor = torch.rand((4, seqlen, hidden_size)).to("cuda")

# Measure inference performance
torch.cuda.synchronize()
st_time = time.time()
for iter_idx in range(100):
    output = torch_layer(input_tensor)
torch.cuda.synchronize()
print("No tp model time = {}".format(time.time() - st_time))
