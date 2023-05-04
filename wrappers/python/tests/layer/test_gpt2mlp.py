# Here we use transformers-4.28.1 from https://github.com/huggingface/transformers 

import torch
import torch.nn as nn
from nntile.model.gpt2mlp import GPT2MLP as GPT2MLP_nntile
import time
import nntile
import numpy as np
from typing import Optional, Tuple
from huggingface_activations import ACT2FN
from gpt2_config import GPT2Config

batch_size = 100


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        # self.bias = nn.Parameter(torch.zeros(nf))
        self.bias = torch.zeros(nf)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


interm_size = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_config = GPT2Config(activation_function="relu",
                         resid_pdrop=0.)

input_dim = gpt2_config.n_embd
gpt2mlp_hug = GPT2MLP(interm_size, gpt2_config).to(device)

test_input_np = np.array(np.random.randn(batch_size, input_dim), dtype=np.float32, order="F")
test_input = torch.from_numpy(test_input_np).to(device)
hug_result = gpt2mlp_hug(test_input)
print("Norm of the output of PyTorch GPT2MLP", torch.norm(hug_result, "fro").item())

time0 = -time.time()
# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0

nntile_config = {
    "hidden_size": gpt2_config.n_embd,
    "interm_size": interm_size,
    "activation_function": gpt2_config.activation_function
}


x_single_traits = nntile.tensor.TensorTraits([batch_size, test_input.shape[1]], \
        [batch_size, test_input.shape[1]])
x_single_distr = [0]
x_single = nntile.tensor.Tensor_fp32(x_single_traits, x_single_distr, next_tag)
x_single.from_array(test_input.cpu().numpy())
next_tag = x_single.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x_single, x_grad, x_grad_required)

print("Create model...")
# gpt2mlp_nntile = GPT2MLP_nntile(x_moments, nntile_config, next_tag)
# next_tag = gpt2mlp_nntile.next_tag
gpt2mlp_nntile, next_tag = GPT2MLP_nntile.from_torch(gpt2mlp_hug, x_moments,
                                                     batch_size, nntile_config, next_tag)
print("Create model...done")
# print("Init model...")
# gpt2mlp_nntile.init_randn_async()
# print("Init model...done")
print("Forward model...")
gpt2mlp_nntile.forward_async()
print("Forward model...done")


output_traits = nntile.tensor.TensorTraits([batch_size, input_dim], \
        [batch_size, input_dim])
output_single_distr = [0]
output_single = nntile.tensor.Tensor_fp32(output_traits, output_single_distr, next_tag)
next_tag = output_single.next_tag
nntile.tensor.gather_async(gpt2mlp_nntile.activations[-1].value, output_single)

output = np.zeros(output_single.shape, order="F", dtype=np.float32)
# to_array causes y_single to finish gather procedure
output_single.to_array(output)
print("Norm of the output of Nntile GPT2MLP", np.linalg.norm(output, "fro"))

x_single.unregister()
output_single.unregister()
gpt2mlp_nntile.unregister()

