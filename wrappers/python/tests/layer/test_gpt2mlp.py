# Here we use transformers-4.28.1 from https://github.com/huggingface/transformers 

import transformers.models.gpt2.modeling_gpt2 as gpt2_blocks
import torch
from nntile.model.gpt2mlp import GPT2MLP as GPT2MLP_nntile
import time
import nntile
import numpy as np

batch_size = 100

interm_size = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_config = gpt2_blocks.GPT2Config(activation_function="relu",
                                resid_pdrop=0.)
input_dim = gpt2_config.n_embd
gpt2mlp_hug = gpt2_blocks.GPT2MLP(interm_size, gpt2_config).to(device)

test_input_np = np.array(np.random.randn(batch_size, input_dim), dtype=np.float32, order="F")
test_input = torch.from_numpy(test_input_np).to(device)
hug_result = gpt2mlp_hug(test_input)
print(torch.norm(hug_result, "fro").item())

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
print("Init model...")
gpt2mlp_nntile.init_randn_async()
print("Init model...done")
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
print(np.linalg.norm(output, "fro"))

x_single.unregister()
output_single.unregister()
gpt2mlp_nntile.unregister()

