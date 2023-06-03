# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_training.py
# GPT2 training example
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-05-17

# Imports
import nntile
import numpy as np
import time
import sys
import torch
from transformers import GPT2Tokenizer, TextDataset, GPT2LMHeadModel
from transformers import GPT2Model, GPT2Config
# pip3 install datasets
from datasets import load_dataset
from nntile.model.gpt2 import GPT2
from nntile.tensor import copy_async
from nntile.loss import Frob
import pdb 

# Describe dataset
dataset_path = "./data"
dataset = "WikiText-103"
subdataset = np.arange(1000)

# Describe GPT2 neural network
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
seq_len = 512
seq_len_tile = 512
batch_size = 1
batch_size_tile = 1

# Read dataset
if dataset == "WikiText-103":
    train_dataset = load_dataset("wikitext", "wikitext-103-v1", \
            split='train', cache_dir=dataset_path).select(subdataset)
else:
    raise ValueError("{} dataset is not supported yet!".format(dataset))

# Tokenize and store as a single numpy array
map_tokens = map(lambda x: tokenizer(x["text"])["input_ids"], \
        train_dataset)
list_tokens = []
for seq in map_tokens:
    list_tokens.extend(seq)
num_tokens = len(list_tokens)
num_seq = num_tokens // (seq_len+1)
num_batches = num_seq // batch_size
num_tokens_truncated = num_batches * batch_size * (seq_len+1)
tokens = np.array(list_tokens[:num_tokens_truncated], order='F', \
        dtype=np.int64)
tokens = tokens.reshape(num_batches, batch_size, seq_len+1)
print("Number of train sequences: {}".format(num_batches * batch_size))
print("Number of train batches: {}".format(num_batches))

# PyTorch model
# model_torch = GPT2LMHeadModel.from_pretrained("gpt2")
config = GPT2Config()
config.attn_pdrop = 0
config.embd_pdrop = 0
config.resid_pdrop = 0
config.n_head=1
config.num_hidden_layers = 0
model_torch = GPT2Model(config)

vocab_size = model_torch.config.vocab_size
print(model_torch)
# input_ids = tokenizer('I enjoy walking with my cute dog', return_tensors='pt')
# print(input_ids, input_ids["input_ids"].shape)
print(tokens.shape, tokens.dtype)
# pdb.set_trace()
output = model_torch(torch.from_numpy(tokens[0, :, :-1]))
torch_loss = 0.5 * torch.sum(torch.square(output.last_hidden_state))
torch_loss.backward()
print(output.last_hidden_state.shape, torch_loss.item())
# print(output.logits.shape)




time0 = -time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0

# Prepare input batches for NNTile
time0 = -time.time()
batch_input = []
batch_output = []
x_single_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
        [batch_size, seq_len])
x_single_distr = [0]
x_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
        next_tag)
next_tag = x_single.next_tag
y_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
        next_tag)
next_tag = y_single.next_tag
x_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
        [batch_size_tile, seq_len_tile])
x_distr = [0] * x_traits.grid.nelems
for i in range(num_batches):
    x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x_single.from_array(tokens[i, :, :-1])
    nntile.tensor.scatter_async(x_single, x)
    batch_input.append(x)
    y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
    next_tag = y.next_tag
    y_single.from_array(tokens[i, :, 1:])
    nntile.tensor.scatter_async(y_single, y)
    batch_output.append(y)


nntile_model, next_tag = GPT2.from_torch(model_torch, batch_size, seq_len, 
                                         config.layer_norm_epsilon, next_tag)
copy_async(batch_input[0], nntile_model.activations[0].value) 
nntile_model.forward_async()

nntile_model.clear_gradients()

fro_loss, next_tag = Frob.generate_simple(nntile_model.activations[-1], next_tag)
fro_loss.y.from_array(np.zeros((1, seq_len, config.n_embd), order="F", dtype=np.float32))

fro_loss.calc_async()

nntile_model.backward_async()

val_np = np.zeros((1,), order="F", dtype=np.float32)
fro_loss.val.to_array(val_np)
print("NNTile loss = {}".format(val_np[0]))
print("Relative difference between PyTorch and NNTile losses = {}".format(
    abs(val_np[0] - torch_loss.item()) / torch_loss.item()))

for i, (p_nntile, p_torch) in enumerate(zip(nntile_model.parameters, model_torch.parameters())):
    p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
    p_nntile.grad.to_array(p_nntile_grad_np)
    if len(p_nntile.grad.shape) == 1:
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
    elif len(p_nntile.grad.shape) == 2:
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np).T) / torch.norm(p_torch.grad)
    print("Relative error in gradient in layer {} = {}".format(i, rel_error.item()))



# Wait for all scatters to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("From PyTorch loader to NNTile batches in {} seconds".format(time0))

# Define tensor X for input batches
#time0 = -time.time()
#x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
#next_tag = x.next_tag
#x_moments = nntile.tensor.TensorMoments(x, None, False)

# Unregister single-tile tensors for data scattering/gathering
x_single.unregister()
y_single.unregister()

# Unregister all tensors related to model
nntile_model.unregister()

# Unregister optimizer states
#optimizer.unregister()

# Unregister loss function
fro_loss.unregister()

# Unregister input/output batches
for x in batch_input:
    x.unregister()
for x in batch_output:
    x.unregister()

