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
# @date 2023-06-19

# Imports
import nntile
import math
import numpy as np
import time
import sys
import torch
from torch import Tensor
import torch.nn as nn
from transformers import GPT2Tokenizer, TextDataset, GPT2LMHeadModel
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from nntile.model.gpt2 import GPT2
from nntile.tensor import copy_async
from nntile.loss import Frob
import pdb 
from typing import Optional, Tuple, List
from packaging import version
import copy
import argparse

# Create argument parser
parser = argparse.ArgumentParser(prog="GPT2-based neural networks", \
        description="This example presents an NNTile implementation of a " \
        "GPT2-family of models and compares it against the Huggingface. " \
        "It checks relative accuracy of a forward pass (values of " \
        "activations) and backward pass (gradients of parameters and " \
        "activations) and a throughput of inference and training. It can " \
        "also fine-tune a pretrained NNTile model on a chosen dataset.")
parser.add_argument("--model", choices=["gpt2", "gpt2-small", "gpt2-medium", \
        "gpt2-large", "gpt2-xl"], default="gpt2")
parser.add_argument("--model-path")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--batch-size-tile", type=int, default=1)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--nforward", type=int, default=10)
parser.add_argument("--nbackward", type=int, default=10)
#parser.add_argument("--dataset", choices=["WikiText-103"], \
#        default="WikiText-103")
#parser.add_argument("--dataset_path")
#parser.add_argument("--epoch", type=int)
#parser.add_argument("--epoch_warmup", type=int)
#parser.add_argument("--lr", type=float)

# Parse arguments
args = parser.parse_args()
print(args)

# Check arguments
assert args.seq_len > 0
assert args.seq_len_tile > 0
assert args.seq_len % args.seq_len_tile == 0
assert args.batch_size > 0
assert args.batch_size_tile > 0
assert args.batch_size % args.batch_size_tile == 0
assert args.nforward > 0
assert args.nbackward > 0

# Load named pretrained PyTorch model
pretrained_model_torch = GPT2LMHeadModel.from_pretrained(args.model, \
        cache_dir=args.model_path)

# Create a new PyTorch model with adjusted config and load weights from the
# pretrained one. This step is requried as some operations of GPT2 are still
# pending in NNTile implementation (bias in Linear layers and entire Attention
# layers).
config = copy.deepcopy(pretrained_model_torch.config)
config.attn_pdrop = 0
config.embd_pdrop = 0
config.resid_pdrop = 0
config.activation_function = "tanh"
model_torch = GPT2LMHeadModel(config)
# Current version splits lm_head and wte parameters, shared parameters will be
# supported soon
model_torch.lm_head.weight = nn.Parameter(pretrained_model_torch.lm_head \
        .weight.detach().clone())
model_torch.transformer.wte.weight = pretrained_model_torch.transformer \
        .wte.weight
model_torch.transformer.wpe.weight = pretrained_model_torch.transformer \
        .wpe.weight
model_torch.transformer.ln_f.weight = pretrained_model_torch.transformer \
        .ln_f.weight
model_torch.transformer.ln_f.bias = pretrained_model_torch.transformer \
        .ln_f.bias

# Identity module that simply outputs input without changes to substitue
# Attention
class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            output_attentions=None,):
        return x, None, None

# Linear layer without bias
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and
    also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = torch.zeros((), device=args.device)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

# GPT2 MLP block with Linear layers without bias
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) \
            -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# Prepare PyTorch model: remove attention and use MLP without bias
identity = IdentityModule()
inner_dim = config.n_inner if config.n_inner is not None \
        else 4 * config.hidden_size
for h_idx in range(config.num_hidden_layers):
    model_torch.transformer.h[h_idx].attn = identity
    model_torch.transformer.h[h_idx].mlp = GPT2MLP(inner_dim, config)

# Print altered PyTorch model to be tested
print("PyTorch model:")
print(model_torch)

# Run forward pass with random input once as a warmup and then measure
# throughput
input_value = torch.randint(config.vocab_size, \
        (args.batch_size, args.seq_len), dtype=torch.int64, device=args.device)
model_torch = model_torch.to(args.device)
output_value = model_torch(input_value)
time0 = time.time()
for i in range(args.nforward):
    output_value = model_torch(input_value)
if args.device == "cuda":
    torch.cuda.synchronize()
time1 = time.time() - time0
print("Torch forward throughput (sequence/sec): ", \
        args.nforward * args.batch_size / time1)

# Run backward pass with random gradient of output once as a warmup and then
# measure throughput while retaining the backward graph
output_grad = torch.randn_like(output_value.logits, device=args.device)
loss = (output_value.logits * output_grad).sum()
loss.backward(retain_graph=True)
time0 = time.time()
for i in range(args.nbackward):
    loss.backward(retain_graph=True)
if args.device == "cuda":
    torch.cuda.synchronize()
time1 = time.time() - time0
print("Torch backward throughput (sequence/sec): ", \
        args.nbackward * args.batch_size / time1)

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1))
next_tag = 0

# Prepare GPT2 model based on the NNTile backend
nntile_model, next_tag = GPT2.from_torch(model_torch, args.batch_size, \
        args.seq_len, config, next_tag)

# Measure throughput of the forward pass by NNTile with a single run as a
# warmup
nntile_model.activations[0].value.from_array(input_value.cpu().numpy())
nntile_model.forward_async()
nntile.starpu.wait_for_all()
time0 = time.time()
for i in range(args.nforward):
    nntile_model.forward_async()
nntile.starpu.wait_for_all()
time1 = time.time() - time0
print("NNTile forward throughput (sequence/sec): ", \
        args.nforward * args.batch_size / time1)

nntile_model.activations[-1].grad.from_array(output_grad.cpu().numpy())
nntile_model.clear_gradients()
nntile_model.backward_async()
nntile.starpu.wait_for_all()
time0 = time.time()
for i in range(args.nbackward):
    nntile_model.clear_gradients()
    nntile_model.backward_async()
nntile.starpu.wait_for_all()
time1 = time.time() - time0
print("NNTile backward throughput (sequence/sec): ", \
        args.nbackward * args.batch_size / time1)

# Check backward via Frob (MSE) loss
#nntile_model.clear_gradients()
#
#fro_loss, next_tag = Frob.generate_simple(nntile_model.activations[-1], \
#        next_tag)
#fro_loss.y.from_array(np.zeros((1, seq_len, config.vocab_size), order="F", \
#        dtype=np.float32))
## fro_loss.y.from_array(np.array(trial_true_output.detach().numpy(), \
##        order="F", dtype=np.float32))
#fro_loss.calc_async()
#
#nntile_model.backward_async()
#
## Describe GPT2 neural network
#tokenizer = GPT2Tokenizer.from_pretrained(args.model)
#
## Read dataset
#if dataset == "WikiText-103":
#    train_dataset = load_dataset("wikitext", "wikitext-103-v1", \
#            split='train', cache_dir=dataset_path).select(subdataset)
#    test_dataset = load_dataset("wikitext", "wikitext-103-v1", \
#            split='test', cache_dir=dataset_path).select(subdataset)
#else:
#    raise ValueError("{} dataset is not supported yet!".format(dataset))
#
## Tokenize and store as a single numpy array
#map_train_tokens = map(lambda x: tokenizer(x["text"])["input_ids"], \
#        train_dataset)
#list_train_tokens = []
#for seq in map_train_tokens:
#    list_train_tokens.extend(seq)
#num_train_tokens = len(list_train_tokens)
#num_train_seq = num_train_tokens // (seq_len+1)
#num_train_batches = num_train_seq // batch_size
#num_train_tokens_truncated = num_train_batches * batch_size * (seq_len+1)
#train_tokens = np.array(list_train_tokens[:num_train_tokens_truncated], \
#        order='F', dtype=np.int64)
#train_tokens = train_tokens.reshape(num_train_batches, batch_size, seq_len+1)
#print("Number of train sequences: {}".format(num_train_batches * batch_size))
#print("Number of train batches: {}".format(num_train_batches))

#
#
#val_np = np.zeros((1,), order="F", dtype=np.float32)
#fro_loss.val.to_array(val_np)
#print("NNTile loss = {}".format(val_np[0]))
#print("Relative difference between PyTorch and NNTile losses = {}".format(
#    abs(val_np[0] - torch_loss.item()) / torch_loss.item()))
#
#for i, (p_nntile, (name, p_torch)) in enumerate(zip(nntile_model.parameters, \
#        model_torch.named_parameters())):
#    p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", \
#            dtype=np.float32)
#    p_nntile.grad.to_array(p_nntile_grad_np)
#    layer_type = name.split(".")[-2]
#    if len(p_nntile.grad.shape) == 1 or layer_type in ("c_proj", "c_fc"):
#        rel_error = torch.norm(p_torch.grad \
#                - torch.from_numpy(p_nntile_grad_np)) \
#                / torch.norm(p_torch.grad)
#    elif len(p_nntile.grad.shape) == 2:
#        rel_error = torch.norm(p_torch.grad \
#                - torch.from_numpy(p_nntile_grad_np).T) \
#                / torch.norm(p_torch.grad)
#    print("Relative error in gradient in parameter {} = {}".format(i, \
#            rel_error.item()))
#
#p_nntile_grad_np = np.zeros(nntile_model.parameters[-1].grad.shape, \
#        order="F", dtype=np.float32)
#nntile_model.parameters[-1].grad.to_array(p_nntile_grad_np)
#rel_error = torch.norm(model_torch.lm_head.weight.grad \
#        - torch.from_numpy(p_nntile_grad_np).T) \
#        / torch.norm(model_torch.lm_head.weight.grad)
#print("Relative error in gradient in lm_head = {}".format(rel_error.item()))
#
## Wait for all scatters to finish
#nntile.starpu.wait_for_all()
#time0 += time.time()
#print("From PyTorch loader to NNTile batches in {} seconds".format(time0))
#
## Set up learning rate and optimizer for training
## optimizer = nntile.optimizer.SGD(nntile_model.get_parameters(), lr, \
##        next_tag, momentum=0.9, nesterov=False, weight_decay=0.)
#optimizer = nntile.optimizer.Adam(nntile_model.get_parameters(), lr, next_tag)
#next_tag = optimizer.get_next_tag()
#
## Define Cross Entropy loss function
#loss, next_tag = nntile.loss.CrossEntropy.generate_simple( \
#        nntile_model.activations[-1], next_tag)
#
## Set up training pipeline
#pipeline = nntile.pipeline.Pipeline(batch_input, batch_output, nntile_model, \
#        optimizer, loss, n_epochs)
#
## Compute test accuracy of the pretrained model using the test dataset
#test_top1_accuracy = 0
#total_num_samples = 0
#z_single_distr = [0]
#z_single = nntile.tensor.Tensor_fp32(x_single_traits, z_single_distr, next_tag)
#next_tag = z_single.next_tag
#for test_batch_data, test_batch_label in test_loader:
#    x_single.from_array(test_batch_data.view(-1, n_pixels).numpy())
#    nntile.tensor.scatter_async(x_single, m.activations[0].value)
#    m.forward_async()
#    nntile.tensor.gather_async(m.activations[-1].value, z_single)
#    output = np.zeros(z_single.shape, order="F", dtype=np.float32)
#    # to_array causes y_single to finish gather procedure
#    z_single.to_array(output)
#    pred_labels = np.argmax(output, 1)
#    test_top1_accuracy += np.sum(pred_labels == test_batch_label.numpy())
#    total_num_samples += test_batch_label.shape[0]
#test_top1_accuracy /= total_num_samples
#
#print("Test accuracy of the pretrained GPT model =", test_top1_accuracy)
#
## Prepare input and output batches for training by NNTile
#time0 = -time.time()
#batch_input = []
#batch_output = []
#x_single_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
#        [batch_size, seq_len])
#x_single_distr = [0]
#x_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
#        next_tag)
#next_tag = x_single.next_tag
#y_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
#        next_tag)
#next_tag = y_single.next_tag
#x_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
#        [batch_size_tile, seq_len_tile])
#x_distr = [0] * x_traits.grid.nelems
#for i in range(num_batches):
#    x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
#    next_tag = x.next_tag
#    x_single.from_array(tokens[i, :, :-1])
#    nntile.tensor.scatter_async(x_single, x)
#    batch_input.append(x)
#    y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
#    next_tag = y.next_tag
#    y_single.from_array(tokens[i, :, 1:])
#    nntile.tensor.scatter_async(y_single, y)
#    batch_output.append(y)
#
## Unregister single-tile tensors for data scattering/gathering
#x_single.unregister()
#y_single.unregister()

# Unregister all tensors related to model
nntile_model.unregister()

## Unregister optimizer states
#optimizer.unregister()
#
## Unregister loss function
#fro_loss.unregister()
#loss.unregister()
#
## Unregister input/output batches
#for x in batch_input:
#    x.unregister()
#for x in batch_output:
#    x.unregister()
#
