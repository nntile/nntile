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
# @date 2023-06-21

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
from typing import Union, Optional, Tuple, List
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
parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0", "cuda:1"], \
        default="cpu")
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
#config.n_head = 2
#config.num_hidden_layers = 3
model_torch = GPT2LMHeadModel(config)
# Current version splits lm_head and wte parameters, shared parameters will be
# supported soon
model_torch.lm_head.weight = nn.Parameter(pretrained_model_torch.lm_head \
        .weight.detach().clone())
model_torch.transformer.wte.weight = nn.Parameter(pretrained_model_torch \
        .transformer.wte.weight.detach().clone())
model_torch.transformer.wpe.weight = nn.Parameter(pretrained_model_torch \
        .transformer.wpe.weight.detach().clone())
model_torch.transformer.ln_f.weight = nn.Parameter(pretrained_model_torch \
        .transformer.ln_f.weight.detach().clone())
model_torch.transformer.ln_f.bias = nn.Parameter(pretrained_model_torch \
        .transformer.ln_f.bias.detach().clone())

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

# Prepare PyTorch model: use MLP without bias and unmasked attention
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        #     persistent=False,
        # )
        self.register_buffer(
            "bias",
            torch.ones((max_positions, max_positions), dtype=torch.bool).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # print('Scale attn weights',  config.scale_attn_weights)
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
    #     index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

    #     # Prune conv1d layers
    #     self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    #     self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

    #     # Update hyper params
    #     self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
    #     self.num_heads = self.num_heads - len(heads)
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
    #     # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
    #     bsz, num_heads, q_seq_len, dk = query.size()
    #     _, _, k_seq_len, _ = key.size()

    #     # Preallocate attn_weights for `baddbmm`
    #     attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

    #     # Compute Scale Factor
    #     scale_factor = 1.0
    #     if self.scale_attn_weights:
    #         scale_factor /= float(value.size(-1)) ** 0.5

    #     if self.scale_attn_by_inverse_layer_idx:
    #         scale_factor /= float(self.layer_idx + 1)

    #     # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
    #     with autocast(enabled=False):
    #         q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
    #         attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
    #         attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

    #     if not self.is_cross_attention:
    #         # if only "normal" attention layer implements causal mask
    #         query_length, key_length = query.size(-2), key.size(-2)
    #         causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    #         mask_value = torch.finfo(attn_weights.dtype).min
    #         # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    #         # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    #         mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    #         attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    #     if attention_mask is not None:
    #         # Apply the attention mask
    #         attn_weights = attn_weights + attention_mask

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    #     # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
    #     if attn_weights.dtype != torch.float32:
    #         raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
    #     attn_weights = attn_weights.type(value.dtype)
    #     attn_weights = self.attn_dropout(attn_weights)

    #     # Mask heads if we want to
    #     if head_mask is not None:
    #         attn_weights = attn_weights * head_mask

    #     attn_output = torch.matmul(attn_weights, value)

    #     return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
inner_dim = config.n_inner if config.n_inner is not None \
        else 4 * config.hidden_size

for h_idx in range(config.num_hidden_layers):
    model_torch.transformer.h[h_idx].ln_1.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].ln_1.weight.detach().clone())
    model_torch.transformer.h[h_idx].ln_1.bias = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].ln_1.bias.detach().clone())
    model_torch.transformer.h[h_idx].ln_2.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].ln_2.weight.detach().clone())
    model_torch.transformer.h[h_idx].ln_2.bias = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].ln_2.bias.detach().clone())
    model_torch.transformer.h[h_idx].attn = GPT2Attention(config)
    model_torch.transformer.h[h_idx].attn.c_attn.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].attn.c_attn.weight.detach().clone())
    model_torch.transformer.h[h_idx].attn.c_proj.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].attn.c_proj.weight.detach().clone())
    model_torch.transformer.h[h_idx].mlp = GPT2MLP(inner_dim, config)
    model_torch.transformer.h[h_idx].mlp.c_fc.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].mlp.c_fc.weight.detach().clone())
    model_torch.transformer.h[h_idx].mlp.c_proj.weight = nn.Parameter(pretrained_model_torch.transformer.h[h_idx].mlp.c_proj.weight.detach().clone())

# Print altered PyTorch model to be tested
print("PyTorch model:")
print(model_torch)

# Get output from a random input through the forward pass
input_value = torch.randint(config.vocab_size, \
        (args.batch_size, args.seq_len), dtype=torch.int64, device=args.device)
model_torch = model_torch.to(args.device)
output_value = model_torch(input_value)
output_value_np = output_value.logits.detach().cpu().numpy()

# Get gradients of parameters through the backward pass
output_grad = torch.ones_like(output_value.logits, device=args.device)
loss = (output_value.logits * output_grad).sum()
loss.backward(retain_graph=True)

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

# Check accuracy of the forward pass by the output activation
nntile_model.activations[0].value.from_array(input_value.cpu().numpy().T)
nntile_model.forward_async()
nntile.starpu.wait_for_all()
nntile_output_np = np.zeros_like(output_value_np.T, order='F')
nntile_model.activations[-1].value.to_array(nntile_output_np)
diff = np.linalg.norm(nntile_output_np.T - output_value_np)
norm = np.linalg.norm(output_value_np)
print("NNTile forward pass relative accuracy: {}".format(diff/norm))

# Run backward pass by the NNTile to get gradients of parameters
nntile_output_shape = nntile_model.activations[-1].value.shape
nntile_output_traits = nntile.tensor.TensorTraits(nntile_output_shape, \
        nntile_output_shape)
nntile_output = nntile.tensor.Tensor_fp32(nntile_output_traits, [0], next_tag)
next_tag = nntile_output.next_tag
output_grad_np = output_grad.detach().cpu().numpy().T
nntile_output.from_array(output_grad_np)
nntile_model.clear_gradients()
nntile.tensor.copy_async(nntile_output, nntile_model.activations[-1].grad)
nntile_model.backward_async()
nntile.starpu.wait_for_all()

# Now compare gradients
nntile_par_idx = 0
for name, p_torch in model_torch.named_parameters():
    p_torch_grad_np = p_torch.grad.detach().cpu().numpy()
    torch.cuda.synchronize()
    layer_name = name.split(".")[-2]
    if len(p_torch.shape) == 1 or layer_name in ("lm_head",):
        p_nntile = nntile_model.parameters[nntile_par_idx]
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", \
                dtype=np.float32)
        p_nntile.grad.to_array(p_nntile_grad_np)
        diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
        norm = np.linalg.norm(p_torch_grad_np)
        nntile_par_idx += 1
    elif layer_name == "c_attn":
        attn_head_size = config.n_embd // config.n_head
        diff = 0
        norm = np.linalg.norm(p_torch_grad_np)
        for i_head in range(3*config.n_head):
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", \
                    dtype=np.float32)
            p_nntile.grad.to_array(p_nntile_grad_np)
            current_grad_block = p_torch_grad_np[:, \
                    i_head*attn_head_size:(i_head+1)*attn_head_size]
            diff += np.linalg.norm(current_grad_block.T-p_nntile_grad_np) ** 2
            nntile_par_idx += 1
        diff = diff ** 0.5
    elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
        attn_head_size = config.n_embd // config.n_head
        diff = 0
        norm = np.linalg.norm(p_torch_grad_np)
        for i_head in range(config.n_head):
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", \
                    dtype=np.float32)
            p_nntile.grad.to_array(p_nntile_grad_np)
            current_grad_block = p_torch_grad_np[i_head*attn_head_size: \
                    (i_head+1)*attn_head_size, :]
            diff += np.linalg.norm(current_grad_block.T-p_nntile_grad_np) ** 2
            nntile_par_idx += 1
        diff = diff ** 0.5
    elif len(p_torch.shape) == 2:
        p_nntile = nntile_model.parameters[nntile_par_idx]
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", \
                dtype=np.float32)
        p_nntile.grad.to_array(p_nntile_grad_np)
        diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np.T)
        norm = np.linalg.norm(p_torch_grad_np)
        nntile_par_idx += 1
    print("Relative error of gradient of {} = {}".format(name, diff/norm))

# Measure throughput of Torch forward pass
time0 = time.time()
for i in range(args.nforward):
    output_value = model_torch(input_value)
if args.device == "cuda":
    torch.cuda.synchronize()
time1 = time.time() - time0
print("Torch forward throughput (sequence/sec): ", \
        args.nforward * args.batch_size / time1)

# Measure throughput of Torch backward pass
time0 = time.time()
for i in range(args.nbackward):
    loss.backward(retain_graph=True)
if args.device == "cuda":
    torch.cuda.synchronize()
time1 = time.time() - time0
print("Torch backward throughput (sequence/sec): ", \
        args.nbackward * args.batch_size / time1)

# Measure throughput of the forward pass by NNTile
time0 = time.time()
for i in range(args.nforward):
    nntile_model.forward_async()
nntile.starpu.wait_for_all()
time1 = time.time() - time0
print("NNTile forward throughput (sequence/sec): ", \
        args.nforward * args.batch_size / time1)

time0 = time.time()
for i in range(args.nbackward):
    nntile_model.clear_gradients()
    nntile.tensor.copy_async(nntile_output, nntile_model.activations[-1].grad)
    nntile_model.backward_async()
nntile.starpu.wait_for_all()
time1 = time.time() - time0
print("NNTile backward throughput (sequence/sec): ", \
        args.nbackward * args.batch_size / time1)

nntile_output.unregister()

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
