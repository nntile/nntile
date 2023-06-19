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
import math
import numpy as np
import time
import sys
import torch
from torch import Tensor
import torch.nn as nn
from transformers import GPT2Tokenizer, TextDataset, GPT2LMHeadModel
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
# pip3 install datasets
from datasets import load_dataset
from nntile.model.gpt2 import GPT2
from nntile.tensor import copy_async
from nntile.loss import Frob
import pdb 
from typing import Optional, Tuple, Union

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
pretrained_model_torch = GPT2LMHeadModel.from_pretrained("gpt2")
config = GPT2Config()
config.attn_pdrop = 0
config.embd_pdrop = 0
config.resid_pdrop = 0
config.n_head=1
config.num_hidden_layers = 1
model_torch = GPT2LMHeadModel(config)
# Current version splits lm_head and wte parameters, shared parameters will be supported soon
model_torch.lm_head.weight = nn.Parameter(pretrained_model_torch.lm_head.weight.detach().clone())
model_torch.transformer.wte.weight = pretrained_model_torch.transformer.wte.weight
model_torch.transformer.wpe.weight = pretrained_model_torch.transformer.wpe.weight
model_torch.transformer.ln_f.weight = pretrained_model_torch.transformer.ln_f.weight
model_torch.transformer.ln_f.bias = pretrained_model_torch.transformer.ln_f.bias

class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            output_attentions=None,):
        return x, None, None

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

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

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
    
# identity = IdentityModule()
# torch_attn = TorchMHAttention(config.n_embd, config.n_head)
inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
for h_idx in range(config.num_hidden_layers):
    model_torch.transformer.h[h_idx].attn = GPT2Attention(config)
    model_torch.transformer.h[h_idx].mlp = GPT2MLP(inner_dim, config)

vocab_size = model_torch.config.vocab_size
print(model_torch)
# input_ids = tokenizer('I enjoy walking with my cute dog', return_tensors='pt')
# print(input_ids, input_ids["input_ids"].shape)
print(tokens.shape, tokens.dtype)
# pdb.set_trace()
output = model_torch(torch.from_numpy(tokens[10, :, :-1]))
# output_logits = output.logits.detach().clone()
# trial_true_output = output_logits + torch.randn(output.logits.shape)
torch_loss = 0.5 * torch.sum(torch.square(output.logits))
torch_loss.backward()
print(output.logits.shape, torch_loss.item())
# print(output.logits.shape)

for p in model_torch.parameters():
    print(p.shape)



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
                                         config, next_tag)
copy_async(batch_input[10], nntile_model.activations[0].value) 
nntile_model.forward_async()

nntile_model.clear_gradients()

fro_loss, next_tag = Frob.generate_simple(nntile_model.activations[-1], next_tag)
fro_loss.y.from_array(np.zeros((1, seq_len, config.vocab_size), order="F", dtype=np.float32))
# fro_loss.y.from_array(np.array(trial_true_output.detach().numpy(), order="F", dtype=np.float32))
fro_loss.calc_async()

nntile_model.backward_async()

val_np = np.zeros((1,), order="F", dtype=np.float32)
fro_loss.val.to_array(val_np)
print("NNTile loss = {}".format(val_np[0]))
print("Relative difference between PyTorch and NNTile losses = {}".format(
    abs(val_np[0] - torch_loss.item()) / torch_loss.item()))

for i, (p_nntile, (name, p_torch)) in enumerate(zip(nntile_model.parameters, model_torch.named_parameters())):
    p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
    p_nntile.grad.to_array(p_nntile_grad_np)
    layer_type = name.split(".")[-2]
    if len(p_nntile.grad.shape) == 1 or layer_type in ("c_proj", "c_fc"):
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
    elif len(p_nntile.grad.shape) == 2:
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np).T) / torch.norm(p_torch.grad)
    print("Relative error in gradient in parameter {} = {}".format(i, rel_error.item()))

p_nntile_grad_np = np.zeros(nntile_model.parameters[-1].grad.shape, order="F", dtype=np.float32)
nntile_model.parameters[-1].grad.to_array(p_nntile_grad_np)
rel_error = torch.norm(model_torch.lm_head.weight.grad - torch.from_numpy(p_nntile_grad_np).T) / torch.norm(model_torch.lm_head.weight.grad)
print("Relative error in gradient in lm_head = {}".format(rel_error.item()))

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

