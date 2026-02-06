# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/gpt2_attention.py
# GPT2Attention submodule of NNTile Python package
#
# @version 1.1.0

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention as GPT2Attention_torch, GPT2Config as GPT2ConfigTorch)

import nntile.utils.constructors as nntc
from nntile.functions import (
    add_fiber_inplace_async, copy_intersection_async, sum_fiber_async,
    transpose_async)
from nntile.tensor import (
    Tensor_bool, TensorMoments, TensorTraits, notrans, to_numpy)

from ..layer.cache_utils import KVCache
from ..layer.linear import Linear
from ..layer.sdpa import Sdpa
from .base_model import BaseModel
from .gpt2_config import GPT2ConfigNNTile


class GPT2Attention(BaseModel):
    """
    GPT-2 self-attention with SDPA (vanilla/flash).
    """

    def __init__(self, x: TensorMoments, config: GPT2ConfigNNTile):
        self.config = config
        self.flash_attention = bool(config.flash_attention)
        self.redux = 1 if config.redux else 0

        # Model shapes
        hidden_size, seq_len, n_batch = x.value.shape
        hidden_size_tile, seq_len_tile, n_batch_tile = x.value.basetile_shape
        n_head = config.n_head
        n_head_tile = config.n_head_tile
        head_size = hidden_size // n_head
        if hidden_size != head_size * n_head:
            raise ValueError("hidden_size must be divisible by n_head")
        head_size_tile = max(1, hidden_size_tile // n_head_tile)
        if head_size % head_size_tile != 0:
            raise ValueError("head_size must be divisible by head_size_tile")

        self.n_head = n_head
        self.n_head_tile = n_head_tile
        self.head_size = head_size
        self.head_size_tile = head_size_tile
        if x.grad is not None:
            x.grad.set_reduction_add()

        # Create mask for SDPA
        mask_shape = [seq_len, seq_len]
        mask_basetile = [seq_len_tile, seq_len_tile]
        mask_traits = TensorTraits(mask_shape, mask_basetile)
        mask_distr = [0] * mask_traits.grid.nelems
        if self.flash_attention:
            mask_type = type(x.value)
            mask_values = np.tril(
                np.float32("-inf") * np.ones((seq_len, seq_len),
                                             dtype=np.float32, order="F"),
                -1,
            )
            mask_tensor = mask_type(mask_traits, mask_distr)
            mask_tensor.from_array(mask_values)
        else:
            mask_type = Tensor_bool
            mask_values = np.triu(np.ones((seq_len, seq_len)), k=0)
            mask_tensor = mask_type(mask_traits, mask_distr)
            mask_tensor.from_array(
                np.array(mask_values, dtype=bool, order="F")
            )

        # Projections for Q/K/V: shape [1, n_head, head_size, seq, batch]
        out_shape_qkv = [1, n_head, head_size]
        out_tile_qkv = [1, n_head_tile, head_size_tile]
        gemm_ndim = 1
        self.q_proj = Linear.generate_simple(
            x, "R", notrans, gemm_ndim, out_shape_qkv, out_tile_qkv,
            bias=False, redux=config.redux
        )
        self.k_proj = Linear.generate_simple(
            x, "R", notrans, gemm_ndim, out_shape_qkv, out_tile_qkv,
            bias=False, redux=config.redux
        )
        self.v_proj = Linear.generate_simple(
            x, "R", notrans, gemm_ndim, out_shape_qkv, out_tile_qkv,
            bias=False, redux=config.redux
        )
        self.q_proj_out = self.q_proj.activations_output[0]
        self.k_proj_out = self.k_proj.activations_output[0]
        self.v_proj_out = self.v_proj.activations_output[0]
        self.q_proj_out.grad.set_reduction_add()
        self.k_proj_out.grad.set_reduction_add()
        self.v_proj_out.grad.set_reduction_add()

        # Biases for Q/K/V shaped [head_size, n_head]
        bias_traits = TensorTraits(
            [head_size, n_head], [head_size_tile, n_head_tile]
        )
        bias_distr = [0] * bias_traits.grid.nelems
        bias_type = type(x.value)
        self.q_bias = TensorMoments(
            bias_type(bias_traits, bias_distr),
            bias_type(bias_traits, bias_distr),
            True,
        )
        self.k_bias = TensorMoments(
            bias_type(bias_traits, bias_distr),
            bias_type(bias_traits, bias_distr),
            True,
        )
        self.v_bias = TensorMoments(
            bias_type(bias_traits, bias_distr),
            bias_type(bias_traits, bias_distr),
            True,
        )
        self.q_bias.grad.set_reduction_add()
        self.k_bias.grad.set_reduction_add()
        self.v_bias.grad.set_reduction_add()

        # Tensors for SDPA input: [head_size, seq, batch, kv_group(=1), n_head]
        qkv_shape = [head_size, seq_len, n_batch, 1, n_head]
        qkv_basetile = [head_size_tile, seq_len_tile, n_batch_tile, 1,
                        n_head_tile]
        qkv_traits = TensorTraits(qkv_shape, qkv_basetile)
        qkv_distr = [0] * qkv_traits.grid.nelems
        tensor_type = type(x.value)
        self.q = TensorMoments(
            tensor_type(qkv_traits, qkv_distr),
            tensor_type(qkv_traits, qkv_distr),
            True,
        )
        self.k = TensorMoments(
            tensor_type(qkv_traits, qkv_distr),
            tensor_type(qkv_traits, qkv_distr),
            True,
        )
        self.v = TensorMoments(
            tensor_type(qkv_traits, qkv_distr),
            tensor_type(qkv_traits, qkv_distr),
            True,
        )
        self.q.grad.set_reduction_add()
        self.k.grad.set_reduction_add()
        self.v.grad.set_reduction_add()

        # Causal mask
        mask_traits = TensorTraits([seq_len, seq_len],
                                   [seq_len_tile, seq_len_tile])
        mask_distr = [0] * mask_traits.grid.nelems
        if self.flash_attention:
            mask_type = tensor_type
            mask_values = np.tril(
                np.float32("-inf") * np.ones((seq_len, seq_len),
                                             dtype=np.float32, order="F"),
                -1,
            )
            mask_tensor = mask_type(mask_traits, mask_distr)
            mask_tensor.from_array(mask_values)
        else:
            mask_type = Tensor_bool
            mask_values = np.triu(np.ones((seq_len, seq_len)), k=0)
            mask_tensor = mask_type(mask_traits, mask_distr)
            mask_tensor.from_array(
                np.array(mask_values, dtype=bool, order="F")
            )

        # SDPA
        self.sdpa = Sdpa.generate_simple(
            self.q,
            self.k,
            self.v,
            mask=mask_tensor,
            flash_attention=self.flash_attention,
            redux=config.redux,
        )
        self.context = self.sdpa.activations_output[0]

        # Transposed context for output proj: [1, n_head, head, seq, batch]
        ctx_tr_shape = [1, n_head, head_size, seq_len, n_batch]
        ctx_tr_basetile = [1, n_head_tile, head_size_tile,
                           seq_len_tile, n_batch_tile]
        ctx_tr_traits = TensorTraits(ctx_tr_shape, ctx_tr_basetile)
        ctx_tr_distr = [0] * ctx_tr_traits.grid.nelems
        self.context_transposed = TensorMoments(
            tensor_type(ctx_tr_traits, ctx_tr_distr),
            tensor_type(ctx_tr_traits, ctx_tr_distr),
            True,
        )
        self.context_transposed.grad.set_reduction_add()

        # Output projection: input features ndim=3 -> [hidden_size, seq, batch]
        self.out_proj = Linear.generate_simple(
            self.context_transposed,
            "R",
            notrans,
            3,
            [hidden_size],
            [hidden_size_tile],
            bias=True,
            redux=config.redux,
        )
        self.output = self.out_proj.activations_output[0]
        if self.output.grad is not None:
            self.output.grad.set_reduction_add()

        activations = [
            x,
            self.q_proj_out,
            self.k_proj_out,
            self.v_proj_out,
            self.q,
            self.k,
            self.v,
            self.context,
            self.context_transposed,
            self.output,
        ]
        layers = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.sdpa,
            self.out_proj,
        ]
        super().__init__(activations, layers, config)
        # Preserve BaseLayer-like API expected by GPT2Block
        self.activations_output = [self.output]
        # Add biases to parameter list manually (projections have bias=False)
        self.parameters.extend([self.q_bias, self.k_bias, self.v_bias])

        # Allow shared input gradient accumulation
        if x.grad is not None:
            x.grad.set_reduction_add()

    def forward_async(self):
        # Q/K/V projections
        self.q_proj.forward_async()
        self.k_proj.forward_async()
        self.v_proj.forward_async()

        # Reorder to SDPA layout and add biases
        transpose_async(1.0, self.q_proj_out.value, self.q.value, 2)
        add_fiber_inplace_async(1.0, self.q_bias.value, 1.0,
                                self.q.value, 0, 1)

        transpose_async(1.0, self.k_proj_out.value, self.k.value, 2)
        add_fiber_inplace_async(1.0, self.k_bias.value, 1.0,
                                self.k.value, 0, 1)

        transpose_async(1.0, self.v_proj_out.value, self.v.value, 2)
        add_fiber_inplace_async(1.0, self.v_bias.value, 1.0,
                                self.v.value, 0, 1)

        # SDPA and output projection
        self.sdpa.forward_async()
        transpose_async(1.0, self.context.value,
                        self.context_transposed.value, 3)
        self.out_proj.forward_async()

    def forward_dynamic(
        self,
        x: TensorMoments,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[TensorMoments, Optional[KVCache]]:
        tensor_type = type(x.value)
        _, seq_len, batch_size = x.value.shape
        _, seq_len_tile, batch_tile = x.value.basetile_shape

        # Q/K/V projections
        q_proj_out = self.q_proj.forward_dynamic(x)
        k_proj_out = self.k_proj.forward_dynamic(x)
        v_proj_out = self.v_proj.forward_dynamic(x)

        # Allocate temporary tensors for SDPA layout
        qkv_shape = [self.head_size, seq_len, batch_size, 1, self.n_head]
        qkv_basetile = [
            self.head_size_tile, seq_len_tile, batch_tile, 1, self.n_head_tile
        ]
        qkv_traits = TensorTraits(qkv_shape, qkv_basetile)
        qkv_distr = [0] * qkv_traits.grid.nelems

        q = TensorMoments(tensor_type(qkv_traits, qkv_distr), None, False)
        k = TensorMoments(tensor_type(qkv_traits, qkv_distr), None, False)
        v = TensorMoments(tensor_type(qkv_traits, qkv_distr), None, False)

        # Transpose to SDPA layout and add biases
        transpose_async(1.0, q_proj_out.value, q.value, 2)
        add_fiber_inplace_async(1.0, self.q_bias.value, 1.0, q.value, 0, 1)

        transpose_async(1.0, k_proj_out.value, k.value, 2)
        add_fiber_inplace_async(1.0, self.k_bias.value, 1.0, k.value, 0, 1)

        transpose_async(1.0, v_proj_out.value, v.value, 2)
        add_fiber_inplace_async(1.0, self.v_bias.value, 1.0, v.value, 0, 1)

        # Handle KV cache for incremental decoding
        kv_size_before = len(kv_cache) if kv_cache is not None else 0
        k_for_attn = k.value
        v_for_attn = v

        # Handle KV cache - always append current K/V first,
        # then use cache for attention
        if kv_cache is not None:
            kv_cache.append(k.value, v.value)

        kv_size_after = len(kv_cache) if kv_cache is not None else 0

        # If we have cached data, use it for attention
        if kv_cache is not None and kv_size_after > seq_len:
            k_for_attn = kv_cache.k_partial
            v_for_attn = kv_cache.v_partial
        else:
            k_for_attn = k.value
            v_for_attn = v

        # For GPT-2, use flash attention when possible
        # but handle masks correctly
        original_flash = self.sdpa.flash_attention
        q_seq = q.value.shape[1]
        q_tile = q.value.basetile_shape[1]
        q_seq_tiles = (
            q.value.grid.shape[1]
            if hasattr(q.value, "grid")
            else None
        )
        flash_active = (
            self.flash_attention
            and kv_size_before == 0
            and (q_seq % q_tile == 0 or q_seq_tiles == 1)
        )
        mask_tmp_allocated = False

        # Create mask for SDPA
        mask_arg = self.sdpa.mask
        if mask_arg is not None:
            if flash_active:
                # For flash attention, create dynamic mask
                # if sequence length doesn't match
                k_seq = k_for_attn.shape[1]
                q_seq = q.value.shape[1]
                mask_bt = (
                    min(mask_arg.basetile_shape[0], k_seq),
                    min(mask_arg.basetile_shape[1], q_seq),
                )
                if (
                    tuple(mask_arg.shape) != (k_seq, q_seq)
                    or tuple(mask_arg.basetile_shape) != mask_bt
                ):
                    mask_tmp_allocated = True
                    mask_tmp = nntc.zeros(
                        (k_seq, q_seq),
                        basetile_shape=mask_bt,
                        dtype=type(mask_arg),
                    )
                    copy_intersection_async(
                        mask_arg,
                        [0, 0],
                        mask_tmp,
                        [0, max(k_seq - q_seq, 0)],
                    )
                    mask_arg = mask_tmp
            else:
                # For vanilla attention, prepare mask for KV cache
                if kv_size_before > 0:
                    mask_arg, mask_tmp_allocated = (
                        self.sdpa._prepare_mask_for_kv_cache(
                            mask_arg,
                            k_for_attn.shape[1],
                            q.value.shape[1],
                            k_for_attn.basetile_shape[1],
                            q.value.basetile_shape[1],
                        )
                    )

        # SDPA - use cached K/V if available
        if kv_cache is not None and kv_size_after > seq_len:
            # Using cached data - wrap as TensorMoments
            k_tensor = TensorMoments(k_for_attn, None, False)
            v_tensor = TensorMoments(v_for_attn, None, False)
        else:
            # Using current data
            k_tensor = k
            v_tensor = v

        self.sdpa.flash_attention = flash_active
        try:
            sdpa_out = self.sdpa.forward_dynamic(
                q, k_tensor, v_tensor, mask=mask_arg
            )
        finally:
            self.sdpa.flash_attention = original_flash

        # Transpose context for output projection
        ctx_tr_shape = [1, self.n_head, self.head_size, seq_len, batch_size]
        ctx_tr_basetile = [
            1, self.n_head_tile, self.head_size_tile, seq_len_tile, batch_tile
        ]
        ctx_tr_traits = TensorTraits(ctx_tr_shape, ctx_tr_basetile)
        ctx_tr_distr = [0] * ctx_tr_traits.grid.nelems
        context_transposed = TensorMoments(
            tensor_type(ctx_tr_traits, ctx_tr_distr), None, False
        )
        transpose_async(1.0, sdpa_out.value, context_transposed.value, 3)

        # Output projection
        out = self.out_proj.forward_dynamic(context_transposed)

        # Invalidate temporaries
        q.value.invalidate_submit()
        k.value.invalidate_submit()
        v.value.invalidate_submit()
        sdpa_out.value.invalidate_submit()
        context_transposed.value.invalidate_submit()
        if mask_tmp_allocated:
            mask_arg.invalidate_submit()

        # Invalidate cached tensors if they were used
        if kv_cache is not None and kv_size_before > 0:
            kv_cache.k_partial.invalidate_submit()
            kv_cache.v_partial.invalidate_submit()

        return out, kv_cache

    def backward_async(self):
        # Output projection backward
        self.out_proj.backward_async()
        transpose_async(1.0, self.context_transposed.grad,
                        self.context.grad, 2)

        # SDPA backward
        self.sdpa.backward_async()

        # Bias gradients
        sum_fiber_async(1.0, self.q.grad, 1.0,
                        self.q_bias.grad, 0, 1, redux=self.redux)
        sum_fiber_async(1.0, self.k.grad, 1.0,
                        self.k_bias.grad, 0, 1, redux=self.redux)
        sum_fiber_async(1.0, self.v.grad, 1.0,
                        self.v_bias.grad, 0, 1, redux=self.redux)

        # Propagate through transposes to projections
        transpose_async(1.0, self.q.grad, self.q_proj_out.grad, 3)
        transpose_async(1.0, self.k.grad, self.k_proj_out.grad, 3)
        transpose_async(1.0, self.v.grad, self.v_proj_out.grad, 3)

        # Q/K/V projections backward
        self.v_proj.backward_async()
        self.k_proj.backward_async()
        self.q_proj.backward_async()

    @classmethod
    def from_torch(
        cls,
        torch_layer: GPT2Attention_torch,
        x: TensorMoments,
        config: GPT2ConfigNNTile,
    ) -> "GPT2Attention":
        layer = cls(x, config)
        n_emb = config.hidden_size
        head_size = layer.head_size

        # Split combined QKV weights/biases
        weight_torch_np = torch_layer.c_attn.weight.detach().cpu().numpy()
        bias_torch_np = torch_layer.c_attn.bias.detach().cpu().numpy()

        w_q_np = weight_torch_np[:, 0:n_emb].T.reshape(
            1, layer.n_head, head_size, n_emb
        )
        w_k_np = weight_torch_np[:, n_emb:2 * n_emb].T.reshape(
            1, layer.n_head, head_size, n_emb
        )
        w_v_np = weight_torch_np[:, 2 * n_emb:3 * n_emb].T.reshape(
            1, layer.n_head, head_size, n_emb
        )
        layer.q_proj.parameters[0].value.from_array(w_q_np)
        layer.k_proj.parameters[0].value.from_array(w_k_np)
        layer.v_proj.parameters[0].value.from_array(w_v_np)

        b_q_np = bias_torch_np[0:n_emb].reshape(
            layer.n_head, head_size
        ).T
        b_k_np = bias_torch_np[n_emb:2 * n_emb].reshape(
            layer.n_head, head_size
        ).T
        b_v_np = bias_torch_np[2 * n_emb:3 * n_emb].reshape(
            layer.n_head, head_size
        ).T
        layer.q_bias.value.from_array(b_q_np)
        layer.k_bias.value.from_array(b_k_np)
        layer.v_bias.value.from_array(b_v_np)

        # Output projection
        w_out_np = (
            torch_layer.c_proj.weight.detach().cpu().numpy().T.reshape(
                n_emb, layer.n_head, head_size
            )
        ).reshape(n_emb, 1, layer.n_head, head_size)
        layer.out_proj.parameters[0].value.from_array(w_out_np)
        layer.out_proj.parameters[1].value.from_array(
            torch_layer.c_proj.bias.detach().cpu().numpy()
        )
        return layer

    def to_torch(self) -> GPT2Attention_torch:
        config_torch = GPT2ConfigTorch(
            n_embd=self.config.hidden_size,
            n_head=self.config.n_head,
            use_cache=False,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            scale_attn_weights=True,
        )
        torch_layer = GPT2Attention_torch(
            config_torch, is_cross_attention=False, layer_idx=0
        )

        n_emb = self.config.hidden_size
        weight_torch_np = np.empty((n_emb, 3 * n_emb))
        weight_torch_np[:, :n_emb] = to_numpy(
            self.q_proj.parameters[0].value
        ).reshape(n_emb, n_emb).T
        weight_torch_np[:, n_emb:2 * n_emb] = to_numpy(
            self.k_proj.parameters[0].value
        ).reshape(n_emb, n_emb).T
        weight_torch_np[:, 2 * n_emb:3 * n_emb] = to_numpy(
            self.v_proj.parameters[0].value
        ).reshape(n_emb, n_emb).T
        torch_layer.c_attn.weight.data = torch.tensor(
            weight_torch_np, requires_grad=True
        )

        bias_torch_np = np.empty((3 * n_emb,))
        bias_torch_np[:n_emb] = to_numpy(self.q_bias.value).T.reshape(
            n_emb,
        )
        bias_torch_np[n_emb:2 * n_emb] = to_numpy(self.k_bias.value).T.reshape(
            n_emb,
        )
        bias_torch_np[2 * n_emb:3 * n_emb] = to_numpy(
            self.v_bias.value
        ).T.reshape(
            n_emb,
        )
        torch_layer.c_attn.bias.data = torch.tensor(
            bias_torch_np, requires_grad=True
        )

        torch_layer.c_proj.weight.data = torch.tensor(
            to_numpy(self.out_proj.parameters[0].value).reshape(
                n_emb, n_emb
            ).T,
            requires_grad=True,
        )
        torch_layer.c_proj.bias.data = torch.tensor(
            to_numpy(self.out_proj.parameters[1].value), requires_grad=True
        )
        return torch_layer

    def to_torch_with_grads(self) -> GPT2Attention_torch:
        torch_layer = self.to_torch()
        n_emb = self.config.hidden_size

        weight_torch_np = np.empty((n_emb, 3 * n_emb))
        weight_torch_np[:, :n_emb] = to_numpy(
            self.q_proj.parameters[0].grad
        ).reshape(n_emb, n_emb).T
        weight_torch_np[:, n_emb:2 * n_emb] = to_numpy(
            self.k_proj.parameters[0].grad
        ).reshape(n_emb, n_emb).T
        weight_torch_np[:, 2 * n_emb:3 * n_emb] = to_numpy(
            self.v_proj.parameters[0].grad
        ).reshape(n_emb, n_emb).T
        torch_layer.c_attn.weight.grad = torch.tensor(weight_torch_np)

        bias_torch_np = np.empty((3 * n_emb,))
        bias_torch_np[:n_emb] = to_numpy(self.q_bias.grad).T.reshape(n_emb,)
        bias_torch_np[n_emb:2 * n_emb] = to_numpy(self.k_bias.grad).T.reshape(
            n_emb,
        )
        bias_torch_np[2 * n_emb:3 * n_emb] = to_numpy(self.v_bias.grad).T \
        .reshape(
            n_emb,
        )
        torch_layer.c_attn.bias.grad = torch.tensor(bias_torch_np)

        torch_layer.c_proj.weight.grad = torch.tensor(
            to_numpy(self.out_proj.parameters[0].grad).reshape(
                n_emb, n_emb
            ).T
        )
        torch_layer.c_proj.bias.grad = torch.tensor(
            to_numpy(self.out_proj.parameters[1].grad)
        )
        return torch_layer

    def unregister(self):
        # Manually owned biases are not tied to any BaseLayer,
        # so drop them here
        super().unregister()
        for bias in (self.q_bias, self.k_bias, self.v_bias):
            bias.unregister()

    def get_forward_flops(self):
        n_emb = self.head_size * self.n_head
        n_seq = self.activations[0].value.shape[1]
        n_batch = self.activations[0].value.shape[2]
        proj_flops = 3 * 2 * n_emb * n_emb * n_seq * n_batch
        attn_scores = 2 * self.head_size * n_seq * n_seq * self.n_head \
            * n_batch
        attn_ctx = 2 * self.head_size * n_seq * n_seq * self.n_head * n_batch
        out_proj = 2 * n_emb * n_emb * n_seq * n_batch
        return proj_flops + attn_scores + attn_ctx + out_proj

    def get_backward_flops(self):
        total_backward_flops = 0
        # Add backward flops from all linear layers
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            total_backward_flops += layer.get_backward_flops()
        # SDPA layer returns 0 for backward flops
        total_backward_flops += self.sdpa.get_backward_flops()
        return total_backward_flops
