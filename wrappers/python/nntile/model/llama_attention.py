# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/llama_attention.py
# LlamaAttention submodule of NNTile Python package
#
# @version 1.1.0

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig as LlamaConfig_torch)

import nntile.utils.constructors as nntc

from nntile.tensor import (
    Tensor_bool, TensorMoments, TensorTraits, add_slice_inplace_async,
    copy_intersection_async, notrans, rope_async, rope_backward_async,
    sum_slice_async, to_numpy,
    transpose_async)

from ..layer.linear import Linear
from ..layer.sdpa import Sdpa
from ..layer.cache_utils import KVCache
from .base_model import BaseModel
from .llama_config import LlamaConfigNNTile


class LlamaAttention(BaseModel):
    """
    LLaMA self-attention implemented with explicit Q/K/V/O Linear projections
    and the Sdpa layer (with optional FlashAttention backend).

    Input/output shapes follow the NNTile convention:
        x: (hidden_size, seq_len, batch)
        y: (hidden_size, seq_len, batch)
    """

    def __init__(
        self,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: Optional[np.ndarray],
        config: LlamaConfigNNTile,
    ):
        self.config = config
        self.n_emb = config.hidden_size
        self.n_head = config.n_attention_head
        self.n_head_kv = config.num_key_value_heads
        if self.n_head % self.n_head_kv != 0:
            raise ValueError("n_attention_head must be divisible by "
                "num_key_value_heads")
        self.kv_group_size = self.n_head // self.n_head_kv
        self.head_size = self.n_emb // self.n_head
        if self.n_emb != self.head_size * self.n_head:
            raise ValueError("hidden_size must be divisible by "
                "n_attention_head")
        self.n_emb_kv = self.head_size * self.n_head_kv
        redux = config.redux

        n_emb_tile, n_seq_tile, n_batch_tile = x.value.basetile_shape
        head_size_tile = self.head_size
        kv_group_size_tile = self.kv_group_size
        if config.n_head_tile % self.kv_group_size != 0:
            raise ValueError("n_head_tile must be divisible by kv_group_size")
        n_head_kv_tile = config.n_head_tile // self.kv_group_size

        n_seq = x.value.shape[1]
        n_batch = x.value.shape[2]

        # Q/K/V projections
        if config.attention_bias:
            raise NotImplementedError("Bias in Q/K/V projections is "
                "not supported")
        self.q_proj = Linear.generate_simple(
            x,
            "R",
            notrans,
            1,
            [self.kv_group_size, self.n_head_kv, self.head_size],
            [kv_group_size_tile, n_head_kv_tile, head_size_tile],
            bias=False,
            redux=redux,
        )
        self.k_proj = Linear.generate_simple(
            x,
            "R",
            notrans,
            1,
            [self.n_head_kv, self.head_size],
            [n_head_kv_tile, head_size_tile],
            bias=False,
            redux=redux,
        )
        self.v_proj = Linear.generate_simple(
            x,
            "R",
            notrans,
            1,
            [self.n_head_kv, self.head_size],
            [n_head_kv_tile, head_size_tile],
            bias=False,
            redux=redux,
        )

        # Allocate intermediate tensors
        q_traits = TensorTraits(
            [
                self.head_size,
                n_seq,
                n_batch,
                self.kv_group_size,
                self.n_head_kv
            ],
            [
                head_size_tile,
                n_seq_tile,
                n_batch_tile,
                kv_group_size_tile,
                n_head_kv_tile
            ],
        )
        k_traits = TensorTraits(
            [self.head_size, n_seq, n_batch, self.n_head_kv],
            [head_size_tile, n_seq_tile, n_batch_tile, n_head_kv_tile],
        )
        k_rep_traits = TensorTraits(
            [
                self.head_size,
                n_seq,
                n_batch,
                self.kv_group_size,
                self.n_head_kv,
            ],
            [
                head_size_tile,
                n_seq_tile,
                n_batch_tile,
                kv_group_size_tile,
                n_head_kv_tile,
            ],
        )
        v_traits = TensorTraits(
            [self.head_size, n_seq, n_batch, self.n_head_kv],
            [head_size_tile, n_seq_tile, n_batch_tile, n_head_kv_tile],
        )
        v_rep_traits = TensorTraits(
            [
                self.head_size,
                n_seq,
                n_batch,
                self.kv_group_size,
                self.n_head_kv,
            ],
            [
                head_size_tile,
                n_seq_tile,
                n_batch_tile,
                kv_group_size_tile,
                n_head_kv_tile,
            ],
        )
        b_transposed_traits = TensorTraits(
            [
                self.kv_group_size,
                self.n_head_kv,
                self.head_size,
                n_seq,
                n_batch,
            ],
            [
                kv_group_size_tile,
                n_head_kv_tile,
                head_size_tile,
                n_seq_tile,
                n_batch_tile,
            ],
        )
        cos_traits = TensorTraits(
            [self.head_size // 2, n_seq, n_batch],
            [head_size_tile // 2, n_seq_tile, n_batch_tile],
        )
        sin_traits = TensorTraits(
            [self.head_size // 2, n_seq, n_batch],
            [head_size_tile // 2, n_seq_tile, n_batch_tile],
        )

        tensor_type = type(x.value)
        q_value = tensor_type(q_traits, [0] * q_traits.grid.nelems)
        q_grad = tensor_type(q_traits, [0] * q_traits.grid.nelems)
        self.q = TensorMoments(q_value, q_grad, True)

        q_rope_value = tensor_type(q_traits, [0] * q_traits.grid.nelems)
        q_rope_grad = tensor_type(q_traits, [0] * q_traits.grid.nelems)
        self.q_rope = TensorMoments(q_rope_value, q_rope_grad, True)

        k_value = tensor_type(k_traits, [0] * k_traits.grid.nelems)
        k_grad = tensor_type(k_traits, [0] * k_traits.grid.nelems)
        self.k = TensorMoments(k_value, k_grad, True)

        k_rope_value = tensor_type(k_traits, [0] * k_traits.grid.nelems)
        k_rope_grad = tensor_type(k_traits, [0] * k_traits.grid.nelems)
        self.k_rope = TensorMoments(k_rope_value, k_rope_grad, True)

        k_rep_value = tensor_type(k_rep_traits, [0] * k_rep_traits.grid.nelems)
        k_rep_grad = tensor_type(k_rep_traits, [0] * k_rep_traits.grid.nelems)
        self.k_rep = TensorMoments(k_rep_value, k_rep_grad, True)

        v_value = tensor_type(v_traits, [0] * v_traits.grid.nelems)
        v_grad = tensor_type(v_traits, [0] * v_traits.grid.nelems)
        self.v = TensorMoments(v_value, v_grad, True)

        v_rep_value = tensor_type(v_rep_traits, [0] * v_rep_traits.grid.nelems)
        v_rep_grad = tensor_type(v_rep_traits, [0] * v_rep_traits.grid.nelems)
        self.v_rep = TensorMoments(v_rep_value, v_rep_grad, True)

        self.q.value.set_reduction_add()
        self.k.value.set_reduction_add()
        self.v.value.set_reduction_add()
        self.k_rep.value.set_reduction_add()
        self.v_rep.value.set_reduction_add()
        if self.q.grad is not None:
            self.q.grad.set_reduction_add()
        if self.k.grad is not None:
            self.k.grad.set_reduction_add()
        if self.v.grad is not None:
            self.v.grad.set_reduction_add()
        if self.k_rep.grad is not None:
            self.k_rep.grad.set_reduction_add()
        if self.v_rep.grad is not None:
            self.v_rep.grad.set_reduction_add()

        # Rotary embeddings
        self.cos = tensor_type(cos_traits, [0] * cos_traits.grid.nelems)
        self.sin = tensor_type(sin_traits, [0] * sin_traits.grid.nelems)
        self._fill_sin_cos(position_ids)

        # Mask handling differs for flash/non-flash SDPA
        sdpa_mask = None
        if mask is not None:
            mask = np.array(mask, dtype=bool, order="F")
            mask_traits = TensorTraits(
                [n_seq, n_seq],
                [n_seq_tile, n_seq_tile],
            )
            if config.flash_attention:
                mask_tensor = tensor_type(
                    mask_traits, [0] * mask_traits.grid.nelems
                )
                mask_tensor.from_array(
                    np.where(mask, 0.0, -np.inf).astype(np.float32, order="F")
                )
            else:
                mask_tensor = Tensor_bool(
                    mask_traits, [0] * mask_traits.grid.nelems
                )
                mask_tensor.from_array(mask)
            sdpa_mask = mask_tensor

        self.sdpa = Sdpa.generate_simple(
            self.q_rope,
            self.k_rep,
            self.v_rep,
            mask=sdpa_mask,
            flash_attention=config.flash_attention,
            redux=redux,
        )
        self.attn_output = self.sdpa.y

        self.attn_output_transposed = TensorMoments(
            tensor_type(
                b_transposed_traits, [0] * b_transposed_traits.grid.nelems
            ),
            tensor_type(
                b_transposed_traits, [0] * b_transposed_traits.grid.nelems
            ),
            True,
        )
        self.attn_output_transposed.value.set_reduction_add()
        if self.attn_output_transposed.grad is not None:
            self.attn_output_transposed.grad.set_reduction_add()

        self.out_proj = Linear.generate_simple(
            self.attn_output_transposed,
            "R",
            notrans,
            3,
            [self.n_emb],
            [n_emb_tile],
            bias=config.attention_bias,
            redux=redux,
        )

        activations = [
            x,
            self.q_proj.activations_output[0],
            self.q,
            self.q_rope,
            self.k_proj.activations_output[0],
            self.k,
            self.k_rope,
            self.k_rep,
            self.v_proj.activations_output[0],
            self.v,
            self.v_rep,
            self.attn_output,
            self.attn_output_transposed,
            self.out_proj.activations_output[0],
        ]
        layers = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.sdpa,
            self.out_proj,
        ]
        super().__init__(activations, layers, config)
        self.temporaries.extend([self.sin, self.cos])

    def _rope_slice(
        self, seq_len: int, batch: int, kv_cache_size: int = 0
    ):
        if (
            kv_cache_size == 0
            and seq_len == self.sin.shape[1]
            and batch == self.sin.shape[2]
        ):
            return self.sin, self.cos

        sin_bt = [
            self.sin.basetile_shape[0],
            min(seq_len, self.sin.basetile_shape[1]),
            self.sin.basetile_shape[2],
        ]
        cos_bt = [
            self.cos.basetile_shape[0],
            min(seq_len, self.cos.basetile_shape[1]),
            self.cos.basetile_shape[2],
        ]
        sin_slice = nntc.zeros(
            (self.head_size // 2, seq_len, batch),
            basetile_shape=sin_bt,
            dtype=type(self.sin),
        )
        cos_slice = nntc.zeros(
            (self.head_size // 2, seq_len, batch),
            basetile_shape=cos_bt,
            dtype=type(self.cos),
        )
        copy_intersection_async(
            self.sin, [0, kv_cache_size, 0], sin_slice, [0, 0, 0]
        )
        copy_intersection_async(
            self.cos, [0, kv_cache_size, 0], cos_slice, [0, 0, 0]
        )
        return sin_slice, cos_slice

    # ---- Forward / Backward -------------------------------------------------

    def forward_async(self):
        # Q = q_proj(x)
        self.q_proj.forward_async()
        transpose_async(
            1.0, self.q_proj.activations_output[0].value, self.q.value, 2
        )
        # Apply RoPE on Q
        rope_async(self.sin, self.cos, self.q.value, self.q_rope.value)

        # K = k_proj(x)
        self.k_proj.forward_async()
        transpose_async(
            1.0, self.k_proj.activations_output[0].value, self.k.value, 1
        )
        # Apply RoPE on K
        rope_async(self.sin, self.cos, self.k.value, self.k_rope.value)
        # Repeat K to have the same number of heads, as Q
        add_slice_inplace_async(
            1.0, self.k_rope.value, 0.0, self.k_rep.value, 3
            )

        # V = v_proj(x)
        self.v_proj.forward_async()
        transpose_async(
            1.0, self.v_proj.activations_output[0].value, self.v.value, 1
        )
        # Repeat V to have the same number of heads, as Q
        add_slice_inplace_async(1.0, self.v.value, 0.0, self.v_rep.value, 3)

        # Attention
        self.sdpa.forward_async()

        # Output projection
        transpose_async(
            1.0, self.attn_output.value, self.attn_output_transposed.value, 3
        )
        self.out_proj.forward_async()

    def forward_dynamic(
        self,
        x: TensorMoments,
        kv_cache: Optional[KVCache] = None,
    ):
        # Allocate temporaries on the fly (no backward support here)
        q_tmp = self.q_proj.forward_dynamic(x)
        if q_tmp.value.shape[0] == self.head_size:
            q_transposed = q_tmp.value
            q_shape = list(q_tmp.value.shape)
            q_bt = list(q_tmp.value.basetile_shape)
        else:
            q_shape = [
                self.head_size,
                q_tmp.value.shape[3],
                q_tmp.value.shape[4],
                self.kv_group_size,
                self.n_head_kv,
            ]
            q_bt = [
                self.q.value.basetile_shape[0],
                min(q_shape[1], self.q.value.basetile_shape[1]),
                self.q.value.basetile_shape[2],
                self.q.value.basetile_shape[3],
                self.q.value.basetile_shape[4],
            ]
            q_transposed = nntc.empty(
                q_shape,
                basetile_shape=q_bt,
                dtype=type(q_tmp.value),
            )
            transpose_async(1.0, q_tmp.value, q_transposed, 2)
            q_tmp.value.invalidate_submit()

        q_rope_tmp = TensorMoments(
            nntc.empty(
                q_shape,
                basetile_shape=q_bt,
                dtype=type(q_tmp.value),
            ),
            None,
            False,
        )
        kv_size = len(kv_cache) if kv_cache is not None else 0
        sin_local, cos_local = self._rope_slice(
            q_transposed.shape[1], q_transposed.shape[2], kv_size
        )
        rope_async(sin_local, cos_local, q_transposed, q_rope_tmp.value)
        q_transposed.invalidate_submit()

        k_tmp = self.k_proj.forward_dynamic(x)
        if k_tmp.value.shape[0] == self.head_size:
            k_transposed = k_tmp.value
            k_shape = list(k_tmp.value.shape)
            k_bt = list(k_tmp.value.basetile_shape)
        else:
            k_shape = [
                self.head_size,
                k_tmp.value.shape[2],
                k_tmp.value.shape[3],
                self.n_head_kv,
            ]
            k_bt = [
                self.k.value.basetile_shape[0],
                min(k_shape[1], self.k.value.basetile_shape[1]),
                self.k.value.basetile_shape[2],
                self.k.value.basetile_shape[3],
            ]
            k_transposed = nntc.empty(
                k_shape,
                basetile_shape=k_bt,
                dtype=type(k_tmp.value),
            )
            transpose_async(1.0, k_tmp.value, k_transposed, 1)
            k_tmp.value.invalidate_submit()

        k_rope_tmp = TensorMoments(
            nntc.empty(
                k_shape,
                basetile_shape=k_bt,
                dtype=type(k_tmp.value),
            ),
            None,
            False,
        )
        rope_async(sin_local, cos_local, k_transposed, k_rope_tmp.value)
        k_transposed.invalidate_submit()
        if sin_local is not self.sin:
            sin_local.invalidate_submit()
            cos_local.invalidate_submit()

        v_tmp = self.v_proj.forward_dynamic(x)
        if v_tmp.value.shape[0] == self.head_size:
            v_transposed = v_tmp.value
        else:
            v_shape = [
                self.head_size,
                v_tmp.value.shape[2],
                v_tmp.value.shape[3],
                self.n_head_kv,
            ]
            v_bt = [
                self.v.value.basetile_shape[0],
                min(v_shape[1], self.v.value.basetile_shape[1]),
                self.v.value.basetile_shape[2],
                self.v.value.basetile_shape[3],
            ]
            v_transposed = nntc.empty(
                v_shape,
                basetile_shape=v_bt,
                dtype=type(v_tmp.value),
            )
            transpose_async(1.0, v_tmp.value, v_transposed, 1)
            v_tmp.value.invalidate_submit()

        k_src = k_rope_tmp.value
        v_src = v_transposed
        if kv_cache is not None:
            if k_src.shape[1] + len(kv_cache) > self.k.value.shape[1]:
                raise Exception(
                    "Overload internal state: "
                    f"try add {k_src.shape[1]} "
                    f"to {len(kv_cache)}, max: {self.k.value.shape[1]}. "
                )
            kv_cache.append(k_src, v_src)
            k_src = kv_cache.k_partial
            v_src = kv_cache.v_partial

        seq_len = k_src.shape[1]
        k_rep_shape = [
            self.head_size,
            seq_len,
            self.k_rep.value.shape[2],
            self.kv_group_size,
            self.n_head_kv,
        ]
        k_rep_bt = [
            self.k_rep.value.basetile_shape[0],
            k_src.basetile_shape[1],
            self.k_rep.value.basetile_shape[2],
            self.k_rep.value.basetile_shape[3],
            self.k_rep.value.basetile_shape[4],
        ]
        k_rep_tmp = TensorMoments(
            nntc.empty(k_rep_shape, basetile_shape=k_rep_bt, dtype=type(k_src)),
            None,
            False,
        )

        v_rep_shape = [
            self.head_size,
            seq_len,
            self.v_rep.value.shape[2],
            self.kv_group_size,
            self.n_head_kv,
        ]
        v_rep_bt = [
            self.v_rep.value.basetile_shape[0],
            v_src.basetile_shape[1],
            self.v_rep.value.basetile_shape[2],
            self.v_rep.value.basetile_shape[3],
            self.v_rep.value.basetile_shape[4],
        ]
        v_rep_tmp = TensorMoments(
            nntc.empty(v_rep_shape, basetile_shape=v_rep_bt, dtype=type(v_src)),
            None,
            False,
        )

        add_slice_inplace_async(1.0, k_src, 0.0, k_rep_tmp.value, 3)
        add_slice_inplace_async(1.0, v_src, 0.0, v_rep_tmp.value, 3)
        k_src.wont_use()
        v_src.wont_use()
        k_rope_tmp.value.invalidate_submit()
        v_transposed.invalidate_submit()

        mask_local = self.sdpa.mask
        if kv_cache is not None and self.sdpa.mask is not None:
            q_seq = q_rope_tmp.value.shape[1]
            k_seq = k_rep_shape[1]
            mask_bt = [
                min(k_src.basetile_shape[1], k_seq),
                min(q_rope_tmp.value.basetile_shape[1], q_seq),
            ]
            mask_traits = TensorTraits([k_seq, q_seq], mask_bt)
            mask_tensor = Tensor_bool(mask_traits, [0] * mask_traits.grid.nelems)
            mask_np = np.zeros((k_seq, q_seq), dtype=bool, order="F")
            for j in range(q_seq):
                cutoff = kv_size + j
                if cutoff + 1 < k_seq:
                    mask_np[cutoff + 1 :, j] = True
            mask_tensor.from_array(mask_np)
            mask_local = mask_tensor

        sdpa_out = self.sdpa.forward_dynamic(
            q_rope_tmp,
            k_rep_tmp,
            v_rep_tmp,
            mask=mask_local,
        )

        attn_transposed_tmp = nntc.empty(
            [
                sdpa_out.value.shape[3],
                sdpa_out.value.shape[4],
                sdpa_out.value.shape[0],
                sdpa_out.value.shape[1],
                sdpa_out.value.shape[2],
            ],
            basetile_shape=[
                sdpa_out.value.basetile_shape[3],
                sdpa_out.value.basetile_shape[4],
                sdpa_out.value.basetile_shape[0],
                sdpa_out.value.basetile_shape[1],
                sdpa_out.value.basetile_shape[2],
            ],
            dtype=type(sdpa_out.value),
        )
        transpose_async(
            1.0,
            sdpa_out.value,
            attn_transposed_tmp,
            3,
        )
        sdpa_out.value.invalidate_submit()

        out = self.out_proj.forward_dynamic(
            TensorMoments(attn_transposed_tmp, None, False)
        )

        # Invalidate temporaries to hint release
        q_rope_tmp.value.invalidate_submit()
        k_rep_tmp.value.invalidate_submit()
        v_rep_tmp.value.invalidate_submit()

        return out, kv_cache

    def backward_async(self):
        # Output projection
        self.out_proj.backward_async()
        transpose_async(
            1.0,
            self.attn_output_transposed.grad,
            self.attn_output.grad,
            2,
        )

        # SDPA backward into Q_rope, K_rep, V_rep
        self.sdpa.backward_async()

        # V path
        sum_slice_async(1.0, self.v_rep.grad, 0.0, self.v.grad, 3)
        transpose_async(
            1.0, self.v.grad, self.v_proj.activations_output[0].grad, 3
        )
        self.v_proj.backward_async()

        # K path
        sum_slice_async(1.0, self.k_rep.grad, 0.0, self.k_rope.grad, 3)
        rope_backward_async(self.sin, self.cos, self.k_rope.grad, self.k.grad)
        transpose_async(
            1.0, self.k.grad, self.k_proj.activations_output[0].grad, 3
        )
        self.k_proj.backward_async()

        # Q path
        rope_backward_async(self.sin, self.cos, self.q_rope.grad, self.q.grad)
        transpose_async(
            1.0, self.q.grad, self.q_proj.activations_output[0].grad, 3
        )
        self.q_proj.backward_async()

    # ---- Torch interoperability --------------------------------------------

    @staticmethod
    def rotate_tensor_in(x: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            new_shape = (1, x.shape[0], np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), x.shape[-1], 1)
        else:
            new_shape = (
                np.prod(x.shape[:axis]),
                x.shape[axis],
                np.prod(x.shape[axis + 1 :]),
            )
        x_reshaped = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, 0::2, :] = x_reshaped[:, :mid, :]
        y_reshaped[:, 1::2, :] = x_reshaped[:, mid:, :]
        return y_reshaped.reshape(x.shape)

    @staticmethod
    def rotate_tensor_out(x: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            new_shape = (1, x.shape[0], np.prod(x.shape[1:]))
        elif axis == x.ndim - 1:
            new_shape = (np.prod(x.shape[:-1]), x.shape[-1], 1)
        else:
            new_shape = (
                np.prod(x.shape[:axis]),
                x.shape[axis],
                np.prod(x.shape[axis + 1 :]),
            )
        x_reshaped = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_reshaped = np.empty_like(x_reshaped)
        y_reshaped[:, :mid, :] = x_reshaped[:, 0::2, :]
        y_reshaped[:, mid:, :] = x_reshaped[:, 1::2, :]
        return y_reshaped.reshape(x.shape)

    @classmethod
    def from_torch(
        cls,
        torch_layer: LlamaAttention_torch,
        x: TensorMoments,
        position_ids: np.ndarray,
        mask: np.ndarray,
        config: LlamaConfigNNTile,
    ) -> "LlamaAttention":
        if torch_layer.q_proj.bias is not None:
            raise NotImplementedError("Q/K/V projection bias is not supported")
        layer = cls(x, position_ids, mask, config)

        tmp_q_shape = layer.q_proj.w.value.shape.copy()
        tmp_q_shape[:2] = tmp_q_shape[1::-1]
        layer.q_proj.w.value.from_array(
            cls.rotate_tensor_in(
                np.moveaxis(
                    torch_layer.q_proj.weight.detach()
                    .cpu()
                    .numpy()
                    .reshape(*tmp_q_shape),
                    0,
                    1,
                ),
                2,
            )
        )
        layer.k_proj.w.value.from_array(
            cls.rotate_tensor_in(
                torch_layer.k_proj.weight.detach()
                .cpu()
                .numpy()
                .reshape(*layer.k_proj.w.value.shape),
                1,
            )
        )
        layer.v_proj.w.value.from_array(
            torch_layer.v_proj.weight.detach()
            .cpu()
            .numpy()
            .reshape(*layer.v_proj.w.value.shape)
        )

        tmp_w_shape = layer.out_proj.w.value.shape.copy()
        tmp_w_shape[1:3] = tmp_w_shape[2:0:-1]
        layer.out_proj.w.value.from_array(
            np.moveaxis(
                torch_layer.o_proj.weight.detach()
                .cpu()
                .numpy()
                .reshape(*tmp_w_shape),
                1,
                2,
            )
        )
        if torch_layer.o_proj.bias is not None:
            if layer.out_proj.b is None:
                raise NotImplementedError("Output projection bias is "
                    "not allocated")
            layer.out_proj.b.value.from_array(
                torch_layer.o_proj.bias.detach()
                .cpu()
                .numpy()
                .reshape(*layer.out_proj.b.value.shape)
            )

        return layer

    def to_torch(self) -> LlamaAttention_torch:
        bias = self.out_proj.b is not None
        torch_layer_config = LlamaConfig_torch(
            hidden_size=self.n_emb,
            num_attention_heads=self.n_head,
            num_key_value_heads=self.n_head_kv,
            attention_bias=bias,
            use_cache=False,
            attention_dropout=0.0,
        )
        torch_layer = LlamaAttention_torch(torch_layer_config, layer_idx=0)
        torch_layer.q_proj.weight.data = torch.tensor(
            np.moveaxis(
                __class__.rotate_tensor_out(to_numpy(self.q_proj.w.value), 2),
                0,
                1
            ).reshape(self.n_emb, self.n_emb),
            requires_grad=True,
        )
        torch_layer.k_proj.weight.data = torch.tensor(
            __class__.rotate_tensor_out(to_numpy(self.k_proj.w.value), 1)
            .reshape(
                self.n_emb_kv, self.n_emb
            ),
            requires_grad=True,
        )
        torch_layer.v_proj.weight.data = torch.tensor(
            to_numpy(self.v_proj.w.value).reshape(self.n_emb_kv, self.n_emb),
            requires_grad=True,
        )
        torch_layer.o_proj.weight.data = torch.tensor(
            np.moveaxis(to_numpy(self.out_proj.w.value), 1, 2).reshape(
                self.n_emb, self.n_emb
            ),
            requires_grad=True,
        )
        if bias:
            torch_layer.o_proj.bias.data = torch.tensor(
                to_numpy(self.out_proj.b.value).flatten(),
                requires_grad=True,
            )
        return torch_layer

    def to_torch_with_grads(self) -> LlamaAttention_torch:
        torch_layer = self.to_torch()
        if self.out_proj.b is not None:
            torch_layer.o_proj.bias.grad = torch.tensor(
                to_numpy(self.out_proj.b.grad).flatten()
            )
        torch_layer.q_proj.weight.grad = torch.tensor(
            np.moveaxis(
                __class__.rotate_tensor_out(to_numpy(self.q_proj.w.grad), 2),
                0,
                1
            ).reshape(self.n_emb, self.n_emb)
        )
        torch_layer.k_proj.weight.grad = torch.tensor(
            __class__.rotate_tensor_out(to_numpy(self.k_proj.w.grad), 1)
            .reshape(
                self.n_emb_kv, self.n_emb
            )
        )
        torch_layer.v_proj.weight.grad = torch.tensor(
            to_numpy(self.v_proj.w.grad).reshape(self.n_emb_kv, self.n_emb)
        )
        torch_layer.o_proj.weight.grad = torch.tensor(
            np.moveaxis(to_numpy(self.out_proj.w.grad), 1, 2).reshape(
                self.n_emb, self.n_emb
            )
        )
        return torch_layer

    # ---- Helpers -----------------------------------------------------------

    def _fill_sin_cos(self, position_ids: np.ndarray):
        n_seq = position_ids.shape[1]
        if n_seq != self.q.value.shape[1]:
            raise ValueError("position_ids must match the sequence length")
        tmp = np.arange(0, self.head_size, 2, dtype=np.float32)
        inv_freq = 1.0 / (self.config.rope_theta ** (tmp / self.head_size))
        freq_frame = np.empty(
            (self.head_size // 2, position_ids.shape[1], position_ids.shape[0])
        )
        for i in range(position_ids.shape[0]):
            freq_frame[:, :, i] = np.outer(inv_freq, position_ids[i, :])
        self.cos.from_array(np.cos(freq_frame).astype(np.float32, order="F"))
        self.sin.from_array(np.sin(freq_frame).astype(np.float32, order="F"))
