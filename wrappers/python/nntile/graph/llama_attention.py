# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/graph/llama_attention.py
# Python wrapper for Graph API LlamaAttention with HuggingFace interop.
#
# @version 1.1.0

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch,
    LlamaConfig as LlamaConfig_torch,
)

from ..nntile_graph import DataType, NNGraph, llama


class GraphLlamaAttention:
    """
    Python wrapper around the C++ Graph API ``LlamaAttention``.

    Provides weight conversion helpers (``from_torch`` / ``to_torch`` /
    ``to_torch_with_grads``) that mirror the existing
    ``nntile.model.llama_attention.LlamaAttention`` interface, and
    convenience methods to build a graph, compile a Runtime, and execute
    forward/backward passes.
    """

    def __init__(
        self,
        graph: NNGraph,
        module: llama.LlamaAttention,
        config: llama.LlamaConfig,
        dtype: DataType,
        output_node,
        *,
        x_node=None,
        sin_node=None,
        cos_node=None,
        mask_node=None,
    ):
        self.graph = graph
        self.module = module
        self.config = config
        self.dtype = dtype
        self.output_node = output_node
        self.x_node = x_node
        self.sin_node = sin_node
        self.cos_node = cos_node
        self.mask_node = mask_node

        self.n_emb = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_head_kv = config.num_key_value_heads
        self.head_size = config.head_dim
        self.kv_group_size = self.n_heads // self.n_head_kv
        self.use_gqa = self.n_head_kv < self.n_heads

    # ------------------------------------------------------------------
    # RoPE interleave helpers (identical to the existing wrapper)
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_tensor_in(x: np.ndarray, axis: int) -> np.ndarray:
        """Interleave first/second halves along *axis* for NNTile RoPE."""
        if axis == 0:
            new_shape = (1, x.shape[0], int(np.prod(x.shape[1:])))
        elif axis == x.ndim - 1:
            new_shape = (int(np.prod(x.shape[:-1])), x.shape[-1], 1)
        else:
            new_shape = (
                int(np.prod(x.shape[:axis])),
                x.shape[axis],
                int(np.prod(x.shape[axis + 1:])),
            )
        x_r = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_r = np.empty_like(x_r)
        y_r[:, 0::2, :] = x_r[:, :mid, :]
        y_r[:, 1::2, :] = x_r[:, mid:, :]
        return y_r.reshape(x.shape)

    @staticmethod
    def rotate_tensor_out(x: np.ndarray, axis: int) -> np.ndarray:
        """De-interleave along *axis* (inverse of ``rotate_tensor_in``)."""
        if axis == 0:
            new_shape = (1, x.shape[0], int(np.prod(x.shape[1:])))
        elif axis == x.ndim - 1:
            new_shape = (int(np.prod(x.shape[:-1])), x.shape[-1], 1)
        else:
            new_shape = (
                int(np.prod(x.shape[:axis])),
                x.shape[axis],
                int(np.prod(x.shape[axis + 1:])),
            )
        x_r = x.reshape(new_shape)
        mid = x.shape[axis] // 2
        y_r = np.empty_like(x_r)
        y_r[:, :mid, :] = x_r[:, 0::2, :]
        y_r[:, mid:, :] = x_r[:, 1::2, :]
        return y_r.reshape(x.shape)

    # ------------------------------------------------------------------
    # Sin / cos for RoPE
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sin_cos(
        position_ids: np.ndarray,
        head_size: int,
        rope_theta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute sin/cos arrays in NNTile layout (head_size/2, seq, batch)."""
        n_batch, n_seq = position_ids.shape
        tmp = np.arange(0, head_size, 2, dtype=np.float32)
        inv_freq = 1.0 / (rope_theta ** (tmp / head_size))
        freq = np.empty((head_size // 2, n_seq, n_batch), dtype=np.float32)
        for i in range(n_batch):
            freq[:, :, i] = np.outer(inv_freq, position_ids[i, :])
        cos_arr = np.cos(freq).astype(np.float32)
        sin_arr = np.sin(freq).astype(np.float32)
        return sin_arr, cos_arr

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        config: llama.LlamaConfig,
        seq_len: int,
        n_batch: int,
        position_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dtype: DataType = DataType.FP32,
        graph_name: str = "llama_attn",
    ) -> "GraphLlamaAttention":
        """Build the NNGraph for a LlamaAttention layer."""
        graph = NNGraph(graph_name)

        x_node = graph.tensor(
            [config.hidden_size, seq_len, n_batch],
            "x", dtype, True,
        )

        sin_arr, cos_arr = cls.compute_sin_cos(
            position_ids, config.head_dim, config.rope_theta,
        )
        sin_node = graph.tensor(
            [config.head_dim // 2, seq_len, n_batch],
            "sin", dtype, False,
        )
        cos_node = graph.tensor(
            [config.head_dim // 2, seq_len, n_batch],
            "cos", dtype, False,
        )

        mask_node = None
        if mask is not None:
            mask_node = graph.tensor(
                list(mask.shape), "mask", DataType.BOOL, False,
            )

        module = llama.LlamaAttention(graph, "attn", config, dtype)
        output_node = module.forward(x_node, sin_node, cos_node, mask_node)

        x_node.mark_input()
        sin_node.mark_input()
        cos_node.mark_input()
        if mask_node is not None:
            mask_node.mark_input()
        output_node.mark_output()

        for _, param in module.named_parameters():
            param.mark_input()

        wrapper = cls(
            graph, module, config, dtype, output_node,
            x_node=x_node,
            sin_node=sin_node,
            cos_node=cos_node,
            mask_node=mask_node,
        )
        wrapper._sin_arr = sin_arr
        wrapper._cos_arr = cos_arr
        wrapper._mask = mask
        return wrapper

    # ------------------------------------------------------------------
    # Weight conversion: HuggingFace -> NNTile numpy arrays
    # ------------------------------------------------------------------

    def weights_from_torch(
        self, torch_layer: LlamaAttention_torch,
    ) -> dict[str, np.ndarray]:
        """
        Convert HuggingFace weight tensors into NNTile-format numpy arrays
        keyed by the C++ parameter name (``attn.q_weight``, etc.).
        """
        result = {}
        prefix = self.module.name + "_"

        q_np = torch_layer.q_proj.weight.detach().cpu().numpy()
        if self.use_gqa:
            q_shape = (
                self.n_head_kv, self.kv_group_size,
                self.head_size, self.n_emb,
            )
            q_np = q_np.reshape(q_shape)
            q_np = np.moveaxis(q_np, 0, 1)
            q_np = self.rotate_tensor_in(q_np, 2)
        else:
            q_shape = (self.n_heads, self.head_size, self.n_emb)
            q_np = q_np.reshape(q_shape)
            q_np = self.rotate_tensor_in(q_np, 1)
        result[prefix + "q_weight"] = np.ascontiguousarray(q_np)

        k_np = torch_layer.k_proj.weight.detach().cpu().numpy()
        k_np = k_np.reshape(self.n_head_kv, self.head_size, self.n_emb)
        k_np = self.rotate_tensor_in(k_np, 1)
        result[prefix + "k_weight"] = np.ascontiguousarray(k_np)

        v_np = torch_layer.v_proj.weight.detach().cpu().numpy()
        v_np = v_np.reshape(self.n_head_kv, self.head_size, self.n_emb)
        result[prefix + "v_weight"] = np.ascontiguousarray(v_np)

        o_np = torch_layer.o_proj.weight.detach().cpu().numpy()
        if self.use_gqa:
            o_shape_tmp = (
                self.n_emb, self.n_head_kv,
                self.kv_group_size, self.head_size,
            )
            o_np = np.moveaxis(
                o_np.reshape(o_shape_tmp), 1, 2,
            )
        else:
            o_np = o_np.reshape(self.n_emb, self.n_heads, self.head_size)
        result[prefix + "o_weight"] = np.ascontiguousarray(o_np)

        return result

    # ------------------------------------------------------------------
    # Weight conversion: NNTile numpy arrays -> HuggingFace
    # ------------------------------------------------------------------

    def _make_torch_layer(self, bias: bool = False) -> LlamaAttention_torch:
        torch_config = LlamaConfig_torch(
            hidden_size=self.n_emb,
            num_attention_heads=self.n_heads,
            num_key_value_heads=self.n_head_kv,
            attention_bias=bias,
            use_cache=False,
            attention_dropout=0.0,
        )
        return LlamaAttention_torch(torch_config, layer_idx=0)

    def weights_to_torch(
        self, weight_arrays: dict[str, np.ndarray],
    ) -> LlamaAttention_torch:
        """Build HuggingFace LlamaAttention and load NNTile weights."""
        torch_layer = self._make_torch_layer()
        prefix = self.module.name + "_"

        q_np = weight_arrays[prefix + "q_weight"].copy()
        if self.use_gqa:
            q_np = self.rotate_tensor_out(q_np, 2)
            q_np = np.moveaxis(q_np, 0, 1)
        else:
            q_np = self.rotate_tensor_out(q_np, 1)
        torch_layer.q_proj.weight.data = torch.tensor(
            q_np.reshape(self.n_emb, self.n_emb), requires_grad=True,
        )

        k_np = self.rotate_tensor_out(
            weight_arrays[prefix + "k_weight"].copy(), 1,
        )
        n_emb_kv = self.n_head_kv * self.head_size
        torch_layer.k_proj.weight.data = torch.tensor(
            k_np.reshape(n_emb_kv, self.n_emb), requires_grad=True,
        )

        v_np = weight_arrays[prefix + "v_weight"]
        torch_layer.v_proj.weight.data = torch.tensor(
            v_np.reshape(n_emb_kv, self.n_emb), requires_grad=True,
        )

        o_np = weight_arrays[prefix + "o_weight"]
        if self.use_gqa:
            o_np = np.moveaxis(o_np, 1, 2)
        torch_layer.o_proj.weight.data = torch.tensor(
            o_np.reshape(self.n_emb, self.n_emb), requires_grad=True,
        )
        return torch_layer

    def grads_to_torch(
        self,
        torch_layer: LlamaAttention_torch,
        grad_arrays: dict[str, np.ndarray],
    ) -> None:
        """Load NNTile gradient arrays into a HuggingFace model's ``.grad``."""
        prefix = self.module.name + "_"
        n_emb_kv = self.n_head_kv * self.head_size

        q_g = grad_arrays[prefix + "q_weight"].copy()
        if self.use_gqa:
            q_g = self.rotate_tensor_out(q_g, 2)
            q_g = np.moveaxis(q_g, 0, 1)
        else:
            q_g = self.rotate_tensor_out(q_g, 1)
        torch_layer.q_proj.weight.grad = torch.tensor(
            q_g.reshape(self.n_emb, self.n_emb),
        )

        k_g = self.rotate_tensor_out(
            grad_arrays[prefix + "k_weight"].copy(), 1,
        )
        torch_layer.k_proj.weight.grad = torch.tensor(
            k_g.reshape(n_emb_kv, self.n_emb),
        )

        v_g = grad_arrays[prefix + "v_weight"]
        torch_layer.v_proj.weight.grad = torch.tensor(
            v_g.reshape(n_emb_kv, self.n_emb),
        )

        o_g = grad_arrays[prefix + "o_weight"].copy()
        if self.use_gqa:
            o_g = np.moveaxis(o_g, 1, 2)
        torch_layer.o_proj.weight.grad = torch.tensor(
            o_g.reshape(self.n_emb, self.n_emb),
        )

    # ------------------------------------------------------------------
    # Convenience: build from a HuggingFace LlamaAttention
    # ------------------------------------------------------------------

    @classmethod
    def from_torch(
        cls,
        torch_layer: LlamaAttention_torch,
        seq_len: int,
        n_batch: int,
        position_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dtype: DataType = DataType.FP32,
        graph_name: str = "llama_attn",
    ) -> "GraphLlamaAttention":
        """Build graph + return wrapper with weights ready for binding."""
        hf_cfg = torch_layer.config
        config = llama.LlamaConfig()
        config.hidden_size = hf_cfg.hidden_size
        config.num_attention_heads = hf_cfg.num_attention_heads
        config.num_key_value_heads = hf_cfg.num_key_value_heads
        config.vocab_size = hf_cfg.vocab_size
        config.intermediate_size = hf_cfg.intermediate_size
        config.num_hidden_layers = hf_cfg.num_hidden_layers
        config.max_position_embeddings = hf_cfg.max_position_embeddings
        config.rms_norm_eps = hf_cfg.rms_norm_eps
        config.rope_theta = hf_cfg.rope_theta
        config.attention_bias = getattr(hf_cfg, "attention_bias", False)
        config.mlp_bias = getattr(hf_cfg, "mlp_bias", False)
        config.compute_head_dim()
        config.validate()

        wrapper = cls.build(
            config, seq_len, n_batch, position_ids, mask, dtype, graph_name,
        )
        wrapper._torch_weights = wrapper.weights_from_torch(torch_layer)
        return wrapper

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def compile(self):
        """Compile the forward + backward tensor graph and return a Runtime."""
        tg = self.graph.tensor_graph()
        from ..nntile_graph import Runtime
        rt = Runtime(tg)
        rt.compile()
        return rt

    def bind_weights(self, rt, weight_arrays: dict[str, np.ndarray]):
        """Bind all weight arrays into the runtime.

        Keys must be the full NNGraph tensor names (e.g. ``attn_q_weight``).
        Data is raveled in Fortran order to match NNTile's column-major layout.
        """
        for name, arr in weight_arrays.items():
            rt.bind_data(name, np.asfortranarray(arr).ravel(order='F'))

    def get_weight_grads(self, rt) -> dict[str, np.ndarray]:
        """Read weight gradients from the runtime after backward execution."""
        grads = {}
        for local_name, param in self.module.named_parameters():
            if param.grad is None:
                continue
            grad_flat = np.array(rt.get_output(param.grad.name))
            grads[param.name] = grad_flat.reshape(param.shape, order='F')
        return grads
