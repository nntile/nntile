# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/graph/model/llama/test_llama_attention.py
# Test for Graph API LlamaAttention against HuggingFace reference.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaRotaryEmbedding,
)

import nntile
from nntile.graph import DataType, NNGraph, Runtime
from nntile.graph.llama_attention import GraphLlamaAttention


@dataclass
class Params:
    head_size: int
    n_head: int
    n_head_kv: int
    seq_len: int
    n_batch: int


single_tile = Params(
    head_size=64,
    n_head=8,
    n_head_kv=4,
    seq_len=64,
    n_batch=3,
)

multiple_heads = Params(
    head_size=128,
    n_head=8,
    n_head_kv=4,
    seq_len=128,
    n_batch=4,
)


def _build_torch_layer(params: Params):
    """Create an HF LlamaAttention with random weights."""
    hidden_size = params.head_size * params.n_head
    cfg = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=params.n_head,
        num_key_value_heads=params.n_head_kv,
        attention_bias=False,
        pretraining_tp=1,
    )
    return LlamaAttention(cfg, layer_idx=0), cfg


def _run_torch_forward(torch_layer, x_np, params, pos_ids, mask):
    """Run HF LlamaAttention forward and return the output tensor."""
    x_torch = torch.tensor(x_np.T.copy(), requires_grad=True)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long)
    rotary = LlamaRotaryEmbedding(config=torch_layer.config)
    pos_embs = rotary(torch_layer.v_proj.weight, pos_ids_torch)

    mask_torch = (
        torch.tensor(np.array(1 - mask, dtype=np.float32)).T
        * torch.finfo(torch.float32).min
    )
    mask_torch = mask_torch[None, None, :, :].expand(
        params.n_batch, 1, -1, -1
    )
    y = torch_layer(
        x_torch, position_embeddings=pos_embs,
        attention_mask=mask_torch, position_ids=pos_ids_torch,
    )[0]
    return y, x_torch, pos_embs, mask_torch


def _build_nntile_graph(
    torch_layer, params, x_np, pos_ids, mask, gen,
):
    """Build an NNTile GraphLlamaAttention from an HF layer and return
    everything needed to run forward + backward."""
    wrapper = GraphLlamaAttention.from_torch(
        torch_layer, params.seq_len, params.n_batch,
        pos_ids, mask, DataType.FP32,
    )
    return wrapper


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        pytest.param(multiple_heads, id="multiple_heads"),
    ],
)
class TestGraphLlamaAttention:

    def test_coercion(self, context, torch_rng, params: Params):
        """Weights survive the round-trip HF -> NNTile -> HF."""
        torch_layer, _ = _build_torch_layer(params)
        gen = np.random.default_rng(42)
        hidden_size = params.head_size * params.n_head
        pos_ids = gen.integers(
            params.seq_len,
            size=(params.n_batch, params.seq_len),
            dtype=np.int64,
        )
        mask = np.triu(
            np.ones((params.seq_len, params.seq_len)), k=0
        ).astype(bool)

        wrapper = GraphLlamaAttention.from_torch(
            torch_layer, params.seq_len, params.n_batch,
            pos_ids, mask, DataType.FP32,
        )
        weights = wrapper._torch_weights
        torch_layer_rt = wrapper.weights_to_torch(weights)

        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(),
            torch_layer_rt.named_parameters(),
        ):
            assert n1 == n2, f"Name mismatch: {n1} vs {n2}"
            diff = torch.norm(p1 - p2)
            ref = torch.norm(p1)
            assert diff <= 1e-5 * ref, (
                f"Round-trip mismatch for {n1}: diff={diff:.3e}, ref={ref:.3e}"
            )

    def test_forward(self, context, torch_rng, params: Params):
        """Graph API forward matches HF forward."""
        torch_layer, _ = _build_torch_layer(params)
        gen = np.random.default_rng(42)
        hidden_size = params.head_size * params.n_head
        x_shape = [hidden_size, params.seq_len, params.n_batch]
        x_np = gen.standard_normal(x_shape).astype(np.float32)

        pos_ids = gen.integers(
            params.seq_len,
            size=(params.n_batch, params.seq_len),
            dtype=np.int64,
        )
        mask = np.triu(
            np.ones((params.seq_len, params.seq_len)), k=0
        ).astype(bool)

        y_torch, _, _, _ = _run_torch_forward(
            torch_layer, x_np, params, pos_ids, mask,
        )

        wrapper = GraphLlamaAttention.from_torch(
            torch_layer, params.seq_len, params.n_batch,
            pos_ids, mask, DataType.FP32,
        )
        tg = wrapper.graph.tensor_graph()
        rt = Runtime(tg)
        rt.compile()

        rt.bind_data("x", np.asfortranarray(x_np).ravel(order='F'))
        rt.bind_data("sin", np.asfortranarray(
            wrapper._sin_arr).ravel(order='F'))
        rt.bind_data("cos", np.asfortranarray(
            wrapper._cos_arr).ravel(order='F'))
        rt.bind_data(
            "mask",
            np.asfortranarray(mask).ravel(order='F').astype(np.uint8),
        )
        wrapper.bind_weights(rt, wrapper._torch_weights)

        rt.execute()
        rt.wait()

        out_name = wrapper.output_node.name
        y_flat = np.array(rt.get_output(out_name))
        y_nntile_np = np.array(
            y_flat, dtype=np.float32,
        ).reshape(x_shape, order='F')
        y_nntile = torch.tensor(y_nntile_np.T.copy())

        nntile.starpu.wait_for_all()

        rtol = 1e-5
        diff = torch.norm(y_torch - y_nntile)
        ref = torch.norm(y_torch)
        assert diff <= rtol * ref, (
            f"Forward mismatch: diff={diff:.3e}, ref={ref:.3e}"
        )

    def test_forward_backward(self, context, torch_rng, params: Params):
        """Graph API forward+backward matches HF forward+backward."""
        torch_layer, _ = _build_torch_layer(params)
        gen = np.random.default_rng(42)
        hidden_size = params.head_size * params.n_head
        x_shape = [hidden_size, params.seq_len, params.n_batch]
        x_np = gen.standard_normal(x_shape).astype(np.float32)

        pos_ids = gen.integers(
            params.seq_len,
            size=(params.n_batch, params.seq_len),
            dtype=np.int64,
        )
        mask = np.triu(
            np.ones((params.seq_len, params.seq_len)), k=0
        ).astype(bool)

        y_torch, x_torch, pos_embs, mask_torch = _run_torch_forward(
            torch_layer, x_np, params, pos_ids, mask,
        )

        y_grad_np = gen.standard_normal(x_shape).astype(np.float32)
        y_grad_torch = torch.tensor(y_grad_np.T.copy())
        loss = (y_torch * y_grad_torch).sum()
        loss.backward()

        wrapper = GraphLlamaAttention.from_torch(
            torch_layer, params.seq_len, params.n_batch,
            pos_ids, mask, DataType.FP32,
        )

        output_grad_node = wrapper.graph.get_or_create_grad(
            wrapper.output_node, "output_grad",
        )
        output_grad_node.mark_input()
        wrapper.output_node.backward()

        wrapper.x_node.grad.mark_output()
        for _, param in wrapper.module.named_parameters():
            if param.grad is not None:
                param.grad.mark_output()

        tg = wrapper.graph.tensor_graph()
        rt = Runtime(tg)
        rt.compile()

        rt.bind_data("x", np.asfortranarray(x_np).ravel(order='F'))
        rt.bind_data("sin", np.asfortranarray(
            wrapper._sin_arr).ravel(order='F'))
        rt.bind_data("cos", np.asfortranarray(
            wrapper._cos_arr).ravel(order='F'))
        rt.bind_data("mask", np.asfortranarray(
            mask).ravel(order='F').astype(np.uint8))
        wrapper.bind_weights(rt, wrapper._torch_weights)
        rt.bind_data("output_grad", np.asfortranarray(
            y_grad_np).ravel(order='F'))

        rt.execute()
        rt.wait()
        nntile.starpu.wait_for_all()

        # Compare x gradient
        x_grad_flat = np.array(rt.get_output(wrapper.x_node.grad.name))
        x_grad_nntile = torch.tensor(
            x_grad_flat.reshape(x_shape, order='F').T.copy()
        )
        rtol = 1e-5
        diff = torch.norm(x_torch.grad - x_grad_nntile)
        ref = torch.norm(x_torch.grad)
        assert diff <= rtol * ref, (
            f"x.grad mismatch: diff={diff:.3e}, ref={ref:.3e}"
        )

        # Compare weight gradients via round-trip
        weight_grads = wrapper.get_weight_grads(rt)
        torch_layer_rt = wrapper.weights_to_torch(wrapper._torch_weights)
        wrapper.grads_to_torch(torch_layer_rt, weight_grads)

        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(),
            torch_layer_rt.named_parameters(),
        ):
            assert n1 == n2
            if p1.requires_grad and p1.grad is not None:
                g1, g2 = p1.grad, p2.grad
                diff = torch.norm(g1 - g2)
                ref = torch.norm(g1)
                assert diff <= rtol * ref, (
                    f"Grad mismatch for {n1}: diff={diff:.3e}, ref={ref:.3e}"
                )
