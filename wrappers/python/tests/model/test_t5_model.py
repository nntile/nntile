# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_t5_model.py
# Test for nntile.model.t5_model - T5Model
#
# @version 1.1.0

from dataclasses import dataclass

import nntile.functions
import numpy as np
import pytest
import torch
from transformers.models.t5.modeling_t5 import (
    T5Model as T5ModelTorch,
    T5Config as T5ConfigTorch,
    T5ForSequenceClassification as T5ForSequenceClassificationTorch,
)
from transformers import T5Tokenizer

import nntile
from nntile.model.t5_config import T5ConfigNNTile
from nntile.model.t5_model import T5Model, T5ForSequenceClassification
from nntile.tensor import TensorMoments, TensorTraits
import nntile.utils.constructors as nntc

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "bf16": nntile.tensor.Tensor_bf16,
}

dtype2tol = {
    "fp32": {"rtol": 6e-4},
    "fp32_fast_tf32": {"rtol": 7e-4},
    "bf16": {"rtol": 1.2e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")


@dataclass
class T5ModelTestParams:
    num_layers: int
    d_model: int
    d_model_tile: int
    d_kv: int
    d_kv_tile: int
    d_ff: int
    d_ff_tile: int
    n_head: int
    n_head_tile: int
    n_batch: int
    n_batch_tile: int
    enc_seq_len: int
    enc_seq_len_tile: int
    dec_seq_len: int
    dec_seq_len_tile: int
    redux: bool = True
    is_gated_act: bool = True


single_tile = T5ModelTestParams(
    d_model=512,
    d_model_tile=512,
    d_kv=64,
    d_kv_tile=64,
    d_ff=1024,
    d_ff_tile=1024,
    n_head=8,
    n_head_tile=8,
    enc_seq_len=64,
    enc_seq_len_tile=64,
    # dec_seq_len=32,
    # dec_seq_len_tile=32,
    dec_seq_len=64,
    dec_seq_len_tile=64,
    n_batch=1,
    n_batch_tile=1,
    is_gated_act=True,
    num_layers=3,
)

multiple_tiles = T5ModelTestParams(
    d_model=512,
    d_model_tile=128,
    d_kv=64,
    d_kv_tile=64,
    d_ff=1024,
    d_ff_tile=256,
    n_head=8,
    n_head_tile=2,
    enc_seq_len=64,
    enc_seq_len_tile=32,
    dec_seq_len=64,
    dec_seq_len_tile=32,
    n_batch=4,
    n_batch_tile=1,
    is_gated_act=True,
    num_layers=4,
)


def generate_inputs(params: T5ModelTestParams, dtype: str):
    # Configure PyTorch T5 model
    torch_config = T5ConfigTorch(
        d_model=params.d_model,
        d_ff=params.d_ff,
        d_kv=params.d_kv,
        num_heads=params.n_head,
        dropout_rate=0.0,
        dense_act_fn="gelu_new",
        is_gated_act=params.is_gated_act,
        num_layers=params.num_layers,
        attn_implementation="eager",
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    torch_model = T5ModelTorch(torch_config)

    # Configure NNTile T5 model config
    nntile_config = T5ConfigNNTile(
        d_model=params.d_model,
        d_model_tile=params.d_model_tile,
        d_kv=params.d_kv,
        d_kv_tile=params.d_kv_tile,
        d_ff=params.d_ff,
        d_ff_tile=params.d_ff_tile,
        n_head=params.n_head,
        n_head_tile=params.n_head_tile,
        redux=params.redux,
        num_layers=params.num_layers,
        is_gated_act=params.is_gated_act,
    )

    # Make sure all dropout layers are disabled
    for name, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    torch_model.eval()  # Set to evaluation mode

    # Generator for random values
    gen = np.random.default_rng(42)

    # Set encoder input tensor dimensions
    enc_shape = [params.d_model, params.enc_seq_len, params.n_batch]
    enc_basetile = [params.d_model_tile, params.enc_seq_len_tile, params.n_batch_tile]
    enc_traits = TensorTraits(enc_shape, enc_basetile)
    enc_distr = [0] * enc_traits.grid.nelems
    enc_type = dtype2nntile[dtype]
    enc_value = enc_type(enc_traits, enc_distr, 0)
    enc_grad = enc_type(enc_traits, enc_distr, 0)
    enc_X = TensorMoments(enc_value, enc_grad, grad_required=True)
    nntile.functions.fill_async(0.0, enc_X.grad)

    # Generate random encoder input data
    enc_random = gen.standard_normal(enc_shape, dtype=np.float32)
    enc_nntile = np.array(enc_random, dtype=np.float32, order="F")
    enc_value.from_array(enc_nntile)
    enc_torch = torch.Tensor(enc_nntile.T)
    enc_torch.requires_grad_()

    # Set decoder input tensor dimensions
    dec_shape = [params.d_model, params.dec_seq_len, params.n_batch]
    dec_basetile = [params.d_model_tile, params.dec_seq_len_tile, params.n_batch_tile]
    dec_traits = TensorTraits(dec_shape, dec_basetile)
    dec_distr = [0] * dec_traits.grid.nelems
    dec_type = dtype2nntile[dtype]
    dec_value = dec_type(dec_traits, dec_distr, 0)
    dec_grad = dec_type(dec_traits, dec_distr, 0)
    dec_X = TensorMoments(dec_value, dec_grad, grad_required=True)
    nntile.functions.fill_async(0.0, dec_X.grad)

    # Generate random decoder input data
    dec_random = gen.standard_normal(dec_shape, dtype=np.float32)
    dec_nntile = np.array(dec_random, dtype=np.float32, order="F")
    dec_value.from_array(dec_nntile)
    dec_torch = torch.Tensor(dec_nntile.T)
    dec_torch.requires_grad_()

    # Initialize NNTile model from PyTorch model
    nntile_model, _ = T5Model.from_torch(
        torch_model, enc_X, dec_X, nntile_config, next_tag=0
    )

    # for test_forward_decoder_only
    nntile_model.encoder.activations[-1].value.from_array(enc_nntile)

    # Generate random gradient for backward pass
    dec_grad_random = gen.standard_normal(dec_shape, dtype=np.float32)
    dec_grad_nntile = np.array(dec_grad_random, dtype=np.float32, order="F")

    nntile_model.clear_gradients()
    nntile_model.activations[-1].grad.from_array(dec_grad_nntile)

    dec_grad_torch = torch.Tensor(dec_grad_nntile.T)

    # Create encoder attention mask for PyTorch model
    enc_attn_mask = torch.ones((params.n_batch, params.enc_seq_len), dtype=torch.long)
    dec_attn_mask = torch.ones((params.n_batch, params.dec_seq_len), dtype=torch.long)

    return (
        torch_model,
        nntile_model,
        enc_torch,
        dec_torch,
        dec_grad_torch,
        enc_attn_mask,
        dec_attn_mask,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        pytest.param(multiple_tiles, id="multiple_tiles"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
        pytest.param("fp32_fast_tf32", marks=nocuda),
        pytest.param("bf16", marks=nocuda),
    ],
)
class TestT5Model:
    def test_decoder_forward_only(
        self, starpu_simple, torch_rng, params: T5ModelTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_model, nntile_model, enc_x, dec_x, _, enc_attn_mask, dec_attn_mask = (
            generate_inputs(params, dtype)
        )

        # PyTorch forward pass
        y = torch_model(
            input_ids=None,
            decoder_input_ids=None,
            encoder_outputs=[enc_x],
            inputs_embeds=None,
            decoder_inputs_embeds=dec_x,
        )

        y_encoder = y.encoder_last_hidden_state
        y = y.last_hidden_state

        # NNTile forward pass
        nntile_model.decoder.forward_async()

        y_encoder_nntile = torch.Tensor(
            nntc.to_numpy(nntile_model.encoder.activations[-1].value).T
        )
        y_nntile = torch.Tensor(
            nntc.to_numpy(nntile_model.decoder.activations[-1].value).T
        )

        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y_encoder - y_encoder_nntile) <= rtol * torch.norm(y_encoder)
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

        # Clean up
        nntile_model.unregister()

    def test_forward(
        self, starpu_simple, torch_rng, params: T5ModelTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_model, nntile_model, enc_x, dec_x, _, enc_attn_mask, dec_attn_mask = (
            generate_inputs(params, dtype)
        )

        # PyTorch forward pass
        y = torch_model(
            input_ids=None,
            decoder_input_ids=None,
            encoder_outputs=None,
            inputs_embeds=enc_x,
            decoder_inputs_embeds=dec_x,
        )
        y_encoder = y.encoder_last_hidden_state
        y = y.last_hidden_state

        nntile_model.clear_gradients()

        # NNTile forward pass
        nntile_model.forward_async()

        y_encoder_nntile = torch.Tensor(
            nntc.to_numpy(nntile_model.encoder.activations[-1].value).T
        )
        y_nntile = torch.Tensor(nntc.to_numpy(nntile_model.activations[-1].value).T)

        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y_encoder - y_encoder_nntile) <= rtol * torch.norm(y_encoder)
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

        # Clean up
        nntile_model.unregister()

    def test_backward(
        self, starpu_simple, torch_rng, params: T5ModelTestParams, dtype: str
    ):
        """Test that backward pass gives same results in PyTorch and NNTile"""
        (
            torch_model,
            nntile_model,
            enc_x,
            dec_x,
            dec_grad,
            enc_attn_mask,
            dec_attn_mask,
        ) = generate_inputs(params, dtype)

        # PyTorch forward and backward pass
        y = torch_model(
            input_ids=None,
            attention_mask=enc_attn_mask,
            decoder_input_ids=None,
            decoder_attention_mask=dec_attn_mask,
            encoder_outputs=None,
            inputs_embeds=enc_x,
            decoder_inputs_embeds=dec_x,
        ).last_hidden_state

        res = (y * dec_grad).sum()
        res.backward()

        # NNTile forward and backward pass

        nntile_model.forward_async()
        nntile_model.backward_async()

        # Compare encoder gradients
        dec_grad_nntile = torch.Tensor(
            nntc.to_numpy(nntile_model.decoder.activations[0].grad).T
        )
        enc_grad_nntile = torch.Tensor(
            nntc.to_numpy(nntile_model.activations[0].grad).T
        )

        rtol = dtype2tol[dtype]["rtol"]

        assert torch.norm(enc_x.grad - enc_grad_nntile) <= rtol * torch.norm(enc_x.grad)
        assert torch.norm(dec_x.grad - dec_grad_nntile) <= rtol * torch.norm(dec_x.grad)

        # Clean up
        nntile_model.unregister()

    def test_t5_to_torch_for_sequence_classification(
        self, starpu_simple, torch_rng, params: T5ModelTestParams, dtype: str
    ):
        """Test that converting T5ForSequenceClassification to torch and back gives same results"""
        # Create PyTorch T5ForSequenceClassification model
        torch_config = T5ConfigTorch(
            d_model=params.d_model,
            d_ff=params.d_ff,
            d_kv=params.d_kv,
            num_heads=params.n_head,
            dropout_rate=0.0,
            dense_act_fn="gelu_new",
            is_gated_act=params.is_gated_act,
            num_layers=params.num_layers,
            num_labels=2,  # Binary classification for testing
            attn_implementation="eager",
            decoder_start_token_id=0,
            pad_token_id=0,
        )
        torch_model = T5ForSequenceClassificationTorch(torch_config)
        torch_model.eval()

        # Configure NNTile T5 model config
        nntile_config = T5ConfigNNTile(
            d_model=params.d_model,
            d_model_tile=params.d_model_tile,
            d_kv=params.d_kv,
            d_kv_tile=params.d_kv_tile,
            d_ff=params.d_ff,
            d_ff_tile=params.d_ff_tile,
            n_head=params.n_head,
            n_head_tile=params.n_head_tile,
            redux=params.redux,
            num_layers=params.num_layers,
            is_gated_act=params.is_gated_act,
            dtype=dtype,
        )

        # Set encoder input tensor dimensions
        # Set encoder input tensor dimensions
        enc_shape = [params.enc_seq_len, params.n_batch]
        enc_basetile = [params.enc_seq_len_tile, params.n_batch_tile]
        enc_traits = TensorTraits(enc_shape, enc_basetile)
        enc_distr = [0] * enc_traits.grid.nelems
        enc_value = nntile.tensor.Tensor_int64(enc_traits, enc_distr, 0)
        enc_grad = nntile.tensor.Tensor_int64(enc_traits, enc_distr, 0)
        enc_X = TensorMoments(enc_value, None, False)

        # Set decoder input tensor dimensions
        dec_shape = [params.dec_seq_len, params.n_batch]
        dec_basetile = [params.dec_seq_len_tile, params.n_batch_tile]
        dec_traits = TensorTraits(dec_shape, dec_basetile)
        dec_distr = [0] * dec_traits.grid.nelems
        dec_value = nntile.tensor.Tensor_int64(dec_traits, dec_distr, 0)
        dec_grad = nntile.tensor.Tensor_int64(dec_traits, dec_distr, 0)
        dec_X = TensorMoments(dec_value, None, False)

        # Convert to NNTile model
        nntile_model, _ = T5ForSequenceClassification.from_torch(
            torch_model, enc_X, dec_X, nntile_config, next_tag=0
        )

        # Convert back to PyTorch
        torch_model_converted = nntile_model.to_torch()

        # Generate random input data
        gen = np.random.default_rng(42)
        enc_random = gen.integers(0, 100, enc_shape, dtype=np.int64)
        dec_random = gen.integers(0, 100, dec_shape, dtype=np.int64)
        enc_torch = torch.Tensor(enc_random.T).long()
        dec_torch = torch.Tensor(dec_random.T).long()

        text = "This movie was fantastic!"
        # Initialize tokenizer from transformers
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        # Run forward pass on both models
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            y_original = torch_model(
                **inputs,
                use_cache=False,
            ).logits

            inputs = tokenizer(text, return_tensors="pt")
            y_converted = torch_model_converted(
                **inputs,
                use_cache=False,
            ).logits

        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y_original - y_converted) <= rtol * torch.norm(y_original)

        # Clean up
        nntile_model.unregister()
