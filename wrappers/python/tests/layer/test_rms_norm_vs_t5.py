# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_rms_norm_vs_t5.py
# Comparison test for NNTile's RMSNorm vs HuggingFace's T5LayerNorm
#
# @version 1.1.0

import numpy as np
import pytest
import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm as T5LayerNormTorch

import nntile
from nntile.layer.rms_norm import RMSNorm
from nntile.tensor import TensorMoments, TensorTraits
import nntile.utils.constructors as nntc

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "bf16": nntile.tensor.Tensor_bf16,
}

dtype2tol = {
    "fp32": {"rtol": 1e-5, "atol": 1e-7},
    "fp32_fast_tf32": {"rtol": 1e-3, "atol": 1e-5},
    "bf16": {"rtol": 1e-2, "atol": 1e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")


def generate_test_data(d_model=512, seq_len=64, batch_size=2, dtype="fp32", redux=True):
    """Generate test data for comparing T5LayerNorm and RMSNorm"""
    # Configure PyTorch T5LayerNorm
    torch_norm = T5LayerNormTorch(d_model)
    
    # Set to evaluation mode
    torch_norm.eval()
    
    # Generator for random values
    gen = np.random.default_rng(42)
    
    # Create input tensor dimensions for NNTile
    shape = [d_model, seq_len, batch_size]
    basetile = shape.copy()  # Single tile for simplicity
    traits = TensorTraits(shape, basetile)
    distr = [0] * traits.grid.nelems
    tensor_type = dtype2nntile[dtype]
    value = tensor_type(traits, distr, 0)
    grad = tensor_type(traits, distr, 0)
    x = TensorMoments(value, grad, grad_required=True)
    nntile.functions.fill_async(0.0, x.grad)
    
    # Generate random input data
    random_data = gen.standard_normal(shape, dtype=np.float32)
    nntile_data = np.array(random_data, dtype=np.float32, order="F")
    value.from_array(nntile_data)
    
    # Create equivalent PyTorch tensor (note the transpose for dimension ordering)
    torch_data = torch.tensor(nntile_data.T, requires_grad=True)
    
    # Initialize NNTile RMSNorm layer
    rms_norm, _ = RMSNorm.from_torch(torch_norm, x, 0, 1e-6, 0, redux=redux)
    rms_norm.clear_gradients()
    
    # Generate random gradient for backward pass
    grad_random = gen.standard_normal(shape, dtype=np.float32)
    grad_nntile = np.array(grad_random, dtype=np.float32, order="F")
    rms_norm.activations_output[0].grad.from_array(grad_nntile)
    grad_torch = torch.tensor(grad_nntile.T)
    
    return torch_norm, rms_norm, torch_data, x, grad_torch


@pytest.mark.parametrize(
    "d_model, seq_len, batch_size", 
    [
        (512, 64, 1),  # Simple case
        (768, 128, 4),  # Larger model
        (1024, 32, 8),  # Different dimensions
    ]
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
        # pytest.param("fp32_fast_tf32", marks=nocuda),
        # pytest.param("bf16", marks=nocuda),
    ],
)
@pytest.mark.parametrize("redux", [False]) # [True, False])
class TestRMSNormVsT5:
    
    def test_forward(self, starpu_simple, d_model, seq_len, batch_size, dtype, redux):
        """Test that forward pass gives the same results for T5LayerNorm and RMSNorm"""
        torch_norm, rms_norm, torch_data, x, _ = generate_test_data(
            d_model, seq_len, batch_size, dtype, redux
        )
        
        # PyTorch forward pass
        torch_output = torch_norm(torch_data)
        
        # NNTile forward pass
        rms_norm.forward_async()
        nntile_output = torch.tensor(nntc.to_numpy(rms_norm.activations_output[0].value).T)
        
        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        atol = dtype2tol[dtype]["atol"]
        
        # Print some sample values for debugging
        print(f"PyTorch output sample: {torch_output.flatten()[:5]}")
        print(f"NNTile output sample: {nntile_output.flatten()[:5]}")
        
        assert torch.allclose(torch_output, nntile_output, rtol=rtol, atol=atol), \
            f"Forward pass mismatch: {torch.max(torch.abs(torch_output - nntile_output))}"
        
        # Clean up
        rms_norm.unregister()
    
    def test_backward(self, starpu_simple, d_model, seq_len, batch_size, dtype, redux):
        """Test that backward pass gives the same gradients for T5LayerNorm and RMSNorm"""
        torch_norm, rms_norm, torch_data, x, grad_torch = generate_test_data(
            d_model, seq_len, batch_size, dtype, redux
        )
        
        # PyTorch forward and backward pass
        torch_output = torch_norm(torch_data)
        loss = (torch_output * grad_torch).sum()
        loss.backward()
        
        # NNTile forward and backward pass
        rms_norm.forward_async()
        rms_norm.backward_async()
        
        # Compare input gradients
        nntile_grad = torch.tensor(nntc.to_numpy(x.grad).T)
        
        rtol = dtype2tol[dtype]["rtol"]
        atol = dtype2tol[dtype]["atol"]
        
        # Print some sample gradients for debugging
        print(f"PyTorch input grad sample: {torch_data.grad.flatten()[:5]}")
        print(f"NNTile input grad sample: {nntile_grad.flatten()[:5]}")
        
        assert torch.allclose(torch_data.grad, nntile_grad, rtol=rtol, atol=atol), \
            f"Backward pass mismatch: {torch.max(torch.abs(torch_data.grad - nntile_grad))}"
        
        # Clean up
        rms_norm.unregister()

    def test_numerical_stability(self, starpu_simple, d_model, seq_len, batch_size, dtype, redux):
        """Test numerical stability with small values"""
        torch_norm, rms_norm, torch_data, x, _ = generate_test_data(
            d_model, seq_len, batch_size, dtype, redux
        )
        
        # Create very small values
        small_values = torch.ones_like(torch_data) * 1e-10
        small_values.requires_grad_(True)
        
        # Convert to NNTile format
        small_nntile = np.array(small_values.detach().numpy().T, dtype=np.float32, order="F")
        x.value.from_array(small_nntile)
        
        # PyTorch forward pass
        torch_output = torch_norm(small_values)
        
        # NNTile forward pass
        rms_norm.forward_async()
        nntile_output = torch.tensor(nntc.to_numpy(rms_norm.activations_output[0].value).T)
        
        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        atol = dtype2tol[dtype]["atol"]
        
        assert torch.allclose(torch_output, nntile_output, rtol=rtol, atol=atol), \
            f"Small values test failed: {torch.max(torch.abs(torch_output - nntile_output))}"
        
        # Clean up
        rms_norm.unregister() 