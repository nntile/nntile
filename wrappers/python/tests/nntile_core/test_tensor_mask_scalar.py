# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_mask_scalar.py
# Test for tensor::mask_scalar<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @author Aleksandr Mikhalev
# @date 2023-06-22

# All necesary imports
import nntile
import numpy as np
import torch
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64,
        bool: nntile.tensor.Tensor_bool}
# Define mapping between tested function and numpy type
mask_scalar_func = {np.float32: nntile.nntile_core.tensor.mask_scalar_fp32,
                    np.float64: nntile.nntile_core.tensor.mask_scalar_fp64}

# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    shape = [3, 3, 1]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    # Tensor objects
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    mask_traits = nntile.tensor.TensorTraits(shape[:2], shape[:2])
    mask = Tensor[bool](mask_traits, mpi_distr, next_tag)
    # Set initial values of tensors
    rand_A = np.random.randn(*shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    print(np_A[:, :, 0])
    A.from_array(np_A)
    
    causal_mask = torch.tril(torch.ones((3, 3), dtype=torch.bool))
    mask_value = -1000.
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    if dtype == np.float32:
        mask_value_t = torch.full([], mask_value, dtype=torch.float32)
        torch_res = torch.where(causal_mask, torch.tensor(rand_A, dtype=torch.float32).permute(2,1,0), mask_value_t).permute(2,1,0)
    elif dtype == np.float64:
        mask_value_t = torch.full([], mask_value, dtype=torch.float64)
        torch_res = torch.where(causal_mask, torch.tensor(rand_A, dtype=torch.float64), mask_value_t)

    mask.from_array(np.array(causal_mask.numpy(), dtype=bool, order="F"))
    print(np.array(causal_mask.numpy(), dtype=bool, order="F"))
    mask_scalar_func[dtype](mask, dtype(mask_value), A)
    A.to_array(np_A)
    nntile.starpu.wait_for_all()
    A.unregister()
    mask.unregister()
    # Compare results
    print(np_A[:, :, 0], torch_res[:, :, 0])
    return (torch_res.numpy() == np_A).all()

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()
