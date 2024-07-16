# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# Test for nntile.layer.rotary_embedding
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2024-07-14

# All necesary imports
import nntile
import numpy as np
from numpy.random import default_rng
import torch
from nntile.torch_models.llama_attn import apply_rotary_pos_emb
from nntile.torch_models.llama_attn import LlamaRotaryEmbedding as TorchRope

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}

# Get MixerMlp layer from nntile
rope_layer = nntile.layer.RotaryEmbedding

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    rng = default_rng()

    head_size, n_seq, n_batch, n_head = 16, 4, 4, 4
    half_hs = int(head_size / 2)
    max_pos_emb = 256
    theta = 2.0
    rand_input = np.random.rand(n_batch, n_head, n_seq, head_size)
    np_input = np.array(rand_input, dtype=dtype, order='F')

    pos = [rng.choice(max_pos_emb-1, size=n_seq, replace=False) for _ in range(n_batch)]
    pos = np.array(pos)
    
    # PyTorch part
    torch_input = torch.from_numpy(np_input)
    pos_ix = torch.tensor(pos, dtype=torch.long)
    RopeLayer = TorchRope(dim=head_size,max_position_embeddings=max_pos_emb,base=theta)

    permuted_input = torch.empty(*torch_input.shape)
    for i in range(half_hs):
        permuted_input[..., i] = torch_input[..., 2*i]
        permuted_input[..., half_hs + i] = torch_input[..., 2*i + 1]
    
    cos, sin = RopeLayer.forward(permuted_input,pos_ix)
    torch_permuted_output = apply_rotary_pos_emb(permuted_input, cos, sin)

    torch_output = torch.empty(*torch_permuted_output.shape)
    for i in range(half_hs):
        torch_output[..., 2*i] = torch_permuted_output[..., i]
        torch_output[..., 2*i+1] = torch_permuted_output[..., half_hs + i]
    np_torch_output = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)


    # NNTile part
    head_size_tile = 8
    n_seq_tile = 2
    n_batch_tile = 2
    n_head_tile = 2
    A_shape = [head_size, n_seq, n_batch, n_head]
    tile_shape = [head_size_tile, n_seq_tile, n_batch_tile, n_head_tile]
    A_traits = nntile.tensor.TensorTraits(A_shape, tile_shape)
    mpi_distr = [0] * A_traits.grid.nelems
    next_tag = 0

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag

    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    np_tmp = np.swapaxes(np_input, 2,3)
    nntile_input = np.moveaxis(np_tmp, [0,1,2,3], [2,3,0,1])

    # Define layer
    layer, next_tag = rope_layer.generate_simple(A_moments, pos, theta, next_tag)
    A.from_array(nntile_input)

    # inv_freq = 1.0 / (2 ** (np.arange(0, head_size, 2, dtype=dtype) / head_size))
    # freq_frame = np.empty((int(half_hs), n_seq, n_batch))
    # for i in range(n_batch):
    #     freq_frame[:,:,i] = np.outer(inv_freq, pos_ix[i, :])
    # np_freqs = np.array(freq_frame, dtype=dtype, order='F')
    # np_cos = np.cos(np_freqs)
    # np_sin = np.sin(np_freqs)
 
    # layer.cos.from_array(np_cos)
    # layer.sin.from_array(np_sin)

    # layer.clear_gradients()
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_nntile_out = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
    layer.y.value.to_array(np_nntile_out)
    
    np_tmp = np.copy(np.moveaxis(np_nntile_out, [2,3,0,1], [0,1,2,3]))
    np_nntile_out = np.swapaxes(np_tmp, 3,2)

    if np.linalg.norm(np_torch_output-np_nntile_out)/np.linalg.norm(np_torch_output) > tol:
        print('whoooops')
        A_moments.unregister()
        layer.unregister()
        return False

    A_moments.unregister()
    layer.unregister()
    print("test complete")
    assert True

# Test runner for different precisions
def test():
    dtype = np.float32
    helper(dtype)


if __name__ == "__main__":
    test()
