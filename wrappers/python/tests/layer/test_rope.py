import nntile
import numpy as np
import torch.nn.functional as F
import torch


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
    # Describe single-tile tensor, located at node 0
    hs, n_seq, n_b, num_head = 8, 4, 1, 2
    n_emb = hs * num_head 
    A_shape = [hs, n_seq, n_b, num_head]
    tile_shape = [4,2,1,2]
    A_traits = nntile.tensor.TensorTraits(A_shape, tile_shape)
    mpi_distr = [0] * A_traits.grid.nelems
    next_tag = 0
    
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag

    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    print('Before generate simple')
    # Define mlp_mixer layer
    layer, next_tag = rope_layer.generate_simple(A_moments, next_tag)
    print('generate simple complete')
    A.from_array(np_A)

    # inv_freq = 1.0 / (10000 ** (np.arange(0, 8, 2, dtype=np.int64).float() / n_emb))
    # t = np.arange(n_seq, dtype=np.int64).type_as(inv_freq)
    # freqs = np.outer(inv_freq, t)

    # freqs_tensor = np.empty(hs, n_seq, nh)
    # for i in range(nh):
    #     freqs_tensor[:,:,i] = freqs[hs*i : hs*(i+1)]

    rand_freqs = np.random.normal(loc=0, scale=1,size=(4,4,2))
    np_cos = np.array(np.cos(rand_freqs), dtype=dtype, order='F')
    np_sin = np.array(np.sin(rand_freqs), dtype=dtype, order='F')
    layer.cos.from_array(np_cos)
    layer.sin.from_array(np_sin)

    print("Right before forward")
    # layer.clear_gradients()
    layer.forward_async()
    print('forward done')
    nntile.starpu.wait_for_all()

    np_output = np.empty(np_A.shape)
    for k in range(num_head):
        for b in range(n_b):
            for j in range(n_seq):
                for i in range(int(hs/2)):
                    x = np.copy(np_output[2*i, j, b, k])
                    y = np.copy(np_output[2*i+1, j, b, k])
                    np_output[2*i, j, b, k] = np_cos[i,j,k] * x - np_sin[i,j,k] * y
                    np_output[2*i+1, j, b, k] = np_sin[i,j,k] * x + np_cos[i, j, k] * y


    np_nntile_out = np.zeros_like(np_output, order='F')
    layer.y.value.to_array(np_nntile_out)
    if np.linalg.norm(np_output-np_nntile_out)/np.linalg.norm(np_output) > tol:
        A_moments.unregister()
        layer.unregister()
        return False 

    A_moments.unregister()
    layer.unregister()
    print("test complete")
    assert True


# Test runner for different precisions
def test():
    for dtype in dtypes:
        helper(dtype)


if __name__ == "__main__":
    test()