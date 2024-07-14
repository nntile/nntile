import nntile
import numpy as np


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
    head_size, n_seq, n_b, num_head = 8, 4, 4, 2
    n_emb = head_size * num_head
    head_size_tile = 4
    n_seq_tile = 2
    n_batch_tile = 2
    num_head_tile = 1
    A_shape = [head_size, n_seq, n_b, num_head]
    tile_shape = [head_size_tile, n_seq_tile, n_batch_tile, num_head_tile]
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

    # Define mlp_mixer layer
    layer, next_tag = rope_layer.generate_simple(A_moments, next_tag)
    A.from_array(np_A)

    rand_freqs = np.random.normal(loc=0, scale=1,size=(int(head_size / 2),n_seq,n_b))
    np_cos = np.array(np.cos(rand_freqs), dtype=dtype, order='F')
    np_sin = np.array(np.sin(rand_freqs), dtype=dtype, order='F')
    layer.cos.from_array(np_cos)
    layer.sin.from_array(np_sin)

    # layer.clear_gradients()
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_output = np.empty(np_A.shape)
    for k in range(num_head):
        for b in range(n_b):
            for j in range(n_seq):
                for i in range(int(head_size/2)):
                    x = np.copy(np_A[2*i, j, b, k])
                    y = np.copy(np_A[2*i+1, j, b, k])
                    np_output[2*i, j, b, k] = np_cos[i,j,b] * x - np_sin[i,j,b] * y
                    np_output[2*i+1, j, b, k] = np_sin[i,j,b] * x + np_cos[i, j, b] * y


    np_nntile_out = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
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
