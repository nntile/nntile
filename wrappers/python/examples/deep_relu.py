# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_relu.py
# Example of using Deep ReLU network of NNTile Python package
#
# @version 1.1.0

import time

import numpy as np

# Imports
import nntile
import nntile.optimizer as opt

time0 = -time.time()

# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0

# Define matrix A as a Hilbert matrix
n_rows = 1024
n_cols = 1024
n_batches = 10  # Number of batches in a single epoch
batch_size = 128
n_cols_tile = 512
n_rows_tile = 512
batch_size_tile = 128
gemm_ndim = 1
hidden_layer_dim = 128  # Rank of approximation
hidden_layer_dim_tile = 128
nlayers = 2
n_epochs = 1000
# Number of FLOPs for 2 layers only
n_flops = (
    n_epochs
    * 2
    * hidden_layer_dim
    * (3 * n_rows + 2 * n_cols)
    * batch_size
    * n_batches
)
lr = 1e-7
A = np.zeros((n_rows, n_cols), order="F", dtype=np.float32)
for i in range(n_rows):
    for j in range(n_cols):
        A[i, j] = 1.0 / (i + j + 1)

# Define tensors for batches of X and Y
batch_input = []
batch_output = []
x_traits_full = nntile.tensor.TensorTraits(
    [n_cols, batch_size], [n_cols, batch_size]
)
x_full = nntile.tensor.Tensor_fp32(x_traits_full, [0], next_tag)
next_tag = x_full.next_tag
y_traits_full = nntile.tensor.TensorTraits(
    [n_rows, batch_size], [n_rows, batch_size]
)
y_full = nntile.tensor.Tensor_fp32(y_traits_full, [0], next_tag)
next_tag = y_full.next_tag
rng = np.random.Generator(44)

# Define traits of distributed input and output batches
x_traits = nntile.tensor.TensorTraits(
    [n_cols, batch_size], [n_cols_tile, batch_size_tile]
)
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits(
    [n_rows, batch_size], [n_rows_tile, batch_size_tile]
)
y_distr = [0] * y_traits.grid.nelems

# Define tensor X for input batches
# It shall move into DeepLinear generator in some future
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x_grad.next_tag
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
m = nntile.model.DeepReLU(
    x_moments,
    "R",
    gemm_ndim,
    hidden_layer_dim,
    hidden_layer_dim_tile,
    nlayers,
    next_tag,
)
next_tag = m.next_tag

# Set up learning rate and optimizer for training
# optimizer = opt.SGD(m.get_parameters(), lr, next_tag, momentum=0.9,
#        nesterov=False, weight_decay=1e-6)
optimizer = opt.Adam(m.get_parameters(), lr, next_tag)
next_tag = optimizer.get_next_tag()

# Set up Frobenius loss function for the model
frob, next_tag = nntile.loss.Frob.generate_simple(m.activations[-1], next_tag)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(
    batch_input, batch_output, m, optimizer, frob, n_epochs, lr
)

for i in range(n_batches):
    # Generate input and output batches
    X = rng.standard_normal(n_cols, batch_size)
    Y = A @ X
    # Wrap numpy tensor into NNTile tensor
    x_full.from_array(X)
    # Scatter input batch
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    nntile.nntile_core.tensor.scatter_fp32(x_full, x)
    batch_input.append(x)
    # Wrap numpy tensor into NNTile tensor
    y_full.from_array(Y)
    # Scatter output batch
    y = nntile.tensor.Tensor_fp32(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    nntile.nntile_core.tensor.scatter_fp32(y_full, y)
    batch_output.append(y)

# Full tensors (stored as a single tile) are not needed anymore
x_full.unregister()
y_full.unregister()

time0 += time.time()
print("Finish generating in {} seconds".format(time0))
# Wait for all computations to finish
nntile.starpu.wait_for_all()

# Randomly init weights of deep linear network
time0 = -time.time()
m.init_randn_async()

# Wait for all computations to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("Finish random weights init in {} seconds".format(time0))

# Start timer and run training
time0 = -time.time()
pipeline.train_async()
time0 += time.time()
print("Finish adding tasks in {} seconds".format(time0))

# Wait for all computations to finish
time0 = -time.time()
nntile.starpu.wait_for_all()
time0 += time.time()
print("Done in {} seconds".format(time0))
np_val = np.array([1], order="F", dtype=np.float32)
np_val[0] = 0
frob.val.to_array(np_val)
nntile.starpu.wait_for_all()
print("Loss is {}".format(np_val[0]))
print("Norm is {}".format(np.linalg.norm(Y, "fro")))
print("Total GFLOP/s: {}".format(n_flops * 1e-9 / time0))

# Unregister all tensors related to model
m.unregister()

# Unregister optimizer states
optimizer.unregister()

# Unregister loss function
frob.y.unregister()
frob.val.unregister()
frob.tmp.unregister()

# Unregister input/output batches
for x in batch_input:
    x.unregister()
for x in batch_output:
    x.unregister()
