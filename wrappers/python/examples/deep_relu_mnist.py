# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_relu.py
# Deep ReLU network for digist classification of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-21

# Imports
import nntile
import numpy as np
import time
import sys

time0 = -time.time()

# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0

# Describe MNIST dataset
mnist_images_train = "train-images-idx3-ubyte"
mnist_labels_train = "train-labels-idx1-ubyte"
mnist_images_test = "t10k-images-idx3-ubyte"
mnist_labels_test = "t10k-labels-idx1-ubyte"

# Function to read images
def read_images(fname):
    print("Start reading ", fname)
    with open(fname, "rb") as fd:
        raw = fd.read()
    print("Finish reading ", fname)
    print("Start parsing ", fname)
    magic_number = int(raw[:4].hex(), 16)
    assert(magic_number == 2051)
    n_images = int(raw[4:8].hex(), 16)
    n_pixels_row = int(raw[8:12].hex(), 16)
    n_pixels_column = int(raw[12:16].hex(), 16)
    n_pixels = n_pixels_row * n_pixels_column
    data_uint = np.frombuffer(raw, "B", offset=16).reshape(n_pixels, n_images)
    data = np.array(data_uint, dtype=np.float32, order='F') / 255
    print("Finish parsing ", fname)
    return data

# Function to read labels
def read_labels(fname):
    print("Start reading ", fname)
    with open(fname, "rb") as fd:
        raw = fd.read()
    print("Finish reading ", fname)
    print("Start parsing ", fname)
    magic_number = int(raw[:4].hex(), 16)
    assert(magic_number == 2049)
    n_images = int(raw[4:8].hex(), 16)
    data_uint = np.frombuffer(raw, "B", offset=8)
    data = np.array(data_uint, dtype=np.int64, order='F')
    print("Finish parsing ", fname)
    return data

# Read train and test images
time0 = -time.time()
print("Start reading data")
data_train = read_images(mnist_images_train)
data_test = read_images(mnist_images_test)
labels_train = read_labels(mnist_labels_train)
labels_test = read_labels(mnist_labels_test)
time0 += time.time()
print("Finish reading data in {} seconds".format(time0))
time0 = -time.time()

# Get sizes from data set
n_images_train = data_train.shape[1]
assert(labels_train.shape[0] == n_images_train)
n_images_test = data_test.shape[1]
assert(labels_test.shape[0] == n_images_test)
n_pixels = data_train.shape[0]
assert(data_test.shape[0] == n_pixels)
n_images_batch = 10000
n_batches = n_images_train // n_images_batch
if n_images_train != n_batches*n_images_batch:
    raise ValueError("Wrong batch size")
n_labels = labels_train.max() + 1

# Define tile sizes
n_pixels_tile = 392
n_images_train_tile = 1000
n_images_test_tile = 1000

# Describe neural network
gemm_ndim = 1
hidden_layer_dim = 1000
hidden_layer_dim_tile = 500
n_layers = 10
n_epochs = 10
lr = 1e-7

# Number of FLOPs for training
n_flops_train_first_layer = 2 * 2 * n_pixels * n_images_batch \
        * hidden_layer_dim # once for forward, once for backward
n_flops_train_mid_layer = 3 * 2 * hidden_layer_dim * n_images_batch \
        * hidden_layer_dim # once for forward, twice for backward
n_flops_train_last_layer = 3 * 2 * n_labels * n_images_batch \
        * hidden_layer_dim # once for forward, twice for backward

# Define tensors for batches of X and Y
data_train_traits = nntile.tensor.TensorTraits(data_train.shape, \
        data_train.shape)
data_train_tensor = nntile.tensor.Tensor_fp32(data_train_traits, [0], next_tag)
next_tag = data_train_tensor.next_tag
data_train_tensor.from_array(data_train)
labels_train_traits = nntile.tensor.TensorTraits(labels_train.shape, \
        labels_train.shape)
labels_train_tensor = nntile.tensor.Tensor_fp32(labels_train_traits, [0], \
        next_tag)
labels_train_tensor.from_array(labels_train)
next_tag = labels_train_tensor.next_tag
batch_data = []
batch_labels = []
x_traits = nntile.tensor.TensorTraits([n_pixels, n_images_batch], \
        [n_pixels_tile, n_images_train_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits([n_images_batch], [n_images_train_tile])
y_distr = [0] * y_traits.grid.nelems
for i in range(n_batches):
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    nntile.tensor.copy_intersection_async(data_train_tensor, [0, 0], x, \
            [0, i*n_images_batch])
    batch_data.append(x)
    y = nntile.tensor.Tensor_fp32(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    nntile.tensor.copy_intersection_async(labels_train_tensor, [0], y, \
            [i*n_images_batch])
    batch_labels.append(y)

# Unregister single-tile tensors
nntile.starpu.wait_for_all()
data_train_tensor.unregister()
labels_train_tensor.unregister()

# Define tensor X for input batches
# It shall move into DeepLinear generator in some future
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
m = nntile.model.DeepReLU(x_moments, 'R', gemm_ndim, hidden_layer_dim,
        hidden_layer_dim_tile, n_layers, next_tag)
next_tag = m.next_tag

# Set up learning rate and optimizer for training
#optimizer = opt.SGD(m.get_parameters(), lr, next_tag, momentum=0.9,
#        nesterov=False, weight_decay=1e-6)
optimizer = nntile.optimizer.Adam(m.get_parameters(), lr, next_tag)
next_tag = optimizer.get_next_tag()

# Set up Frobenius loss function for the model
frob, next_tag = nntile.loss.Frob.generate_simple(m.activations[-1], next_tag)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_data, batch_labels, m, optimizer,
        frob, n_epochs, lr)

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

## Start timer and run training
#time0 = -time.time()
#pipeline.train_async()
#time0 += time.time()
#print("Finish adding tasks in {} seconds".format(time0))
#
## Wait for all computations to finish
#time0 = -time.time()
#nntile.starpu.wait_for_all()
#time0 += time.time()
#print("Done in {} seconds".format(time0))
#np_val = np.array([1], order='F', dtype=np.float32)
#np_val[0] = 0
#frob.val.to_array(np_val)
#print("Loss is {}".format(np_val[0]))
#print("Norm is {}".format(np.linalg.norm(Y, 'fro')))
#print("Total GFLOP/s: {}".format(n_flops*1e-9/time0))

# Unregister all tensors related to model
m.unregister()

# Unregister optimizer states
optimizer.unregister()

# Unregister loss function
frob.y.unregister()
frob.val.unregister()
frob.tmp.unregister()

# Unregister input/output batches
for x in batch_data:
    x.unregister()
for x in batch_labels:
    x.unregister()

