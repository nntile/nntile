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
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch

batch_size = 1000

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])
mnisttrainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
trainldr = torch.utils.data.DataLoader(mnisttrainset, batch_size=batch_size, shuffle=True)
mnist_test_set = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=batch_size, shuffle=True)

time0 = -time.time()

# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0

n_pixels = 28 * 28
# Define tile sizes
n_pixels_tile = 784
n_images_train_tile = 1000
n_images_test_tile = 1000

# Describe neural network
gemm_ndim = 1
hidden_layer_dim = 100
hidden_layer_dim_tile = 10
n_layers = 5
n_epochs = 6
lr = 1e-2
n_classes = 10

# Number of FLOPs for training
n_flops_train_first_layer = 2 * 2 * n_pixels * batch_size \
        * hidden_layer_dim # once for forward, once for backward
n_flops_train_mid_layer = 3 * 2 * hidden_layer_dim * batch_size \
        * hidden_layer_dim # once for forward, twice for backward
n_flops_train_last_layer = 3 * 2 * n_classes * batch_size \
        * hidden_layer_dim # once for forward, twice for backward

batch_data = []
batch_labels = []
x_traits = nntile.tensor.TensorTraits([batch_size, n_pixels], \
        [n_images_train_tile, n_pixels_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits([batch_size], [n_images_train_tile])
y_distr = [0] * y_traits.grid.nelems
for train_batch, train_labels in trainldr:
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x.from_array(train_batch.view(-1, n_pixels).numpy() / 255.)
    batch_data.append(x)
    y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    y.from_array(train_labels.numpy())
    batch_labels.append(y)

# Unregister single-tile tensors
nntile.starpu.wait_for_all()

# Define tensor X for input batches
# It shall move into DeepLinear generator in some future
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
m = nntile.model.DeepReLU(x_moments, 'L', gemm_ndim, hidden_layer_dim,
        hidden_layer_dim_tile, n_layers, n_classes, next_tag)
next_tag = m.next_tag
print("Model is initialized!")
# Set up learning rate and optimizer for training
# optimizer = nntile.optimizer.SGD(m.get_parameters(), lr, next_tag, momentum=0.9,
#        nesterov=False, weight_decay=0.)
optimizer = nntile.optimizer.Adam(m.get_parameters(), lr, next_tag)
next_tag = optimizer.get_next_tag()

# Set up Frobenius loss function for the model
# frob, next_tag = nntile.loss.Frob.generate_simple(m.activations[-1], next_tag)
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(m.activations[-1],
                                                          next_tag)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_data, batch_labels, m, optimizer,
        loss, n_epochs)

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
time0 = -time.time()
pipeline.train_async()
time0 += time.time()
print("Finish adding tasks in {} seconds".format(time0))
#
## Wait for all computations to finish
time0 = -time.time()
nntile.starpu.wait_for_all()
time0 += time.time()
print("Done in {} seconds".format(time0))

# Compute test accuracy of the trained model
test_accuracy = 0
total_num_samples = 0
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
for test_batch, test_label in test_loader:
    x.from_array(test_batch.view(-1, n_pixels).numpy() / 255.)
    nntile.tensor.copy_async(x, m.activations[0].value)
    m.forward_async()
    output = np.zeros(m.activations[-1].value.shape, order="F", dtype=np.float32)
    m.activations[-1].value.to_array(output)
    pred_labels = np.argmax(output, 1)
    test_accuracy += np.sum(pred_labels == test_label.numpy())
    total_num_samples += test_label.shape[0]
test_accuracy /= total_num_samples
print("Test accuracy of the trained Deep ReLU model =", test_accuracy)
x.unregister()
#print("Total GFLOP/s: {}".format(n_flops*1e-9/time0))

# Unregister all tensors related to model
m.unregister()

# Unregister optimizer states
optimizer.unregister()

# Unregister loss function
loss.unregister()
# frob.y.unregister()
# frob.val.unregister()
# frob.tmp.unregister()

# Unregister input/output batches
for x in batch_data:
    x.unregister()
for x in batch_labels:
    x.unregister()

