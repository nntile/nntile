# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_relu.py
# Deep ReLU network for image classification with NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-03-27

# Imports
import nntile
import numpy as np
import time
import sys
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch

dataset = "mnist"
#dataset = "cifar10"

if dataset == "mnist":
        batch_size = 2000
        trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0,), (255,))])
        mnist_train_set = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
        train_loader = torch.utils.data.DataLoader(mnist_train_set, batch_size=batch_size, shuffle=True)
        mnist_test_set = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
        test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=batch_size, shuffle=True)

        n_pixels = 28 * 28
        # Define tile sizes
        n_pixels_tile = 392
        n_classes = 10
elif dataset == "cifar10":
        batch_size = 2000
        transform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.1307,), (0.3081,))])
        cifar10_train_set = dts.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True)
        cifar10_test_set = dts.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size, shuffle=True)
        n_pixels = 32 * 32 * 3
        n_pixels_tile = n_pixels // 2
        n_classes = 10
else:
     raise ValueError("{} dataset is not supported yet!".format(dataset))

time0 = -time.time()
# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0


n_images_train_tile = 500
n_images_test_tile = 500

# Describe neural network
gemm_ndim = 1
hidden_layer_dim = 2000
hidden_layer_dim_tile = 500
n_layers = 5
n_epochs = 10
lr = 1e-2


# Number of FLOPs for training per batch
n_flops_train_first_layer = 2 * 2 * n_pixels * batch_size \
        * hidden_layer_dim # once for forward, once for backward
n_flops_train_mid_layer = 3 * 2 * hidden_layer_dim * batch_size \
        * hidden_layer_dim # once for forward, twice for backward
n_flops_train_last_layer = 3 * 2 * n_classes * batch_size \
        * hidden_layer_dim # once for forward, twice for backward
n_flops = n_flops_train_first_layer + (n_layers-2)*n_flops_train_mid_layer \
        + n_flops_train_last_layer
# Multiply by number of epochs and batches
n_flops *= n_epochs * len(train_loader)

time0 = -time.time()
batch_data = []
batch_labels = []
x_single_traits = nntile.tensor.TensorTraits([batch_size, n_pixels], \
        [batch_size, n_pixels])
x_single_distr = [0]
x_single = nntile.tensor.Tensor_fp32(x_single_traits, x_single_distr, next_tag)
next_tag = x_single.next_tag
y_single_traits = nntile.tensor.TensorTraits([batch_size], [batch_size])
y_single_distr = [0]
y_single = nntile.tensor.Tensor_int64(y_single_traits, y_single_distr, \
        next_tag)
next_tag = y_single.next_tag
x_traits = nntile.tensor.TensorTraits([batch_size, n_pixels], \
        [n_images_train_tile, n_pixels_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits([batch_size], [n_images_train_tile])
y_distr = [0] * y_traits.grid.nelems
for train_batch_data, train_batch_labels in train_loader:
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x_single.from_array(train_batch_data.view(batch_size, n_pixels).numpy())
    nntile.tensor.scatter_async(x_single, x)
    batch_data.append(x)
    y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    y_single.from_array(train_batch_labels.numpy())
    nntile.tensor.scatter_async(y_single, y)
    batch_labels.append(y)

# Wait for all scatters to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("From PyTorch loader to NNTile batches in {} seconds".format(time0))

# Define tensor X for input batches
# It shall move into DeepLinear generator in some future
time0 = -time.time()
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
m = nntile.model.DeepReLU(x_moments, 'L', gemm_ndim, hidden_layer_dim, \
        hidden_layer_dim_tile, n_layers, n_classes, next_tag)
next_tag = m.next_tag
# Set up learning rate and optimizer for training
# optimizer = nntile.optimizer.SGD(m.get_parameters(), lr, next_tag, momentum=0.9,
#        nesterov=False, weight_decay=0.)
optimizer = nntile.optimizer.Adam(m.get_parameters(), lr, next_tag)
next_tag = optimizer.get_next_tag()

# Set up Frobenius loss function for the model
# frob, next_tag = nntile.loss.Frob.generate_simple(m.activations[-1], next_tag)
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(m.activations[-1], \
        next_tag)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_data, batch_labels, m, optimizer, \
        loss, n_epochs)

time0 += time.time()
print("Finish generating pipeline (model, loss and optimizer) in {} seconds" \
        .format(time0))

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
print("Finish adding tasks (computations are running) in {} seconds" \
        .format(time0))
#
## Wait for all computations to finish
time1 = -time.time()
nntile.starpu.wait_for_all()
time1 += time.time()
print("All computations done in {} + {} = {} seconds".format(time0, time1, \
        time0 + time1))
print("Train GFLOPs/s (based on gemms): {}" \
        .format(n_flops * 1e-9 / (time0+time1)))

# Get inference rate based on train data
time0 = -time.time()
for x in batch_data:
    nntile.tensor.copy_async(x, m.activations[0].value)
    m.forward_async()
nntile.starpu.wait_for_all()
time0 += time.time()
# FLOPS for inference over the first layer per batch
n_flops_inference = 2 * n_pixels * batch_size * hidden_layer_dim
# FLOPS for inference over each middle layer per batch
n_flops_inference += (n_layers-2) * 2 * hidden_layer_dim * batch_size \
        * hidden_layer_dim
# FLOPS for inference over the last layer per batch
n_flops_inference += 2 * n_classes * batch_size * hidden_layer_dim
# Multiply FLOPS per number of batches
n_flops_inference *= len(train_loader)
print("Inference speed: {} samples/second".format(\
        len(train_loader) * batch_size / time0))
print("Inference GFLOPs/s (based on gemms): {}" \
        .format(n_flops_inference * 1e-9 / time0))

# Compute test accuracy of the trained model
test_top1_accuracy = 0
total_num_samples = 0
z_single_traits = nntile.tensor.TensorTraits([batch_size, n_classes], \
        [batch_size, n_classes])
z_single_distr = [0]
z_single = nntile.tensor.Tensor_fp32(z_single_traits, z_single_distr, next_tag)
next_tag = z_single.next_tag
for test_batch_data, test_batch_label in test_loader:
    x_single.from_array(test_batch_data.view(-1, n_pixels).numpy())
    nntile.tensor.scatter_async(x_single, m.activations[0].value)
    m.forward_async()
    nntile.tensor.gather_async(m.activations[-1].value, z_single)
    output = np.zeros(z_single.shape, order="F", dtype=np.float32)
    # to_array causes y_single to finish gather procedure
    z_single.to_array(output)
    pred_labels = np.argmax(output, 1)
    test_top1_accuracy += np.sum(pred_labels == test_batch_label.numpy())
    total_num_samples += test_batch_label.shape[0]
test_top1_accuracy /= total_num_samples

print("Test accuracy of the trained Deep ReLU model =", test_top1_accuracy)

# Unregister single-tile tensors for data scattering/gathering
x_single.unregister()
y_single.unregister()
z_single.unregister()

# Unregister all tensors related to model
m.unregister()

# Unregister optimizer states
optimizer.unregister()

# Unregister loss function
loss.unregister()

# Unregister input/output batches
for x in batch_data:
    x.unregister()
for x in batch_labels:
    x.unregister()
