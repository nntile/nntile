# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_relu_image_classification.py
# Deep ReLU network for image classification with NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-06-28

# Imports
import torch
import nntile
import numpy as np
import time
import sys
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import argparse

# Create argument parser
parser = argparse.ArgumentParser(prog="DeepReLU neural network", \
        description="This example trains NNTile version of DeepReLU neural "\
        "network from a scratch for an image classification task")
parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
parser.add_argument("--dataset_dir")
parser.add_argument("--batch", type=int)
parser.add_argument("--batch_tile", type=int)
parser.add_argument("--depth", type=int)
parser.add_argument("--hidden_dim", type=int)
parser.add_argument("--hidden_dim_tile", type=int)
parser.add_argument("--pixels_tile", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--epoch_warmup", type=int)
parser.add_argument("--lr", type=float)

# Parse arguments
args = parser.parse_args()
print(args)
fp32_fast_fp16 = False
fp32_convert_fp16 = True
deep_relu_mp = False # Do not enable it, under construction

if args.dataset == "mnist":
    dataset_transforms = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0,), (255,))])
    train_set = datasets.MNIST(root=args.dataset_dir, train=True, \
            download=True, transform=dataset_transforms)
    test_set = datasets.MNIST(root=args.dataset_dir, train=False, \
            download=True, transform=dataset_transforms)
    n_pixels = 28 * 28
    n_classes = 10
elif args.dataset == "cifar10":
    dataset_transforms = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.CIFAR10(root=args.dataset_dir, train=True, \
            download=True, transform=dataset_transforms)
    test_set = datasets.CIFAR10(root=args.dataset_dir, train=False, \
            download=True, transform=dataset_transforms)
    n_pixels = 32 * 32 * 3
    n_classes = 10
else:
    raise ValueError("{} dataset is not supported yet!".format(dataset))

train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

print("Number of train images: {}".format(len(train_loader) * args.batch))
print("Number of train batches: {}".format(len(train_loader)))
print("Number of test images: {}".format(len(test_loader) * args.batch))
print("Number of test batches: {}".format(len(test_loader)))

time0 = -time.time()
# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0

# Number of FLOPs for training per batch
n_flops_train_first_layer = 2 * 2 * n_pixels * args.batch \
        * args.hidden_dim # once for forward, once for backward
n_flops_train_mid_layer = 3 * 2 * args.hidden_dim * args.batch \
        * args.hidden_dim # once for forward, twice for backward
n_flops_train_last_layer = 3 * 2 * n_classes * args.batch \
        * args.hidden_dim # once for forward, twice for backward
n_flops = n_flops_train_first_layer + (args.depth-2)*n_flops_train_mid_layer \
        + n_flops_train_last_layer
# Multiply by number of epochs and batches
n_flops *= args.epoch * len(train_loader)

# Set up batches of data and labels
time0 = -time.time()
batch_data = []
batch_labels = []
x_traits = nntile.tensor.TensorTraits([n_pixels, args.batch], \
        [args.pixels_tile, args.batch_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits([args.batch], [args.batch_tile])
y_distr = [0] * y_traits.grid.nelems
for train_batch_data, train_batch_labels in train_loader:
    if train_batch_data.shape[0] != args.batch:
        break
    x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x.from_array(train_batch_data.view(args.batch, n_pixels).numpy().T)
    batch_data.append(x)
    y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
    next_tag = y.next_tag
    y.from_array(train_batch_labels.numpy())
    batch_labels.append(y)

# Wait for all scatters to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("From PyTorch loader to NNTile batches in {} seconds".format(time0))

# Define tensor X for input batches
time0 = -time.time()
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
gemm_ndim = 1
if deep_relu_mp:
    m = nntile.model.DeepReLU_mp(x_moments, 'R', gemm_ndim, args.hidden_dim, \
            args.hidden_dim_tile, args.depth, n_classes, next_tag)
    print("GEMM FP16")
else:
    m = nntile.model.DeepReLU(x_moments, 'R', gemm_ndim, args.hidden_dim, \
            args.hidden_dim_tile, args.depth, n_classes, next_tag, \
            fp32_fast_fp16, fp32_convert_fp16)
    print("GEMM FP32_FAST_FP16: {}".format(m.fp32_fast_fp16))
    print("GEMM FP32_CONVERT_FP16: {}".format(m.fp32_convert_fp16))
next_tag = m.next_tag
# Set up learning rate and optimizer for training
optimizer = nntile.optimizer.SGD(m.get_parameters(), lr, next_tag, \
        momentum=0.0, nesterov=False, weight_decay=0.0)
#optimizer = nntile.optimizer.Adam(m.get_parameters(), lr, next_tag)

# Set up Cross Entropy loss function for the model
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(m.activations[-1], \
        next_tag)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_data, batch_labels, m, optimizer, \
        loss, args.epoch)

time0 += time.time()
print("Finish generating pipeline (model, loss and optimizer) in {} seconds" \
        .format(time0))

# Randomly init weights of the DeepReLU model
time0 = -time.time()
m.init_randn_async()

# Wait for all parameters to initialize
nntile.starpu.wait_for_all()
time0 += time.time()
print("Finish random weights init in {} seconds".format(time0))

# Compute test accuracy of the untrained model
test_top1_accuracy = 0
total_num_samples = 0
for test_batch_data, test_batch_label in test_loader:
    m.activations[0].value.from_array( \
            test_batch_data.view(-1, n_pixels).numpy().T)
    m.forward_async()
    output = np.zeros(m.activations[-1].value.shape, order="F", \
            dtype=np.float32)
    m.activations[-1].value.to_array(output)
    pred_labels = np.argmax(output, 0)
    test_top1_accuracy += np.sum(pred_labels == test_batch_label.numpy())
    total_num_samples += test_batch_label.shape[0]
# Report the accuracy if it was computed
if total_num_samples > 0:
    test_top1_accuracy /= total_num_samples
    print("Test accuracy of the untrained Deep ReLU model =", \
            test_top1_accuracy)

# Run a some warmup epochs to let StarPU allocate temp buffers and pin them
pipeline.n_epochs = args.epoch_warmup
print("Start {} warmup epochs to let StarPU allocate and pin buffer" \
        .format(args.epoch_warmup))
time0 = -time.time()
nntile.starpu.pause()
pipeline.train_async()
nntile.starpu.resume()
nntile.starpu.wait_for_all()
time0 += time.time()
print("Finish {} warmup epochs in {} seconds".format(args.epoch_warmup, time0))

# Start timer and run training
pipeline.n_epochs = args.epoch
time0 = -time.time()
nntile.starpu.pause()
pipeline.train_async()
nntile.starpu.resume()
time0 += time.time()
print("Finish adding tasks (computations are running) in {} seconds" \
        .format(time0))

# Wait for all computations to finish
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
    #nntile.tensor.copy_async(x, m.activations[0].value)
    m.forward_async()
    #for t in m.activations:
    #    t.value.wont_use()
    #for p in m.parameters:
    #    p.value.wont_use()
nntile.starpu.wait_for_all()
time0 += time.time()

# FLOPS for inference over the first layer per batch
n_flops_inference = 2 * n_pixels * args.batch * args.hidden_dim
# FLOPS for inference over each middle layer per batch
n_flops_inference += (args.depth-2) * 2 * args.hidden_dim * args.batch \
        * args.hidden_dim
# FLOPS for inference over the last layer per batch
n_flops_inference += 2 * n_classes * args.batch * args.hidden_dim
# Multiply FLOPS per number of batches
n_flops_inference *= len(train_loader)
print("Inference speed: {} samples/second".format(\
        len(train_loader) * args.batch / time0))
print("Inference GFLOPs/s (based on gemms): {}" \
        .format(n_flops_inference * 1e-9 / time0))

# Compute test accuracy of the trained model
test_top1_accuracy = 0
total_num_samples = 0
for test_batch_data, test_batch_label in test_loader:
    m.activations[0].value.from_array( \
            test_batch_data.view(-1, n_pixels).numpy().T)
    m.forward_async()
    output = np.zeros(m.activations[-1].value.shape, order="F", \
            dtype=np.float32)
    m.activations[-1].value.to_array(output)
    pred_labels = np.argmax(output, 0)
    test_top1_accuracy += np.sum(pred_labels == test_batch_label.numpy())
    total_num_samples += test_batch_label.shape[0]
# Report the accuracy if it was computed
if total_num_samples > 0:
    test_top1_accuracy /= total_num_samples
    print("Test accuracy of the trained Deep ReLU model =", test_top1_accuracy)

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

