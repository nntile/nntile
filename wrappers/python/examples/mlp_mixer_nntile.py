# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# Example for NNTile version of MLP-Mixer model
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2023-12-5

# All necesary imports
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dts 
import torchvision.transforms as trnsfrms
import pathlib
import os, sys
import argparse
import nntile
from nntile.model.mlp_mixer import MlpMixer
from nntile.torch_models.mlp_mixer import MlpMixer as TorchMlpMixer


def image_patching(image, patch_size):
    c, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("patch size should be divisor of both image height and width")
    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)
    patched_batch = torch.empty((n_patches, n_channels), dtype=image.dtype)

    n_y = int(w / patch_size)

    for i in range(n_patches):
        x = i // n_y
        y = i % n_y

        for clr in range(c):
            vect_patch = image[clr, x * patch_size: (x+1) * patch_size , y * patch_size: (y+1) * patch_size].flatten()
            patched_batch[i, clr * (patch_size ** 2) : (clr+1) * (patch_size ** 2)] = vect_patch
    return patched_batch


def data_loader_to_nntile(data_set, label_set, batch_input, batch_output, trns, batch_size, minibatch_size, patch_size, next_tag):
    total_len, h, w, c = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)

    X_shape = [channel_size, minibatch_size, num_clr_channels * patch_size ** 2]
    Y_shape = [minibatch_size]

    tmp_data_tensor = torch.empty((n_patches, minibatch_size, n_channels), dtype=torch.float32)
    tmp_label_tensor = torch.empty(minibatch_size, dtype=torch.float32)

    x_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    x_distr = [0] * x_traits.grid.nelems

    y_traits = nntile.tensor.TensorTraits(Y_shape, Y_shape)
    y_distr = [0] * y_traits.grid.nelems

    for i in range(n_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                tmp_data_tensor[:, k, :] = image_patching(trns(data_set[i * batch_size + j * minibatch_size + k, :, :, :]), patch_size)
                tmp_label_tensor[k] = label_set[i * batch_size + j * minibatch_size + k]
            x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            x.from_array(np.asfortranarray(tmp_data_tensor.numpy()))
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(np.asfortranarray(tmp_label_tensor.numpy().reshape(-1)))
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)   


def data_loader_to_tensor(data_set, label_set, trns, batch_size, minibatch_size, patch_size):
    total_len, h, w, c = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)

    train_tensor = torch.empty((n_batches, n_minibatches, n_patches, minibatch_size, n_channels), dtype=torch.float32)
    label_tensor = torch.empty((n_batches, n_minibatches, minibatch_size), dtype=torch.float32)
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, :, k, :] = image_patching(trns(data_set[i * batch_size + j * minibatch_size + k, :, :, :]), patch_size)
                label_tensor[i, j, k] = label_set[i * batch_size + j * minibatch_size + k]
    return train_tensor, label_tensor


# Instantiate the parser
parser = argparse.ArgumentParser(prog="MlpMixer neural network", \
        description="This example trains PyTorch version of MlpMixer neural "\
        "network from a scratch for an image classification task")
parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
parser.add_argument("--batch", type=int)
parser.add_argument("--minibatch", type=int )
parser.add_argument("--depth", type=int)
parser.add_argument("--patch_size", type=int)
parser.add_argument("--hidden_dim", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument('--check', action='store_true', default=False)
parser.add_argument('--save_final_state', action='store_true', default=False)

args = parser.parse_args()

cur_pos = str(pathlib.Path(__file__).parent.absolute())
experiment_folder = "mlp_mixer_data"
init_state_path = os.path.join(cur_pos, experiment_folder, "torch_init_state.pt")
final_state_path = os.path.join(cur_pos, experiment_folder, "nntile_final_state.pt")

# Parse arguments
args = parser.parse_args()
print(args)

batch_size = args.batch
minibatch_size = args.minibatch
patch_size = args.patch_size
num_mixer_layers = args.depth
hidden_dim = args.hidden_dim
lr = args.lr
num_epochs = args.epoch

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])
# For PyTorch testing
# device_for_pytorch = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )
device_for_pytorch="cpu"
# Check consistency of batch and minibatch sizes
if batch_size % minibatch_size != 0:
    raise ValueError("Batch must consist of integer number of minibatches")

if args.dataset == "mnist":
    train_set = dts.MNIST(root='./', train=True, download=True, transform=trnsform)
    test_set = dts.MNIST(root='./', train=False, download=True, transform=trnsform)
    n_classes = 10
    num_clr_channels = 1
    # Check consistency of image and patch sizes
    if 28 % patch_size != 0:
        raise ValueError("Image size must be divisible by patch size without remainder")
    channel_size = int(28 * 28 / patch_size ** 2)
elif args.dataset == "cifar10":
    train_set = dts.CIFAR10(root='./', train=True, download=False, transform=trnsform)
    test_set = dts.CIFAR10(root='./', train=False, download=False)
    n_classes = 10
    num_clr_channels = 3
    # Check consistency of image and patch sizes
    if 32 % patch_size != 0:
        raise ValueError("Image size must be divisible by patch size without remainder")
    channel_size = int(32 * 32 / patch_size ** 2)
else:
    raise ValueError("{} dataset is not supported yet!".format(args.dataset))

# Set up StarPU configuration and init it
config = nntile.starpu.Config(-1, -1, 1)
# Init all NNTile-StarPU codelets
nntile.starpu.init()

X_shape = [channel_size, minibatch_size, num_clr_channels * patch_size ** 2]
Y_shape = [minibatch_size]
next_tag = 0

# Prepare data for NNTile training
batch_input = []
batch_output = []

data_loader_to_nntile(train_set.data, train_set.targets, batch_input, batch_output, trnsform, batch_size, minibatch_size, patch_size, next_tag)

test_data_tensor, test_labels_tensor = data_loader_to_tensor(test_set.data, test_set.targets, trnsform, batch_size, minibatch_size, patch_size)
test_labels_tensor = test_labels_tensor.type(torch.LongTensor)
torch_mixer_model = TorchMlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, n_classes)
optim_torch = torch.optim.Adam(torch_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")
# checkpoint = torch.load(init_state_path)
# torch_mixer_model.load_state_dict(checkpoint['model_state_dict'])

print("Accuracy before training:")
torch_mixer_model.evaluate(test_data_tensor, test_labels_tensor, device_for_pytorch)

nntile_mixer_model, next_tag = MlpMixer.from_torch(torch_mixer_model,minibatch_size,n_classes, next_tag)
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(nntile_mixer_model.activations[-1], next_tag)
optimizer = nntile.optimizer.Adam(nntile_mixer_model.get_parameters(), \
            args.lr, next_tag)
next_tag = optimizer.get_next_tag()

# Check accuracy of output and gradients of parmeters if required
if args.check:
    torch_mixer_model.zero_grad()
    patched_test_sample = test_data_tensor[1,1,:,:,:]
    test_labels = test_labels_tensor[1, 1, :]

    torch_mixer_model.zero_grad()
    torch_output = torch_mixer_model(patched_test_sample)

    np_torch_output = np.array(torch_output.detach().numpy(), order="F", dtype=np.float32)
    loss_local = crit_torch(torch_output, test_labels)
    loss_local.backward()

    data_train_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    test_tensor = nntile.tensor.Tensor_fp32(data_train_traits, [0], next_tag)
    next_tag = test_tensor.next_tag
    test_tensor.from_array(patched_test_sample.numpy())
    nntile.tensor.copy_async(test_tensor, nntile_mixer_model.activations[0].value)

    label_train_traits = nntile.tensor.TensorTraits(Y_shape, Y_shape)
    label_train_tensor = nntile.tensor.Tensor_int64(label_train_traits, [0], next_tag)
    next_tag = label_train_tensor.next_tag
    label_train_tensor.from_array(test_labels.numpy())
    nntile.tensor.copy_async(label_train_tensor, loss.y)    

    nntile_mixer_model.clear_gradients()
    nntile_mixer_model.forward_async()
    loss.calc_async()

    nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
    loss.get_val(nntile_xentropy_np)
    nntile_xentropy_np = nntile_xentropy_np.reshape(-1)
    
    nntile_last_layer_output = np.zeros(nntile_mixer_model.activations[-1].value.shape, order="F", dtype=np.float32)
    nntile_mixer_model.activations[-1].value.to_array(nntile_last_layer_output)

    print("PyTorch loss: {}, NNTile loss: {}".format(loss_local.item(), nntile_xentropy_np[0]))
    print("Norm of inference difference: {}".format(np.linalg.norm(nntile_last_layer_output - np_torch_output.T, 'fro')))

    nntile_mixer_model.backward_async()

    for i, (p_torch, p_nntile) in enumerate(zip(torch_mixer_model.parameters(), nntile_mixer_model.parameters)):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
        p_nntile.grad.to_array(p_nntile_grad_np)
        if p_torch.grad.shape[0] != p_nntile_grad_np.shape[0]:
            p_nntile_grad_np = np.transpose(p_nntile_grad_np)
        
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
        norm = torch.norm(p_torch.grad)
        print("Gradient of {} parameter: norm={} rel_err={}".format(i, norm, \
                rel_error))


# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_input, batch_output, \
        nntile_mixer_model, optimizer, loss, num_epochs)
nntile_mixer_model.clear_gradients()

#Actual training
pipeline.train_async()
#nntile.starpu.resume()
nntile.starpu.wait_for_all()

nntile_mixer_model.to_torch(torch_mixer_model)

if args.save_final_state:
    torch.save({
                'model_state_dict': torch_mixer_model.state_dict(),
                'optimizer_state_dict': optim_torch.state_dict(),
                }, final_state_path)

loss.unregister()
optimizer.unregister()
for batch in batch_input+batch_output:
    for x in batch:
        x.unregister()

# Unregister all tensors related to model
nntile_mixer_model.unregister()

#   Evaluate accuracy of nntile model by uploading trained weights to the torch model
print("Accuracy after training:")
torch_mixer_model.evaluate(test_data_tensor, test_labels_tensor, device_for_pytorch)
