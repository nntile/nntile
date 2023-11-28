# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# Example for torch version of MLP-Mixer model
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2023-11-22

# All necesary imports
import pathlib
import os, sys
import torch
import torch.nn as nn
import torchvision.datasets as dts 
import numpy as np
import argparse

import torchvision.transforms as trnsfrms
from nntile.torch_models.mlp_mixer import MlpMixer


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


def data_loader_to_tensor_rgb(data_set, label_set, trns, batch_size, minibatch_size, patch_size):
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


def data_loader_to_tensor(data_set, label_set, trns, batch_size, minibatch_size):
    total_len, h, w = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)
    data_set = data_set.numpy()
    label_set = label_set.numpy()
    train_tensor = torch.empty((n_batches, n_minibatches, minibatch_size, h, w, 1), dtype=torch.float32)
    label_tensor = torch.empty((n_batches, n_minibatches, minibatch_size), dtype=torch.float32)
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, k, :, :, 0] = trns(data_set[i * batch_size + j * minibatch_size + k, :, :])
                label_tensor[i, j, k] = label_set[i * batch_size + j * minibatch_size + k]
    return train_tensor, label_tensor


# Instantiate the parser
parser = argparse.ArgumentParser(prog="DeepReLU neural network", \
        description="This example trains NNTile version of MlpMixer neural "\
        "network from a scratch for an image classification task")
parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
parser.add_argument("--batch", type=int)
parser.add_argument("--minibatch", type=int)
parser.add_argument("--depth", type=int)
parser.add_argument("--patch_size", type=int)
parser.add_argument("--hidden_dim", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--lr", type=float)


args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

cur_pos = str(pathlib.Path(__file__).parent.absolute())
experiment_name = "mlp_mixer.pt"

full_path = os.path.join(cur_pos, experiment_name)

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

# Check consistency of batch and minibatch sizes
if batch_size % minibatch_size != 0:
    raise ValueError("Batch must consist of integer number of minibatches")
accumulation_steps = int(batch_size / minibatch_size)

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
    train_set = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=True, download=False, transform=trnsform)
    test_set = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=False, download=False)
    n_classes = 10
    num_clr_channels = 3
    # Check consistency of image and patch sizes
    if 32 % patch_size != 0:
        raise ValueError("Image size must be divisible by patch size without remainder")
    channel_size = int(32 * 32 / patch_size ** 2)
else:
    raise ValueError("{} dataset is not supported yet!".format(args.dataset))

mlp_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, n_classes)
# optim_torch = torch.optim.Adam(mlp_mixer_model.parameters(), lr=lr)
optim_torch = torch.optim.SGD(mlp_mixer_model.parameters(), lr=lr, momentum=0.9)
crit_torch = nn.CrossEntropyLoss()


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


train_data_tensor, train_labels = data_loader_to_tensor_rgb(train_set.data, train_set.targets, trnsform, batch_size, minibatch_size, patch_size)
train_labels = train_labels.type(torch.LongTensor) 
num_batch_train, num_minibatch_train = train_data_tensor.shape[0], train_data_tensor.shape[1]
    
test_data_tensor, test_labels = data_loader_to_tensor_rgb(test_set.data, test_set.targets, trnsform, batch_size, minibatch_size, patch_size)
test_labels = test_labels.type(torch.LongTensor)
num_batch_test, num_minibatch_test = test_data_tensor.shape[0], test_data_tensor.shape[1]

interm_train_loss = []
interm_test_loss = []

mlp_mixer_model.zero_grad()

torch.save({
            'model_state_dict': mlp_mixer_model.state_dict(),
            'optimizer_state_dict': optim_torch.state_dict(),
            }, os.path.join(cur_pos, 'init_state.pt'))

mlp_mixer_model = mlp_mixer_model.to(device)
for epoch_counter in range(num_epochs):  
    for batch_iter in range(num_batch_train):
        for minibatch_iter in range(num_minibatch_train):
            patched_train_sample = train_data_tensor[batch_iter,minibatch_iter,:,:,:]
            patched_train_sample = patched_train_sample.to(device)
            true_labels = train_labels[batch_iter, minibatch_iter, :].to(device)
            torch_output = mlp_mixer_model(patched_train_sample)
            torch_loss = crit_torch(torch_output, true_labels)
            # print(torch_loss)
            # normalize loss to account for batch accumulation
            # torch_loss = torch_loss / train_data_tensor.shape[1]
            # interm_train_loss.append(torch_loss.to('cpu').detach().numpy())
            torch_loss.backward()
            
            optim_torch.step()
            mlp_mixer_model.zero_grad()

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        with torch.no_grad():
            test_loss = 0 
            for test_batch_iter in range(num_batch_test):
                for test_minibatch_iter in range(num_minibatch_test):
                    patched_test_sample = test_data_tensor[test_batch_iter,test_minibatch_iter,:,:,:]
                    patched_test_sample = patched_test_sample.to(device)
                    true_test_labels = test_labels[test_batch_iter, test_minibatch_iter, :].to(device)
                
                    torch_output = mlp_mixer_model(patched_test_sample)

                    _, predictions = torch.max(torch_output, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(true_test_labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1

                    test_loss += crit_torch(torch_output, true_test_labels)
            test_loss = test_loss / (num_batch_test * num_minibatch_test)
            interm_test_loss.append(test_loss.to('cpu').detach().numpy())
        print('Epoch: {}, Batch {} out of {}, Last train loss: {}, Avg test loss: {}'.format(epoch_counter, batch_iter, train_data_tensor.shape[0], torch_loss, interm_test_loss[-1]))
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print('Total = {}'.format(sum(correct_pred.values()) / sum(total_pred.values())))
torch.save({
            'epoch': num_epochs,
            'model_state_dict': mlp_mixer_model.state_dict(),
            'optimizer_state_dict': optim_torch.state_dict(),
            }, full_path)
# np.savez('test_loss_cifar10', test_loss = interm_test_loss)


