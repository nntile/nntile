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
# @date 2023-06-09

# All necesary imports
import pathlib
import os, sys
import torch
import torch.nn as nn
import torchvision.datasets as dts 
import numpy as np
import argparse

import torchvision.transforms as trnsfrms
from nntile.torch_models.mlp_mixer import MlpMixer, image_patching_rgb

# Instantiate the parser
parser = argparse.ArgumentParser(description='Parameters to run example')
# Switch
parser.add_argument('--train', action='store_true')
parser.add_argument('--add_name', type=str)

args = parser.parse_args()
print("To train = {}".format(args.train))


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

cur_pos = str(pathlib.Path(__file__).parent.absolute())
add_name = ''
if args.add_name != None:
    add_name = args.add_name
    experiment_name = "mlp_mixer+{}.pt".format(add_name)
else:
    experiment_name = "mlp_mixer.pt"
full_path = os.path.join(cur_pos, experiment_name)

patch_size = 16
minibatch_size = 16
batch_size = 1024
accumulation_steps = int(batch_size / minibatch_size)

channel_size = int(32 * 32 / patch_size ** 2)
hidden_dim = 768
num_mixer_layers = 12
num_classes = 10
num_clr_channels = 3

lr = 1e-3
num_epochs = 10

mlp_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes)
optim_torch = torch.optim.SGD(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")
trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

if args.train:
    mlp_mixer_model = mlp_mixer_model.to(device)
    cifartrainset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=True, download=False, transform=trnsform)
    testldr = torch.utils.data.DataLoader(cifartrainset, batch_size=minibatch_size, shuffle=True)
    total_train_samples = len(list(testldr))
    print(total_train_samples)

    cifartestset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=False, download=False, transform=trnsform)
    testldr = torch.utils.data.DataLoader(cifartestset, batch_size=minibatch_size, shuffle=True)
    total_test_samples = len(testldr)

    interm_train_loss = []
    interm_test_loss = []

    mlp_mixer_model.zero_grad()

    torch.save({
                'model_state_dict': mlp_mixer_model.state_dict(),
                'optimizer_state_dict': optim_torch.state_dict(),
                }, os.path.join(cur_pos, 'init_state.pt'))

    for epoch_counter in range(num_epochs):  
        for train_iter, (train_batch_sample, true_labels) in enumerate(testldr):
            patched_train_sample = image_patching_rgb(train_batch_sample, patch_size)
            patched_train_sample = patched_train_sample.to(device)
            true_labels = true_labels.to(device)
    
            torch_output = mlp_mixer_model(patched_train_sample)
            torch_loss = crit_torch(torch_output, true_labels)
            # normalize loss to account for batch accumulation
            torch_loss = torch_loss / accumulation_steps
            interm_train_loss.append(torch_loss.to('cpu').detach().numpy())
            torch_loss.backward()
            if (train_iter + 1) % accumulation_steps == 0 or (train_iter + 1) == total_train_samples:
                optim_torch.step()
                mlp_mixer_model.zero_grad()
                with torch.no_grad():
                    test_loss = 0
                    for test_batch_sample, true_test_labels in testldr:
                        patched_test_sample = image_patching_rgb(test_batch_sample, patch_size)
                        patched_test_sample = patched_train_sample.to(device)
                        true_test_labels = true_test_labels.to(device)
                        torch_output = mlp_mixer_model(patched_test_sample)
                        test_loss += crit_torch(torch_output, true_test_labels)
                    interm_test_loss.append(test_loss / total_test_samples)
                print('Epoch: {}, Minibatch {} out of {}, Last train loss: {}, Avg test loss: {}'.format(epoch_counter, train_iter, total_train_samples, torch_loss, interm_test_loss[-1]))
    
    torch.save({
                'epoch': num_epochs,
                'model_state_dict': mlp_mixer_model.state_dict(),
                'optimizer_state_dict': optim_torch.state_dict(),
                'loss': interm_train_loss,
                }, full_path)
    np.savez('Loss_cifar10', train_loss=interm_train_loss, test_loss = interm_test_loss)

else:

    checkpoint = torch.load(full_path)
    mlp_mixer_model.load_state_dict(checkpoint['model_state_dict'])
    optim_torch.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    test_loss = 0

    cifartestset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=False, download=False, transform=trnsform)
    testldr = torch.utils.data.DataLoader(cifartestset, batch_size=minibatch_size, shuffle=True)
    total_test_samples = len(testldr)
    mlp_mixer_model = mlp_mixer_model.to(device)
    with torch.no_grad():
        for test_iter, (test_batch_sample, true_labels) in enumerate(testldr):
            patched_train_sample = image_patching_rgb(test_batch_sample, patch_size)
            patched_train_sample = patched_train_sample.to(device)
            true_labels = true_labels.to(device)

            torch_output = mlp_mixer_model(patched_train_sample)
            torch_loss = crit_torch(torch_output, true_labels)
            # normalize loss to account for batch accumulation
            test_loss += torch_loss / total_test_samples
    print(test_loss)
