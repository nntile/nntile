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
import torch
import torch.nn as nn
import torchvision.datasets as dts 
import numpy as np
import torchvision.transforms as trnsfrms
from nntile.torch_models.mlp_mixer import MlpMixer, image_patching


init_patch_size = 7
batch_size = 1
channel_size = int(28 * 28 / init_patch_size ** 2)
hidden_dim = 100
num_mixer_layers = 10
num_classes = 10

lr = 1e-2


X_shape = [channel_size, batch_size, init_patch_size]

mlp_mixer_model = MlpMixer(channel_size, init_patch_size ** 2, hidden_dim, num_mixer_layers, num_classes)
optim_torch = torch.optim.SGD(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

mnisttrainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
trainldr = torch.utils.data.DataLoader(mnisttrainset, batch_size=batch_size, shuffle=True)

training_iteration = 0
for train_batch_sample, true_labels in trainldr:
    train_batch_sample = train_batch_sample.view(-1, 28, 28)
    train_batch_sample = torch.swapaxes(train_batch_sample, 0, 1)
    patched_train_sample = image_patching(train_batch_sample, init_patch_size)
    
    mlp_mixer_model.zero_grad()
    torch_output = mlp_mixer_model(patched_train_sample)
    torch_loss = crit_torch(torch_output, true_labels)
    torch_loss.backward()
    if (training_iteration % 100) == 99:
        print("Intermediate PyTorch loss =", torch_loss.item())
    optim_torch.step()
    training_iteration += 1
    # break

