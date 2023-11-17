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
from nntile.torch_models.mlp_mixer import MlpMixer, image_patching_rgb
from nntile.model.mlp_mixer import MlpMixer as MlpMixerTile

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

patch_size = 16
batch_size = 8
channel_size = int(32 * 32 / patch_size ** 2)
hidden_dim = 1024
num_mixer_layers = 8
num_classes = 10
num_clr_channels = 3

lr = 1e-3
stop_train_iter = 10000
next_tag = 0

X_shape = [channel_size, batch_size, patch_size]

mlp_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes).to(device)
optim_torch = torch.optim.SGD(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

cifartrainset = dts.CIFAR10(root='/datasets/cifar10/', train=True, download=False, transform=trnsform)
trainldr = torch.utils.data.DataLoader(cifartrainset, batch_size=batch_size, shuffle=True)

training_iteration = 0
interm_loss = []
mlp_mixer_model.zero_grad()
for train_batch_sample, true_labels in trainldr:
    # print(true_labels)
    # train_batch_sample = train_batch_sample.view(-1, 32, 32)
    # train_batch_sample = torch.swapaxes(train_batch_sample, 0, 1)
    patched_train_sample = image_patching_rgb(train_batch_sample, patch_size)
    print(patched_train_sample.device)
    mlp_mixer_model.zero_grad()
    torch_output = mlp_mixer_model(patched_train_sample)
    torch_loss = crit_torch(torch_output, true_labels)
    torch_loss.backward()
    print("Intermediate PyTorch loss =", torch_loss.item())
    if (training_iteration % 50) == 49:
        print("Intermediate PyTorch loss =", torch_loss.item())
        interm_loss.append(torch_loss.item())
    optim_torch.step()

    if training_iteration == stop_train_iter:
            break    
    training_iteration += 1

np.savez('Loss_cifar10', loss=interm_loss)

