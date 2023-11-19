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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

patch_size = 16
minibatch_size = 16
batch_size = 512
accumulation_steps = int(batch_size / minibatch_size)

channel_size = int(32 * 32 / patch_size ** 2)
hidden_dim = 768
num_mixer_layers = 12
num_classes = 10
num_clr_channels = 3

lr = 1e-3
num_epochs = 10

mlp_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes).to(device)
optim_torch = torch.optim.SGD(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

cifartrainset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=True, download=False, transform=trnsform)
trainldr = torch.utils.data.DataLoader(cifartrainset, batch_size=minibatch_size, shuffle=True)
total_train_samples = len(trainldr)

minibatch_counter = 0
interm_loss = []

mlp_mixer_model.zero_grad()

torch.save({
            'model_state_dict': mlp_mixer_model.state_dict(),
            'optimizer_state_dict': optim_torch.state_dict(),
            }, 'init_state.pt')

for epoch_counter in range(num_epochs):  
    for train_iter, (train_batch_sample, true_labels) in enumerate(trainldr):
        patched_train_sample = image_patching_rgb(train_batch_sample, patch_size)
        patched_train_sample = patched_train_sample.to(device)
        true_labels = true_labels.to(device)
 
        torch_output = mlp_mixer_model(patched_train_sample)
        torch_loss = crit_torch(torch_output, true_labels)
        # normalize loss to account for batch accumulation
        torch_loss = torch_loss / accumulation_steps
        interm_loss.append(torch_loss.to('cpu').detach().numpy())

        torch_loss.backward()
        if (train_iter + 1) % accumulation_steps == 0 or (train_iter + 1) == total_train_samples:
            optim_torch.step()
            mlp_mixer_model.zero_grad()
            print('Epoch: {}, Processing minibatch {} out of {}'.format(epoch_counter, train_iter, total_train_samples))
        
torch.save({
            'epoch': num_epochs,
            'model_state_dict': mlp_mixer_model.state_dict(),
            'optimizer_state_dict': optim_torch.state_dict(),
            'loss': interm_loss,
            }, 'final_state.pt')
np.savez('Loss_cifar10', loss=interm_loss)

