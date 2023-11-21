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


def data_loader_to_tensor(data_set, label_set, trns, batch_size, minibatch_size):
    total_len, h, w, c = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)
  
    train_tensor = torch.empty((n_batches, n_minibatches, minibatch_size, h, w, c), dtype=torch.float32)
    label_tensor = torch.empty((n_batches, n_minibatches, minibatch_size), dtype=torch.float32)
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                for clr_counter in range(c):
                    train_tensor[i, j, k, :, :, clr_counter] = trns(data_set[i * batch_size + j * minibatch_size + k, :, :, clr_counter])
                label_tensor[i, j, k] = label_set[i * batch_size + j * minibatch_size + k]
    return train_tensor, label_tensor


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
batch_size = 512
accumulation_steps = int(batch_size / minibatch_size)

channel_size = int(32 * 32 / patch_size ** 2)
hidden_dim = 768
num_mixer_layers = 12
num_classes = 10
num_clr_channels = 3

lr = 1e-2
num_epochs = 12

mlp_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes)
optim_torch = torch.optim.Adam(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss()
trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.train:
    mlp_mixer_model = mlp_mixer_model.to(device)
    cifartrainset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=True, download=False, transform=trnsform)
    train_data_tensor, train_labels = data_loader_to_tensor(cifartrainset.data, cifartrainset.targets, trnsform, batch_size, minibatch_size)
    train_labels = train_labels.type(torch.LongTensor) 

    cifartestset = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=False, download=False)
    test_data_tensor, test_labels = data_loader_to_tensor(cifartestset.data, cifartestset.targets, trnsform, batch_size, minibatch_size)
    test_labels = test_labels.type(torch.LongTensor)

    interm_train_loss = []
    interm_test_loss = []

    mlp_mixer_model.zero_grad()

    torch.save({
                'model_state_dict': mlp_mixer_model.state_dict(),
                'optimizer_state_dict': optim_torch.state_dict(),
                }, os.path.join(cur_pos, 'init_state.pt'))

    for epoch_counter in range(num_epochs):  
        for batch_iter in range(train_data_tensor.shape[0]):
            for minibatch_iter in range(train_data_tensor.shape[1]):
                patched_train_sample, _, _ = image_patching_rgb(train_data_tensor[batch_iter,minibatch_iter,:,:,:,:], patch_size)
                patched_train_sample = patched_train_sample.to(device)
                true_labels = train_labels[batch_iter, minibatch_iter, :].to(device)
                torch_output = mlp_mixer_model(patched_train_sample)
                torch_loss = crit_torch(torch_output, true_labels)

                # normalize loss to account for batch accumulation
                torch_loss = torch_loss / train_data_tensor.shape[1]
                interm_train_loss.append(torch_loss.to('cpu').detach().numpy())
                torch_loss.backward()
                
            optim_torch.step()
            mlp_mixer_model.zero_grad()

            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            with torch.no_grad():
                test_loss = 0 
                for test_batch_iter in range(test_data_tensor.shape[0]):
                    for test_minibatch_iter in range(test_data_tensor.shape[1]):
                        patched_test_sample, _, _ = image_patching_rgb(test_data_tensor[test_batch_iter,test_minibatch_iter,:,:,:,:], patch_size)
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
                interm_test_loss.append(test_loss / test_data_tensor.shape[0])
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
    np.savez('Loss_cifar10', train_loss=interm_train_loss, test_loss = interm_test_loss)

else:

    checkpoint = torch.load('final_state.pt')
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
            patched_train_sample, ll, pp = image_patching_rgb(test_batch_sample, patch_size)
            print(patched_train_sample.dtype)
            patched_train_sample = patched_train_sample.to(device)
            true_labels = true_labels.to(device)

            torch_output = mlp_mixer_model(patched_train_sample)
            torch_loss = crit_torch(torch_output, true_labels)
            # normalize loss to account for batch accumulation
            test_loss += torch_loss / total_test_samples
    print(test_loss)
