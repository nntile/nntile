# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/from_torch_to_nntile.py
# Example of comparison of Deep ReLU network of NNTile Python package and PyTorch
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @date 2023-03-27

import torch
import torch.nn as nn
import torchvision.datasets as dts 
import numpy as np
import random
import time
import nntile
import torchvision.transforms as trnsfrms

torch.set_num_threads(10)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(121)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28*28, 4000, bias=False) 
        self.linear2 = nn.Linear(4000, 4000, bias=False)
        self.final = nn.Linear(4000, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a

n_classes = 10
batch_size = 5000
lr = 1e-3
mlp_model = MLP()
optim_torch = torch.optim.SGD(mlp_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

mnisttrainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
trainldr = torch.utils.data.DataLoader(mnisttrainset, batch_size=batch_size, shuffle=True)

for train_batch_sample, true_labels in trainldr:
    # true_labels = torch.randint(0, n_classes, (batch_size, ))
    # train_batch_sample = torch.randn((batch_size, 28*28))
    train_batch_sample = train_batch_sample.view(-1, 28*28)
    break

config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0

mlp_nntile, next_tag = nntile.model.DeepReLU.from_torch(mlp_model, batch_size, n_classes, "relu", next_tag)
optimizer = nntile.optimizer.SGD(mlp_nntile.get_parameters(), lr, next_tag, momentum=0.0,
       nesterov=False, weight_decay=0.)
# optimizer = nntile.optimizer.Adam(m.get_parameters(), lr, next_tag)
next_tag = optimizer.get_next_tag()

data_train_traits = nntile.tensor.TensorTraits(train_batch_sample.shape, \
        train_batch_sample.shape)
data_train_tensor = nntile.tensor.Tensor_fp32(data_train_traits, [0], next_tag)
next_tag = data_train_tensor.next_tag
data_train_tensor.from_array(train_batch_sample)
crit_nntile, next_tag = nntile.loss.CrossEntropy.generate_simple(mlp_nntile.activations[-1], next_tag)
nntile.tensor.copy_async(data_train_tensor, mlp_nntile.activations[0].value)

mlp_model.zero_grad()
time_torch_forward = -time.time()
torch_output = mlp_model(train_batch_sample)
time_torch_forward += time.time()
print("PyTorch model forward requires {} seconds".format(time_torch_forward))
torch_loss = crit_torch(torch_output, true_labels)
print("PyTorch loss =", torch_loss.item())

time_torch_backward = -time.time()
torch_loss.backward()
time_torch_backward += time.time()
print("PyTorch model backward requires {} seconds".format(time_torch_backward))

time_nntile_forward = -time.time()
mlp_nntile.forward_async()
time_nntile_forward += time.time()

time1 = -time.time()
nntile.starpu.wait_for_all()
time1 += time.time()
print("NNTile model forward done in {} + {} = {} seconds".format(time_nntile_forward, time1, \
        time_nntile_forward + time1))

label_train_traits = nntile.tensor.TensorTraits(true_labels.shape, \
        true_labels.shape)
label_train_tensor = nntile.tensor.Tensor_int64(label_train_traits, [0], next_tag)
next_tag = label_train_tensor.next_tag
label_train_tensor.from_array(true_labels.numpy())

nntile.tensor.copy_async(label_train_tensor, crit_nntile.y)
crit_nntile.calc_async()
nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
crit_nntile.get_val(nntile_xentropy_np)
print("NNTile loss =", nntile_xentropy_np[0])

time_nntile_backward = -time.time()
mlp_nntile.backward_async()
time_nntile_backward += time.time()

time1 = -time.time()
nntile.starpu.wait_for_all()
time1 += time.time()
print("NNTile model backward done in {} + {} = {} seconds".format(time_nntile_backward, time1, \
        time_nntile_backward + time1))


for i, (p_torch, p_nntile) in enumerate(zip(mlp_model.parameters(), mlp_nntile.parameters)):
    p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
    p_nntile.grad.to_array(p_nntile_grad_np)
    print("Relative error in gradient of layer {} = {}".format(i,
        np.linalg.norm(p_nntile_grad_np.T - p_torch.grad.numpy(), "fro") / np.linalg.norm(p_nntile_grad_np.T, "fro")))


# Make optimizer step and compare updated losses
optim_torch.step()
torch_output = mlp_model(train_batch_sample)
torch_loss = crit_torch(torch_output, true_labels)
print("PyTorch loss after optimizer step =", torch_loss.item())
_, pred_labels = torch.max(torch_output, 1)
torch_accuracy = torch.sum(true_labels == pred_labels) / true_labels.shape[0]
print("PyTorch accuracy =", torch_accuracy.item())

optimizer.step()
nntile.tensor.copy_async(data_train_tensor, mlp_nntile.activations[0].value)
mlp_nntile.forward_async()

nntile_last_layer_output = np.zeros(mlp_nntile.activations[-1].value.shape, order="F", dtype=np.float32)
mlp_nntile.activations[-1].value.to_array(nntile_last_layer_output)
pred_labels = np.argmax(nntile_last_layer_output, 1)
nntile_accuracy = np.sum(true_labels.numpy() == pred_labels) / true_labels.shape[0]
print("NNTile accuracy =", nntile_accuracy)

nntile.tensor.copy_async(label_train_tensor, crit_nntile.y)
crit_nntile.calc_async()
nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
crit_nntile.get_val(nntile_xentropy_np)
print("NNTile loss after optimizer step =", nntile_xentropy_np[0])


mlp_nntile.unregister()
crit_nntile.unregister()
data_train_tensor.unregister()
label_train_tensor.unregister()