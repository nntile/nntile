import torch.optim as optim
import torch.nn as nn
import torch
import nntile
import time
import copy
from nntile.tensor import copy_async, clear_async
import numpy as np


# Create simple PyTorch model and make some optimizer steps
class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, n_layers, n_classes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, dim_hidden)])
        self.layers.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(n_layers - 2)])
        self.layers.append(nn.Linear(dim_hidden, n_classes))
        self.relu = nn.ReLU()

    def forward(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = self.relu(x)
        x = self.layers[-1](x)
        return x

input_dim = 5
hidden_dim = 100
n_classes = 10
n_layers = 5
device = "cuda"
lr = 1e-3
n_epoch = 3

torch_mlp = MLP(input_dim, hidden_dim, n_layers, n_classes).to(device)
init_torch_mlp = copy.deepcopy(torch_mlp)

n_samples = 1024
batch_size = 32
minibatch_size = 8
num_batches = n_samples // batch_size
num_minibatches = num_batches // minibatch_size
X_torch = torch.randn(num_batches, num_minibatches, minibatch_size, input_dim, device=device, dtype=torch.float32)
print("Shape of the random data", X_torch.shape)
y_torch = torch.randint(n_classes, (num_batches, num_minibatches, minibatch_size), device=device, dtype=torch.int64)
torch_loss = nn.CrossEntropyLoss(reduction="sum")
torch_optim = optim.Adam(torch_mlp.parameters(), lr=lr)

torch_loss_history = []

for i_epoch in range(n_epoch):
    for i_batch in range(num_batches):
        loss_val = 0
        torch_optim.zero_grad()
        for i_minibatch in range(num_minibatches):
            output = torch_mlp(X_torch[i_batch][i_minibatch])
            loss_val += torch_loss(output, y_torch[i_batch][i_minibatch])
        loss_val.backward(retain_graph=True)
        torch_optim.step()
        torch_loss_history.append(loss_val.item())

# Set up StarPU+MPI and init codelets
time0 = -time.time()
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
nntile.starpu.restrict_cuda()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0

time0 = -time.time()
batch_data = []
batch_labels = []
minibatch_tile = minibatch_size
input_dim_tile = input_dim
x_traits = nntile.tensor.TensorTraits([input_dim, minibatch_size], \
        [input_dim_tile, minibatch_tile])
x_distr = [0] * x_traits.grid.nelems
y_traits = nntile.tensor.TensorTraits([minibatch_size], [minibatch_tile])
y_distr = [0] * y_traits.grid.nelems
for i_batch in range(num_batches):
    minibatch_x = []
    minibatch_y = []
    for i_minibatch in range(num_minibatches):
        x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_np = np.array(X_torch[i_batch][i_minibatch].cpu().numpy(), order="F", dtype=np.float32)
        x.from_array(x_np.T)
        minibatch_x.append(x)
        y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
        next_tag = y.next_tag
        y.from_array(y_torch[i_batch][i_minibatch].cpu().numpy())
        minibatch_y.append(y)
    batch_data.append(minibatch_x)
    batch_labels.append(minibatch_y)

# Wait for all scatters to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("From PyTorch loader to NNTile batches in {} seconds".format(time0))

# Define tensor X for input batches
time0 = -time.time()

# Define deep ReLU network
nntile_model, next_tag = nntile.model.DeepReLU.from_torch(init_torch_mlp, minibatch_size, n_classes,
                                                          "relu", next_tag)

# Set up Cross Entropy loss function for the model
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(nntile_model.activations[-1],
                                                          next_tag)
nntile_loss_hist = []

nntile_optimizer = nntile.optimizer.Adam(nntile_model.get_parameters(), lr, next_tag)
next_tag = nntile_optimizer.get_next_tag()

pipeline = nntile.pipeline.Pipeline(batch_data, batch_labels, nntile_model, nntile_optimizer,
                                    loss, n_epoch)
pipeline.train_async()

nntile.starpu.wait_for_all()

nntile_loss_hist = copy.deepcopy(pipeline.loss_hist)
for i in range(len(torch_loss_history)):
    # print(abs(torch_loss_history[i] - nntile_loss_hist[i]) / torch_loss_history[i])
    assert abs(torch_loss_history[i] - nntile_loss_hist[i]) / torch_loss_history[i] < 1e-6

loss.unregister()
for batch in batch_data + batch_labels:
    for x in batch:
        x.unregister()
nntile_optimizer.unregister()
nntile_model.unregister()