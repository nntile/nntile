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
device = "cpu"

torch_mlp = MLP(input_dim, hidden_dim, n_layers, n_classes).to(device)

n_samples = 1024
batch_size = 32
minibatch_size = 8
num_batches = n_samples // batch_size
num_minibatches = num_batches // minibatch_size
X_torch = torch.randn(num_batches, num_minibatches, minibatch_size, input_dim, device=device, dtype=torch.float32)

y_torch = torch.randint(n_classes, (num_batches, num_minibatches, minibatch_size), device=device, dtype=torch.int64)
torch_loss = nn.CrossEntropyLoss(reduction="sum")
torch_loss_history = []

for i_batch in range(num_batches):
    loss_val = 0
    for i_minibatch in range(num_minibatches):
        output = torch_mlp(X_torch[i_batch][i_minibatch])
        loss_val += torch_loss(output, y_torch[i_batch][i_minibatch])
    torch_loss_history.append(loss_val.item())

# Set up StarPU+MPI and init codelets
time0 = -time.time()
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
nntile.starpu.restrict_cpu()
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
    for i_minimatch in range(num_minibatches):
        x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x.from_array(X_torch[i_batch][i_minibatch].cpu().numpy().T)
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
x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
next_tag = x.next_tag
x_grad = None
x_grad_required = False
x_moments = nntile.tensor.TensorMoments(x, x_grad, x_grad_required)

# Define deep ReLU network
gemm_ndim = 1
hidden_dim_tile = hidden_dim
depth = n_layers
nntile_model, next_tag = nntile.model.DeepReLU.from_torch(torch_mlp, minibatch_size, n_classes, "relu", next_tag)

# Set up Cross Entropy loss function for the model
# print("shape of the last activation", nntile_model.activations[-1].value.shape)
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(nntile_model.activations[-1], \
        next_tag)
nntile_loss_hist = []

for x_batch, y_batch in zip(batch_data, batch_labels):
    clear_async(loss.val)
    # Accumulate gradients from minibatches
    for x_minibatch, y_minibatch in zip(x_batch, y_batch):
        copy_async(x_minibatch, nntile_model.activations[0].value)
        # Perform forward pass
        nntile_model.forward_async()
        # Copy true result into loss function
        copy_async(y_minibatch, loss.y)
        # Loss function shall be instatiated to read X from
        # activations[-1].value of the model and write gradient
        # into activations[-1].grad
        loss.calc_async()
    loss_np = np.zeros((1,), dtype=np.float32, order="F")
    loss.get_val(loss_np)
    nntile_loss_hist.append(loss_np[0])

nntile.starpu.wait_for_all()

for i in range(len(torch_loss_history)):
    print(abs(torch_loss_history[i] - nntile_loss_hist[i]) / torch_loss_history[i])

loss.unregister()
for batch in batch_data + batch_labels:
    for x in batch:
        x.unregister()
        
nntile_model.unregister()