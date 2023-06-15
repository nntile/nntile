import torch.nn as nn
import torch

from nntile.model.base_model import BaseModel
from nntile.layer.linear import Linear
from nntile.tensor import TensorMoments, TensorTraits, Tensor_int64, Tensor_fp32
from nntile.tensor import notrans
from nntile.layer.act import Act
from nntile.layer.add_slice import AddSlice
import nntile
from nntile.loss.frob import Frob
import numpy as np


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(15, 20, bias=False)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(10, 20, bias=False)
        self.lin3 = nn.Linear(20, 10, bias=False)

    def forward(self, x, y):
        x = self.lin1(x)
        x = self.relu(x)
        y = self.lin2(y)
        x = x + y.unsqueeze(0)
        x = self.lin3(x)
        return x

batch_size = 10
input_dim= 15
x_input_torch = torch.randn(batch_size, input_dim)
y_input_torch = torch.randn(10)
torch_model = ToyModel()
torch_output = torch_model(x_input_torch, y_input_torch)
torch_loss = torch.sum(torch.square(torch_output)) * 0.5
torch_loss.backward()
torch_loss_val = torch_loss.item()
print("Torch loss = {}".format(torch_loss_val))

class NNTileToyModel(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, y: TensorMoments, axis: int, next_tag: int):
        activations = [x, y]
        layers = []
        new_layer, next_tag = Linear.generate_simple_mpiroot(x, "L", notrans,
                1, [20], [20], next_tag)
        
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Act.generate_simple(activations[-1], "relu",
                                                      next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple_mpiroot(activations[1], "L", notrans,
                1, [20], [20], next_tag)
        
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = AddSlice.generate_simple(activations[3], activations[-1], axis,
                                                      next_tag)
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        new_layer, next_tag = Linear.generate_simple_mpiroot(activations[-1], "L", notrans,
                1, [10], [10], next_tag)
        
        layers.append(new_layer)
        activations.extend(new_layer.activations_output)

        self.next_tag = next_tag
        super().__init__(activations, layers)
        
    
    @staticmethod
    def from_torch(torch_model, batch_size, input_dim, axis, next_tag):

        x_traits = TensorTraits([batch_size, input_dim], \
        [batch_size, input_dim])
        x_distr = [0] * x_traits.grid.nelems
        x = Tensor_fp32(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x_grad = None
        x_grad_required = False
        x = TensorMoments(x, x_grad, x_grad_required)

        y_traits = TensorTraits([10], [10])
        y_distr = [0] * y_traits.grid.nelems
        y = Tensor_fp32(y_traits, y_distr, next_tag)
        next_tag = y.next_tag
        y_grad = None
        y_grad_required = False
        y = TensorMoments(y, y_grad, y_grad_required)

        nntile_model = NNTileToyModel(x, y, axis, next_tag)

        for p_nntile, p_torch in zip(nntile_model.parameters, torch_model.parameters()):
            p_nntile.value.from_array(p_torch.detach().numpy().T)

        return nntile_model, nntile_model.next_tag


config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
next_tag = 0
axis = 0
nntile_model, next_tag = NNTileToyModel.from_torch(torch_model, batch_size, input_dim, axis, next_tag)

nntile_model.activations[0].value.from_array(x_input_torch.numpy())
nntile_model.activations[1].value.from_array(y_input_torch.numpy())

nntile_model.forward_async()
nntile_model.clear_gradients()

nntile_loss_func, next_tag =  Frob.generate_simple(nntile_model.activations[-1], next_tag)
nntile_loss_func.y.from_array(np.zeros((batch_size, 10), order="F", dtype=np.float32))

nntile_loss_func.calc_async()


nntile_model.backward_async()


val_np = np.zeros((1,), order="F", dtype=np.float32)
nntile_loss_func.val.to_array(val_np)
print("NNTile loss = {}".format(val_np[0]))
print("Relative diff between Pytorch and NNTile models losses = {}".format(abs(val_np[0] - torch_loss_val) / torch_loss_val))

for i, (p_nntile, p_torch) in enumerate(zip(nntile_model.parameters, torch_model.parameters())):
    p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
    p_nntile.grad.to_array(p_nntile_grad_np)
    # print(p_nntile_grad_np)
    rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np).T) / torch.norm(p_torch.grad)
    print("Relative error in gradient in layer {} = {}".format(i, rel_error.item()))

nntile_model.unregister()
nntile_loss_func.unregister()