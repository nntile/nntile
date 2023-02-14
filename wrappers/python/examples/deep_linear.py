# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/deep_linear.py
# Example of using Deep Linear netwokr of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-14

# Imports
import nntile

# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define tensor X for input batches
next_tag = 0
x_traits = nntile.tensor.TensorTraits([8, 8], [2, 2])
x = nntile.tensor.Tensor_fp32(x_traits, [0]*x_traits.grid.nelems, next_tag)
next_tag = x.next_tag
dx = nntile.tensor.Tensor_fp32(x_traits, [0]*x_traits.grid.nelems, next_tag)
next_tag = dx.next_tag
x_moments = nntile.tensor.TensorMoments(x, dx, False)

# Define deep linear network
m = nntile.model.DeepLinear(x_moments, 'R', 1, 4, 3, 3, next_tag)

# Randomly init weights of deep linear network
m.init_randn_async()

# Randomly init input batch
seed = 100
mean = 1.0
dev = 2.0
nntile.tensor.randn_async(x, [0]*len(x_traits.shape), x_traits.shape, seed,
        mean, dev)

# Run forward propagation
m.forward_async()

# Randomly generate gradient of loss by the output of model
grad = m.activations[-1].grad
seed_grad = 10000000
mean_grad = -1.0
dev_grad = 1.0
nntile.tensor.randn_async(grad, [0]*len(grad.shape), grad.shape, seed_grad,
        mean_grad, dev_grad)

# Run backward propagation
m.backward_async()

# Wait for all computations to finish
nntile.starpu.wait_for_all()

