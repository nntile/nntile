# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/pipeline.py
# TRaining pipeline of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-02-18

from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        copy_async, axpy_async
from nntile.layer.base_layer import BaseLayer
from nntile.model.base_model import BaseModel
import numpy as np
from typing import List, Any

class Pipeline(object):
    x: List[Tensor]
    y: List[Tensor]
    model: BaseModel
    opt: Any
    loss: Any
    n_epoch: int
    lr: float

    def __init__(self, x: List[Tensor], y: List[Tensor], model: BaseModel, opt,
            loss, n_epochs, lr):
        self.x = x
        self.y = y
        self.model = model
        self.opt = opt
        self.loss = loss
        self.n_epochs = n_epochs
        self.lr = lr

    def train_async(self):
        for i_epoch in range(self.n_epochs):
            for x_batch, y_batch in zip(self.x, self.y):
                # Copy input batch into activation[0] of the model
                # print("Copy async")
                copy_async(x_batch, self.model.activations[0].value)
                # Perform forward pass
                # print("Model forward")
                self.model.forward_async()
                # Copy true result into loss function
                # print("Copy true labels in loss")
                copy_async(y_batch, self.loss.y)

                # Loss function shall be instatiated to read X from
                # activations[-1].value of the model and write gradient into
                # activations[-1].grad
                # print("Compute loss")
                self.loss.calc_async()
                nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
                self.loss.get_val(nntile_xentropy_np)
                print("Loss in {} epoch = {}".format(i_epoch, nntile_xentropy_np[0]))
                # Now do the backward pass
                # print("backward")
                self.model.backward_async()
                # Apply optimizer here
                # print("Optimizer step")
                self.opt.step()
                break
            nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
            self.loss.get_val(nntile_xentropy_np)
            print("Last batch loss after in {} epoch = {}".format(i_epoch, nntile_xentropy_np[0]))
            break

