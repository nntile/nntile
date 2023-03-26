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
            loss, n_epochs):
        self.x = x
        self.y = y
        self.model = model
        self.opt = opt
        self.loss = loss
        self.n_epochs = n_epochs

    def train_async(self):
        output = np.zeros(self.model.activations[-1].value.shape, order="F", dtype=np.float32)
        true_labels = np.zeros(self.model.activations[-1].value.shape[0], order="F", dtype=np.int64)
        for i_epoch in range(self.n_epochs):
            print("Epoch ", i_epoch)
            for x_batch, y_batch in zip(self.x, self.y):
                # Copy input batch into activation[0] of the model
                # print("Copy async")
                copy_async(x_batch, self.model.activations[0].value)
                # Perform forward pass
                # print("Model forward")
                self.model.forward_async()
                self.model.activations[-1].value.to_array(output)
                print(output)
                y_batch.to_array(true_labels)
                print("Accuracy in the current batch =", np.sum(true_labels == np.argmax(output, axis=1)) / true_labels.shape[0])
                import ipdb
                ipdb.set_trace()
                # Copy true result into loss function
                # print("Copy true labels in loss")
                copy_async(y_batch, self.loss.y)

                # Loss function shall be instatiated to read X from
                # activations[-1].value of the model and write gradient into
                # activations[-1].grad
                # print("Compute loss")
                self.loss.calc_async()
                # nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
                # self.loss.get_val(nntile_xentropy_np)
                # print("Loss in {} epoch = {}".format(i_epoch, nntile_xentropy_np[0]))
                # Now do the backward pass
                # print("backward")
                self.model.backward_async()
                # print("Gradient of last activation")
                # self.model.activations[-1].grad.to_array(output)
                # print(output)
                # Apply optimizer here
                # print("Optimizer step")
                print("Model parameters before opt step")
                for i, p in enumerate(self.model.parameters):
                    print("Parameter", i)
                    p_np = np.zeros(p.value.shape, order="F", dtype=np.float32)
                    p.grad.to_array(p_np)
                    print(p_np.max(), p_np.min())
                self.opt.step()
                # print("Model parameters after opt step")
                # for i, p in enumerate(self.model.parameters):
                #     print("Parameter", i)
                #     p_np = np.zeros(p.value.shape, order="F", dtype=np.float32)
                #     p.value.to_array(p_np)
                #     print(p_np)
            nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
            self.loss.get_val(nntile_xentropy_np)
            print("Last batch loss after in {} epoch = {}".format(i_epoch, nntile_xentropy_np[0]))

