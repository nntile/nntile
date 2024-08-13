# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/pipeline.py
# TRaining pipeline of NNTile Python package
#
# @version 1.1.0

from typing import Any, List

from nntile.model.base_model import BaseModel
from nntile.tensor import Tensor, clear_async, copy_async


class Pipeline(object):
    x: List[List[Tensor]]
    y: List[List[Tensor]]
    model: BaseModel
    opt: Any
    loss: Any
    n_epochs: int
    lr: float

    def __init__(self, x: List[List[Tensor]], y: List[List[Tensor]],
            model: BaseModel, opt, loss, n_epochs):
        self.x = x
        self.y = y
        self.model = model
        self.opt = opt
        self.loss = loss
        self.n_epochs = n_epochs
        self.loss_hist = []

    def train_async(self):
        for i_epoch in range(self.n_epochs):
            # print("Epoch ", i_epoch)
            num_batches = len(self.x)
            for i_batch, (x_batch, y_batch) in enumerate(zip(self.x, self.y)):
                # Zero out gradients of all weights and activations
                self.model.clear_parameters_grads()
                clear_async(self.loss.val)
                # Accumulate gradients from subbatches
                for x_minibatch, y_minibatch in zip(x_batch, y_batch):
                    # Clear gradients of inter-layer activations
                    self.model.clear_activations_grads()
                    # Copy input batch into activation[0] of the model
                    copy_async(x_minibatch, self.model.activations[0].value)
                    # Perform forward pass
                    self.model.forward_async()
                    # Copy true result into loss function
                    copy_async(y_minibatch, self.loss.y)
                    # Loss function shall be instatiated to read X from
                    # activations[-1].value of the model and write gradient
                    # into activations[-1].grad
                    self.loss.calc_async()
                    # Now do the backward pass
                    self.model.backward_async()
                    # Invalidate activations[2:]. We have to keep
                    # activations[1] as it holds positional embedding indices,
                    # that are computed once
                    for t in self.model.activations[2:]:
                        t.value.invalidate_submit()
                    # Invalidate gradients of activations
                    for t in self.model.activations:
                        if t.grad_required:
                            t.grad.invalidate_submit()
                # Apply optimizer after gradients for entire batch are
                # accumulated
                self.opt.step()
                # Invalidate gradients of parameters and hint to offload
                # parameters
                for p in self.model.parameters:
                    p.value.wont_use()
                    if p.grad_required:
                        p.grad.invalidate_submit()
                # Limit parallelism through value of loss
                loss_np = self.loss.get_val()
                self.loss_hist.append(loss_np[0])
                # print("Loss in {} epoch = {}".format(i_epoch, loss_np[0]))
                print("Batch={}/{} Epoch={}/{} Loss={}".format(
                        i_batch + 1, num_batches, i_epoch + 1, self.n_epochs,
                        loss_np[0]), flush=True)
            # nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
            # self.loss.get_val(nntile_xentropy_np)
            # print("Last batch loss after in {} epoch = {}".format(
            #       i_epoch, nntile_xentropy_np[0]))

    def print_meminfo(self):
        params_nbytes = 0
        for params in self.model.parameters:
            params_nbytes += params.get_nbytes()

        acts_nbytes = 0
        for acts in self.model.activations:
            acts_nbytes += acts.get_nbytes()

        opts_nbytes = self.opt.get_nbytes()

        persistent_nbytes = params_nbytes + acts_nbytes + opts_nbytes

        temps_nbytes = 0
        for layer in self.model.layers:
            for temps in layer.temporaries:
                temps_nbytes += temps.get_nbytes()

        print(f"Params+grads (GB): {params_nbytes / 2**30:.3f}")
        print(f"Activations  (GB): {acts_nbytes / 2**30:.3f}")
        print(f"Optimizer    (GB): {opts_nbytes / 2**30:.3f}")
        print(f"Persistent   (GB): {persistent_nbytes / 2**30:.3f}")
        print(f"Temporaries  (GB): {temps_nbytes / 2**30:.3f}")
