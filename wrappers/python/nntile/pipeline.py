# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/pipeline.py
# Training pipeline of NNTile Python package
#
# @version 1.1.0

from typing import Any, List

import numpy as np

import nntile.utils.constructors as nntc
from nntile.graph_capture_sched import (
    graph_recording_begin, graph_recording_end)
from nntile.model.base_model import BaseModel
from nntile.nntile_core.starpu import iteration_pop, iteration_push
from nntile.tensor import (
    Tensor, Tensor_bool, TensorTraits, clear_async, copy_async, isfinite_async,
    log_scalar_async, scale_inplace_async)


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

    def train_async(self, log_loss=True):
        for i_epoch in range(self.n_epochs):
            num_batches = len(self.x)
            for i_batch, (x_batch, y_batch) in enumerate(zip(self.x, self.y)):
                # Provide batch number to the FXT trace
                iteration_push(i_batch)
                # StarPU graph batch capture (SGOC DSO; no-op for e.g. dmdasd).
                graph_recording_begin()
                # Minibatch 0 clears parameter grads and output loss.
                i_minibatch = 0
                iteration_push(i_minibatch)
                # Zero out gradients of all weights
                self.model.clear_parameters_grads()
                clear_async(self.loss.val)
                # Mark end of minibatch number 0
                iteration_pop()
                # Accumulate gradients from subbatches
                for x_minibatch, y_minibatch in zip(x_batch, y_batch):
                    # Next minibatch: mark start of forward pass.
                    i_minibatch += 1
                    iteration_push(i_minibatch)
                    # Copy input batch into activation[0] of the model
                    copy_async(x_minibatch, self.model.activations[0].value)
                    # Perform forward pass
                    self.model.forward_async()
                    # Copy true result into loss function
                    copy_async(y_minibatch, self.loss.y)
                    # Loss function shall be instatiated to read X from
                    # activations[-1].value of the model and write gradient
                    # into activations[-1].grad
                    # Mark end of the forward pass
                    iteration_pop()
                    # Mark start of the backward pass
                    i_minibatch += 1
                    iteration_push(i_minibatch)
                    # Clear gradients of inter-layer activations
                    self.model.clear_activations_grads()
                    # We cound loss as backward operation
                    self.loss.calc_async()
                    # Perform backward pass
                    self.model.backward_async()
                    # Invalidate activations[2:]. We have to keep
                    # activations[1] as it holds positional embedding indices,
                    # that are computed once
                    if (self.model.config.name == "bert" or
                        self.model.config.name == "roberta"):
                        for t in self.model.activations[3:]:
                            t.value.invalidate_submit()
                    else:
                        for t in self.model.activations[2:]:
                            t.value.invalidate_submit()
                    # Invalidate gradients of activations
                    for t in self.model.activations:
                        if t.grad_required:
                            t.grad.invalidate_submit()
                    # Mark end of the backward pass
                    iteration_pop()
                # Define optimizer step as minibatch number 4,294,967,295
                # (maximal unsigned 32-bit integer)
                i_minibatch = 4_294_967_295
                iteration_push(i_minibatch)
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
                if log_loss:
                    log_scalar_async("Train loss", self.loss.val)
                # End graph batch capture (see graph_capture_sched).
                graph_recording_end()
                loss_np = self.loss.get_val()
                self.loss_hist.append(loss_np[0])
                # print("Loss in {} epoch = {}".format(i_epoch, loss_np[0]))
                print("Batch={}/{} Epoch={}/{} Loss={}".format(
                        i_batch + 1, num_batches, i_epoch + 1, self.n_epochs,
                        loss_np[0]), flush=True)
                # Mark end of optimizer step
                iteration_pop()
                # Mark end of batch
                iteration_pop()

    def train_with_scaler_async(self, init_scale: float,
                                      downscale_step: float,
                                      upscale_step: float,
                                      plateau_scale_counter: float,
                                      log_loss: bool = True):
        loss_scale = init_scale
        traits_flag = TensorTraits([], [])
        flag = Tensor_bool(traits_flag)
        flag_init_val = 1
        np_dst_init = np.array([flag_init_val], dtype=bool)
        flag.from_array(np_dst_init)
        good_scale_counter = 0
        for i_epoch in range(self.n_epochs):
            # Provide epoch number to the FXT trace
            iteration_push(i_epoch)
            # print("Epoch ", i_epoch)
            num_batches = len(self.x)
            for i_batch, (x_batch, y_batch) in enumerate(zip(self.x, self.y)):
                # Provide batch number to the FXT trace
                iteration_push(i_batch)
                num_loss_scale_updates = 0
                while True:
                    flag.from_array(np_dst_init)
                    # Scale loss for further de-scale
                    self.loss.scale *= loss_scale
                    # Zero out gradients of all weights
                    self.model.clear_parameters_grads()
                    clear_async(self.loss.val)
                    # Accumulate gradients from subbatches
                    for x_minibatch, y_minibatch in zip(x_batch, y_batch):
                        # Clear gradients of inter-layer activations
                        self.model.clear_activations_grads()
                        # Copy input batch into activation[0] of the model
                        copy_async(x_minibatch,
                                   self.model.activations[0].value)
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
                        # activations[1] as it holds positional embedding
                        # indices, that are computed once
                        if (self.model.config.name == "bert" or
                            self.model.config.name == "roberta"):
                            for t in self.model.activations[3:]:
                                t.value.invalidate_submit()
                        else:
                            for t in self.model.activations[2:]:
                                t.value.invalidate_submit()
                        # Invalidate gradients of activations
                        for t in self.model.activations:
                            if t.grad_required:
                                t.grad.invalidate_submit()
                    isfinite_grads = True
                    loss_np = self.loss.get_val()
                    for p in self.model.parameters:
                        if p.grad_required:
                            isfinite_async(p.grad, flag)
                    isfinite_grads = nntc.to_numpy(flag)[0]
                    self.loss.scale /= loss_scale
                    if not isfinite_grads:
                        print("Inf/NaN in gradients are found for scale {}!"
                        " Reduce the loss_scale...".format(loss_scale))
                        loss_scale /= downscale_step
                        num_loss_scale_updates += 1
                        good_scale_counter = 0
                    else:
                        print("Accept the current loss scale = {}!"
                        " Keep training...".format(loss_scale))
                        break
                if num_loss_scale_updates == 0:
                    good_scale_counter += 1
                # De-scale gradients
                for p in self.model.parameters:
                    if p.grad_required:
                        scale_inplace_async(1. / loss_scale, p.grad)
                # Apply optimizer after gradients for entire batch are
                # accumulated
                self.opt.step()
                # for p in self.opt.first_moments:
                #     isfinite_async(p, flag)
                # isfinite_grads = nntc.to_numpy(flag)[0]
                # print("Isfinite first monents", isfinite_grads)
                # for p in self.opt.second_moments:
                #     isfinite_async(p, flag)
                # isfinite_grads = nntc.to_numpy(flag)[0]
                # print("Isfinite second monents", isfinite_grads)
                # for p in self.model.parameters:
                #     isfinite_async(p.value, flag)
                # isfinite_grads = nntc.to_numpy(flag)[0]
                # print("Isfinite parameters", isfinite_grads)
                # # Invalidate gradients of parameters and hint to offload
                # parameters
                for p in self.model.parameters:
                    p.value.wont_use()
                    if p.grad_required:
                        p.grad.invalidate_submit()
                # Limit parallelism through value of loss
                if log_loss:
                    log_scalar_async("Train loss", self.loss.val)
                # De-scaling
                loss_np = self.loss.get_val() / loss_scale
                self.loss_hist.append(loss_np[0])
                # print("Loss in {} epoch = {}".format(i_epoch, loss_np[0]))
                print("Batch={}/{} Epoch={}/{} Loss={}".format(
                        i_batch + 1, num_batches, i_epoch + 1, self.n_epochs,
                        loss_np[0]), flush=True)
                if good_scale_counter == plateau_scale_counter:
                    good_scale_counter = 0
                    if loss_scale * upscale_step < init_scale:
                        loss_scale *= upscale_step
                        print("Increase loss scale to {}...".format(
                            loss_scale))
                # Finish current batch in the FXT trace
                iteration_pop()
            # nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
            # self.loss.get_val(nntile_xentropy_np)
            # print("Last batch loss after in {} epoch = {}".format(
            #       i_epoch, nntile_xentropy_np[0]))
            # Finish current epoch in the FXT trace
            iteration_pop()

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
