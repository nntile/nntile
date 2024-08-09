# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/test_gpt2.py
# GPT2 performance analysis workflow
#
# @version 1.1.0

import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from transformers import GPT2Config, GPT2LMHeadModel

import nntile
from nntile.model.gpt2 import (
    GPT2Config as GPT2Config_nntile, GPT2Model as GPT2Model_nntile)

data_dir = Path(__file__).parent / 'testdata'
nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


def check_grads(config, model_torch, nntile_model):
    nntile_par_idx = 0
    for name, p_torch in model_torch.named_parameters():
        p_torch_grad_np = p_torch.grad.cpu().detach().numpy()
        layer_name = name.split(".")[-2]
        if layer_name == "lm_head":
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                    dtype=np.float32)
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        elif layer_name == "c_attn" and name.split(".")[-1] == "weight":
            diff = 0
            norm = np.linalg.norm(p_torch_grad_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                        dtype=np.float32)
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.reshape(-1, config.n_embd)
                current_grad_block_ix = slice(i_tensor * config.n_embd,
                                              (i_tensor + 1) * config.n_embd)
                current_grad_block = p_torch_grad_np[:, current_grad_block_ix]
                diff_tensor = current_grad_block.T - p_nntile_grad_np
                diff += np.linalg.norm(diff_tensor) ** 2
                nntile_par_idx += 1
            diff = diff ** 0.5
        elif layer_name == "c_attn" and name.split(".")[-1] == "bias":
            diff = 0
            norm = np.linalg.norm(p_torch_grad_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                        dtype=np.float32)
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.transpose().reshape(-1)
                current_grad_block = p_torch_grad_np[i_tensor * config.n_embd:
                        (i_tensor + 1) * config.n_embd]
                diff_tensor = current_grad_block - p_nntile_grad_np
                diff += np.linalg.norm(diff_tensor) ** 2
                nntile_par_idx += 1
            diff = diff ** 0.5
        elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
            if name.split(".")[-1] == "weight":
                diff = 0
                norm = np.linalg.norm(p_torch_grad_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                        dtype=np.float32)
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.reshape(config.n_embd, -1)
                diff = np.linalg.norm(p_torch_grad_np.T - p_nntile_grad_np)
                nntile_par_idx += 1
            elif name.split(".")[-1] == "bias":
                norm = np.linalg.norm(p_torch_grad_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                        dtype=np.float32)
                p_nntile.grad.to_array(p_nntile_grad_np)
                diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
                nntile_par_idx += 1
        elif len(p_torch.shape) == 2:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                    dtype=np.float32)
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np.T)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        elif len(p_torch.shape) == 1:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F",
                    dtype=np.float32)
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        print("Gradient of {}: norm={} rel_err={}".format(name, norm,
                diff / norm))


def check_params(config, model_torch, nntile_model):
    nntile_par_idx = 0
    for name, p_torch in model_torch.named_parameters():
        p_torch_np = p_torch.cpu().detach().numpy()
        layer_name = name.split(".")[-2]
        if layer_name == "lm_head":
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                    dtype=np.float32)
            p_nntile.value.to_array(p_nntile_np)
            diff = np.linalg.norm(p_torch_np - p_nntile_np)
            norm = np.linalg.norm(p_torch_np)
            nntile_par_idx += 1
        elif layer_name == "c_attn" and name.split(".")[-1] == "weight":
            diff = 0
            norm = np.linalg.norm(p_torch_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                        dtype=np.float32)
                p_nntile.value.to_array(p_nntile_np)
                p_nntile_np = p_nntile_np.reshape(-1, config.n_embd)
                current_block_ix = slice(i_tensor * config.n_embd,
                                         (i_tensor + 1) * config.n_embd)
                current_block = p_torch_np[:, current_block_ix]
                diff += np.linalg.norm(current_block.T - p_nntile_np) ** 2
                nntile_par_idx += 1
            diff = diff ** 0.5
        elif layer_name == "c_attn" and name.split(".")[-1] == "bias":
            diff = 0
            norm = np.linalg.norm(p_torch_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                        dtype=np.float32)
                p_nntile.value.to_array(p_nntile_np)
                p_nntile_np = p_nntile_np.transpose().reshape(-1)
                current_block = p_torch_np[i_tensor * config.n_embd:
                        (i_tensor + 1) * config.n_embd]
                diff += np.linalg.norm(current_block - p_nntile_np) ** 2
                nntile_par_idx += 1
            diff = diff ** 0.5
        elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
            if name.split(".")[-1] == "weight":
                diff = 0
                norm = np.linalg.norm(p_torch_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                        dtype=np.float32)
                p_nntile.value.to_array(p_nntile_np)
                p_nntile_np = p_nntile_np.reshape(config.n_embd, -1)
                diff = np.linalg.norm(p_torch_np.T - p_nntile_np)
                nntile_par_idx += 1
            elif name.split(".")[-1] == "bias":
                norm = np.linalg.norm(p_torch_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                        dtype=np.float32)
                p_nntile.value.to_array(p_nntile_np)
                diff = np.linalg.norm(p_torch_np - p_nntile_np)
                nntile_par_idx += 1
        elif len(p_torch.shape) == 2:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                    dtype=np.float32)
            p_nntile.value.to_array(p_nntile_np)
            diff = np.linalg.norm(p_torch_np - p_nntile_np.T)
            norm = np.linalg.norm(p_torch_np)
            nntile_par_idx += 1
        elif len(p_torch.shape) == 1:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_np = np.zeros(p_nntile.value.shape, order="F",
                    dtype=np.float32)
            p_nntile.value.to_array(p_nntile_np)
            diff = np.linalg.norm(p_torch_np - p_nntile_np)
            norm = np.linalg.norm(p_torch_np)
            nntile_par_idx += 1
        print("Parameter {}: norm={} rel_err={}".format(name, norm,
                diff / norm))


@pytest.mark.xfail(reason='not implemented')
@pytest.mark.slow
@pytest.mark.parametrize(
    'num_samples,batch_size,minibatch_size,minibatch_size_tile,seq_len_tile,'
    'optimizer,lr,nepochs', [
        (8, 4, 2, 2, 1024, 'adam', 1e-4, 3),
        (8, 4, 2, 2, 1024, 'sgd', 1e-4, 3),
    ])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cpu', marks=nocuda)])
def test_gpt2(device, num_samples, batch_size, minibatch_size,
              minibatch_size_tile, seq_len_tile, optimizer, lr, nepochs):
    assert num_samples % batch_size == 0
    assert batch_size % minibatch_size == 0
    num_minibatch = batch_size // minibatch_size
    assert minibatch_size % minibatch_size_tile == 0

    # Init model locally or remote and save the corresponding checkpoint for
    # further processing.
    config = GPT2Config.from_json_file(data_dir / 'gpt2_test_config.json')

    n_head_tile = config.n_head
    assert config.n_positions % seq_len_tile == 0
    config.attn_pdrop = 0
    config.embd_pdrop = 0
    config.resid_pdrop = 0
    if config.n_inner is not None:
        inner_dim = config.n_inner
    else:
        inner_dim = 4 * config.hidden_size
    config.n_inner = inner_dim
    model_torch = GPT2LMHeadModel(config).to(device)
    model_torch.lm_head.weight = nn.Parameter(model_torch.lm_head
        .weight.detach().clone())

    # Initialize NNTile and StarPU
    time0 = time.time()
    # Set up StarPU+MPI and init codelets
    _nntile_config = nntile.starpu.Config(-1, -1, 1)
    nntile.starpu.init()
    # Restrict computations to CUDA if possible
    if device == "cuda":
        nntile.starpu.restrict_cuda()
    elif device == "cpu":
        nntile.starpu.restrict_cpu()
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
    next_tag = 0
    nntile_flashattention = False
    nntile_use_redux = False
    n_embd_tile = 384
    n_inner_tile = 1536
    nntile_model_config = GPT2Config_nntile(config.vocab_size, n_embd_tile,
        config.n_embd, n_embd_tile, config.max_position_embeddings,
        config.n_inner, n_inner_tile, config.layer_norm_epsilon,
        config.num_hidden_layers, config.n_head, n_head_tile,
        "gelutanh", nntile_flashattention, nntile_use_redux)
    nntile_model, next_tag = GPT2Model_nntile.from_torch(model_torch,
            minibatch_size, minibatch_size_tile, config.n_positions,
            seq_len_tile, nntile_model_config, next_tag)
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
            nntile_model.activations[-1], next_tag)

    if optimizer == "adam":
        nntile_optimizer = nntile.optimizer \
            .FusedAdam(nntile_model.get_parameters(), lr, next_tag)
        torch_optimizer = Adam(model_torch.parameters(), lr)
    elif optimizer == "sgd":
        nntile_optimizer = nntile.optimizer.SGD(nntile_model.get_parameters(),
                                                lr, next_tag)
        torch_optimizer = SGD(model_torch.parameters(), lr)
    next_tag = nntile_optimizer.get_next_tag()

    # Create random dataset for train sumulation
    num_train_batches = num_samples // batch_size
    num_minibatch = batch_size // minibatch_size
    torch.manual_seed(0)
    random_dataset_shape = (num_train_batches, num_minibatch, minibatch_size,
                            config.n_positions + 1)
    random_dataset = torch.randint(config.vocab_size, random_dataset_shape,
                                   dtype=torch.int64, device=device)
    torch_input = random_dataset[:, :, :, :-1]
    torch_output = random_dataset[:, :, :, 1:]
    # Run train loop for n_epoch for PyTorch model and report loss after every
    # epoch.
    torch_loss_func = nn.CrossEntropyLoss(reduction="sum")
    # Define Cross Entropy loss function
    nntile_loss_func, next_tag = nntile.loss.CrossEntropy.generate_simple(
            nntile_model.activations[-1], next_tag)
    torch_loss_hist = []

    for i in range(nepochs):
        for j in range(num_train_batches):
            torch_optimizer.zero_grad()
            loss = torch.zeros(1, dtype=torch.float32, device=device)
            for k in range(num_minibatch):
                train_input = torch_input[j][k].to(device)
                train_labels = torch_output[j][k].to(device).reshape(-1)
                train_output = model_torch(train_input)
                train_logits = train_output.logits \
                    .reshape(-1, config.vocab_size)
                loss_local = torch_loss_func(train_logits, train_labels)
                loss += loss_local
            torch_loss_hist.append(loss.item())
            loss.backward(retain_graph=True)
            torch_optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    # Train with NNtile model
    batch_input = []
    batch_output = []
    x_traits = nntile.tensor.TensorTraits(
            [config.n_positions, minibatch_size],
            [seq_len_tile, minibatch_size_tile])
    x_distr = [0] * x_traits.grid.nelems
    for i in range(num_train_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(num_minibatch):
            x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            x_buf = random_dataset[i, j, :, :-1].cpu().T
            x.from_array(np.asfortranarray(x_buf))
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = y.next_tag
            y_buf = random_dataset[i, j, :, 1:].cpu().T
            y.from_array(np.asfortranarray(y_buf))
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)
    time1 = time.time() - time0
    print("From PyTorch loader to NNTile batches in {} seconds".format(time1))

    # Set up training pipeline
    pipeline = nntile.pipeline.Pipeline(batch_input, batch_output,
            nntile_model, nntile_optimizer, nntile_loss_func, nepochs)
    pipeline.train_async()
    nntile.starpu.wait_for_all()
    for i in range(len(torch_loss_hist)):
        rel_error = (abs(torch_loss_hist[i] - pipeline.loss_hist[i]) /
                     torch_loss_hist[i])
        assert rel_error < 1e-4

    nntile_loss_func.unregister()
    nntile_optimizer.unregister()
    for batch in batch_input + batch_output:
        for x in batch:
            x.unregister()

    # Unregister all tensors related to model
    nntile_model.unregister()
