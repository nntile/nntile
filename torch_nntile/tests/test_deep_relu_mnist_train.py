# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_deep_relu_mnist_train.py
# Full-batch MNIST training: CPU vs nntile loss and weight parity.

import torch
import pytest

pytest.importorskip("torchvision")
from torchvision import datasets

import torch_nntile
from torch_nntile import _C
from torch_nntile.models import DeepReLU
from torch_nntile.training import (
    clone_model_weights,
    max_weight_delta,
    train_full_batch_step,
)


pytestmark = [
    pytest.mark.skipif(
        not _C.has_libnntile(),
        reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def mnist_full_batch(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mnist")
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
    )
    images = dataset.data.reshape(len(dataset), -1).to(torch.float32) / 255.0
    labels = dataset.targets.clone()
    assert images.shape == (60_000, 28 * 28)
    assert labels.shape == (60_000,)
    return images, labels


def test_mnist_full_batch_training_matches_cpu(mnist_full_batch):
    """Train DeepReLU on all 60k MNIST images; compare CPU vs nntile."""
    images, labels = mnist_full_batch
    seed = 42
    epochs = 3
    lr = 0.1

    torch.manual_seed(seed)
    model_cpu = DeepReLU.mnist()
    model_cpu.init_kaiming_uniform_(seed=seed)

    model_nnt = DeepReLU.mnist()
    model_nnt.load_state_dict(model_cpu.state_dict())
    model_nnt = model_nnt.to("nntile")

    assert max_weight_delta(
        clone_model_weights(model_cpu),
        clone_model_weights(model_nnt),
    ) == 0.0

    cpu_losses: list[float] = []
    nnt_losses: list[float] = []
    x_nnt = images.to("nntile")
    y_nnt = labels.to("nntile")

    for _ in range(epochs):
        cpu_losses.append(
            train_full_batch_step(model_cpu, images, labels, lr)
        )
        nnt_losses.append(
            train_full_batch_step(model_nnt, x_nnt, y_nnt, lr)
        )

    for loss_cpu, loss_nnt in zip(cpu_losses, nnt_losses):
        assert abs(loss_cpu - loss_nnt) < 1e-3

    weight_delta = max_weight_delta(
        clone_model_weights(model_cpu),
        clone_model_weights(model_nnt),
    )
    assert weight_delta < 1e-3
