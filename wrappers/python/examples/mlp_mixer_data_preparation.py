# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/mlp_mixer_data_preparation.py
# Helper module to load MNIST data for MLP-Mixer training
#
# @version 1.1.0

import numpy as np
import torch

import nntile


def color_image_patching(image, patch_size):
    c, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "patch size should be divisor of both image height and width"
        )
    n_patches = int(h * w / (patch_size**2))
    n_channels = c * (patch_size**2)
    patched_batch = torch.empty((n_patches, n_channels), dtype=image.dtype)

    n_y = int(w / patch_size)

    for i in range(n_patches):
        x = i // n_y
        y = i % n_y

        for clr in range(c):
            vect_patch = image[
                clr,
                x * patch_size : (x + 1) * patch_size,
                y * patch_size : (y + 1) * patch_size,
            ].flatten()
            patched_batch[
                i, clr * (patch_size**2) : (clr + 1) * (patch_size**2)
            ] = vect_patch
    return patched_batch


def grayscale_image_patching(image, patch_size):
    _, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            "patch size should be divisor of both image height and width"
        )
    n_patches = int(h * w / (patch_size**2))
    n_channels = patch_size**2
    patched_batch = torch.empty((n_patches, n_channels), dtype=image.dtype)

    n_y = int(w / patch_size)

    for i in range(n_patches):
        x = i // n_y
        y = i % n_y

        vect_patch = image[
            0,
            x * patch_size : (x + 1) * patch_size,
            y * patch_size : (y + 1) * patch_size,
        ].flatten()
        patched_batch[i, :] = vect_patch
    return patched_batch


def cifar_data_loader_to_nntile(
    data_set,
    label_set,
    batch_input,
    batch_output,
    trns,
    batch_size,
    minibatch_size,
    patch_size,
    next_tag,
):
    total_len, h, w, num_clr_channels = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size**2))
    n_channels = num_clr_channels * (patch_size**2)

    X_shape = [n_patches, minibatch_size, num_clr_channels * patch_size**2]
    Y_shape = [minibatch_size]

    tmp_data_tensor = torch.empty(
        (n_patches, minibatch_size, n_channels), dtype=torch.float32
    )
    tmp_label_tensor = torch.empty(minibatch_size, dtype=torch.float32)

    x_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    x_distr = [0] * x_traits.grid.nelems

    y_traits = nntile.tensor.TensorTraits(Y_shape, Y_shape)
    y_distr = [0] * y_traits.grid.nelems

    for i in range(n_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                tmp_data_tensor[:, k, :] = color_image_patching(
                    trns(
                        data_set[
                            i * batch_size + j * minibatch_size + k, :, :, :
                        ]
                    ),
                    patch_size,
                )
                tmp_label_tensor[k] = label_set[
                    i * batch_size + j * minibatch_size + k
                ]
            x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            x.from_array(np.asfortranarray(tmp_data_tensor.numpy()))
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(
                np.asfortranarray(tmp_label_tensor.numpy().reshape(-1))
            )
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)


def mnist_data_loader_to_nntile(
    data_set,
    label_set,
    batch_input,
    batch_output,
    trns,
    batch_size,
    minibatch_size,
    patch_size,
    next_tag,
):
    total_len, h, w = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size**2))
    n_channels = patch_size**2

    X_shape = [n_patches, minibatch_size, patch_size**2]
    Y_shape = [minibatch_size]

    data_set = data_set.numpy()
    tmp_data_tensor = torch.empty(
        (n_patches, minibatch_size, n_channels), dtype=torch.float32
    )
    tmp_label_tensor = np.empty(minibatch_size, dtype=np.float32)

    x_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    x_distr = [0] * x_traits.grid.nelems

    y_traits = nntile.tensor.TensorTraits(Y_shape, Y_shape)
    y_distr = [0] * y_traits.grid.nelems

    for i in range(n_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                tmp_data_tensor[:, k, :] = grayscale_image_patching(
                    trns(
                        data_set[i * batch_size + j * minibatch_size + k, :, :]
                    ),
                    patch_size,
                )
                tmp_label_tensor[k] = label_set[
                    i * batch_size + j * minibatch_size + k
                ]
            x = nntile.tensor.Tensor_fp32(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            x.from_array(np.asfortranarray(tmp_data_tensor.numpy()))
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(y_traits, y_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(np.asfortranarray(tmp_label_tensor.reshape(-1)))
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)


def cifar_data_loader_to_tensor(
    data_set, label_set, trns, batch_size, minibatch_size, patch_size
):
    total_len, h, w, c = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size**2))
    n_channels = c * (patch_size**2)

    train_tensor = torch.empty(
        (n_batches, n_minibatches, n_patches, minibatch_size, n_channels),
        dtype=torch.float32,
    )
    label_tensor = torch.empty(
        (n_batches, n_minibatches, minibatch_size), dtype=torch.float32
    )
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, :, k, :] = color_image_patching(
                    trns(
                        data_set[
                            i * batch_size + j * minibatch_size + k, :, :, :
                        ]
                    ),
                    patch_size,
                )
                label_tensor[i, j, k] = label_set[
                    i * batch_size + j * minibatch_size + k
                ]
    return train_tensor, label_tensor


def mnist_data_loader_to_tensor(
    data_set, label_set, trns, batch_size, minibatch_size, patch_size
):
    total_len, h, w = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size**2))
    n_channels = patch_size**2

    data_set = data_set.numpy()
    train_tensor = torch.empty(
        (n_batches, n_minibatches, n_patches, minibatch_size, n_channels),
        dtype=torch.float32,
    )
    label_tensor = torch.empty(
        (n_batches, n_minibatches, minibatch_size), dtype=torch.float32
    )
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, :, k, :] = grayscale_image_patching(
                    trns(
                        data_set[i * batch_size + j * minibatch_size + k, :, :]
                    ),
                    patch_size,
                )
                label_tensor[i, j, k] = label_set[
                    i * batch_size + j * minibatch_size + k
                ]
    return train_tensor, label_tensor
