# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# Test for comparison of torch and NNTile versions of MLP-Mixer model
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2023-11-16

# All necesary imports
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dts 
import torchvision.transforms as trnsfrms
import nntile
from nntile.torch_models.mlp_mixer import MlpMixer, image_patching_rgb
from nntile.model.mlp_mixer import MlpMixer as MlpMixerTile
import pathlib
import os, sys


def image_patching(image, patch_size):
    c, h, w = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("patch size should be divisor of both image height and width")
    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)
    patched_batch = torch.empty((n_patches, n_channels), dtype=image.dtype)

    n_y = int(w / patch_size)

    for i in range(n_patches):
        x = i // n_y
        y = i % n_y

        for clr in range(c):
            vect_patch = image[clr, x * patch_size: (x+1) * patch_size , y * patch_size: (y+1) * patch_size].flatten()
            patched_batch[i, clr * (patch_size ** 2) : (clr+1) * (patch_size ** 2)] = vect_patch
    return patched_batch


def data_loader_to_tensor(data_set, label_set, trns, batch_size, minibatch_size, patch_size):
    total_len, h, w, c = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)

    n_patches = int(h * w / (patch_size ** 2))
    n_channels = c * (patch_size ** 2)
  
    train_tensor = torch.empty((n_batches, n_minibatches, n_patches, minibatch_size, n_channels), dtype=torch.float32)
    label_tensor = torch.empty((n_batches, n_minibatches, minibatch_size), dtype=torch.float32)
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, :, k, :] = image_patching(trns(data_set[i * batch_size + j * minibatch_size + k, :, :, :]), patch_size)
                label_tensor[i, j, k] = label_set[i * batch_size + j * minibatch_size + k]
    return train_tensor, label_tensor


# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()


def helper():
    patch_size = 16
    batch_size = 32
    minibatch_size = 16
    channel_size = int(32 * 32 / patch_size ** 2)
    hidden_dim = 2048
    num_mixer_layers = 10
    num_classes = 10
    num_clr_channels = 3

    lr = 1e-2
    next_tag = 0
    tol = 1e-4

    cur_pos = str(pathlib.Path(__file__).parent.absolute())

    init_state_path = os.path.join(cur_pos, "mlp_mixer_init_state.pt")
    final_state_path = os.path.join(cur_pos, "mlp_mixer_final_state.pt")

    X_shape = [channel_size, minibatch_size, num_clr_channels * patch_size ** 2]
    Y_shape = [minibatch_size]

    torch_mixer_model = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes)
    optim_torch = torch.optim.SGD(torch_mixer_model.parameters(), lr=lr)
    crit_torch = nn.CrossEntropyLoss(reduction="sum")

    trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

    train_set = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=True, download=False, transform=trnsform)
    test_set = dts.CIFAR10(root='/mnt/local/dataset/by-domain/cv/CIFAR10/', train=False, download=False, transform=trnsform)
    

    train_data_tensor, train_labels = data_loader_to_tensor(train_set.data, train_set.targets, trnsform, batch_size, minibatch_size, patch_size)
    train_labels = train_labels.type(torch.LongTensor) 

    test_data_tensor, test_labels = data_loader_to_tensor(test_set.data, test_set.targets, trnsform, batch_size, minibatch_size, patch_size)
    test_labels = test_labels.type(torch.LongTensor) 
    num_batch_train, num_minibatch_train = train_data_tensor.shape[0], train_data_tensor.shape[1]

    torch_mixer_model.zero_grad()

    torch.save({
            'model_state_dict': torch_mixer_model.state_dict(),
            'optimizer_state_dict': optim_torch.state_dict(),
            }, init_state_path)

    # num_batch_to_go = 3
    # for batch_iter in range(num_batch_to_go):
    #     torch_train_loss = torch.zeros(1, dtype=torch.float32)
    #     for minibatch_iter in range(num_minibatch_train):
    #         patched_train_sample = train_data_tensor[batch_iter,minibatch_iter,:,:,:]
    #         true_labels = train_labels[batch_iter, minibatch_iter, :]
    #         torch_output = torch_mixer_model(patched_train_sample)
    #         loss_local = crit_torch(torch_output, true_labels)
    #         loss_local.backward()
    #         torch_train_loss += loss_local            
    #     optim_torch.step()
    #     torch_mixer_model.zero_grad()


    # torch.save({
    #         'model_state_dict': torch_mixer_model.state_dict(),
    #         'optimizer_state_dict': optim_torch.state_dict(),
    #         }, final_state_path)
    

    patched_test_sample = test_data_tensor[1,1,:,:,:]
    true_labels = train_labels[1, 1, :]

    torch_mixer_model_loaded = MlpMixer(channel_size, num_clr_channels * patch_size ** 2, hidden_dim, num_mixer_layers, num_classes)
    checkpoint = torch.load(init_state_path)
    torch_mixer_model_loaded.load_state_dict(checkpoint['model_state_dict'])

    nntile_mixer_model, next_tag = MlpMixerTile.from_torch(torch_mixer_model_loaded,minibatch_size,num_classes, next_tag)
    crit_nntile, next_tag = nntile.loss.CrossEntropy.generate_simple(nntile_mixer_model.activations[-1], next_tag)

    torch_mixer_model_loaded.zero_grad()
    torch_output = torch_mixer_model_loaded(patched_test_sample)
    _, pred_labels_torch = torch.max(torch_output, 1)

    np_torch_output = np.array(torch_output.detach().numpy(), order="F", dtype=np.float32)
    loss_local = crit_torch(torch_output, true_labels)
    loss_local.backward()


    data_train_traits = nntile.tensor.TensorTraits(X_shape, X_shape)
    test_tensor = nntile.tensor.Tensor_fp32(data_train_traits, [0], next_tag)
    next_tag = test_tensor.next_tag
    test_tensor.from_array(patched_test_sample.numpy())
    nntile.tensor.copy_async(test_tensor, nntile_mixer_model.activations[0].value)

    label_train_traits = nntile.tensor.TensorTraits(Y_shape, Y_shape)
    label_train_tensor = nntile.tensor.Tensor_int64(label_train_traits, [0], next_tag)
    next_tag = label_train_tensor.next_tag
    label_train_tensor.from_array(true_labels.numpy())
    nntile.tensor.copy_async(label_train_tensor, crit_nntile.y)    

    nntile_mixer_model.clear_gradients()
    nntile_mixer_model.forward_async()
    crit_nntile.calc_async()

    nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
    crit_nntile.get_val(nntile_xentropy_np)
    nntile_xentropy_np = nntile_xentropy_np.reshape(-1)
    

    nntile_last_layer_output = np.zeros(nntile_mixer_model.activations[-1].value.shape, order="F", dtype=np.float32)
    nntile_mixer_model.activations[-1].value.to_array(nntile_last_layer_output)
    pred_labels_nntile = np.argmax(nntile_last_layer_output, axis=0)

    print("PyTorch loss: {}, NNTile loss: {}".format(loss_local.item(), nntile_xentropy_np[0]))
    # print("PyTorch predicted label: {}".format(pred_labels_torch))
    # print("NNTile predicted label: {}".format(pred_labels_nntile))
    print("Norm of inference difference: {}".format(np.linalg.norm(nntile_last_layer_output - np_torch_output.T, 'fro')))

    nntile_mixer_model.backward_async()

    for i, (p_torch, p_nntile) in enumerate(zip(torch_mixer_model_loaded.parameters(), nntile_mixer_model.parameters)):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
        p_nntile.grad.to_array(p_nntile_grad_np)
        if p_torch.grad.shape[0] != p_nntile_grad_np.shape[0]:
                p_nntile_grad_np = np.transpose(p_nntile_grad_np)
        
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
        if rel_error > tol:
            crit_nntile.unregister()
            test_tensor.unregister()
            nntile_mixer_model.unregister()
            return False
            
    crit_nntile.unregister()
    test_tensor.unregister()
    nntile_mixer_model.unregister()
    print("Test successful")
    assert True


# Test runner
def test():
    helper()


if __name__ == "__main__":
    test()
