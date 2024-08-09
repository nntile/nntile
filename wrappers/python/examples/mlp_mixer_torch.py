# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/mlp_mixer_torch.py
# Example for torch version of MLP-Mixer model
#
# @version 1.1.0

import argparse
# All necesary imports
import pathlib

import torch
import torch.nn as nn
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms

from nntile.torch_models.mlp_mixer import MlpMixer


def image_patching(image, patch_size):
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


def data_loader_to_tensor_rgb(
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
                train_tensor[i, j, :, k, :] = image_patching(
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


def data_loader_to_tensor(
    data_set, label_set, trns, batch_size, minibatch_size
):
    total_len, h, w = data_set.shape
    n_batches = total_len // batch_size
    n_minibatches = int(batch_size / minibatch_size)
    data_set = data_set.numpy()
    label_set = label_set.numpy()
    train_tensor = torch.empty(
        (n_batches, n_minibatches, minibatch_size, h, w, 1),
        dtype=torch.float32,
    )
    label_tensor = torch.empty(
        (n_batches, n_minibatches, minibatch_size), dtype=torch.float32
    )
    for i in range(n_batches):
        for j in range(n_minibatches):
            for k in range(minibatch_size):
                train_tensor[i, j, k, :, :, 0] = trns(
                    data_set[i * batch_size + j * minibatch_size + k, :, :]
                )
                label_tensor[i, j, k] = label_set[
                    i * batch_size + j * minibatch_size + k
                ]
    return train_tensor, label_tensor


def evaluate(torch_model, test_data_tensor, test_label_tensor):
    num_batch_test, num_minibatch_test = (
        test_data_tensor.shape[0],
        test_data_tensor.shape[1],
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        torch_test_loss = torch.zeros(1, dtype=torch.float32).to(device)
        for test_batch_iter in range(num_batch_test):
            for test_minibatch_iter in range(num_minibatch_test):
                patched_test_sample = test_data_tensor[
                    test_batch_iter, test_minibatch_iter, :, :, :
                ]
                patched_test_sample = patched_test_sample.to(device)
                true_test_labels = test_label_tensor[
                    test_batch_iter, test_minibatch_iter, :
                ].to(device)

                torch_output = torch_model(patched_test_sample)

                _, predictions = torch.max(torch_output, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(true_test_labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

                loss_local = crit_torch(torch_output, true_test_labels)
                torch_test_loss += loss_local
        test_loss = torch_test_loss.item() / (
            num_batch_test * num_minibatch_test
        )
    print("Avg test loss: {}".format(test_loss))
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
    print(
        "Total accuracy = {}".format(
            sum(correct_pred.values()) / sum(total_pred.values())
        )
    )


# Instantiate the parser
parser = argparse.ArgumentParser(
    prog="DeepReLU neural network",
    description="This example trains PyTorch version of MlpMixer neural "
    "network from a scratch for an image classification task",
)
parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
parser.add_argument("--batch", type=int)
parser.add_argument("--minibatch", type=int)
parser.add_argument("--depth", type=int)
parser.add_argument("--patch_size", type=int)
parser.add_argument("--hidden_dim", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--test_each_batch", action=argparse.BooleanOptionalAction)
parser.add_argument("--test_each_epoch", action=argparse.BooleanOptionalAction)
parser.add_argument("--no_save", action="store_true", default=False)


args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

cur_pos = pathlib.Path(__file__).parent.absolute()
experiment_folder = "mlp_mixer_data"

init_state_path = cur_pos / experiment_folder / "torch_init_state.pt"
final_state_path = cur_pos / experiment_folder / "torch_final_state.pt"
final_nntile_path = cur_pos / experiment_folder / "nntile_final_state.pt"
# Parse arguments
args = parser.parse_args()
print(args)

batch_size = args.batch
minibatch_size = args.minibatch
patch_size = args.patch_size
num_mixer_layers = args.depth
hidden_dim = args.hidden_dim
lr = args.lr
num_epochs = args.epoch

trnsform = trnsfrms.Compose([trnsfrms.ToTensor()])

# Check consistency of batch and minibatch sizes
if batch_size % minibatch_size != 0:
    raise ValueError("Batch must consist of integer number of minibatches")
accumulation_steps = int(batch_size / minibatch_size)

if args.dataset == "mnist":
    train_set = dts.MNIST(
        root="./", train=True, download=True, transform=trnsform
    )
    test_set = dts.MNIST(
        root="./", train=False, download=True, transform=trnsform
    )
    n_classes = 10
    num_clr_channels = 1
    # Check consistency of image and patch sizes
    if 28 % patch_size != 0:
        raise ValueError(
            "Image size must be divisible by patch size without remainder"
        )
    channel_size = int(28 * 28 / patch_size**2)
elif args.dataset == "cifar10":
    train_set = dts.CIFAR10(
        root="/mnt/local/dataset/by-domain/cv/CIFAR10/",
        train=True,
        download=False,
        transform=trnsform,
    )
    test_set = dts.CIFAR10(
        root="/mnt/local/dataset/by-domain/cv/CIFAR10/",
        train=False,
        download=False,
    )
    n_classes = 10
    num_clr_channels = 3
    # Check consistency of image and patch sizes
    if 32 % patch_size != 0:
        raise ValueError(
            "Image size must be divisible by patch size without remainder"
        )
    channel_size = int(32 * 32 / patch_size**2)
else:
    raise ValueError("{} dataset is not supported yet!".format(args.dataset))

mlp_mixer_model = MlpMixer(
    channel_size,
    num_clr_channels * patch_size**2,
    hidden_dim,
    num_mixer_layers,
    n_classes,
)
optim_torch = torch.optim.Adam(mlp_mixer_model.parameters(), lr=lr)
crit_torch = nn.CrossEntropyLoss(reduction="sum")

train_data_tensor, train_labels = data_loader_to_tensor_rgb(
    train_set.data,
    train_set.targets,
    trnsform,
    batch_size,
    minibatch_size,
    patch_size,
)
train_labels = train_labels.type(torch.LongTensor)
num_batch_train, num_minibatch_train = (
    train_data_tensor.shape[0],
    train_data_tensor.shape[1],
)

test_data_tensor, test_labels = data_loader_to_tensor_rgb(
    test_set.data,
    test_set.targets,
    trnsform,
    batch_size,
    minibatch_size,
    patch_size,
)
test_labels = test_labels.type(torch.LongTensor)
# checkpoint = torch.load(final_nntile_path)
# mlp_mixer_model.load_state_dict(checkpoint['model_state_dict'])
# evaluate(mlp_mixer_model, test_data_tensor, test_labels)

if not args.no_save:
    torch.save(
        {
            "model_state_dict": mlp_mixer_model.state_dict(),
            "optimizer_state_dict": optim_torch.state_dict(),
        },
        init_state_path,
    )
mlp_mixer_model = mlp_mixer_model.to(device)
mlp_mixer_model.zero_grad()
for epoch_counter in range(num_epochs):
    for batch_iter in range(num_batch_train):
        torch_train_loss = torch.zeros(1, dtype=torch.float32).to(device)
        for minibatch_iter in range(num_minibatch_train):
            patched_train_sample = train_data_tensor[
                batch_iter, minibatch_iter, :, :, :
            ]
            patched_train_sample = patched_train_sample.to(device)
            true_labels = train_labels[batch_iter, minibatch_iter, :].to(
                device
            )
            torch_output = mlp_mixer_model(patched_train_sample)
            loss_local = crit_torch(torch_output, true_labels)
            loss_local.backward()
            torch_train_loss += loss_local
        optim_torch.step()
        mlp_mixer_model.zero_grad()
        print(
            "Epoch: {}, batch {} out of {}, train loss: {}".format(
                epoch_counter,
                batch_iter,
                train_data_tensor.shape[0],
                torch_train_loss.item(),
            )
        )

        if args.test_each_batch:
            evaluate(mlp_mixer_model, test_data_tensor, test_labels)

    if args.test_each_epoch:
        evaluate(mlp_mixer_model, test_data_tensor, test_labels)
if not args.no_save:
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": mlp_mixer_model.state_dict(),
            "optimizer_state_dict": optim_torch.state_dict(),
        },
        final_state_path,
    )
