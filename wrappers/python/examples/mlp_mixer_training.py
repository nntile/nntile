# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/mlp_mixer_training.py
# MLP-Mixer image classification training example (local model only)
#
# @version 1.1.0

import argparse
import os
import time

import numpy as np
import torch
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
from mlp_mixer_data_preparation import (
    DTYPE_TO_ACTIVATION_TENSOR, cifar_data_loader_to_nntile,
    mnist_data_loader_to_nntile,
)

import nntile
from nntile.model.mlp_mixer import MlpMixer
from nntile.model.mlp_mixer_config import MlpMixerConfig
from nntile.torch_models.mlp_mixer import MlpMixer as TorchMlpMixer

_DATASETS = {
    "mnist": {"image_size": 28, "num_channels": 1, "n_classes": 10},
    "fashion_mnist": {"image_size": 28, "num_channels": 1, "n_classes": 10},
    "cifar10": {"image_size": 32, "num_channels": 3, "n_classes": 10},
}
_GRAYSCALE_DATASETS = frozenset({"mnist", "fashion_mnist"})


def _env_starpu_profiling_on() -> bool:
    """True when STARPU_PROFILING=1 is set before Context (notebook env)."""
    return os.environ.get("STARPU_PROFILING", "0") == "1"

parser = argparse.ArgumentParser(
    prog="MLP-Mixer image classifier",
    description="Train a locally initialized MLP-Mixer on MNIST, Fashion-MNIST, "
    "or CIFAR-10 "
    "with NNTile (no remote/HF model). Like gpt_neo_training with "
    "pretrained=local: start from random weights, or pass "
    "--checkpoint-path to load weights produced by --save-checkpoint-path "
    "(dict with key model_state_dict) and continue training.",
)
parser.add_argument("--dataset", choices=list(_DATASETS), default="mnist")
parser.add_argument("--data-root", type=str, default="./data")

parser.add_argument("--batch-size", type=int, default=60)
parser.add_argument("--minibatch-size", type=int, default=3)
parser.add_argument("--patch-size", type=int, default=7)
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="Projected patch dimension (MLP-Mixer width)")
parser.add_argument("--num-mixer-layers", type=int, default=8)

parser.add_argument(
    "--checkpoint-path",
    type=str,
    default="",
    help="Path to a .pt file from a previous run (model_state_dict); "
    "architecture flags must match (--patch-size, --hidden-dim, ...).",
)
parser.add_argument("--save-checkpoint-path", type=str, default="mlp_mixer.pt")

parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nepochs", type=int, default=1)

parser.add_argument(
    "--dtype",
    choices=["fp32", "tf32", "bf16", "fp32_fast_tf32"],
    default="fp32",
    help="NNTile compute dtype; tf32 is Tensor_fp32_fast_tf32.",
)
parser.add_argument("--restrict", choices=["cpu", "cuda", None], default=None)

parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str, default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

args = parser.parse_args()
print(args)

assert args.batch_size > 0
assert args.minibatch_size > 0
assert args.batch_size % args.minibatch_size == 0
assert args.nepochs > 0
assert args.patch_size > 0
if args.dtype not in DTYPE_TO_ACTIVATION_TENSOR:
    raise ValueError(f"Unsupported dtype {args.dtype}")
if args.dtype == "bf16" and args.restrict != "cuda":
    raise ValueError(
        "bf16 uses Tensor_bf16; run with --restrict cuda (CPU bf16 is not "
        "supported for this example).",
    )

ds = _DATASETS[args.dataset]
image_size = ds["image_size"]
num_channels = ds["num_channels"]
n_classes = ds["n_classes"]
if image_size % args.patch_size != 0:
    raise ValueError(
        f"Image side {image_size} must be divisible by patch size "
        f"{args.patch_size}"
    )

channel_dim = int(image_size * image_size / args.patch_size**2)
init_patch_dim = num_channels * args.patch_size**2
num_minibatch = args.batch_size // args.minibatch_size

# Local PyTorch reference model (weights source for NNTile)
torch_model = TorchMlpMixer(
    channel_dim,
    init_patch_dim,
    args.hidden_dim,
    args.num_mixer_layers,
    n_classes,
)
if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    torch_model.load_state_dict(checkpoint["model_state_dict"])
print(torch_model)

# StarPU + NNTile
time0 = time.time()
context = nntile.Context(
    ncpu=-1,
    ncuda=-1,
    ooc=0,
    logger=args.logger,
    logger_addr=args.logger_server_addr,
    logger_port=args.logger_server_port,
    verbose=0,
)
if args.restrict == "cuda":
    context.restrict_cuda()
elif args.restrict == "cpu":
    context.restrict_cpu()
# Do not mix env STARPU_PROFILING=1 with profiling_enable/disable here:
# that triggers worker profiling_registered_start asserts on exit.
if not _env_starpu_profiling_on():
    nntile.starpu.profiling_init()
    nntile.starpu.profiling_disable()
print("StarPU + NNTile init in {:.3f} s".format(time.time() - time0))

config = MlpMixerConfig(
    channel_dim=channel_dim,
    init_patch_dim=init_patch_dim,
    projected_patch_dim=args.hidden_dim,
    num_mixer_layers=args.num_mixer_layers,
    n_classes=n_classes,
    dtype=args.dtype,
)

time0 = time.time()
model = MlpMixer.from_torch(
    torch_model, args.minibatch_size, n_classes, config,
)
print("PyTorch -> NNTile in {:.3f} s".format(time.time() - time0))
del torch_model

# Image batches: [n_patches, minibatch, patch_vector] and class labels
transform = trnsfrms.Compose([trnsfrms.ToTensor()])
if args.dataset == "mnist":
    train_set = dts.MNIST(
        root=args.data_root, train=True, download=True,
        transform=transform,
    )
elif args.dataset == "fashion_mnist":
    train_set = dts.FashionMNIST(
        root=args.data_root, train=True, download=True,
        transform=transform,
    )
else:
    train_set = dts.CIFAR10(
        root=args.data_root, train=True, download=True,
        transform=transform,
    )

batch_input = []
batch_output = []
loader = (
    mnist_data_loader_to_nntile
    if args.dataset in _GRAYSCALE_DATASETS
    else cifar_data_loader_to_nntile
)
time0 = time.time()
loader(
    train_set.data,
    train_set.targets,
    batch_input,
    batch_output,
    transform,
    args.batch_size,
    args.minibatch_size,
    args.patch_size,
    activation_dtype=args.dtype,
)
num_batches = len(batch_input)
print("Dataset -> NNTile batches in {:.3f} s ({} batches)".format(
    time.time() - time0, num_batches,
))

if args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(model.get_parameters(), args.lr)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(model.get_parameters(), args.lr)
else:
    optimizer = nntile.optimizer.SGD(model.get_parameters(), args.lr)

loss = nntile.loss.CrossEntropy.generate_simple(
    model.activations[-1],
    scale=1.0 / args.minibatch_size,
)

pipeline = nntile.pipeline.Pipeline(
    batch_input, batch_output, model, optimizer, loss, args.nepochs,
)
pipeline.print_meminfo()

nntile.starpu.wait_for_all()
time0 = time.time()
if not _env_starpu_profiling_on():
    nntile.starpu.profiling_enable()
pipeline.train_async()
nntile.starpu.wait_for_all()
if not _env_starpu_profiling_on():
    nntile.starpu.profiling_disable()
train_time = time.time() - time0
print("Training time: {:.3f} s".format(train_time))
print("Throughput: {:.1f} images/s".format(
    args.nepochs * num_batches * args.batch_size / train_time,
))

loss_np = np.zeros((1,), dtype=np.float32)
loss.val.to_array(loss_np)
print("Loss on last batch: {}".format(loss_np[0]))

nflops = model.get_flops_forward() + model.get_flops_backward()
print("Performance: {:.3f} Tflops/s".format(
    nflops * args.nepochs * num_batches * num_minibatch / train_time * 1e-12,
))

torch_model = model.to_torch()
torch.save({"model_state_dict": torch_model.state_dict()},
           args.save_checkpoint_path)
del torch_model

loss.unregister()
optimizer.unregister()
for batch in batch_input + batch_output:
    for t in batch:
        t.unregister()
model.unregister()
