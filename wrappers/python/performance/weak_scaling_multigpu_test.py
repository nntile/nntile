# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/weak_scaling_multigpu_test.py
# Weak scaling test for multi-GPU performance
#
# @version 1.1.0

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--backend",
                    choices=["nntile"],
                    default="nntile")
parser.add_argument("--mode",
                    choices=["fwd", "fwd-bwd"],
                    default="fwd")

parser.add_argument("--device",
                    type=str,
                    default="0")

args = parser.parse_args()
print(args)

backend = args.backend
if args.device == "cpu":
    device = "cpu"
    cuda_device = 0
    num_cuda = 1
else:
    device = "cuda"
    cuda_device = args.device
    num_cuda = len(cuda_device.split(","))
print(num_cuda)

n_iters = 30
mode = args.mode
num_warmup_calls = 10

num_layers = num_cuda
minibatch_size = 1

hidden_size = 16384
seqlen = 4096
intermediatesize = 53248
if num_cuda == 1:
    hidden_size_tiles = [hidden_size // ht for ht in [1, 2, 4]]
    seqlen_tiles = [seqlen // stile for stile in [1, 2]]
    intermsize_tiles = [intermediatesize // itile for itile in [1, 2, 4, 8]]
elif num_cuda == 4:
    hidden_size_tiles = [hidden_size // ht for ht in [2]]
    seqlen_tiles = [seqlen // stile for stile in [4]]
    intermsize_tiles = [intermediatesize // itile for itile in [1]]
elif num_cuda == 2:
    hidden_size_tiles = [hidden_size // ht for ht in [2]]
    seqlen_tiles = [seqlen // stile for stile in [4]]
    intermsize_tiles = [intermediatesize // itile for itile in [4]]

config_path = "./llama_405b_config.json"
model_name = config_path.split("/")[1][:-5]

cmd_string = "STARPU_ENABLE_STATS=1 STARPU_PROFILING=1 STARPU_SCHED=dmdasd "
cmd_string = cmd_string + "CUDA_VISIBLE_DEVICES={} STARPU_WORKERS_NOBIND=1 " \
    "STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA={}" \
    .format(cuda_device, num_cuda)
cmd_string = cmd_string + " python3 llama_perf.py --config-path=" + config_path
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + \
    " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + \
    " --submodule=llama"
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
elif backend == "torch-compile":
    cmd_string = cmd_string + " --use-torch --torch-compile"
current_cmd = cmd_string + " --seq-len={}".format(seqlen)
current_cmd = current_cmd + " --hidden-size={}".format(hidden_size)
current_cmd = current_cmd + " --num-layers={}".format(num_layers)
current_cmd = current_cmd + " --minibatch-size={}".format(minibatch_size)
current_cmd = current_cmd + " --minibatch-size-tile={}".format(1)
current_cmd = current_cmd + " --results-folder=.results/weak_scaling/gpu" + \
    str(num_cuda) + "/" + model_name + \
    "/num-layers_{}_minibatch-size_{}_hsize_{}" \
    .format(num_layers, minibatch_size, hidden_size) + \
    "/seqlen_{}".format(seqlen)

for hsize_tile in hidden_size_tiles:
    current_cmd_h = current_cmd + " --hidden-size-tile=" + str(hsize_tile)
    for slen_tile in seqlen_tiles:
        current_cmd_h_seqlen = current_cmd_h + " --seq-len-tile=" + \
            str(slen_tile)
        for intermtile in intermsize_tiles:
            current_cmd_h_seqlen_interm = current_cmd_h_seqlen + \
                " --intermediate-size-tile=" + str(intermtile)
            print(current_cmd_h_seqlen_interm)
            os.system(current_cmd_h_seqlen_interm)
