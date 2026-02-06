# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/mlp_multigpu_scaling_test.py
# Multi-GPU scaling test for MLP layer
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

n_iters = 20
mode = args.mode
num_warmup_calls = 5

config_path = "./llama_405b_config.json"
model_name = config_path.split("/")[1][:-5]

cmd_string = "STARPU_MAX_MEMORY_USE=1 STARPU_ENABLE_STATS=1 " \
    "STARPU_PROFILING=1 STARPU_SCHED=dmdasd "
cmd_string = cmd_string + "CUDA_VISIBLE_DEVICES={} STARPU_WORKERS_NOBIND=1 " \
    "STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA={}" \
    .format(cuda_device, num_cuda)
cmd_string = cmd_string + " python3 llama_perf.py --config-path=" + config_path
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + \
    " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + \
    " --submodule=mlp"
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
elif backend == "torch-compile":
    cmd_string = cmd_string + " --use-torch --torch-compile"
current_cmd = cmd_string + " --seq-len=4096"
current_cmd = current_cmd + " --hidden-size=-1"
current_cmd = current_cmd + " --results-folder=.results/gpu" + \
    str(num_cuda) + "/" + model_name + "/mlp_" + mode

hidden_size = 16384
intermediate_size = 53248

hidden_size_tiles = [hidden_size // (2 ** i) for i in range(0, 5)]
intermediate_size_tiles = [intermediate_size // 52]

for j, hsize_tile in enumerate(hidden_size_tiles):
    current_cmd_h = current_cmd + " --hidden-size-tile=" + str(hsize_tile)
    for i, interm_size_tile in enumerate(intermediate_size_tiles):
        current_cmd_hinterm = current_cmd_h + " --intermediate-size-tile=" + \
            str(interm_size_tile)
        os.system(current_cmd_hinterm)
