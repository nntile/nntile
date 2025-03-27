import os
import argparse

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

parser.add_argument("--submodule",
                    choices=["attention", "decoder", "mlp", "causal-llama"],
                    default="attention")

args = parser.parse_args()
print(args)

# seq_len_list = [1024 * i for i in range(4, 5)]#[1024, 2048, 3072, 4096]
hidden_size_list = [1024 * i for i in range(10, 17, 2)]
intermediate_size_list = [4 * h_size for h_size in hidden_size_list]
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

config_path = "./llama_1.3b_config.json"
model_name = config_path.split("/")[1][:-5]

cmd_string = "STARPU_MAX_MEMORY_USE=1 STARPU_ENABLE_STATS=1 STARPU_PROFILING=1 STARPU_SCHED=dmdasd "
cmd_string = cmd_string + "CUDA_VISIBLE_DEVICES={} STARPU_WORKERS_NOBIND=1 STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA={}".format(cuda_device, num_cuda)
cmd_string = cmd_string + " python3 llama_perf.py --config-path=" + config_path
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + " --submodule=" + args.submodule
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
elif backend == "torch-compile":
    cmd_string = cmd_string + " --use-torch --torch-compile"
current_cmd = cmd_string + " --seq-len=4096"

for j, hidden_size in enumerate(hidden_size_list):
    current_cmd = current_cmd + " --hidden-size=" + str(hidden_size)
    current_cmd = current_cmd + " --results-folder=.results/gpu" + str(num_cuda) + "/" + args.submodule + "_" + mode
    current_cmd = current_cmd +"/hsize_" + str(hidden_size)
    hidden_size_tiles = [hidden_size // (2 ** i) for i in range(0, 3)]
    intermediate_size_tiles = [intermediate_size_list[j] // (2 ** i) for i in range(0, 3)]
    for i, hsize_tile in enumerate(hidden_size_tiles):
        current_cmd_h  = current_cmd + " --hidden-size-tile=" + str(hsize_tile)
        current_cmd_h = current_cmd_h + " --intermediate-size-tile=" + str(intermediate_size_tiles[i])
        current_cmd_h = current_cmd_h + " --intermediate-size=" + str(intermediate_size_list[j])
        os.system(current_cmd_h)