import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--backend",
                    choices=["nntile", "torch", "torch-compile"],
                    default="torch")
parser.add_argument("--mode",
                    choices=["fwd", "fwd-bwd"],
                    default="fwd")
parser.add_argument("--submodule",
                    choices=["attention", "decoder", "mlp"],
                    default="attention")

parser.add_argument("--device",
                    choices=["cpu", "cuda:0", "cuda:1", "cuda:3"],
                    default="cuda:0")

args = parser.parse_args()
print(args)

hidden_size_list = [512 * i for i in range(1, 31)]
seq_len_list = [1024, 2048, 3072, 4096]
backend = args.backend
if args.device == "cpu":
    device = "cpu"
    num_cuda = 0
else:
    device, num_cuda = args.device.split(":")
    num_cuda = int(num_cuda)

n_iters = 100
submodule = args.submodule
mode = args.mode
num_warmup_calls = 5

cmd_string = "CUDA_VISIBLE_DEVICES={} STARPU_WORKERS_NOBIND=1 STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=1".format(num_cuda)
cmd_string = cmd_string + " python3 llama_perf.py --config-path=./llama_default_config.json"
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + " --submodule=" + submodule
# cmd_string = cmd_string + " --results-folder=.results/" + submodule + "_" + mode
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
elif backend == "torch-compile":
    cmd_string = cmd_string + " --use-torch --torch-compile"
for seq_len in seq_len_list:
    current_cmd = cmd_string + " --seq-len=" + str(seq_len)
    current_cmd = current_cmd + " --results-folder=.results/fixed_numheads/" + submodule + "_" + mode
    current_cmd = current_cmd +"/seq-len_" + str(seq_len)
    for h_size in hidden_size_list:
        current_cmd_h = current_cmd + " --hidden-size=" + str(h_size)
        print(current_cmd_h)
        os.system(current_cmd_h)