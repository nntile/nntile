import os

hidden_size_list = [512 * i for i in range(1, 11)]
seq_len_list = [1024, 2048, 3072, 4096]
head_dims = [64, 128, 256]
backend = "nntile"
device = "cuda"
n_iters = 100
submodule = "attention"
mode = "fwd-bwd"
num_warmup_calls = 5

cmd_string = "CUDA_VISIBLE_DEVICES=1 STARPU_WORKERS_NOBIND=1 STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=1 python3 llama_perf.py --config-path=./llama_default_config.json"
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + " --submodule=" + submodule
# cmd_string = cmd_string + " --results-folder=.results/" + submodule + "_" + mode
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
for seq_len in seq_len_list:
    current_cmd = cmd_string + " --seq-len=" + str(seq_len)
    current_cmd = current_cmd + " --results-folder=.results/" + submodule + "_" + mode
    current_cmd = current_cmd +"/seq-len_" + str(seq_len)
    for h_d in head_dims:
        current_cmd_hd = current_cmd + "/head-dim_" + str(h_d) + " --head-dim=" + str(h_d)
        for h_size in hidden_size_list:
            current_cmd_h = current_cmd_hd + " --hidden-size=" + str(h_size)
            print(current_cmd_h)
            os.system(current_cmd_h)