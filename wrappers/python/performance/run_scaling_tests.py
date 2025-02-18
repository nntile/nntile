import os

hidden_size_list = [1024, 2048, 4096, 8192]
backend = "nntile"
device = "cuda"
n_iters = 100
submodule = "attention"
mode = "fwd-bwd"
num_warmup_calls = 5

cmd_string = "STARPU_SILENT=1 STARPU_NCPU=1 STARPU_NCUDA=1 python3 llama_perf.py --config-path=./llama_default_config.json"
cmd_string = cmd_string + " --restrict=" + device + " --mode=" + mode + " --n-iters=" + str(n_iters)
cmd_string = cmd_string + " --num-warmup-calls=" + str(num_warmup_calls) + " --submodule=" + submodule
cmd_string = cmd_string + " --results-folder=.results/" + submodule + "_" + mode
if backend == "torch":
    cmd_string = cmd_string + " --use-torch"
elif backend == "nntile":
    cmd_string = cmd_string + " --use-nntile"
for h_size in hidden_size_list:
    current_cmd = cmd_string + " --hidden-size=" + str(h_size)
    print(current_cmd)
    os.system(current_cmd)